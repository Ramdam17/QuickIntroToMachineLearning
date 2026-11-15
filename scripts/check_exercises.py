"""Verify the exercises of the notebooks: their form, and that each solution passes its check.

The charter (``docs/notebook_template.md``) gives every exercise three cells, in this order: a
code cell to complete, whose first line is ``# Exercise N`` and whose missing part is marked
``# YOUR CODE``; a code cell that calls ``ml_course.check``; a markdown cell holding the solution,
collapsed under "Show the solution". This guard does three things for each notebook:

1. it verifies that form, and that the solution block starts with the same ``# Exercise N`` line;
2. it verifies that no later cell uses a name defined in a cell to complete, which is what lets
   the notebook run with its exercises unfilled;
3. it puts each solution in place of its cell to complete, executes the notebook on a fresh
   kernel, and verifies that every check passes.

The notebook files are never modified. Run before committing::

    uv run python scripts/check_exercises.py "11_MLP/*.ipynb"

Exit code 0 means every exercise is well formed and every solution passes, 1 means at least one
problem was found (a notebook without any exercise is one), 2 means no notebook matched.
"""

from __future__ import annotations

import argparse
import ast
import logging
import os
import re
import symtable
import sys
from dataclasses import dataclass
from pathlib import Path

import nbformat
from run_notebooks import DEFAULT_CELL_TIMEOUT_S, execute_notebook

from ml_course.check import STRICT_ENV

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"
DEFAULT_PATTERN = "*/*.ipynb"

PLACEHOLDER = "# YOUR CODE"
HEADER = re.compile(r"^# Exercise (\d+)$")
CHECK_CALL = re.compile(r"\bcheck\.(equal|returns)\(")
SOLUTION_SUMMARY = "<summary>Show the solution</summary>"
PYTHON_BLOCK = re.compile(r"```python\n(.*?)\n```", re.DOTALL)
# First characters of the IPython lines (magics, shell calls) that are not Python syntax.
IPYTHON_PREFIXES = ("%", "!")

logger = logging.getLogger("check_exercises")


@dataclass(frozen=True)
class Exercise:
    """One well-formed exercise of a notebook.

    Attributes
    ----------
    number : int
        The N of its ``# Exercise N`` line.
    cell : int
        Zero-based index of the cell to complete. The check is at ``cell + 1`` and the solution
        at ``cell + 2``.
    solution : str
        The Python block of the solution: the whole cell to complete, filled in.
    """

    number: int
    cell: int
    solution: str


def _first_line(source: str) -> str:
    return source.split("\n", 1)[0].strip()


def find_exercises(nb: nbformat.NotebookNode) -> tuple[list[Exercise], list[str]]:
    """Find the exercises of a notebook and the problems of form.

    Parameters
    ----------
    nb : nbformat.NotebookNode
        The notebook, as read by ``nbformat``.

    Returns
    -------
    exercises : list of Exercise
        The exercises whose three cells are well formed, in notebook order.
    problems : list of str
        One sentence per problem of form.
    """
    exercises: list[Exercise] = []
    problems: list[str] = []
    seen: set[int] = set()
    claimed_solutions: set[int] = set()
    cells = nb.cells

    for index, cell in enumerate(cells):
        if cell.cell_type != "code" or PLACEHOLDER not in cell.source:
            continue
        header = HEADER.match(_first_line(cell.source))
        if header is None:
            problems.append(f"[cell {index}] a cell to complete must start with `# Exercise N`")
            continue
        number = int(header.group(1))
        if number in seen:
            problems.append(f"[cell {index}] `# Exercise {number}` is used twice")
            continue
        seen.add(number)

        check = cells[index + 1] if index + 1 < len(cells) else None
        if check is None or check.cell_type != "code" or not CHECK_CALL.search(check.source):
            problems.append(
                f"[cell {index}] Exercise {number} must be followed by a code cell that calls "
                "`check.equal` or `check.returns`"
            )
            continue

        solution = cells[index + 2] if index + 2 < len(cells) else None
        if (
            solution is None
            or solution.cell_type != "markdown"
            or SOLUTION_SUMMARY not in solution.source
            or "<details>" not in solution.source
        ):
            problems.append(
                f"[cell {index}] the check of Exercise {number} must be followed by a markdown "
                f"cell with `<details>` and `{SOLUTION_SUMMARY}`"
            )
            continue
        claimed_solutions.add(index + 2)

        blocks = PYTHON_BLOCK.findall(solution.source)
        if len(blocks) != 1:
            problems.append(
                f"[cell {index + 2}] the solution of Exercise {number} must hold exactly one "
                f"Python block (found {len(blocks)})"
            )
            continue
        block = blocks[0]
        if _first_line(block) != _first_line(cell.source):
            problems.append(
                f"[cell {index + 2}] the solution block must start with `# Exercise {number}`"
            )
            continue
        if PLACEHOLDER in block:
            problems.append(
                f"[cell {index + 2}] the solution of Exercise {number} still holds `{PLACEHOLDER}`"
            )
            continue
        exercises.append(Exercise(number=number, cell=index, solution=block))

    for index, cell in enumerate(cells):
        if (
            cell.cell_type == "markdown"
            and SOLUTION_SUMMARY in cell.source
            and index not in claimed_solutions
        ):
            problems.append(f"[cell {index}] a solution with no well-formed exercise before it")
    return exercises, problems


def _python_source(source: str) -> str:
    """Return a code cell without its IPython lines (magics, shell calls)."""
    return "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith(IPYTHON_PREFIXES)
    )


def _top_level_names(source: str) -> set[str]:
    """Return the names a cell defines at its top level; empty when it is not valid Python."""
    try:
        table = symtable.symtable(_python_source(source), "<cell>", "exec")
    except SyntaxError:
        return set()
    return {symbol.get_name() for symbol in table.get_symbols() if symbol.is_assigned()}


def _module_level_lines(tree: ast.Module, name: str) -> tuple[int | None, int | None]:
    """Return the first line that reads a name and the first that defines it, at top level.

    The bodies of functions, classes and lambdas are left out: they run later, in their own scope.
    """
    first_read: int | None = None
    first_definition: int | None = None
    pending: list[ast.AST] = list(tree.body)
    while pending:
        node = pending.pop()
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            if node.name == name:
                first_definition = min(first_definition or node.lineno, node.lineno)
            continue
        if isinstance(node, ast.Lambda):
            continue
        if isinstance(node, ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp):
            loop_names = {
                target.id
                for generator in node.generators
                for target in ast.walk(generator.target)
                if isinstance(target, ast.Name)
            }
            if name in loop_names:
                # The name is the comprehension's own loop variable. Only the first iterable is
                # evaluated at the top level.
                pending.append(node.generators[0].iter)
                continue
        if isinstance(node, ast.Name) and node.id == name:
            if isinstance(node.ctx, ast.Load):
                first_read = min(first_read or node.lineno, node.lineno)
            else:
                first_definition = min(first_definition or node.lineno, node.lineno)
        pending.extend(ast.iter_child_nodes(node))
    return first_read, first_definition


def _nested_scope_reads(table: symtable.SymbolTable, name: str) -> bool:
    """Return True when a function or class of the cell reads the name from the top level."""
    for child in table.get_children():
        try:
            symbol = child.lookup(name)
        except KeyError:
            symbol = None
        if symbol is not None and symbol.is_global() and symbol.is_referenced():
            return True
        if _nested_scope_reads(child, name):
            return True
    return False


def name_use(source: str, name: str) -> str | None:
    """Say how a code cell uses a name defined earlier in the notebook.

    Parameters
    ----------
    source : str
        Source of the code cell.
    name : str
        The name to look for.

    Returns
    -------
    str or None
        ``"reads"`` when the cell reads the earlier value; ``"redefines"`` when the cell defines
        the name itself before any read; None when the cell does not use the name, or is not valid
        Python.

    Examples
    --------
    >>> name_use("print(score * 2)", "score")
    'reads'
    >>> name_use("score = 0.5\\nprint(score)", "score")
    'redefines'
    >>> name_use("score = score + 1", "score")
    'reads'
    >>> name_use("def double(score):\\n    return 2 * score", "score") is None
    True
    """
    python = _python_source(source)
    try:
        tree = ast.parse(python)
        table = symtable.symtable(python, "<cell>", "exec")
    except SyntaxError:
        return None
    first_read, first_definition = _module_level_lines(tree, name)
    if first_definition is not None:
        # On one line (`x = x + 1`) the right-hand side is evaluated first: the old value is read.
        if first_read is not None and first_read <= first_definition:
            return "reads"
        return "redefines"
    if first_read is not None or _nested_scope_reads(table, name):
        return "reads"
    return None


def later_uses(nb: nbformat.NotebookNode, exercises: list[Exercise]) -> list[str]:
    """Find the cells that read a name defined in a cell to complete.

    The check of an exercise, the cell right after it, may read its names. No other later cell
    may, until a cell defines the name again under the lesson's own control.

    Parameters
    ----------
    nb : nbformat.NotebookNode
        The notebook.
    exercises : list of Exercise
        Its well-formed exercises, as returned by :func:`find_exercises`.

    Returns
    -------
    list of str
        One sentence per cell that depends on a learner's answer.
    """
    problems: list[str] = []
    for exercise in exercises:
        tracked = _top_level_names(nb.cells[exercise.cell].source)
        for index in range(exercise.cell + 2, len(nb.cells)):
            cell = nb.cells[index]
            if cell.cell_type != "code":
                continue
            for name in sorted(tracked):
                use = name_use(cell.source, name)
                if use == "reads":
                    problems.append(
                        f"[cell {index}] uses `{name}`, defined in the cell to complete of "
                        f"Exercise {exercise.number}"
                    )
                elif use == "redefines":
                    tracked = tracked - {name}
    return problems


def with_solutions(nb: nbformat.NotebookNode, exercises: list[Exercise]) -> nbformat.NotebookNode:
    """Return a copy of the notebook with each solution in place of its cell to complete."""
    solved = nbformat.from_dict(nb)
    for exercise in exercises:
        solved.cells[exercise.cell].source = exercise.solution
    return solved


def check_notebook(notebook: Path, cell_timeout_s: int = DEFAULT_CELL_TIMEOUT_S) -> list[str]:
    """Return the exercise problems of one notebook.

    Parameters
    ----------
    notebook : Path
        Notebook file to check. It is read and never written.
    cell_timeout_s : int, default 1500
        Seconds allowed for a single cell when the solved copy is executed.

    Returns
    -------
    list of str
        One sentence per problem, empty when the exercises are well formed, independent of the
        rest of the notebook, and passed by their solutions.
    """
    nb = nbformat.read(notebook, as_version=4)
    exercises, problems = find_exercises(nb)
    if not exercises and not problems:
        return ["no cell to complete: the notebook has no exercise"]
    problems += later_uses(nb, exercises)
    if not exercises:
        return problems

    solved = with_solutions(nb, exercises)
    strict_before = os.environ.get(STRICT_ENV)
    # The kernel inherits this environment: a check that does not pass then raises in its cell.
    os.environ[STRICT_ENV] = "1"
    try:
        failing_cell, error, _ = execute_notebook(solved, notebook.parent, cell_timeout_s)
    finally:
        if strict_before is None:
            del os.environ[STRICT_ENV]
        else:
            os.environ[STRICT_ENV] = strict_before
    if error:
        where = "kernel" if failing_cell is None else f"cell {failing_cell}"
        problems.append(f"[{where}] with the solutions in place: {error}")
    return problems


def find_notebooks(notebook_dir: Path, patterns: list[str]) -> list[Path]:
    """Return the notebooks matching the glob patterns, sorted, without checkpoint copies."""
    found: set[Path] = set()
    for pattern in patterns:
        found.update(
            path
            for path in notebook_dir.glob(pattern)
            if path.suffix == ".ipynb" and ".ipynb_checkpoints" not in path.parts
        )
    return sorted(found)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(
        description="Verify the form of the exercises and that each solution passes its check."
    )
    parser.add_argument(
        "patterns",
        nargs="*",
        default=[DEFAULT_PATTERN],
        help=f"glob patterns relative to the notebook directory (default: {DEFAULT_PATTERN!r})",
    )
    parser.add_argument(
        "--notebook-dir",
        type=Path,
        default=NOTEBOOK_DIR,
        help="directory holding the notebooks (default: notebooks/ of the repository)",
    )
    parser.add_argument(
        "--cell-timeout",
        type=int,
        default=DEFAULT_CELL_TIMEOUT_S,
        help=f"seconds allowed for a single cell (default: {DEFAULT_CELL_TIMEOUT_S})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Check the selected notebooks, log each problem, and return the process exit code."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    notebooks = find_notebooks(args.notebook_dir, args.patterns)
    if not notebooks:
        logger.error("No notebook matches %s under %s.", args.patterns, args.notebook_dir)
        return 2

    total = 0
    faulty = 0
    for position, notebook in enumerate(notebooks, start=1):
        relative = notebook.relative_to(args.notebook_dir)
        logger.info("[%d/%d] %s ...", position, len(notebooks), relative)
        problems = check_notebook(notebook, args.cell_timeout)
        if problems:
            faulty += 1
            total += len(problems)
            for problem in problems:
                logger.info("    %s", problem)
        else:
            logger.info("    ok")

    logger.info(
        "%d notebook(s) checked: %d problem(s) in %d notebook(s).", len(notebooks), total, faulty
    )
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
