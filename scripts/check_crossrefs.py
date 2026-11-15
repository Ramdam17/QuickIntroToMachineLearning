"""Verify the cross-references between the notebooks of the course.

The charter (``docs/notebook_template.md``) fixes one wording for a cross-reference, in plain
text: ``module 03, notebook 4``, optionally followed by the subject in parentheses. This guard
reads the markdown cells and reports three kinds of problem:

- a reference to a module and notebook number that does not exist;
- a reference in another wording (``chapter 3``, ``NB 4``, or ``notebook 4`` with no module),
  which cannot be verified;
- a ``Previous:`` or ``Next:`` entry that is missing, or that does not name the notebook that
  comes before or after in the course. The first notebook of the course writes ``Previous: none``
  and the last one ``Next: none``.

Text inside code spans and code blocks is ignored. ``chapter 6 of`` a book is a citation, not a
cross-reference, and is left alone. Run before committing::

    uv run python scripts/check_crossrefs.py "11_MLP/*.ipynb"

Exit code 0 means every cross-reference is verified, 1 means at least one problem was found, 2
means no notebook matched the patterns.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"
DEFAULT_PATTERN = "*/*.ipynb"

# The one wording of the charter: "module 03, notebook 4".
REFERENCE = re.compile(r"\bmodule\s+(\d{1,2}),\s+notebook\s+(\d{1,2})\b", re.IGNORECASE)
# Wordings the course used before the charter fixed one. They name no module, or use another
# word for it, so the target cannot be checked.
OTHER_WORDINGS = (
    re.compile(r"\bchapters?\s+\d+(?!\s+of\b)", re.IGNORECASE),
    re.compile(r"\bNBs?\s*\d+"),
    re.compile(r"\bnotebooks?\s+\d+", re.IGNORECASE),
)
CODE_BLOCK = re.compile(r"```.*?```", re.DOTALL)
CODE_SPAN = re.compile(r"`[^`\n]*`")
# "Previous: ..." runs up to "Next:" or the end of the line; "Next: ..." up to the end of the line.
PREVIOUS = re.compile(r"Previous:(?P<target>.*?)(?=Next:|$)", re.MULTILINE)
NEXT = re.compile(r"Next:(?P<target>.*?)$", re.MULTILINE)
NO_NEIGHBOUR = "none"
FILE_NUMBER = re.compile(r"^(\d+)_")

logger = logging.getLogger("check_crossrefs")

type NotebookId = tuple[int, int]


def notebook_id(notebook: Path) -> NotebookId | None:
    """Return the (module number, notebook number) of ``NN_Module/MM_title.ipynb``.

    Examples
    --------
    >>> notebook_id(Path("notebooks/03_LogisticRegression/04_gradient_descent.ipynb"))
    (3, 4)
    """
    module = FILE_NUMBER.match(notebook.parent.name)
    number = FILE_NUMBER.match(notebook.name)
    if module is None or number is None:
        return None
    return int(module.group(1)), int(number.group(1))


def course_order(notebook_dir: Path) -> list[NotebookId]:
    """Return the identifiers of every notebook of the course, in course order."""
    ids = {
        notebook_id(path)
        for path in notebook_dir.glob("*/*.ipynb")
        if ".ipynb_checkpoints" not in path.parts
    }
    return sorted(identifier for identifier in ids if identifier is not None)


def markdown_text(notebook: Path) -> list[tuple[int, str]]:
    """Return ``(cell index, text)`` for each markdown cell, without its code spans and blocks."""
    cells = json.loads(notebook.read_text()).get("cells", [])
    texts = []
    for index, cell in enumerate(cells):
        if cell.get("cell_type") != "markdown":
            continue
        raw = cell.get("source", "")
        text = "".join(raw) if isinstance(raw, list) else str(raw)
        texts.append((index, CODE_SPAN.sub("", CODE_BLOCK.sub("", text))))
    return texts


def _format(identifier: NotebookId) -> str:
    return f"module {identifier[0]:02d}, notebook {identifier[1]}"


def _neighbour_problem(label: str, target: str | None, expected: NotebookId | None) -> str | None:
    """Return what is wrong with a ``Previous:`` or ``Next:`` entry, or None when it is right."""
    wanted = NO_NEIGHBOUR if expected is None else _format(expected)
    if target is None:
        return f"no `{label}:` entry; expected `{label}: {wanted}`"
    match = REFERENCE.search(target)
    found = (int(match.group(1)), int(match.group(2))) if match else None
    if found is None and expected is None and NO_NEIGHBOUR in target.lower():
        return None
    if found is not None and found == expected:
        return None
    return f"`{label}:` reads `{target.strip()}`; expected `{label}: {wanted}`"


def check_notebook(notebook: Path, order: list[NotebookId]) -> list[str]:
    """Return the cross-reference problems of one notebook.

    Parameters
    ----------
    notebook : Path
        Notebook file to read.
    order : list of (int, int)
        Identifiers of every notebook of the course, in course order, as given by
        :func:`course_order`.

    Returns
    -------
    list of str
        One sentence per problem, empty when every cross-reference is verified.
    """
    existing = set(order)
    problems: list[str] = []
    previous_target: str | None = None
    next_target: str | None = None

    for index, text in markdown_text(notebook):
        for match in REFERENCE.finditer(text):
            cited = (int(match.group(1)), int(match.group(2)))
            if cited not in existing:
                problems.append(f"[cell {index}] `{match.group(0)}` does not exist")
        without_references = REFERENCE.sub("", text)
        for wording in OTHER_WORDINGS:
            for match in wording.finditer(without_references):
                problems.append(
                    f"[cell {index}] `{match.group(0)}` is not in the wording "
                    "`module NN, notebook M`"
                )
        # The entries of the last cell that holds them win: they close the notebook.
        if found := PREVIOUS.search(text):
            previous_target = found.group("target")
        if found := NEXT.search(text):
            next_target = found.group("target")

    identifier = notebook_id(notebook)
    if identifier is None or identifier not in existing:
        problems.append("the file is not named `NN_Module/MM_title.ipynb`")
        return problems
    position = order.index(identifier)
    before = order[position - 1] if position > 0 else None
    after = order[position + 1] if position + 1 < len(order) else None
    for label, target, expected in (
        ("Previous", previous_target, before),
        ("Next", next_target, after),
    ):
        if problem := _neighbour_problem(label, target, expected):
            problems.append(problem)
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
        description="Verify the cross-references between the notebooks of the course."
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Check the selected notebooks, log each problem, and return the process exit code."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    notebooks = find_notebooks(args.notebook_dir, args.patterns)
    if not notebooks:
        logger.error("No notebook matches %s under %s.", args.patterns, args.notebook_dir)
        return 2

    # The whole course is indexed even when one module is checked: a reference may point anywhere.
    order = course_order(args.notebook_dir)
    total = 0
    faulty = 0
    for notebook in notebooks:
        problems = check_notebook(notebook, order)
        if problems:
            faulty += 1
            total += len(problems)
            logger.info("%s", notebook.relative_to(args.notebook_dir))
            for problem in problems:
                logger.info("  %s", problem)

    logger.info(
        "%d notebook(s) checked: %d problem(s) in %d notebook(s).", len(notebooks), total, faulty
    )
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
