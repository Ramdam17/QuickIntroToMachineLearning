"""Report the code cells that are longer than the notebook charter allows.

The charter (``docs/notebook_template.md``) limits a code cell to 25 lines of code. Blank lines
and comment-only lines do not count. One exception is allowed: a cell made of a single definition
(one function or one class), which cannot be split without breaking it. Run before committing::

    uv run python scripts/check_cells.py "11_MLP/*.ipynb"

Exit code 0 means no cell is above the limit apart from single definitions, 1 means at least one
cell must be split, 2 means no notebook matched the patterns. Single definitions above the limit
are listed as well, so that they can be compared with the exceptions named in the notebook plan.
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"
DEFAULT_PATTERN = "*/*.ipynb"
# Largest number of code lines in one cell. A practical rule of the course charter, not an
# established result.
MAX_CODE_LINES = 25
# First characters of the IPython lines (magics, shell calls) that are not Python syntax.
IPYTHON_PREFIXES = ("%", "!")

logger = logging.getLogger("check_cells")


@dataclass(frozen=True)
class LongCell:
    """A code cell above the limit.

    Attributes
    ----------
    notebook : Path
        Notebook that holds the cell.
    index : int
        Zero-based index of the cell in the notebook.
    code_lines : int
        Number of lines of code of the cell, blank and comment-only lines excluded.
    single_definition : bool
        True when the cell is one function or one class, the exception the charter allows.
    """

    notebook: Path
    index: int
    code_lines: int
    single_definition: bool


def count_code_lines(source: str) -> int:
    """Count the lines of code of a cell, without blank lines and comment-only lines.

    A line that holds code followed by a comment counts. The lines of a multi-line string count
    too, since the cell is no shorter to read for it.

    Examples
    --------
    >>> count_code_lines("# set the weights\\nw = 0.5\\n\\nb = 0.25  # the bias\\n")
    2
    """
    return sum(
        1 for line in source.splitlines() if line.strip() and not line.strip().startswith("#")
    )


def is_single_definition(source: str) -> bool:
    """Return True when the cell holds exactly one statement, a function or a class.

    A cell that Python cannot parse is not a single definition.

    Examples
    --------
    >>> is_single_definition("# the neuron\\ndef neuron(x, w, b):\\n    return w * x + b\\n")
    True
    >>> is_single_definition("def f():\\n    return 1\\n\\nvalue = f()\\n")
    False
    """
    python_lines = [
        line for line in source.splitlines() if not line.lstrip().startswith(IPYTHON_PREFIXES)
    ]
    try:
        tree = ast.parse("\n".join(python_lines))
    except SyntaxError:
        return False
    return len(tree.body) == 1 and isinstance(
        tree.body[0], ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
    )


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


def long_cells(notebook: Path, max_code_lines: int = MAX_CODE_LINES) -> list[LongCell]:
    """Return the code cells of a notebook that have more code lines than the limit.

    Parameters
    ----------
    notebook : Path
        Notebook file to read.
    max_code_lines : int, default 25
        Largest number of code lines allowed in one cell.

    Returns
    -------
    list of LongCell
        One entry per cell above the limit, in notebook order.
    """
    cells = json.loads(notebook.read_text()).get("cells", [])
    found = []
    for index, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        raw = cell.get("source", "")
        source = "".join(raw) if isinstance(raw, list) else str(raw)
        code_lines = count_code_lines(source)
        if code_lines > max_code_lines:
            found.append(LongCell(notebook, index, code_lines, is_single_definition(source)))
    return found


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(
        description="Report the code cells longer than the notebook charter allows."
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
        "--max-code-lines",
        type=int,
        default=MAX_CODE_LINES,
        help=f"largest number of code lines in one cell (default: {MAX_CODE_LINES})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Scan the selected notebooks, log the long cells, and return the process exit code."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    notebooks = find_notebooks(args.notebook_dir, args.patterns)
    if not notebooks:
        logger.error("No notebook matches %s under %s.", args.patterns, args.notebook_dir)
        return 2

    to_split: list[LongCell] = []
    allowed: list[LongCell] = []
    for notebook in notebooks:
        for cell in long_cells(notebook, args.max_code_lines):
            (allowed if cell.single_definition else to_split).append(cell)

    for title, cells in (
        (f"Cells above {args.max_code_lines} code lines, to split:", to_split),
        ("Single definitions above the limit (allowed; name each in the notebook plan):", allowed),
    ):
        if cells:
            logger.info(title)
            for cell in cells:
                relative = cell.notebook.relative_to(args.notebook_dir)
                logger.info("  %s [cell %d]: %d code lines", relative, cell.index, cell.code_lines)

    logger.info(
        "%d notebook(s) scanned: %d cell(s) to split, %d single definition(s) above the limit.",
        len(notebooks),
        len(to_split),
        len(allowed),
    )
    return 1 if to_split else 0


if __name__ == "__main__":
    raise SystemExit(main())
