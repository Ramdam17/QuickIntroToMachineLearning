"""Execute every notebook of the course from top to bottom and report which ones fail.

The notebooks in the repository are never modified: each one is read, executed on a fresh kernel
in its own folder, and its executed copy is written to an output directory. Run it in the
environment set up by the install command of the README::

    uv sync --extra dev --extra boosting --extra deep
    uv run python scripts/run_notebooks.py

Restrict the run to one module or one notebook with glob patterns relative to ``notebooks/``::

    uv run python scripts/run_notebooks.py "11_MLP/*.ipynb"

Exit code 0 means every notebook executed, 1 means at least one failed (the failing cell and its
error are logged), 2 means no notebook matched the patterns.
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError, CellTimeoutError

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"
DEFAULT_PATTERN = "*/*.ipynb"
# Seconds allowed for a single cell. The slowest cells of the course (boosting on large tables,
# PyTorch training) finish well below this; a cell that exceeds it is treated as a failure.
DEFAULT_CELL_TIMEOUT_S = 1500
TIMING_FILENAME = "timing.tsv"

logger = logging.getLogger("run_notebooks")


@dataclass(frozen=True)
class RunResult:
    """Outcome of one notebook execution.

    Attributes
    ----------
    notebook : Path
        Path of the notebook that was executed.
    ok : bool
        True when every cell executed without raising.
    seconds : float
        Wall-clock duration of the execution, in seconds.
    failing_cell : int or None
        Zero-based index of the cell that raised or timed out, None when ``ok`` is True or when
        the kernel itself could not start.
    error : str
        One-line description of the failure, empty when ``ok`` is True.
    """

    notebook: Path
    ok: bool
    seconds: float
    failing_cell: int | None = None
    error: str = ""


def find_notebooks(notebook_dir: Path, patterns: list[str]) -> list[Path]:
    """Return the notebooks matching the patterns, sorted, without checkpoint copies.

    Parameters
    ----------
    notebook_dir : Path
        Directory the patterns are relative to.
    patterns : list of str
        Glob patterns, for example ``["*/*.ipynb"]`` or ``["11_MLP/*.ipynb"]``.

    Returns
    -------
    list of Path
        Matching notebook paths in module then notebook order, each listed once.
    """
    found: set[Path] = set()
    for pattern in patterns:
        found.update(
            path
            for path in notebook_dir.glob(pattern)
            if path.suffix == ".ipynb" and ".ipynb_checkpoints" not in path.parts
        )
    return sorted(found)


def _first_failing_cell(nb: nbformat.NotebookNode) -> int | None:
    """Return the index of the first code cell holding an error output, if any."""
    for index, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        if any(output.get("output_type") == "error" for output in cell.get("outputs", [])):
            return index
    return None


def _last_started_cell(nb: nbformat.NotebookNode) -> int | None:
    """Return the index of the last code cell the kernel started executing, if any."""
    started = [
        index
        for index, cell in enumerate(nb.cells)
        if cell.cell_type == "code" and cell.execution_count is not None
    ]
    return started[-1] if started else None


def execute_notebook(
    nb: nbformat.NotebookNode, working_dir: Path, cell_timeout_s: int = DEFAULT_CELL_TIMEOUT_S
) -> tuple[int | None, str, float]:
    """Execute a notebook held in memory on a fresh kernel, filling its outputs in place.

    Parameters
    ----------
    nb : nbformat.NotebookNode
        The notebook to execute. Its cells receive their outputs.
    working_dir : Path
        Working directory of the kernel, normally the folder of the notebook file, so that the
        relative paths of the notebook resolve as they do for a learner.
    cell_timeout_s : int, default 1500
        Seconds allowed for a single cell.

    Returns
    -------
    failing_cell : int or None
        Zero-based index of the cell that raised or timed out; None when every cell executed, or
        when the kernel itself could not start.
    error : str
        One-line description of the failure, empty when every cell executed.
    seconds : float
        Wall-clock duration of the execution, in seconds.
    """
    client = NotebookClient(
        nb,
        timeout=cell_timeout_s,
        kernel_name="python3",
        resources={"metadata": {"path": str(working_dir)}},
    )
    failing_cell: int | None = None
    error = ""
    start = time.perf_counter()
    try:
        client.execute()
    except CellTimeoutError:
        # The kernel numbers a cell when it starts it, so the interrupted cell is the last one
        # that carries an execution count; the cells after it were never started.
        failing_cell = _last_started_cell(nb)
        error = f"TimeoutError: a cell ran for more than {cell_timeout_s} s"
    except CellExecutionError as exc:
        failing_cell = _first_failing_cell(nb)
        error = f"{exc.ename}: {exc.evalue}"
    except Exception as exc:  # noqa: BLE001 - a kernel that cannot start is reported, not raised
        error = f"{type(exc).__name__}: {exc}"
    return failing_cell, error, time.perf_counter() - start


def run_notebook(
    notebook: Path, executed_copy: Path, cell_timeout_s: int = DEFAULT_CELL_TIMEOUT_S
) -> RunResult:
    """Execute one notebook on a fresh kernel and write its executed copy.

    The notebook runs with its own folder as working directory, as it does when a learner opens
    it. The source file is not modified. The executed copy is written even when a cell fails, so
    that the failure can be read in place.

    Parameters
    ----------
    notebook : Path
        Notebook to execute.
    executed_copy : Path
        Where the executed notebook is written. Parent folders are created.
    cell_timeout_s : int, default 1500
        Seconds allowed for a single cell.

    Returns
    -------
    RunResult
        Whether the notebook executed, how long it took, and what failed if it did not.

    Examples
    --------
    >>> result = run_notebook(Path("notebooks/00_GettingStarted/01_what_is_ml.ipynb"),
    ...                       Path("executed/01_what_is_ml.ipynb"))  # doctest: +SKIP
    >>> result.ok  # doctest: +SKIP
    True
    """
    nb = nbformat.read(notebook, as_version=4)
    failing_cell, error, seconds = execute_notebook(nb, notebook.parent, cell_timeout_s)

    executed_copy.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, executed_copy)
    return RunResult(
        notebook=notebook, ok=not error, seconds=seconds, failing_cell=failing_cell, error=error
    )


def write_timing(results: list[RunResult], notebook_dir: Path, timing_path: Path) -> None:
    """Write one line per notebook: relative path, ``ok`` or ``failed``, seconds.

    Parameters
    ----------
    results : list of RunResult
        Outcomes to write, in execution order.
    notebook_dir : Path
        Directory the notebook paths are made relative to.
    timing_path : Path
        Tab-separated file to write.
    """
    lines = ["notebook\tstatus\tseconds"]
    for result in results:
        status = "ok" if result.ok else "failed"
        relative = result.notebook.relative_to(notebook_dir)
        lines.append(f"{relative}\t{status}\t{result.seconds:.1f}")
    timing_path.write_text("\n".join(lines) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(
        description="Execute the course notebooks from top to bottom and report failures."
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
        "--out-dir",
        type=Path,
        default=None,
        help="where the executed copies and the timing file go (default: a new temporary folder)",
    )
    parser.add_argument(
        "--cell-timeout",
        type=int,
        default=DEFAULT_CELL_TIMEOUT_S,
        help=f"seconds allowed for a single cell (default: {DEFAULT_CELL_TIMEOUT_S})",
    )
    parser.add_argument(
        "--fail-fast", action="store_true", help="stop at the first notebook that fails"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Execute the selected notebooks, log each outcome, and return the process exit code."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    notebooks = find_notebooks(args.notebook_dir, args.patterns)
    if not notebooks:
        logger.error("No notebook matches %s under %s.", args.patterns, args.notebook_dir)
        return 2

    out_dir = args.out_dir or Path(tempfile.mkdtemp(prefix="ml_course_executed_"))
    logger.info("Executing %d notebook(s). Executed copies go to %s", len(notebooks), out_dir)

    results: list[RunResult] = []
    for position, notebook in enumerate(notebooks, start=1):
        relative = notebook.relative_to(args.notebook_dir)
        logger.info("[%d/%d] %s ...", position, len(notebooks), relative)
        result = run_notebook(notebook, out_dir / relative, args.cell_timeout)
        results.append(result)
        if result.ok:
            logger.info("    ok in %.1f s", result.seconds)
        else:
            where = "kernel" if result.failing_cell is None else f"cell {result.failing_cell}"
            logger.error("    FAILED in %.1f s at %s: %s", result.seconds, where, result.error)
            if args.fail_fast:
                break

    timing_path = out_dir / TIMING_FILENAME
    write_timing(results, args.notebook_dir, timing_path)

    failed = [result for result in results if not result.ok]
    total_s = sum(result.seconds for result in results)
    logger.info(
        "%d of %d notebook(s) executed in %.0f s. Timings: %s",
        len(results) - len(failed),
        len(results),
        total_s,
        timing_path,
    )
    for result in failed:
        logger.error("  failed: %s", result.notebook.relative_to(args.notebook_dir))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
