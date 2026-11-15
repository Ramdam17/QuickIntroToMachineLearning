"""Tests for the notebook runner: it must tell a notebook that executes from one that fails."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import nbformat
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_notebooks.py"


def _load_runner() -> ModuleType:
    """Import the script by path, since ``scripts/`` is not an installed package."""
    spec = importlib.util.spec_from_file_location("run_notebooks", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered before execution so that the dataclass can resolve its own module.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


def _write_notebook(path: Path, sources: list[str]) -> Path:
    """Write a notebook made of one code cell per source string."""
    nb = nbformat.v4.new_notebook()
    nb.cells = [nbformat.v4.new_code_cell(source) for source in sources]
    path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, path)
    return path


def test_find_notebooks_sorts_and_skips_checkpoints(tmp_path: Path) -> None:
    _write_notebook(tmp_path / "02_B" / "01_first.ipynb", ["1"])
    _write_notebook(tmp_path / "01_A" / "02_second.ipynb", ["1"])
    _write_notebook(tmp_path / "01_A" / "01_first.ipynb", ["1"])
    _write_notebook(tmp_path / "01_A" / ".ipynb_checkpoints" / "01_first.ipynb", ["1"])

    found = runner.find_notebooks(tmp_path, ["*/*.ipynb", "01_A/*.ipynb"])

    assert [str(path.relative_to(tmp_path)) for path in found] == [
        "01_A/01_first.ipynb",
        "01_A/02_second.ipynb",
        "02_B/01_first.ipynb",
    ]


def test_passing_notebook_is_ok_and_source_is_untouched(tmp_path: Path) -> None:
    notebook = _write_notebook(tmp_path / "src" / "ok.ipynb", ["x = 2 + 3", "print(x)"])
    before = notebook.read_text()
    executed = tmp_path / "out" / "ok.ipynb"

    result = runner.run_notebook(notebook, executed, cell_timeout_s=60)

    assert result.ok
    assert result.failing_cell is None
    assert result.error == ""
    assert notebook.read_text() == before
    outputs = nbformat.read(executed, as_version=4).cells[1].outputs
    assert outputs[0]["text"].strip() == "5"


def test_failing_notebook_reports_the_cell_and_the_error(tmp_path: Path) -> None:
    notebook = _write_notebook(
        tmp_path / "src" / "bad.ipynb", ["x = 1", "raise ValueError('boom')", "y = 2"]
    )
    executed = tmp_path / "out" / "bad.ipynb"

    result = runner.run_notebook(notebook, executed, cell_timeout_s=60)

    assert not result.ok
    assert result.failing_cell == 1
    assert result.error == "ValueError: boom"
    assert executed.exists()


def test_timed_out_cell_is_reported(tmp_path: Path) -> None:
    notebook = _write_notebook(
        tmp_path / "src" / "slow.ipynb", ["x = 1", "import time\ntime.sleep(30)", "y = 2"]
    )

    result = runner.run_notebook(notebook, tmp_path / "out" / "slow.ipynb", cell_timeout_s=2)

    assert not result.ok
    assert result.failing_cell == 1
    assert result.error.startswith("TimeoutError")


def test_notebook_runs_in_its_own_folder(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "data.txt").write_text("hello")
    notebook = _write_notebook(tmp_path / "src" / "cwd.ipynb", ["print(open('data.txt').read())"])

    result = runner.run_notebook(notebook, tmp_path / "out" / "cwd.ipynb", cell_timeout_s=60)

    assert result.ok


@pytest.mark.parametrize(("sources", "expected_code"), [(["1 + 1"], 0), (["1 / 0"], 1)])
def test_main_exit_code_and_timing_file(
    tmp_path: Path, sources: list[str], expected_code: int
) -> None:
    _write_notebook(tmp_path / "nbs" / "00_M" / "01_only.ipynb", sources)
    out_dir = tmp_path / "out"

    code = runner.main(["--notebook-dir", str(tmp_path / "nbs"), "--out-dir", str(out_dir)])

    assert code == expected_code
    lines = (out_dir / runner.TIMING_FILENAME).read_text().splitlines()
    assert lines[0] == "notebook\tstatus\tseconds"
    assert lines[1].split("\t")[:2] == [
        "00_M/01_only.ipynb",
        "ok" if expected_code == 0 else "failed",
    ]


def test_main_returns_2_when_nothing_matches(tmp_path: Path) -> None:
    assert runner.main(["--notebook-dir", str(tmp_path), "nothing/*.ipynb"]) == 2
