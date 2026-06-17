"""Fail if any notebook or module hardcodes a hex colour outside the palette.

The graphic charter has a single source of truth — :mod:`ml_course.colors`. Notebooks and modules
must reference named roles from there, never literal ``#rrggbb``. Run before committing::

    uv run python scripts/check_no_hardcoded_hex.py

Exit code 0 = clean, 1 = violations found (paths and offending lines printed).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# colors.py is the ONE place hex literals are allowed.
ALLOWED = {ROOT / "src" / "ml_course" / "colors.py"}
HEX = re.compile(r"#[0-9a-fA-F]{6}\b|#[0-9a-fA-F]{3}\b")


def _scan_text(text: str, path: Path) -> list[str]:
    """Return ``"path:line: snippet"`` strings for each line containing a hex literal."""
    hits = []
    for i, line in enumerate(text.splitlines(), start=1):
        if HEX.search(line):
            hits.append(f"{path.relative_to(ROOT)}:{i}: {line.strip()}")
    return hits


def _scan_notebook(path: Path) -> list[str]:
    """Return hex-literal hits from a notebook's code cells."""
    nb = json.loads(path.read_text())
    hits = []
    for n, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", [])
        text = "".join(src) if isinstance(src, list) else str(src)
        for line in text.splitlines():
            if HEX.search(line):
                hits.append(f"{path.relative_to(ROOT)} [cell {n}]: {line.strip()}")
    return hits


def main() -> int:
    """Scan notebooks/ and src/, print violations, return process exit code."""
    violations: list[str] = []

    for py in (ROOT / "src").rglob("*.py"):
        if py in ALLOWED:
            continue
        violations += _scan_text(py.read_text(), py)

    nb_dir = ROOT / "notebooks"
    if nb_dir.exists():
        for nb in nb_dir.rglob("*.ipynb"):
            if ".ipynb_checkpoints" in nb.parts:
                continue
            violations += _scan_notebook(nb)

    if violations:
        print("Hardcoded hex colours found (use ml_course.colors instead):\n")
        for v in violations:
            print(f"  {v}")
        print(f"\n{len(violations)} violation(s).")
        return 1
    print("No hardcoded hex colours. Charter respected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
