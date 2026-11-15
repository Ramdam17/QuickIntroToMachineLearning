"""ml_course — the small shared library the course notebooks import.

Public surface:

- :mod:`ml_course.colors` — the graphic-charter palette (single source of truth; no hardcoded hex).
- :mod:`ml_course.viz` — the matplotlib course style and plotting helpers.
- :mod:`ml_course.datasets` — small, offline teaching datasets (e.g. Palmer penguins).
- :mod:`ml_course.check` — the kind automatic checks that follow each exercise of a notebook.
"""

from __future__ import annotations

from ml_course import check, colors, datasets, viz

__all__ = ["check", "colors", "datasets", "viz"]
