"""ml_course — the small shared library the course notebooks import.

Public surface:

- :mod:`ml_course.colors` — the graphic-charter palette (single source of truth; no hardcoded hex).
- :mod:`ml_course.viz` — the matplotlib course style and plotting helpers.
"""

from __future__ import annotations

from ml_course import colors, viz

__all__ = ["colors", "viz"]
