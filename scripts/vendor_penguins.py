"""Warm the offline cache for the Palmer penguins teaching dataset.

The course fetches the full Palmer penguins dataset on first use and caches it under the package
(git-ignored); see :mod:`ml_course.datasets`. Run this once to populate that cache ahead of time
(e.g., before going offline or in CI setup), instead of letting the first notebook trigger the
download:

    uv run python scripts/vendor_penguins.py

Nothing is committed to the repo — the cached CSV is git-ignored.

Data: Gorman KB, Williams TD, Fraser WR (2014), PLoS ONE 9(3):e90081; palmerpenguins (Horst AM,
Hill AP, Gorman KB, 2020). Source CSV: the seaborn-data mirror.
"""

from __future__ import annotations

import logging

from ml_course import datasets


def main() -> None:
    """Download (if needed) and report the cached full dataset and the derived teaching subset."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    full = datasets.load_penguins_full()
    print(f"\nfull dataset: {full.shape[0]} rows, {full.shape[1]} columns")
    print("class counts:")
    print(full[datasets.PENGUINS_TARGET].value_counts().to_string())
    print(f"missing values per column:\n{full.isna().sum().to_string()}")

    subset = datasets.load_penguins()
    print(
        f"\nderived module-00 subset (Adelie vs Gentoo, "
        f"{datasets.PENGUINS_FEATURES}): {subset.shape[0]} rows, no missing values."
    )


if __name__ == "__main__":
    main()
