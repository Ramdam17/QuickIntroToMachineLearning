"""Vendor a small, offline Palmer penguins teaching CSV into the package.

One-off data-prep, kept for provenance and reproducibility. Fetches the Palmer penguins dataset once
via seaborn (network required), keeps the binary 2-feature teaching subset used in module
``00_GettingStarted`` (Adélie vs Gentoo; bill length & flipper length), drops missing rows, and
writes ``src/ml_course/data/penguins.csv`` so the course runs offline thereafter.

Run once (or to refresh the vendored file):

    uv run python scripts/vendor_penguins.py

Data: Gorman KB, Williams TD, Fraser WR (2014), PLoS ONE 9(3):e90081; packaged as palmerpenguins
(Horst AM, Hill AP, Gorman KB, 2020).
"""

from __future__ import annotations

from pathlib import Path

import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "src" / "ml_course" / "data" / "penguins.csv"

SPECIES = ["Adelie", "Gentoo"]
FEATURES = ["bill_length_mm", "flipper_length_mm"]
LABEL = "species"


def main() -> None:
    """Fetch, subset, clean, and write the vendored penguins CSV."""
    print("Fetching Palmer penguins via seaborn (network required)...")
    df = sns.load_dataset("penguins")  # raises if unreachable — we want that, loudly
    print(f"  full dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    subset = df[df[LABEL].isin(SPECIES)][[*FEATURES, LABEL]].dropna().reset_index(drop=True)
    print(f"  teaching subset ({' vs '.join(SPECIES)}, {FEATURES}): {subset.shape[0]} rows")
    print("\nClass counts:\n", subset[LABEL].value_counts().to_string())
    print("\nFeature summary:\n", subset[FEATURES].describe().to_string())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(OUT, index=False)
    print(f"\nWrote {OUT.relative_to(ROOT)} ({subset.shape[0]} rows).")


if __name__ == "__main__":
    main()
