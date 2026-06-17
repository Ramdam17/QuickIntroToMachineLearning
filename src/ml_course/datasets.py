"""Small, offline teaching datasets for the course.

Currently the Palmer penguins binary 2-feature subset used in module ``00_GettingStarted``. Data is
**vendored** (a CSV inside the package) so notebooks run offline; the file is produced by
``scripts/vendor_penguins.py``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_DATA_DIR = Path(__file__).parent / "data"
_PENGUINS_CSV = _DATA_DIR / "penguins.csv"

#: The two numeric features used throughout module 00 (millimetres).
PENGUINS_FEATURES: list[str] = ["bill_length_mm", "flipper_length_mm"]
#: The label column (categorical: two penguin species).
PENGUINS_TARGET: str = "species"


def load_penguins() -> pd.DataFrame:
    """Load the offline Palmer penguins teaching subset as a DataFrame.

    Returns the binary, two-feature subset (Adélie vs Gentoo; bill length and flipper length, in mm)
    used across module ``00_GettingStarted`` — 274 complete rows, no missing values. The label
    column ``species`` keeps its readable string values (encoding it is itself a later lesson).

    Returns
    -------
    pandas.DataFrame, shape (274, 3)
        Columns ``bill_length_mm`` (float, mm), ``flipper_length_mm`` (float, mm), and ``species``
        (str, one of ``"Adelie"``/``"Gentoo"``). One row = one penguin.

    When to use
    -----------
    The running dataset for module 00. It is deliberately small, 2-D, and visualizable, with a
    clean-but-not-perfect class separation so evaluation has honest errors to show. For the
    feature/label arrays an estimator expects, pass the frame through :func:`penguins_xy`.

    Examples
    --------
    >>> df = load_penguins()
    >>> df.shape
    (274, 3)
    >>> sorted(df["species"].unique())
    ['Adelie', 'Gentoo']

    References
    ----------
    Gorman KB, Williams TD, Fraser WR (2014). Ecological sexual dimorphism ... Pygoscelis penguins.
    PLoS ONE 9(3):e90081. https://doi.org/10.1371/journal.pone.0090081
    Horst AM, Hill AP, Gorman KB (2020). palmerpenguins R package.
    https://doi.org/10.5281/zenodo.3960218
    """
    if not _PENGUINS_CSV.exists():
        raise FileNotFoundError(
            f"Vendored dataset missing: {_PENGUINS_CSV}. "
            "Run `uv run python scripts/vendor_penguins.py` to create it (needs network once)."
        )
    return pd.read_csv(_PENGUINS_CSV)


def penguins_xy(df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Split the penguins frame into features ``X`` (DataFrame) and label ``y`` (Series).

    Parameters
    ----------
    df : pandas.DataFrame, optional
        A frame as returned by :func:`load_penguins`. Loaded fresh when omitted.

    Returns
    -------
    X : pandas.DataFrame, shape (n_samples, 2)
        The two feature columns (:data:`PENGUINS_FEATURES`), in millimetres.
    y : pandas.Series, shape (n_samples,)
        The ``species`` label (string categories).

    Examples
    --------
    >>> X, y = penguins_xy()
    >>> list(X.columns)
    ['bill_length_mm', 'flipper_length_mm']
    >>> y.name
    'species'
    """
    if df is None:
        df = load_penguins()
    return df[PENGUINS_FEATURES], df[PENGUINS_TARGET]
