"""Small teaching datasets for the course (Palmer penguins).

The full Palmer penguins dataset is **fetched on first use and cached locally** (under the package,
git-ignored), so notebooks run offline thereafter and nothing is committed to the repo. Module
``00_GettingStarted`` works with a binary, two-feature **subset** (Adélie vs Gentoo; bill length and
flipper length), derived on the fly from the full set; later chapters use the full set via
:func:`load_penguins_full`, which carries categorical features, more classes, and missing values.

References
----------
Gorman KB, Williams TD, Fraser WR (2014). Ecological sexual dimorphism and environmental variability
within a community of Antarctic penguins (genus *Pygoscelis*). PLoS ONE 9(3):e90081.
https://doi.org/10.1371/journal.pone.0090081
Horst AM, Hill AP, Gorman KB (2020). palmerpenguins R package.
https://doi.org/10.5281/zenodo.3960218
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

#: Canonical source CSV (the seaborn-data mirror of Palmer penguins).
_PENGUINS_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
_DATA_DIR = Path(__file__).parent / "data"
_FULL_CSV = _DATA_DIR / "penguins_full.csv"

#: The two numeric features used throughout module ``00_GettingStarted`` (millimetres).
PENGUINS_FEATURES: list[str] = ["bill_length_mm", "flipper_length_mm"]
#: The label column (categorical: penguin species).
PENGUINS_TARGET: str = "species"
#: The two species kept in the module-00 binary subset.
PENGUINS_SUBSET_SPECIES: list[str] = ["Adelie", "Gentoo"]
#: Numeric feature columns available in the full dataset (millimetres / grams).
PENGUINS_FULL_NUMERIC: list[str] = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
]
#: Categorical feature columns available in the full dataset.
PENGUINS_FULL_CATEGORICAL: list[str] = ["island", "sex"]


def _ensure_full_csv() -> Path:
    """Return the path to the cached full penguins CSV, downloading it once if absent.

    On a cache miss the dataset is fetched from :data:`_PENGUINS_URL` and written to
    :data:`_FULL_CSV`; on a hit the cached file is reused. Progress is logged at INFO (never
    silenced); enable ``logging.basicConfig(level=logging.INFO)`` to see the one-time network step.
    """
    if _FULL_CSV.exists():
        logger.info("Using cached penguins dataset: %s", _FULL_CSV)
        return _FULL_CSV
    logger.info("Downloading penguins dataset from %s ...", _PENGUINS_URL)
    full = pd.read_csv(_PENGUINS_URL)
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    full.to_csv(_FULL_CSV, index=False)
    logger.info("Cached to %s (%d rows, %d columns).", _FULL_CSV, full.shape[0], full.shape[1])
    return _FULL_CSV


def load_penguins_full() -> pd.DataFrame:
    """Load the full Palmer penguins dataset (fetched once, then cached locally).

    Returns all 344 penguins and 7 columns — three species, two categorical features, four numeric
    features — **with missing values left in**, so preprocessing lessons have real gaps to handle.

    Returns
    -------
    pandas.DataFrame, shape (344, 7)
        Columns ``species`` (str; ``Adelie``/``Gentoo``/``Chinstrap``), ``island`` (str),
        ``sex`` (str; ``MALE``/``FEMALE``, with some missing), and the numeric ``bill_length_mm``,
        ``bill_depth_mm``, ``flipper_length_mm``, ``body_mass_g`` (millimetres / grams). One row =
        one penguin.

    When to use
    -----------
    For chapters that need categorical features, missing values, or more than two classes (encoding,
    preprocessing pipelines, trees and the boosting family). For the small, visualizable binary 2-D
    teaching set used in module 00, use :func:`load_penguins` / :func:`penguins_xy`.

    Notes
    -----
    The first call downloads the CSV (network required) and caches it under the package; later calls
    read the cache and need no network. Run ``scripts/vendor_penguins.py`` to warm it beforehand.

    Examples
    --------
    >>> df = load_penguins_full()  # doctest: +SKIP
    >>> df.shape  # doctest: +SKIP
    (344, 7)
    """
    return pd.read_csv(_ensure_full_csv())


def load_penguins() -> pd.DataFrame:
    """Load the module-00 subset: Adélie vs Gentoo, two numeric features, no missing values.

    Derived from :func:`load_penguins_full` by keeping the two species and the two features and
    dropping rows with missing measurements — a clean, binary, 2-D, visualizable set with a
    clean-but-not-perfect class separation (so evaluation has honest errors to show).

    Returns
    -------
    pandas.DataFrame, shape (274, 3)
        Columns ``bill_length_mm`` (float, mm), ``flipper_length_mm`` (float, mm), and ``species``
        (str, ``Adelie``/``Gentoo``). One row = one penguin.

    When to use
    -----------
    The running dataset for module 00. For the feature/label arrays an estimator expects, pass it
    through :func:`penguins_xy`. For categorical features or more classes, use
    :func:`load_penguins_full`.

    Examples
    --------
    >>> df = load_penguins()  # doctest: +SKIP
    >>> df.shape  # doctest: +SKIP
    (274, 3)
    """
    full = load_penguins_full()
    return (
        full[full[PENGUINS_TARGET].isin(PENGUINS_SUBSET_SPECIES)][
            [*PENGUINS_FEATURES, PENGUINS_TARGET]
        ]
        .dropna()
        .reset_index(drop=True)
    )


def penguins_xy(df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Split the penguins subset into features ``X`` (DataFrame) and label ``y`` (Series).

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
    >>> X, y = penguins_xy()  # doctest: +SKIP
    >>> list(X.columns)  # doctest: +SKIP
    ['bill_length_mm', 'flipper_length_mm']
    """
    if df is None:
        df = load_penguins()
    return df[PENGUINS_FEATURES], df[PENGUINS_TARGET]
