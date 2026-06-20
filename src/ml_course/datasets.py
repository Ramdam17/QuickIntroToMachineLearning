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


def load_newsgroups(
    categories: list[str] | None = None,
    subset: str = "train",
    *,
    remove: tuple[str, ...] = ("headers", "footers", "quotes"),
    random_state: int = 0,
) -> pd.DataFrame:
    """Fetch a subset of the 20 Newsgroups text corpus as a tidy DataFrame.

    Wraps :func:`sklearn.datasets.fetch_20newsgroups` (downloaded once, then cached under
    ``~/scikit_learn_data``) and returns one row per document with its raw ``text`` and its
    ``category`` label. Progress is logged at INFO (never silenced).

    Parameters
    ----------
    categories : list of str, optional
        Newsgroup names to keep (e.g. ``["sci.med", "comp.graphics"]``). ``None`` keeps all 20.
    subset : {"train", "test"}, default "train"
        Which split to load; 20 Newsgroups ships a fixed train/test split (by post date).
    remove : tuple of str, default ("headers", "footers", "quotes")
        Metadata stripped from each document, so the classifier learns the *content* rather than
        headers/signatures that would leak the label.
    random_state : int, default 0
        Seed for the document shuffle.

    Returns
    -------
    pandas.DataFrame, shape (n_documents, 2)
        Columns ``text`` (str, the document) and ``category`` (str, the newsgroup name). One row =
        one document.

    When to use
    -----------
    For the text-classification notebook (module 02): a real, high-dimensional, sparse problem where
    Naive Bayes shines. For the small 2-D penguins teaching set, use :func:`load_penguins`.

    Notes
    -----
    The first call downloads ~14 MB (network required); later calls read the cache and need no
    network — the same fetch-and-cache pattern as :func:`load_penguins_full`.

    References
    ----------
    Lang K (1995). NewsWeeder: learning to filter netnews. Proc. ICML, 331-339.
    https://doi.org/10.1016/B978-1-55860-377-6.50048-7

    Examples
    --------
    >>> df = load_newsgroups(["sci.med", "comp.graphics"], subset="train")  # doctest: +SKIP
    >>> list(df.columns)  # doctest: +SKIP
    ['text', 'category']
    """
    from sklearn.datasets import fetch_20newsgroups

    logger.info("Loading 20 newsgroups (subset=%s, categories=%s) ...", subset, categories or "all")
    bunch = fetch_20newsgroups(
        subset=subset, categories=categories, remove=remove, random_state=random_state
    )
    names = [bunch.target_names[t] for t in bunch.target]
    logger.info(
        "Loaded %d documents across %d categories.", len(bunch.data), len(bunch.target_names)
    )
    return pd.DataFrame({"text": bunch.data, "category": names})


def load_breast_cancer() -> pd.DataFrame:
    """Load the Wisconsin breast-cancer diagnostic dataset as a tidy DataFrame.

    Wraps :func:`sklearn.datasets.load_breast_cancer` (bundled, no download) into a pandas-first
    frame: the 30 named numeric features plus the diagnosis ``target``. One row = one
    tumour's fine-needle-aspirate measurements.

    Returns
    -------
    pandas.DataFrame, shape (569, 31)
        The 30 feature columns (e.g. ``mean radius``, ``worst concave points``) and ``target``
        (int: **0 = malignant, 1 = benign** — scikit-learn's convention). No missing values.

    When to use
    -----------
    For the demanding logistic-regression case (module 03, NB 6): a real diagnostic problem with
    calibration, threshold choice under asymmetric cost, and L1 feature selection. The same dataset
    KNN met in module 01 NB 5.

    Notes
    -----
    scikit-learn encodes **malignant as 0 and benign as 1**. A task that treats *malignant* as the
    positive class (the costly miss) should flip it: ``y = (df["target"] == 0)``.

    References
    ----------
    Street WN, Wolberg WH, Mangasarian OL (1993). Nuclear feature extraction for breast tumor
    diagnosis. Proc. SPIE 1905, Biomedical Image Processing. https://doi.org/10.1117/12.148698

    Examples
    --------
    >>> df = load_breast_cancer()  # doctest: +SKIP
    >>> df.shape  # doctest: +SKIP
    (569, 31)
    """
    from sklearn.datasets import load_breast_cancer as _sk_load_breast_cancer

    logger.info("Loading the Wisconsin breast-cancer dataset (scikit-learn bundled) ...")
    bunch = _sk_load_breast_cancer()
    df = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    df["target"] = bunch.target
    logger.info(
        "Loaded %d tumours, %d features (target: 0=malignant, 1=benign).",
        df.shape[0],
        len(bunch.feature_names),
    )
    return df
