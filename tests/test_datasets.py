"""Tests for the offline teaching datasets.

The full dataset is fetched and cached on first use, so the first run of these tests needs network;
later runs read the local cache and run offline.
"""

from __future__ import annotations

import pandas as pd

from ml_course import datasets


def test_load_penguins_full_schema() -> None:
    full = datasets.load_penguins_full()
    assert isinstance(full, pd.DataFrame)
    assert full.shape == (344, 7)
    assert datasets.PENGUINS_TARGET in full.columns
    for col in (*datasets.PENGUINS_FULL_NUMERIC, *datasets.PENGUINS_FULL_CATEGORICAL):
        assert col in full.columns
    # Three species in the full set (the subset keeps only two).
    assert sorted(full[datasets.PENGUINS_TARGET].unique()) == ["Adelie", "Chinstrap", "Gentoo"]
    # The full set deliberately keeps missing values (for the preprocessing lessons).
    assert full.isna().any().any()


def test_load_penguins_shape_and_columns() -> None:
    df = datasets.load_penguins()
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == [*datasets.PENGUINS_FEATURES, datasets.PENGUINS_TARGET]
    assert df.shape == (274, 3)


def test_load_penguins_is_clean_and_binary() -> None:
    df = datasets.load_penguins()
    assert not df.isna().any().any()  # the derived subset has no missing values
    assert sorted(df[datasets.PENGUINS_TARGET].unique()) == ["Adelie", "Gentoo"]
    # Both classes are well represented (no degenerate split).
    counts = df[datasets.PENGUINS_TARGET].value_counts()
    assert counts.min() >= 100


def test_penguins_xy_split() -> None:
    X, y = datasets.penguins_xy()
    assert list(X.columns) == datasets.PENGUINS_FEATURES
    assert y.name == datasets.PENGUINS_TARGET
    assert len(X) == len(y) == 274


def test_load_newsgroups_schema() -> None:
    # First run fetches ~14 MB and caches it; later runs read the cache (offline).
    df = datasets.load_newsgroups(["sci.med", "comp.graphics"], subset="train")
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["text", "category"]
    assert sorted(df["category"].unique()) == ["comp.graphics", "sci.med"]
    assert len(df) > 0
    assert df["text"].map(type).eq(str).all()


def test_load_breast_cancer_schema() -> None:
    df = datasets.load_breast_cancer()
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (569, 31)  # 30 features + target
    assert "target" in df.columns
    assert "mean radius" in df.columns
    assert "worst concave points" in df.columns
    # scikit-learn convention: target in {0 (malignant), 1 (benign)}; no missing values.
    assert sorted(df["target"].unique()) == [0, 1]
    assert not df.isna().any().any()
