"""Tests for the offline teaching datasets."""

from __future__ import annotations

import pandas as pd

from ml_course import datasets


def test_load_penguins_shape_and_columns() -> None:
    df = datasets.load_penguins()
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == [*datasets.PENGUINS_FEATURES, datasets.PENGUINS_TARGET]
    assert df.shape == (274, 3)


def test_load_penguins_is_clean_and_binary() -> None:
    df = datasets.load_penguins()
    assert not df.isna().any().any()  # vendored subset has no missing values
    assert sorted(df[datasets.PENGUINS_TARGET].unique()) == ["Adelie", "Gentoo"]
    # Both classes are well represented (no degenerate split).
    counts = df[datasets.PENGUINS_TARGET].value_counts()
    assert counts.min() >= 100


def test_penguins_xy_split() -> None:
    X, y = datasets.penguins_xy()
    assert list(X.columns) == datasets.PENGUINS_FEATURES
    assert y.name == datasets.PENGUINS_TARGET
    assert len(X) == len(y) == 274
