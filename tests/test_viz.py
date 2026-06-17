"""Smoke tests for the graphic-charter library — the charter must load and draw without error."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend for CI / tests

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from ml_course import colors, viz


def test_palette_has_required_roles() -> None:
    for role in ("class_a", "model", "train", "test", "error", "correct", "grid", "text"):
        assert role in colors.COLORS
    assert len(colors.CLASS_CYCLE) == 5


def test_use_course_style_is_idempotent() -> None:
    viz.use_course_style()
    viz.use_course_style()  # second call must not raise


def test_plot_decision_boundary_returns_figure() -> None:
    X = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    y = np.array([0, 0, 1, 1])
    clf = KNeighborsClassifier(n_neighbors=1).fit(X, y)
    fig = viz.plot_decision_boundary(clf, X, y, resolution=40)
    assert fig is not None


def test_plot_confusion_matrix_returns_figure() -> None:
    cm = np.array([[5, 1], [0, 4]])
    fig = viz.plot_confusion_matrix(cm, ["negative", "positive"])
    assert fig is not None


def test_plot_class_balance_returns_figure() -> None:
    y = pd.Series(["a", "a", "b", "a", "b"])
    fig = viz.plot_class_balance(y)
    assert fig is not None


def test_plot_feature_histograms_panels_and_split() -> None:
    df = pd.DataFrame(
        {
            "f1": [1.0, 2.0, 3.0, 4.0],
            "f2": [10.0, 20.0, 30.0, 40.0],
            "label": ["a", "a", "b", "b"],
        }
    )
    fig_plain = viz.plot_feature_histograms(df, ["f1", "f2"])
    assert len(fig_plain.axes) == 2

    fig_split = viz.plot_feature_histograms(df, ["f1"], by="label")
    assert len(fig_split.axes) == 1
