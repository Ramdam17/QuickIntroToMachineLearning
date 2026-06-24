"""Smoke tests for the graphic-charter library — the charter must load and draw without error."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend for CI / tests

import numpy as np
import pandas as pd
import pytest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

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


def test_plot_decision_boundary_dataframe_string_labels() -> None:
    df = pd.DataFrame(
        {
            "bill_length_mm": [39.0, 49.0, 40.0, 48.0],
            "flipper_length_mm": [190.0, 216.0, 188.0, 214.0],
        }
    )
    y = pd.Series(["Adelie", "Gentoo", "Adelie", "Gentoo"])
    clf = KNeighborsClassifier(n_neighbors=1).fit(df.to_numpy(), y.to_numpy())
    fig = viz.plot_decision_boundary(clf, df, y, resolution=30)
    ax = fig.axes[0]
    assert ax.get_xlabel() == "bill_length_mm"
    assert ax.get_ylabel() == "flipper_length_mm"


def test_plot_roc_curve_returns_figure() -> None:
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    fig = viz.plot_roc_curve(y, scores, label="demo")
    assert fig is not None


def test_plot_score_threshold_returns_figure() -> None:
    scores = np.array([-1.0, -0.2, 0.3, 0.9])
    y = np.array([0, 0, 1, 1])
    fig = viz.plot_score_threshold(scores, y, threshold=0.0)
    assert fig is not None


def test_plot_train_test_curve_returns_figure() -> None:
    fig = viz.plot_train_test_curve([1, 2, 3], [0.3, 0.2, 0.1], [0.35, 0.25, 0.30], xlabel="degree")
    assert fig is not None


def test_plot_calibration_curve_returns_figure() -> None:
    y = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    p = np.array([0.1, 0.4, 0.35, 0.8, 0.9, 0.2, 0.7, 0.3])
    fig = viz.plot_calibration_curve(y, p, n_bins=3)
    assert fig is not None


def test_plot_feature_importances_returns_figure_and_ranks_top() -> None:
    names = ["a", "b", "c", "d", "e"]
    importances = np.array([0.05, 0.40, 0.10, 0.30, 0.15])
    fig = viz.plot_feature_importances(names, importances, top=3)
    assert fig is not None
    ax = fig.axes[0]
    # Top-3 only (b=0.40, d=0.30, e=0.15), largest at the top: ticks bottom->top = ["e", "d", "b"].
    tick_labels = [t.get_text() for t in ax.get_yticklabels()]
    assert tick_labels == ["e", "d", "b"]
    # An std vector draws without error (error bars).
    fig_std = viz.plot_feature_importances(
        names, importances, std=np.full(5, 0.01), top=5, label="MDI"
    )
    assert fig_std is not None


def test_plot_svm_decision_returns_figure_and_rings_support_vectors() -> None:
    X = np.array([[-2.0, -2.0], [-1.8, -2.1], [-2.1, -1.7], [2.0, 2.0], [1.9, 1.7], [2.1, 2.2]])
    y = np.array([0, 0, 0, 1, 1, 1])
    clf = SVC(kernel="linear", C=1e6).fit(X, y)
    fig = viz.plot_svm_decision(clf, X, y, resolution=40)
    assert fig is not None
    # The street's support vectors are ringed: a legend entry "support vectors" is drawn.
    legends = [ax.get_legend() for ax in fig.axes]
    labels = [t.get_text() for lg in legends if lg is not None for t in lg.get_texts()]
    assert "support vectors" in labels


def test_plot_svm_decision_rejects_multiclass() -> None:
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(center, 0.3, size=(6, 2)) for center in (-3.0, 0.0, 3.0)])
    y = np.array([0] * 6 + [1] * 6 + [2] * 6)
    clf = SVC(kernel="linear").fit(X, y)  # 3 classes -> 2-D decision_function, no single street
    with pytest.raises(ValueError):
        viz.plot_svm_decision(clf, X, y, resolution=30)
