"""Course plotting style and reusable ML visualizations.

A clean, high-contrast light style with soft pastel accents, so figures read well both in notebooks
and when exported. Colours come from :mod:`ml_course.colors` (single source of truth) — never
hardcode hex here or in notebooks.

When to use
-----------
Call :func:`use_course_style` once near the top of every notebook. Use
:func:`plot_decision_boundary` for any 2-D classifier and :func:`plot_confusion_matrix` for results.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

from ml_course.colors import CLASS_CYCLE, CMAP_PROBA, COLORS

_STYLE: dict[str, object] = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": COLORS["grid"],
    "axes.grid": True,
    "grid.color": COLORS["grid"],
    "grid.linewidth": 0.6,
    "text.color": COLORS["text"],
    "axes.labelcolor": COLORS["text"],
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 11,
    "legend.frameon": False,
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
}


def use_course_style() -> None:
    """Apply the course's matplotlib style globally (idempotent).

    Examples
    --------
    >>> from ml_course import viz
    >>> viz.use_course_style()
    """
    mpl.rcParams.update(_STYLE)


def plot_decision_boundary(
    model,
    X,
    y,
    *,
    resolution: int = 300,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot a 2-D classifier's decision regions with the training points on top.

    Evaluates ``model.predict`` on a dense grid spanning the data and fills each region with the
    class colour, then scatters the samples. Makes "what the model decided, everywhere" visible —
    the central figure of the fundamentals notebooks.

    Parameters
    ----------
    model : object
        A fitted classifier exposing ``predict(X) -> labels``, with ``X`` of shape (n, 2).
    X : pandas.DataFrame or numpy.ndarray, shape (n_samples, 2)
        The two features to plot. A DataFrame's column names become the axis labels.
    y : array-like, shape (n_samples,)
        Class labels, integer or string. Region/point colours follow ``sorted(unique(y))``.
    resolution : int, default 300
        Grid points per axis. Higher is smoother and slower.
    ax : matplotlib.axes.Axes, optional
        Axis to draw on; a new figure is created when omitted.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the plot.

    When to use
    -----------
    Any 2-D classification demo (KNN, SVM, trees, logistic regression). For higher dimensions,
    reduce to 2-D first or plot a different diagnostic.

    Examples
    --------
    >>> from sklearn.neighbors import KNeighborsClassifier
    >>> import numpy as np
    >>> X = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    >>> y = np.array([0, 0, 1, 1])
    >>> clf = KNeighborsClassifier(n_neighbors=1).fit(X, y)
    >>> _ = plot_decision_boundary(clf, X, y)
    """
    if hasattr(X, "columns"):  # a DataFrame: take axis labels from the columns
        feature_names = [str(c) for c in X.columns[:2]]
        X_arr = X.to_numpy(dtype=float)
    else:
        X_arr = np.asarray(X, dtype=float)
        feature_names = ["x1", "x2"]
    if X_arr.ndim != 2 or X_arr.shape[1] != 2:
        raise ValueError(f"X must have shape (n_samples, 2); got {X_arr.shape}.")

    y_arr = np.asarray(y)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
    else:
        fig = ax.figure

    pad = 0.5
    x_min, x_max = X_arr[:, 0].min() - pad, X_arr[:, 0].max() + pad
    y_min, y_max = X_arr[:, 1].min() - pad, X_arr[:, 1].max() + pad
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = np.asarray(model.predict(grid))

    # Map labels (int OR string) to integer codes, covering both the data and the predictions.
    classes = sorted(set(y_arr.tolist()) | set(preds.tolist()))
    code = {label: i for i, label in enumerate(classes)}
    n_classes = len(classes)
    class_colors = [CLASS_CYCLE[i % len(CLASS_CYCLE)] for i in range(n_classes)]
    region_cmap = ListedColormap(class_colors)

    zz = np.array([code[p] for p in preds.tolist()]).reshape(xx.shape)
    ax.contourf(xx, yy, zz, alpha=0.25, levels=np.arange(n_classes + 1) - 0.5, cmap=region_cmap)
    for label in classes:
        mask = y_arr == label
        if not mask.any():
            continue
        ax.scatter(
            X_arr[mask, 0],
            X_arr[mask, 1],
            color=class_colors[code[label]],
            edgecolor=COLORS["text"],
            linewidth=0.6,
            s=45,
            label=str(label),
        )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.legend(loc="best")
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str] | None = None,
    *,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot a confusion matrix as an annotated heatmap (true rows, predicted columns).

    Parameters
    ----------
    cm : numpy.ndarray, shape (n_classes, n_classes)
        Counts; ``cm[i, j]`` = samples of true class ``i`` predicted as class ``j``
        (the orientation returned by ``sklearn.metrics.confusion_matrix``).
    class_names : list of str, optional
        Tick labels; defaults to ``"0", "1", ...``.
    ax : matplotlib.axes.Axes, optional
        Axis to draw on; a new figure is created when omitted.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the heatmap.

    Examples
    --------
    >>> import numpy as np
    >>> cm = np.array([[5, 1], [0, 4]])
    >>> _ = plot_confusion_matrix(cm, ["negative", "positive"])
    """
    cm = np.asarray(cm)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f"cm must be square (n_classes, n_classes); got {cm.shape}.")
    n = cm.shape[0]
    names = class_names if class_names is not None else [str(i) for i in range(n)]

    if ax is None:
        fig, ax = plt.subplots(figsize=(1.4 * n + 2, 1.4 * n + 2))
    else:
        fig = ax.figure

    im = ax.imshow(cm, cmap=CMAP_PROBA)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # Annotate each cell; pick a readable text colour against the cell shade.
    threshold = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(n):
        for j in range(n):
            ax.text(
                j,
                i,
                f"{cm[i, j]}",
                ha="center",
                va="center",
                color=COLORS["background"] if cm[i, j] > threshold else COLORS["text"],
            )
    ax.set_xticks(range(n), names)
    ax.set_yticks(range(n), names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.grid(False)
    return fig


def plot_class_balance(y, ax: plt.Axes | None = None) -> plt.Figure:
    """Bar chart of how many examples fall in each class.

    Parameters
    ----------
    y : array-like or pandas.Series, shape (n_samples,)
        Class labels (any hashable type).
    ax : matplotlib.axes.Axes, optional
        Axis to draw on; a new figure is created when omitted.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the bar chart.

    When to use
    -----------
    A first look at class balance before trusting accuracy: under a strong imbalance a constant
    "predict the majority" classifier can score high (see the metrics notebooks).

    Examples
    --------
    >>> import pandas as pd
    >>> _ = plot_class_balance(pd.Series(["a", "a", "b"]))
    """
    counts = pd.Series(y).value_counts().sort_index()
    classes = [str(c) for c in counts.index]
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.figure
    bar_colors = [CLASS_CYCLE[i % len(CLASS_CYCLE)] for i in range(len(classes))]
    ax.bar(classes, counts.to_numpy(), color=bar_colors, edgecolor=COLORS["text"], linewidth=0.6)
    for i, value in enumerate(counts.to_numpy()):
        ax.text(i, value, str(int(value)), ha="center", va="bottom", color=COLORS["text"])
    ax.set_xlabel("class")
    ax.set_ylabel("count")
    ax.grid(False)
    return fig


def plot_feature_histograms(
    df: pd.DataFrame,
    features: list[str],
    by: str | None = None,
    *,
    bins: int = 20,
) -> plt.Figure:
    """Plot one histogram per feature, optionally split by a label column.

    Parameters
    ----------
    df : pandas.DataFrame
        The data; must contain every name in ``features`` (and ``by`` when given).
    features : list of str
        Numeric feature columns to histogram, one panel each.
    by : str, optional
        A categorical column to split on: each class is overlaid as a translucent histogram with
        its own colour. When omitted, a single histogram per feature is drawn.
    bins : int, default 20
        Number of histogram bins.

    Returns
    -------
    matplotlib.figure.Figure
        A figure with one panel per feature (it owns its panels — no ``ax`` argument).

    When to use
    -----------
    A quick read of each feature's shape and spread, and (with ``by``) of how well a single feature
    separates the classes. Bars show raw counts (not densities), so under a strong class imbalance
    the majority class looks taller — read the overlap of the humps, not their absolute height.

    Examples
    --------
    >>> from ml_course import datasets
    >>> df = datasets.load_penguins()
    >>> _ = plot_feature_histograms(df, ["bill_length_mm", "flipper_length_mm"], by="species")
    """
    n = len(features)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.0))
    axes = np.atleast_1d(axes)
    for ax, feature in zip(axes, features, strict=True):
        if by is None:
            ax.hist(
                df[feature].to_numpy(),
                bins=bins,
                color=CLASS_CYCLE[0],
                edgecolor=COLORS["text"],
                linewidth=0.5,
            )
        else:
            for i, group in enumerate(sorted(df[by].unique())):
                values = df.loc[df[by] == group, feature].to_numpy()
                ax.hist(
                    values,
                    bins=bins,
                    color=CLASS_CYCLE[i % len(CLASS_CYCLE)],
                    alpha=0.6,
                    label=str(group),
                    edgecolor="none",
                )
            ax.legend(title=by)
        ax.set_xlabel(feature)
        ax.set_ylabel("count")
    fig.tight_layout()
    return fig
