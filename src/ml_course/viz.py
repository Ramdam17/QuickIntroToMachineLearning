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
    X: np.ndarray,
    y: np.ndarray,
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
    X : numpy.ndarray, shape (n_samples, 2)
        The two features to plot, in plotting units.
    y : numpy.ndarray, shape (n_samples,)
        Integer class labels ``0..n_classes-1`` used to colour the points.
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
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError(f"X must have shape (n_samples, 2); got {X.shape}.")

    n_classes = int(np.max(y)) + 1
    class_colors = CLASS_CYCLE[:n_classes]
    region_cmap = ListedColormap(class_colors)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
    else:
        fig = ax.figure

    pad = 0.5
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = np.asarray(model.predict(grid)).reshape(xx.shape)

    ax.contourf(xx, yy, zz, alpha=0.25, levels=np.arange(n_classes + 1) - 0.5, cmap=region_cmap)
    for cls in range(n_classes):
        mask = y == cls
        ax.scatter(
            X[mask, 0],
            X[mask, 1],
            color=class_colors[cls],
            edgecolor=COLORS["text"],
            linewidth=0.6,
            s=45,
            label=f"class {cls}",
        )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
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
