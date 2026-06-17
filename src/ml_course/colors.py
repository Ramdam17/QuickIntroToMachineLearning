"""Single source of truth for the course's graphic-charter palette.

Soft warm pastels, a family resemblance with the PPSP courses (and the Quantum Optimal Transport
course). Never hardcode hex in notebooks or modules — import from here. Roles are named for
machine-learning use (class identities, train/test, error/correct) so notebook code reads
intention, not hex.

When to use
-----------
Import :data:`COLORS` for a named role, :data:`CLASS_CYCLE` to colour categorical classes in
order, and the ``CMAP_*`` colormaps for heatmaps (probabilities, confusion matrices, residuals).
"""

from __future__ import annotations

from matplotlib.colors import LinearSegmentedColormap

COLORS: dict[str, str] = {
    # Class identities — up to five well-separated categories.
    "class_a": "#9B8FD4",  # soft periwinkle
    "class_b": "#E8B864",  # warm amber
    "class_c": "#88C9A1",  # soft sage
    "class_d": "#7EB8DA",  # sky blue
    "class_e": "#F4A4B8",  # rose
    # Roles
    "model": "#7EB8DA",  # the model / decision surface
    "highlight": "#F4A4B8",  # emphasis / the punchline
    "train": "#9B8FD4",  # training data
    "test": "#E8B864",  # held-out data
    # Diverging — residuals, correlations, correct vs. misclassified.
    "error": "#E17055",  # coral — misclassified / negative
    "zero": "#FFFFFF",
    "correct": "#5BB8B0",  # soft teal — correct / positive
    # Neutrals
    "grid": "#E2E2EC",
    "text": "#2C3E50",
    "muted": "#9AA0AA",
    "background": "#FFFFFF",
}

# Order in which to colour categorical classes (matches class_a..class_e).
CLASS_CYCLE: list[str] = [
    COLORS["class_a"],
    COLORS["class_b"],
    COLORS["class_c"],
    COLORS["class_d"],
    COLORS["class_e"],
]

# Sequential: white -> model blue. For probabilities / decision confidence.
CMAP_PROBA = LinearSegmentedColormap.from_list("ml_proba", [COLORS["background"], COLORS["model"]])
# Sequential: white -> periwinkle. For counts / feature importance.
CMAP_COUNT = LinearSegmentedColormap.from_list(
    "ml_count", [COLORS["background"], COLORS["class_a"]]
)
# Diverging: coral -> white -> teal. For residuals / correlations (zero at white).
CMAP_DIVERGING = LinearSegmentedColormap.from_list(
    "ml_diverging", [COLORS["error"], COLORS["zero"], COLORS["correct"]]
)
