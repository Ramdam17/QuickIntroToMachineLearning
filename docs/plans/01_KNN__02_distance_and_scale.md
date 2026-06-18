# Notebook plan — 01_KNN / 02_distance_and_scale

> Status: **APPROVED** (2026-06-17, by Rémy). Notebook plans validated by Rémy alone; reviewers return
> on the built notebook. Drives `notebooks/01_KNN/02_distance_and_scale.ipynb`.

## Context

NB 2 of the k-NN chapter. NB 1 established the rule (the neighbourhood vote) and that "nearest" is
decided by **distance**. This notebook makes the load-bearing consequence concrete: **because k-NN is
pure distance, the scale of each feature changes who the neighbours are.** A feature measured on a
larger range silently dominates the distance, the neighbours flip, and accuracy collapses — until we
**standardize** (the NB 11 payoff, fit on train only), which recovers it. Along the way we introduce a
second distance, **Manhattan (L1)** vs **Euclidean (L2)**, *in service of* the scale idea (any distance
suffers the trap), and show honestly that here the **scale matters far more than the metric choice**.
One concept: **the scale trap**. Prereqs: NB 1, plus 02 (Euclidean distance) and 11 (standardization,
fit-on-train).

## Design (measured — `make_moons(n_samples=300, noise=0.30, random_state=0)`, stratified 70/30 → 210/90, by-hand k-NN k=5)

- **The scale trap (single stratified split, by-hand k-NN k=5).** Rescale feature 2 by ×50; raw
  accuracy falls and standardizing recovers it **exactly**:

  | feature-2 scale | raw test acc | standardized (train-stats) acc |
  |---|---|---|
  | ×1 (original) | 0.956 | 0.956 |
  | ×10 | 0.878 | 0.956 |
  | ×20 | 0.811 | 0.956 |
  | **×50** | **0.733** | **0.956** |
  | ×100 | 0.700 | 0.956 |

  We use **×50** (raw 0.733 → standardized 0.956, a ~22-point collapse-and-recovery). Standardizing
  recovers the *exact* original accuracy because z-scoring `x·50` equals z-scoring `x` — it erases the
  artificial scale. **Why the big axis dominates:** at ×50, feature-1 span = **4.40** vs feature-2 span
  = **149.9** (~34×), so the squared-difference sum is governed by feature 2.
- **Euclidean vs Manhattan (by hand).** A worked example shows L1 ≠ L2 off-axis (e.g. from `q=(0,1.2)`:
  to `A=(0,0)` both = 1.200; to `B=(1,1)` Euclidean = 1.020 but Manhattan = 1.200 — Euclidean ranks B
  nearer, Manhattan ties). Final coordinates fixed at build to show a **clean ranking flip** (measured).
  But on this isotropic 2-D data the metric barely changes accuracy: **Euclidean 0.956 vs Manhattan
  0.944** (standardized). Honest scoping: **scale ≫ metric** here; the metric as a *tuned* dial is NB 4,
  its deeper geometry (L1/L2/L∞, Mahalanobis) is NB 6.
- **Evaluation choice:** a single stratified 70/30 split (consistent with NB 1) — enough to show the
  collapse/recovery without dragging in CV mechanics; rigorous CV returns in NB 3 / NB 5. Stated.

## Library additions / figures

**None to `src/`.** Reuse `viz.plot_decision_boundary` for the key raw-vs-standardized boundary figure
by passing it a **tiny in-notebook `ByHandKNN` wrapper** (a 3-line object: `fit` stores the data,
`predict` does the NB 1 vote over a grid) — this keeps the course's consistent boundary styling while
staying **entirely by hand** (no `KNeighborsClassifier`; that is NB 4). The by-hand distance functions
(Euclidean, Manhattan) are defined in-notebook. `pytest` stays 14.
*(Alternative if preferred: a one-off in-notebook meshgrid + `contourf`, like NB 1's figure, with no
class. Recommended path is the wrapper + `plot_decision_boundary` for visual consistency.)*

## Cell-by-cell (~19 cells; intuition → by-hand → "Read the figure")

1. (md) **Header** — `# 02 — Distance & the scale trap`; *Module 01 · k-Nearest Neighbours — notebook 2
   of 6*; purpose; `Prerequisites: 01, plus 02 (distance) & 11 (standardization)`; objectives (compute
   Euclidean & Manhattan distance by hand; see that k-NN is pure distance, so feature scale decides the
   neighbours; watch accuracy collapse when one feature dominates; standardize — fit on train — to
   recover it; judge metric vs scale honestly); warm welcome that builds on "your first method".
2. (code) Imports (numpy, matplotlib, `make_moons`, `train_test_split`; `ml_course` viz +
   `CLASS_CYCLE`/`COLORS`) + `np.random.seed(0)` + `viz.use_course_style()` + data + stratified 70/30
   split; a small by-hand `knn_predict(Xtr, ytr, Xq, k=5, metric=...)` (the NB 1 vote, now with a metric
   switch). Print sizes.
3. (md) **Recap & footing** — from NB 1: k-NN predicts by the neighbourhood vote, and "nearest" is set
   by **distance**. From NB 02: Euclidean distance. From NB 11: standardization (z-score, **fit on
   train**). New here: distance depends on each feature's **units/scale**, and k-NN — being nothing but
   distance — is at its mercy. One concept: the scale trap.
4. (md) **Two ways to measure "near"** — Euclidean (straight line, L2) and Manhattan (city-block, L1),
   each with its formula. Re-establish L2 (NB 02); introduce L1 plainly.
5. (code) **by hand**: Euclidean and Manhattan distance from a query to two training points; print both
   and show they can rank the neighbours differently.
6. (md) **Read the output** — along one axis the two agree; off-axis they differ, and the metric can
   change who counts as "nearest". A real but **secondary** dial (we return to its size at the end).
7. (md) **k-NN is pure distance → it is sensitive to units** — intuition: a feature on a much larger
   range (think grams vs kilograms, mm vs km) dominates the squared-difference sum, so the neighbours —
   and the prediction — are decided almost entirely by that one feature. We will provoke this on purpose.
8. (code) **rescale feature 2 ×50**; print the two feature spans (4.40 vs 149.9); scatter **raw vs
   rescaled** (two panels) to *see* the vertical stretch.
9. (md) **Read the figure** — same points, same classes; only the vertical yardstick changed. Distances
   are now governed by the stretched axis, so points that are close horizontally can look far apart.
10. (code) **by-hand k-NN(5) accuracy on the rescaled data**: raw test acc **0.733**, vs the original
    **0.956**. The collapse, measured.
11. (md) **Read the output** — accuracy fell from 0.956 to 0.733 although nothing about the classes
    changed — we only changed the units. k-NN faithfully followed the inflated axis and mislabelled
    points near the (now invisible-to-it) horizontal structure.
12. (md) **The fix: standardize (NB 11)** — subtract the mean, divide by the standard deviation, **fit
    on the training set only** (the test set borrows the train statistics — no leakage). This puts every
    feature on a comparable scale, so none dominates by accident.
13. (code) standardize with **train** mean/std; by-hand k-NN(5) test acc → **0.956** (recovers exactly).
    Print raw vs standardized side by side.
14. (code) **figure (key)** — raw-vs-standardized **decision boundary**, two panels, via
    `viz.plot_decision_boundary` fed a tiny by-hand `ByHandKNN`: left = rescaled-raw (broken, banded),
    right = standardized (the clean two-crescent split recovered).
15. (md) **Read the figure** — left: the boundary is dictated by the big axis → near-horizontal bands
    that cut through a crescent; right: the familiar moons boundary returns. Standardizing did not change
    k-NN — it changed the yardstick so "near" means what we intend.
16. (md) **Does the metric matter too?** — Euclidean vs Manhattan on the standardized data: **0.956 vs
    0.944** — barely. On isotropic 2-D data the **scale** of features matters far more than the **metric**
    choice. (Metric as a tuned dial → NB 4; metric geometry L1/L2/L∞, Mahalanobis → NB 6.)
17. (md) **Your turn** — (a) feature 1 in km, feature 2 in mm: which dominates the distance, and what
    does standardizing do? (b) why must the scaler be fit on the **training set only** (NB 11)? (c) you
    standardize and accuracy does not move at all — what does that tell you about the original scales?
18. (md) **What you built** + vocabulary — Euclidean (L2) / Manhattan (L1) distance, feature scale, the
    **scale trap** (feature domination), standardization (z-score, fit on train), scale-invariance.
19. (md) **References** — ISLR §2.2.3 & §6 (scaling for distance methods), DOI 10.1007/978-1-0716-1418-1;
    ESL §2.5 (k-NN; distance), DOI 10.1007/978-0-387-84858-7; (Minkowski/Manhattan named). `Previous: 01
    — Predict by the neighbourhood vote` · `Next: 03 — The k dial`.

## Honest limits / no pre-emption

- One concept — the **scale trap**. NB 2 does **not** choose k or show the bias-variance k-dial (NB 3),
  does **not** use `KNeighborsClassifier` or tune the metric (NB 4), and does **not** study metric
  geometry / Mahalanobis (NB 6). The metric is introduced only as a second distance to make the point
  that *any* distance suffers the scale trap, and its smallness-vs-scale is stated.
- The ×50 rescale is a **controlled, clearly labelled** demonstration of a real phenomenon (feature
  domination), not a property of `make_moons`. Stated.
- Accuracy uses a **single** stratified split (not CV) to keep the focus on scale; this is named, with
  CV deferred to NB 3 / NB 5.
- Standardization here recovers the *exact* original accuracy because the rescale is a pure axis
  stretch; on real data standardizing usually helps but need not return an identical number — flagged.

## Verification

Measured anchors (×50 raw 0.733 → standardized 0.956; spans 4.40 vs 149.9; Euclidean 0.956 vs Manhattan
0.944; the by-hand distance example) re-run in the notebook and reconciled into prose at build. Runs
top-to-bottom (nbconvert to /tmp; output-free); `check_no_hardcoded_hex` passes (figures use
`ml_course.colors`/`viz`); `gen_llms_txt` re-run; `pytest` green (14, no `src/` change); both reviewers
pass (no BLOCK); Rémy validates visually; commit + merge `notebook → chapter`.
