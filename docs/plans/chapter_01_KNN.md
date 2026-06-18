# Chapter plan — 01_KNN (k-Nearest Neighbours)

> Status: **APPROVED** (2026-06-17, by Rémy; reviewer-gated — ml-expert + pedagogy both REVISE→
> incorporated, no BLOCK). The course's **first real method**. Per the per-method arc: NB 1–3
> fundamentals (one concept each, by hand before the library), NB 4 the estimator & its parameters,
> NB 5 a demanding practical case; **plus an optional NB 6 "Advanced"** (distances & choosing k) — a
> deliberate, Rémy-approved exception to the usual 5-notebook ceiling, justified because k-NN is *the*
> distance method. Drives the notebook loop in `docs/WORKFLOW.md`.

## Context

k-NN is the most intuitive classifier — predict a point's label by the **majority vote of its nearest
neighbours** — and it makes a perfect first method because it ties the whole on-ramp together:
distance (NB 02), the train/test split (NB 04), **standardization** (NB 11 — load-bearing here, since
k-NN is pure distance), **cross-validation** to choose k (NB 10), the **decision boundary**
(`viz.plot_decision_boundary`, NB 05), and **over-/under-fitting** (NB 09 — now the *k* dial). It is
also our first **lazy / instance-based** learner: "fit" only stores the data; all the work happens at
predict — an honest contrast to the eager learners ahead.

## Datasets (measured; penguins is too separable to teach k-NN — see below)

- **NB 1–4: `make_moons`** (2-D, visualizable, with class overlap). Measured: penguins' binary
  2-feature subset is so separable that k-NN(5) gives **identical** predictions raw vs standardized
  (0/69 flips) and k=1 vs large-k barely differ — it cannot show the scale trap or the k-dial. moons
  can: k=1 train 1.00 / test 0.93 (overfit) → large-k underfit; and a **rescaled-axis** moons (one axis
  ×50) shows the scale trap (raw CV 0.79 → standardized 0.92).
- **NB 5: `breast_cancer`** (scikit-learn built-in, **offline**; binary, 30-D, n=569). Scaled k-NN CV
  ≈ 0.965; a realistic medical task for the full honest workflow. The **curse of dimensionality** is
  *felt* by appending pure-noise dimensions and watching k-NN fall (measured 0.965 → 0.78 at +2000).
- All offline; seeds fixed; colours from `ml_course.colors`; "Read the figure" after every figure; a
  "Your turn" per notebook; a running k-NN vocabulary box.

## The notebooks (one concept each for 1–3; `Prereqs` declared)

| NB | Title | Prereqs | The one concept | Done by hand | Key figure → "Read the figure" |
|----|-------|---------|-----------------|--------------|-------------------------------|
| 1 | Predict by the neighbourhood vote | 02, 04 | k-NN: the k nearest neighbours cast a **majority vote**; *k* = neighbourhood size; **lazy** learner (fit = store the data) — and its cost **felt**: time `fit` (≈ instant) vs the per-query distance work, noting predict cost grows with n | distances from a query to all training points → take the k smallest → vote (k = 1, 3, 5); time fit vs a prediction | moons + a query with its k neighbours highlighted and the vote shown |
| 2 | Distance & the scale trap | 1 (02, 11) | **k-NN is pure distance → the larger-range feature dominates → standardize** (the NB 11 payoff, sharper than for nearest centroid). Euclidean vs Manhattan are introduced *in service of* this (different metrics, same scale problem); the metric as a *tunable* dial is NB 4 | compute both distances by hand; rescale one feature → k-NN accuracy collapses → standardize to recover it | k-NN boundary on rescaled moons: raw (broken) vs standardized (recovered) |
| 3 | The *k* dial: under- vs over-fitting | 1–2 (05, 09, 10) | *k* = the **bias–variance knob**: k = 1 overfits (jagged, train 1.0), large k underfits (smooth); **choose k by CV** (odd-k grid for binary; CV *selects* k, the NB 5 held-out test *evaluates* it). k-NN's **inductive bias** vs NB 05's single bisector: the boundary follows the data **locally** | sweep odd k → train vs test error (the k-dial); pick k with `cross_val_score` (reuse NB 10) | boundaries k = 1 / 15 / 151 (contrast NB 05's one hyperplane); train–test error vs k + CV accuracy vs k |
| 4 | The estimator & its parameters | 1–3 (05) | `KNeighborsClassifier`: **n_neighbors**, **weights** (uniform vs distance), **metric / p** (the NB 2 metric, now a *tuned* dial; the deep metric-geometry study is **NB 6**). These change the **boundary shape** more than the headline score on isotropic moons; `weights="distance"` visibly resists over-smoothing **at large k**. Failure mode: **even k → tie**, resolved **deterministically** by sklearn (lowest class label) — *shown* — hence odd k for binary | swap weights & metric → watch the boundary; force an even-k tie and show the deterministic break | boundary panels: uniform vs distance (at large k); a metric comparison |
| 5 | Demanding case: breast cancer & the curse | 1–4 (05, 10, 11) | full honest workflow on a real 30-D set (`Pipeline(StandardScaler, KNN)` under CV to pick k; one held-out eval; error analysis; **when to / not to use k-NN**) **+ the curse of dimensionality, felt**: append noise dims **at the signal's scale** → distances concentrate (near/far ratio collapses) → k-NN degrades | Pipeline under CV; final held-out score; add scaled-noise dims and plot the fall; note predict-time grows | confusion-matrix/scores; **accuracy vs #noise-dims** + the near/far-ratio collapse as the *why* |
| 6 | Advanced: distances & choosing k *(optional)* | 1–5 | **the metric shapes classification** + **rigorous k/metric selection**: Minkowski p = 1/2/∞ (Manhattan/Euclidean/**Chebyshev**) as neighbourhood *shape* (diamond/circle/square), **Mahalanobis** (covariance-aware) & **cosine**; the **metric × curse** (smaller/fractional p in high-d, Aggarwal 2001 — ties to NB 5); choosing k by CV → a peek at **nested CV**; and the honest clarification that **silhouette selects k-means clusters, not k-NN neighbours** (supervised vs unsupervised model selection) | compute distances under each metric; **draw the unit balls** (L1/L2/L∞); measure metric impact where geometry actually matters; a nested-CV pass | unit balls L1/L2/L∞; metric impact on a boundary/accuracy; a nested-CV scheme |

## Library additions

Expected **none** — reuse `viz.plot_decision_boundary` (boundary vs k), `viz.plot_train_test_curve`
(error vs k; accuracy vs #noise-dims), `viz.plot_confusion_matrix` (NB 5), `StandardScaler`/`Pipeline`
(NB 11), `StratifiedKFold`/`cross_val_score` (NB 10). The NB 1 "query + its neighbours" picture is a
one-off in-notebook figure (charter colours). If a per-notebook plan finds a genuinely reusable helper
(e.g. a neighbours-highlight), it is added then with a test — decided at notebook planning, not now.
`pytest` stays 14 unless a helper is added.

## Honest limits stated in the notebooks

- k-NN is **lazy / instance-based**: fit only stores the data; all computation (and the whole dataset
  in memory) happens at predict — slow and memory-heavy on large data. The cost is **felt**, not just
  stated (NB 1 times fit vs a query; NB 5 notes predict-time growing with n) — the honest contrast to
  the eager learners ahead.
- k-NN also does **regression** (average the neighbours' targets) and emits a vote-fraction
  `predict_proba` (coarse — only k+1 levels — and poorly calibrated; the kind of score one would
  threshold à la NB 08). We focus on the classification vote; these faces are **named, not taught**.
- moons is synthetic, used because penguins is **too separable to teach k-NN** (measured: 0/69 flips
  raw-vs-scaled); stated. **wine** is named in one sentence as a real-world echo of the scale trap —
  no load, no figure (the chapter stays binary; "library additions: none").
- The scale trap (rescaled-axis moons) and the curse (noise dims injected **at the signal's scale**)
  are **controlled, clearly labelled** demonstrations of real phenomena — distance domination and
  distance concentration (the curse's "Read the figure" names the near/far-ratio collapse, not just a
  falling curve).
- Even *k* can **tie**; scikit-learn resolves it **deterministically** by class-label order (not
  randomly) — mildly arbitrary, hence odd k is the default for two classes. The 2-D boundary is only a
  picture of a rule that lives in any dimension. **When to use k-NN:** small, low-dimensional data with
  a meaningful, scaled metric; **when not:** large n (slow predict) or high-dimensional/noisy data (the
  curse) — both demonstrated in NB 5.

## References (grounding; DOIs verified at build, per the chapter-00 precedent)

- T. M. Cover, P. E. Hart (1967), *Nearest neighbor pattern classification*, IEEE Trans. Information
  Theory 13(1):21–27. DOI: 10.1109/TIT.1967.1053964 — the method's origin + the asymptotic bound
  R\* ≤ R_NN ≤ 2R\* (why so simple a rule is surprisingly strong).
- E. Fix, J. L. Hodges (1951), *Discriminatory analysis: nonparametric discrimination* — the root.
- ISLR (James, Witten, Hastie, Tibshirani, 2021) §2.2.3 & ch. 4. DOI: 10.1007/978-1-0716-1418-1.
- ESL (Hastie, Tibshirani, Friedman, 2009) §2.5 & §13.3 (k-NN; high-dimensional distance concentration).
  DOI: 10.1007/978-0-387-84858-7.
- K. Beyer, J. Goldstein, R. Ramakrishnan, U. Shaft (1999), *When is "nearest neighbor" meaningful?*,
  ICDT — the distance-concentration result behind the curse (NB 5).
- C. C. Aggarwal, A. Hinneburg, D. A. Keim (2001), *On the surprising behavior of distance metrics in
  high dimensional space*, ICDT — why smaller/fractional p can stay discriminating in high-d (NB 6).
- P. J. Rousseeuw (1987), *Silhouettes: a graphical aid to the interpretation and validation of cluster
  analysis*, J. Comp. Appl. Math. 20:53–65. DOI: 10.1016/0377-0427(87)90125-7 — the silhouette
  coefficient, for **clustering** k (the honest contrast with k-NN's k in NB 6).
- The "k ≈ √n" rule of thumb is flagged as a **heuristic**, not a theorem.

## Verification (per notebook, at its commit)

Measured numbers reconciled into prose at build; runs top-to-bottom; outputs cleared;
`check_no_hardcoded_hex.py` passes; `uv run pytest` green; `gen_llms_txt.py` re-run; `course_map.md`
01_KNN section aligned to these titles. Both reviewers pass (no BLOCK) on each notebook; Rémy validates
visually; commit per notebook; **chapter close via PR into `main`** (protected).

## Per-notebook status (update as the chapter is built)

| NB | Branch | Status |
|----|--------|--------|
| 1 | `notebook/01_KNN__01_neighbourhood_vote` | **done** — built, both reviewers PASS, Rémy validated, merged to `chapter/01_KNN` (query corrected to q=(-0.23,0.75) for honesty) |
| 2 | `notebook/01_KNN__02_distance_and_scale` | **done** — built, both reviewers PASS, Rémy validated, merged to `chapter/01_KNN` |
| 3 | `notebook/01_KNN__03_the_k_dial` | not started |
| 4 | `notebook/01_KNN__04_estimator_and_parameters` | not started |
| 5 | `notebook/01_KNN__05_demanding_case_curse` | not started |
| 6 | `notebook/01_KNN__06_advanced_distances_and_k` | not started (optional Advanced) |
