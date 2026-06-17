# Chapter plan — 00_GettingStarted

> Status: **APPROVED** (2026-06-16, by Rémy). Reviewer-gated: ml-expert + pedagogy both REVISE, no
> BLOCK; all REVISE items incorporated. Drives the notebook loop in `docs/WORKFLOW.md`.

## Context

`00_GettingStarted` is the course's on-ramp: the foundational vocabulary and skills every ML student
must hold *before* meeting the twelve methods. It is **not** one of the twelve methods, so it is
exempt from the per-method arc (NB 1–3 / 4 / 5) and from the 3–5 notebook ceiling — it is a
one-concept-per-notebook **concept progression**. Scope: the **full foundation**, **11 notebooks**
(an EDA notebook was added at the reviewer gate; the validation notebook may split at its own
planning if it exceeds ~20 cells).

Grounded in: ISLR (James, Witten, Hastie, Tibshirani — ch. 2, 4, 5; §2.2 for bias–variance), Géron
*Hands-On ML* (ch. 1–3), Domingos (CACM 2012), Fawcett "An introduction to ROC analysis" (2006) for
NB 08, the nearest-centroid bias from ESL §4.3 / Tibshirani et al. 2002 (PNAS), and the scikit-learn
*Getting Started* + *Common pitfalls* pages. Exact DOIs verified per notebook at the build gate.

## Running choices (the through-line)

- **Dataset:** Palmer penguins, 2 features (`bill_length_mm`, `flipper_length_mm`), **binary** subset
  (Adélie vs Gentoo) — clean-but-not-perfect 2-D separation, so evaluation has honest errors. The
  **positive class is fixed once (Gentoo)** and reused across NB 06–08. Vendored offline.
- **First classifier:** **nearest centroid**, by hand (class means → nearest), then vs
  `sklearn.neighbors.NearestCentroid`. Stated honestly as a *primitive*: its boundary is the
  perpendicular bisector of the two centroids (a hyperplane); it is Bayes-optimal only for equal,
  isotropic-covariance classes with equal priors; it is **scale-sensitive** (flagged in NB 05,
  resolved in NB 11). A named failure case (an elongated/unequal-spread class) is shown.
- **Score for ROC (NB 08):** the **signed squared-distance** `s(x)=d(x,μ₀)²−d(x,μ₁)²`, which is
  **affine** in `x` (`w=−2(μ₀−μ₁)`); sweeping its threshold slides *parallel* boundaries. `predict`
  uses `sign(s)`. By-hand AUC validated against `roc_auc_score` (legit because `NearestCentroid`
  exposes no `decision_function`/`predict_proba`). A teaser of this score closes NB 05.
- **Complexity dial (NB 09–10):** a polynomial-feature decision boundary whose **degree** is the
  knob — a didactic device, **not** one of the twelve methods (pre-empts none of them).
- Everything in English; `viz.use_course_style()`; colours from `ml_course.colors`; seeds fixed and
  documented (a version-echo cell in NB 01); every figure followed by a "Read the figure" paragraph;
  a "Your turn" per notebook (only on material taught so far); a running **ML vocabulary** box.

## The notebooks (one concept each; `Prereqs` declared)

| NB | Title | Prereqs | The one concept | Done by hand | Key figure → "Read the figure" |
|----|-------|---------|-----------------|--------------|-------------------------------|
| 01 | What is machine learning? | — | learn a rule from examples; supervised; classification vs regression; course scope | read the penguins table; name features vs label; pose "bill+flipper → species?"; meet the toolkit (+version echo) | feature-space scatter coloured by species |
| 02 | Features, labels, the feature space | 01 | `X` (n×d), `y`; feature types; row = example; **the mean of a point cloud and Euclidean distance between two points** (the centroid primitives) | build `X`,`y`; compute a mean point and a distance by hand; place a query point | scatter with a class mean and a distance segment drawn |
| 03 | Look before you model (EDA) | 01–02 | inspect before modelling: distributions, class balance, feature scales/ranges | per-feature histograms; class-count bar; a feature-range table; re-read the scatter for overlap | histograms + class-balance bar |
| 04 | Generalize, don't memorize | 01–03 | **stratified** train/test split; the cardinal sin; i.i.d. as a *chosen assumption*; leakage intro; preview fit→predict→evaluate | stratified split with a fixed seed; show "perfect on training" says nothing about new data | train vs test points (charter train/test colours) |
| 05 | Your first classifier: nearest centroid | 02, 04 | fit→predict made concrete; the estimator API; the bisector boundary + its inductive bias & scale-sensitivity | centroids = means (NB 02); assign by nearest distance; wrap `.predict`; predict on test; **close with the signed-score teaser** | decision regions (`plot_decision_boundary`) + centroids + bisector |
| 06 | Is it any good? accuracy + baseline | 04, 05 | accuracy on test; majority baseline (`DummyClassifier`); accuracy's limit under imbalance (the NB 03 imbalance picture); **fix positive = Gentoo** | accuracy by hand; build baseline; a misleading imbalanced case | accuracy vs baseline bar; imbalance illustration |
| 07 | Confusion matrix, precision & recall | 06 | TP/FP/FN/TN; precision, recall, F1; asymmetric error costs | build the confusion matrix; compute P/R/F1 by hand | confusion-matrix heatmap (`plot_confusion_matrix`) |
| 08 | Scores, thresholds, ROC & AUC | 05, 07 | classifiers emit scores; threshold sweep; ROC/AUC; PR curve; threshold-swept vs threshold-fixed | build the affine signed squared-distance score; sweep threshold; ROC/PR; AUC vs `roc_auc_score` | ROC (+AUC) and a score histogram with a movable threshold |
| 09 | Over-/under-fitting, the generalization gap | 04, 06 | complexity; over/underfit; the **generalization gap** (named precisely, ≠ variance); bias–variance (conceptual); learning curve | vary polynomial degree; train vs test error U; learning curve; (optional multi-resample variance fan in "Going further") | boundary at low/mid/high degree; train/test error vs complexity; a learning curve |
| 10 | Validating honestly: cross-validation | 04, 09 | hyperparameters vs parameters; train/val/test; **stratified** k-fold; model selection; tuning-on-test inflation | implement stratified k-fold by hand; select the degree by CV; show tuning-on-test inflation | k-fold scheme; CV score vs degree with the choice marked |
| 11 | Preprocessing & leakage: scaling, encoding, Pipeline | 04, 05, 10 | standardization (and that NB 05's Euclidean centroid was already trusting raw scales); encoding; fit-on-train-only; `Pipeline` | standardize fitting on train only; show scale-before-split leakage; build a `Pipeline` under CV | before/after scaling; leakage-vs-correct estimate comparison |

NB 10 may split into 10a (*hyperparameters vs parameters, the validation set*) and 10b (*k-fold CV*)
at its own notebook-planning if it would exceed ~20 cells — decided then, not now.

## Library additions needed (introduced just-in-time, never bulk-added)

- `src/ml_course/datasets.py` — `load_penguins()` → clean binary 2-feature set, **offline**.
  Requires vendoring a small `penguins.csv` and a scoped **`.gitignore` exception** (current
  `.gitignore` excludes `*.csv`/`data/`). [for NB 01]
- `src/ml_course/viz.py` — new helpers, each taking `ax`/returning a `Figure` and pulling **all**
  colours from `ml_course.colors` (to pass `check_no_hardcoded_hex.py`, like the two existing
  helpers): `plot_feature_histograms` + `plot_class_balance` [NB 03], `plot_roc_curve` + `plot_pr_curve`
  + `plot_score_threshold` [NB 08], `plot_complexity_curve` + `plot_learning_curve` [NB 09].
- `tests/` — `load_penguins` (shape, no NaN, class balance) + smoke tests for each new viz helper.

## Honest limits stated in the notebooks

- Nearest centroid is a primitive (named inductive bias + a failure case + scale-sensitivity), not a
  contender — never oversold.
- The polynomial-degree dial is a didactic device, not a course method.
- The penguins binary 2-feature subset is a deliberate visualization simplification; i.i.d. is framed
  as an assumption we *adopt* (these penguins span islands and years), not a property of the data.
- By-hand numbers (accuracy, P/R/F1, AUC, CV) are cross-checked against the sklearn equivalents in the
  same notebook — the agreement *is* the rigor check (true because NB 08 uses the affine score).

## Verification (per notebook, at its commit)

- Runs top-to-bottom; outputs cleared; `check_no_hardcoded_hex.py` passes; `uv run pytest` green for
  new `src/` code; `gen_llms_txt.py` re-run.
- `docs/course_map.md` 00 section rewritten to this 11-notebook progression (done at approval).

## Per-notebook status (update as the chapter is built)

| NB | Branch | Status |
|----|--------|--------|
| 01 | `notebook/00_GettingStarted__01_what_is_ml` | **done** (reviewers PASS/REVISE→fixed; Rémy validated; merged) |
| 02 | `notebook/00_GettingStarted__02_features_and_feature_space` | **done** (reviewers PASS/PASS; Rémy validated; merged) |
| 03 | `notebook/00_GettingStarted__03_look_before_you_model` | **done** (reviewers PASS/PASS; Rémy validated; merged; +NB01 c06 forward-ref fix) |
| 04 | `notebook/00_GettingStarted__04_generalize_dont_memorize` | **done** (reviewers PASS/PASS; Rémy validated; merged) |
| 05 | `notebook/00_GettingStarted__05_first_classifier_nearest_centroid` | **done** (reviewers REVISE→fixed / PASS; Rémy validated; merged; +viz pandas-first extension) |
| 06 | `notebook/00_GettingStarted__06_accuracy_and_baseline` | **done** (reviewers PASS/PASS; Rémy validated; merged) |
| 07 | `notebook/00_GettingStarted__07_confusion_matrix_precision_recall` | **done** (reviewers PASS/PASS; Rémy validated; merged) |
| 08 | `notebook/00_GettingStarted__08_scores_thresholds_roc` | **done** (reviewers PASS/PASS; Rémy validated; merged; +viz plot_roc_curve/plot_score_threshold) |
| 09 | `notebook/00_GettingStarted__09_overfitting_generalization_gap` | **done** (pedagogy PASS; ml-expert REVISE→fixed; Rémy validated; merged; +viz plot_train_test_curve) |
| 10 | `notebook/00_GettingStarted__10_cross_validation` | **done** (single NB, Rémy chose over 10a/10b; pedagogy PASS; ml-expert REVISE→fixed stratification-exactness MAJOR + minors; Rémy validated; merged; +NB 01–09 ruff cleanup) |
| 11 | — | not started |
