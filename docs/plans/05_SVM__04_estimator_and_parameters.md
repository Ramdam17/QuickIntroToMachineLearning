# Notebook plan — 05_SVM / 04_estimator_and_parameters

> Status: **APPROVED** (2026-06-22, by Rémy; NB plans are validated by Rémy alone — the two reviewers
> gate the *built* notebook, not this plan). Drives the build. Numbers re-measured at build and
> reconciled into prose.

## Context

NB **4 of 5** — the integrative notebook: the real `sklearn.svm.SVC` and the knobs that make it work.
NB 1–3 built the ideas (margin, the cost `C`, the kernel and its reach `gamma`); now we drive the
estimator and learn to *set* those knobs. The headline is the **`C × gamma` bias–variance map** — the
single picture that ties the chapter together. Following the ch-04 discipline, the notebook **shows
four things and names two**, holding a **~24-cell ceiling** so the map gets room.

## Anchors (measured at plan time, sklearn **1.9.0**; re-measured at build)

- **Parity (the by-hand line is the library's):** `SVC(kernel="linear", C=1e6)` on NB 1's separable
  blobs → `‖w‖ = 1.1612`, **2 support vectors** at indices `[23, 26]` — identical to NB 1's by-hand
  street. The library is not magic; it solves the problem we built by hand.
- **The `C × gamma` CV map (RBF) on `make_moons(noise=0.30, seed 0)`:**
  | `C` \ `gamma` | 0.01 | 0.1 | 1 | 10 |
  |----|----|----|----|----|
  | 0.1 | 0.830 | 0.827 | 0.903 | 0.890 |
  | 1 | 0.827 | 0.833 | 0.930 | 0.937 |
  | 10 | 0.823 | 0.877 | 0.933 | 0.930 |
  | 100 | 0.827 | 0.927 | 0.937 | 0.910 |
  Low `gamma` is a flat ~0.83 underfit wall; the sweet spot is `C` 1–100 with `gamma` ≈ 1; very high
  `gamma`+`C` starts to overfit (0.937 → 0.910).
- **`gamma` as the bias–variance dial** (support-vector count + train acc at `C=1`, the boundary grid):
  `gamma` 0.01 → **167 SVs / train 0.833** (smooth, underfit), 1 → **88 / 0.940** (good), 10 →
  **163 / 0.950** (wiggly, overfit) — a U in complexity. `gamma='scale'` = `1/(n_features·X.var())` is
  the default (another reason to standardize).
- **Multiclass = one-vs-one (OvO):** on 3-species `load_penguins_full` (2 numeric features), `SVC`
  trains `n(n−1)/2 = 3` pairwise classifiers; `decision_function` (default `'ovr'` shape) is `(n, 3)`;
  CV **0.956**. The explicit contrast with ch 03's softmax/OvR and ch 04's native multiclass.
- **`decision_function` is a score, not a probability:** to get a probability, **calibrate** (Platt /
  sigmoid). On a moons train/test split, an uncalibrated `sigmoid(decision_function)` scores **Brier
  0.106**; **`CalibratedClassifierCV` (sigmoid)** scores **Brier 0.071** — better calibrated. **Pin the
  deprecation:** `SVC(probability=True)` is deprecated in 1.9 / removed in 1.11; the message itself
  prescribes `CalibratedClassifierCV(SVC(), ensemble=False)`. (Reuses the ch 03 NB 6 pattern.)
- **Honest tuning:** `GridSearchCV` over `C`/`gamma` on the moons **train** split → best
  `{C=10, gamma=1}`, CV **0.919**, **one sealed test 0.944**.
- **Named (not shown):** `LinearSVC` — the linear SVM that scales to large `n` (on moons it manages
  only **0.823**, since a line cannot curve — it is the *scale* tool, used in NB 5), and `class_weight`
  for imbalance (used lightly in NB 5).

## Library / figures

- **No `src/` change** — reuse `viz.plot_svm_decision` (the gamma boundary grid),
  `viz.plot_decision_boundary` (the 3-class OvO regions — `plot_svm_decision` is binary-only, so the
  multiclass figure uses the region helper), and `viz.plot_calibration_curve` (the reliability
  diagram); the `C × gamma` heatmap is a one-off in-notebook figure (`CMAP_PROBA`). Sklearn: `SVC`,
  `LinearSVC`, `GridSearchCV`, `CalibratedClassifierCV`, `FrozenEstimator`, `StandardScaler`,
  `Pipeline`, `StratifiedKFold`, `cross_val_score`, `train_test_split`, `make_moons`,
  `make_blobs`. (`pytest` stays 19.)
- **Four figures** (each + "Read the figure"): **A** the **`C × gamma` CV heatmap** (the bias–variance
  map — the headline); **B** the **gamma boundary grid** (underfit → good → overfit, with the
  support-vector counts); **C** the **OvO 3-class regions** on penguins_full; **D** the **calibration
  reliability diagram** (uncalibrated score vs Platt-calibrated).

## Cell-by-cell (~22 cells; integrative; ~24-cell ceiling; "Read the figure" after every figure)

1. (md) **Header** — `# 04 — The estimator SVC and its parameters`; *notebook 4 of 5*. **Prerequisites:**
   NB 1–3 (margin, cost `C`, kernel & `gamma`); module 00 NB 09 (over/under-fitting & the U), NB 10
   (cross-validation), NB 11 (`Pipeline`); ch 02/03 calibration (`CalibratedClassifierCV`). **What
   you'll be able to do:** drive `sklearn`'s `SVC`; read the **`C × gamma` bias–variance map** and pick
   a cell by CV; turn `gamma` and see under/over-fitting; handle **multiclass** (one-vs-one); turn an
   SVM **score** into a calibrated **probability**; know when to reach for `LinearSVC`.
2. (code) **Imports + seed + style + data + parity** — `make_moons(0.30)` and the NB-1 blobs; show
   `SVC(kernel="linear", C=1e6)` on the blobs reproduces NB 1 (`‖w‖ 1.1612`, support `[23, 26]`).
3. (md) **Recap / footing** — NB 1–3 built the pieces by hand; here we drive the real `SVC` and learn
   to *set* its knobs. Parity first: the library solves exactly the by-hand problem.
4. (md) **Intuition — `C` and `gamma` together** — `C` (NB 2) trades margin width for violations;
   `gamma` (NB 3) is the RBF's reach. They jointly set the model's complexity — best read as a **map**.
5. (code) **Fig A — the `C × gamma` CV heatmap** on moons (`CMAP_PROBA`), annotated with CV accuracy.
6. (md) **Read the figure (A)** — low `gamma` is a flat underfit wall (~0.83 whatever `C`); the bright
   band is `C` 1–100, `gamma` ≈ 1; the top-right corner (high `gamma`+`C`) dims again — overfitting.
   The map is the practitioner's view; pick a cell by CV, never on the test set.
7. (code) **Fig B — the `gamma` boundary grid** at `C=1`: `gamma` 0.01 / 1 / 10 via
   `viz.plot_svm_decision`, titled with support-vector counts (167 / 88 / 163).
8. (md) **Read the figure (B)** — small `gamma` = broad reach = a nearly straight, underfit boundary on
   almost every point (167 SVs); `gamma` ≈ 1 = a clean curved street (88 SVs); large `gamma` = each
   point's reach is tiny = a wiggly, islanded boundary that memorizes (163 SVs). `gamma='scale'`
   (`1/(n_features·X.var())`) is the sensible default — and depends on feature spread, one more reason
   to standardize (NB 5).
9. (md) **Intuition — kernel & multiclass** — `kernel` (`linear`/`rbf`/`poly`, NB 3) is itself a knob;
   here we stay on the RBF. And `SVC` handles **more than two classes** by **one-vs-one**: a classifier
   per pair, then a vote — distinct from ch 03's softmax/OvR and ch 04's native multiclass.
10. (code) **Fig C — OvO on 3-species penguins_full** — `viz.plot_decision_boundary` of the 3-class
    regions; print that `SVC` trained `n(n−1)/2 = 3` pairwise classifiers and CV **0.956**.
11. (md) **Read the figure (C)** — three species, three pairwise contests, voted into the regions
    shown. One-vs-one keeps each contest a clean two-class margin problem; with few classes the
    `n(n−1)/2` count is cheap.
12. (md) **Intuition — score vs probability** — `SVC` outputs a **signed-distance score**
    (`decision_function`), not a probability. When you need a probability (a threshold, a cost), **wrap
    it** in `CalibratedClassifierCV` (Platt / sigmoid scaling) — the ch 03 NB 6 pattern.
13. (code) **Fig D — calibration** — fit `SVC` on moons train; compare an uncalibrated
    `sigmoid(decision_function)` (Brier **0.106**) with `CalibratedClassifierCV(FrozenEstimator(svc),
    method="sigmoid")` (Brier **0.071**) via `viz.plot_calibration_curve`; **print the
    `probability=True` deprecation** and its prescribed replacement `CalibratedClassifierCV(SVC(),
    ensemble=False)`.
14. (md) **Read the figure (D)** — the raw score *ranks* points well but is not a probability; Platt
    scaling pulls the reliability curve onto the diagonal (Brier 0.106 → 0.071). Note: **don't** use the
    deprecated `probability=True`; `CalibratedClassifierCV` is the supported way (and we *use* a
    calibrated probability for a clinical threshold in NB 5).
15. (md) **Named knobs** — **`LinearSVC`** (and `SGDClassifier(loss="hinge")`): a *linear* SVM that
    scales to large `n` where kernel `SVC` cannot (on moons it manages only 0.823 — a line cannot
    curve — but on large data it is the tool; NB 5). **`class_weight`** — reweight under imbalance
    (NB 5).
16. (md) **Intuition — honest tuning** — choose `C`/`gamma` by **cross-validation on the training
    split**, then read the **sealed test once** (module 00 NB 10).
17. (code) **`GridSearchCV`** over `C`/`gamma` on the moons train → best `{C=10, gamma=1}`, CV **0.919**,
    sealed test **0.944**.
18. (md) **Read the result** — the grid picked the bright-band cell of the map; the sealed-test score
    (0.944) confirms it, and we touched the test set only once.
19. (md) **Your turn** (3 tiered) — *easy*: from the `C × gamma` map, name a cell that underfits and one
    that overfits, and say why; *medium*: turn `gamma` to 0.01 and to 10 and describe the boundary you
    expect; *harder*: take a fitted `SVC`, wrap it in `CalibratedClassifierCV`, and report whether the
    Brier score improves.
20. (md) **What you built** — drove `sklearn`'s `SVC` (parity with the by-hand line); read the
    **`C × gamma` bias–variance map** and chose a cell by CV; saw `gamma` swing from underfit to
    overfit; handled **multiclass** with one-vs-one; turned an SVM **score** into a calibrated
    **probability**; learned when to reach for `LinearSVC`. **Vocabulary:** `C × gamma` map ·
    `gamma='scale'` · one-vs-one · `decision_function` · Platt / `CalibratedClassifierCV` · `LinearSVC`.
21. (md) **Going further** — `decision_function_shape` (`'ovr'` vs `'ovo'` outputs), `coef0`/`degree`
    for the polynomial kernel, `nu`-SVM (`NuSVC`) as a reparametrization of `C`. The next notebook puts
    all of this to work on a real dataset, where **scaling** and the **large-`n` limit** bite.
    **References:** Cortes & Vapnik 1995 (DOI 10.1007/BF00994018); Platt 1999 (probabilistic outputs /
    sigmoid calibration); Chang & Lin 2011 LIBSVM (the solver, OvO; DOI 10.1145/1961189.1961199);
    Hsu & Lin 2002 (a comparison of multiclass SVM methods; DOI 10.1109/72.991427); ESL §12 (DOI
    10.1007/978-0-387-84858-7); ISLR §9 (DOI 10.1007/978-1-0716-1418-1). `Previous: 03 — The kernel
    trick` · `Next: 05 — A demanding case: breast cancer`.

## Honest scoping (stated in the notebook)

- **Four shown, two named, ~24-cell ceiling** — the `C × gamma` map is the protected headline; `kernel`
  is recapped (shown in NB 3) not re-figured; calibration is one compact figure; `LinearSVC` /
  `class_weight` are named and deferred to NB 5.
- **Calibration is named here, *used* in NB 5** — `decision_function` is a score; Platt via
  `CalibratedClassifierCV`, with the `probability=True` deprecation pinned; the clinical-threshold use
  is NB 5.
- **`gamma='scale'` depends on feature spread** — named as another reason to standardize (NB 5).
- **Tuning on train, one sealed test** — `GridSearchCV` honesty; no peeking.
- **`LinearSVC` underperforms on moons (0.823)** — stated plainly: it is the *large-`n` linear* tool,
  not a curve-fitter; its payoff is NB 5's scaling story, not accuracy here.

## Verification

Build via `uv run python - < <scratchpad>/build_ch05_nb4.py` (stdin). Re-measure at build: parity
`‖w‖ 1.1612` / SVs `[23,26]`; the `C × gamma` grid; n_SV-vs-gamma at C=1 (167/123/88/163, train
0.833→0.950); OvO penguins_full 3 pairwise / CV 0.956 / decision_function `(n,3)`; calibration Brier
naive 0.106 vs Platt 0.071 + the `probability=True` deprecation message; GridSearch best `{C=10,
gamma=1}` CV 0.919 / sealed test 0.944; LinearSVC moons 0.823. Runs top-to-bottom (nbconvert to
scratchpad; tracked file **output-free**, `--clear-output --inplace`); **banned-word scan over the JSON
real text** = 0; `check_no_hardcoded_hex` passes; `gen_llms_txt` re-run; `ruff` / `black` clean;
`pytest` 19 (no `src/` change). Both reviewers pass (no BLOCK); Rémy validates visually; commit
`feat(05_svm): notebook 04 — the estimator and its parameters`; merge `notebook → chapter`.
