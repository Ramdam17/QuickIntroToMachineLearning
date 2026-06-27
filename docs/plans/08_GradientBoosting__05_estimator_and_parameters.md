# Notebook plan — 08_GradientBoosting / 05_estimator_and_parameters

> Status: **APPROVED by Rémy (2026-06-26) & persisted.** No reviewer gate at the NB-plan stage (Rémy
> validates alone; both reviewers return on the *built* notebook). Next: build from
> `<scratchpad>/build_ch08_nb5.py`, re-measuring every anchor.

## Context

NB **5 of 6** — the **integrative** notebook (like ch 06-NB 4 and ch 07-NB 4): not one new idea but the
**real estimator and its dials**, gathered under one spine — **early stopping**, the principled cure for
the overfit NB 4 made you watch. NB 4 ended on "how many trees? what ν?"; NB 5's answer is *don't guess —
hold out a slice and let the data stop you*. Around that spine we tour the estimator's parameters:
`loss` (current names), `subsample` (stochastic GB), `feature_importances_` (MDI vs permutation), honest
`GridSearchCV`, the **no-`staged_score`** API trap, and we **name** `HistGradientBoosting*` as the fast
modern default (the bridge to ch 09–10). Regression, on the **same `make_friedman1`** as NB 4 (continuity;
and its *known* structure — 5 informative features + 5 noise — makes the importance figure ground-truthed).

## Anchor re-measured vs the chapter-plan estimate (approved 2026-06-26)

The chapter plan estimated early stopping would halt a 2000-tree request at **≈365–390** trees. Re-measured
on the live install it stops at **142** (seed 0; 133–163 across seeds) — *earlier*, at **no cost** (test
R² 0.930 there, a hair **above** the full 2000-tree 0.927). The direction is the same and the lesson is
stronger (early stopping saves ~1860 trees for equal-or-better R²). Rémy approved 142.

## Dataset & anchors (measured on scikit-learn 1.9.0, seed 0 — re-measured at build)

- **Data:** `make_friedman1(n_samples=2000, noise=1.0, random_state=0)`, 70/30 split (seed 0) → train 1400
  / test 600 (the NB 4 set; x₀–x₄ informative incl. the x₀·x₁ interaction, x₅–x₉ pure noise).
- **Parity recap (prose):** the by-hand loop of NB 1–3 *is* this estimator — regression matched
  `GradientBoostingRegressor` to **1e-16** (NB 1), classification to **3.6e-15** with the Newton leaf (NB 3).
- **Early stopping (the spine):** `GradientBoostingRegressor(n_estimators=2000, learning_rate=0.1,
  max_depth=3, validation_fraction=0.1, n_iter_no_change=10, tol=1e-4)` stops at **142 trees**, test R²
  **0.9299**; the full 2000-tree model: **0.9271** (so early stopping is *better* here — the full model has
  begun NB 4's gentle ν=0.1 overfit). Seed-dependent stop: 142 / 163 / 133 (seeds 0/1/2), R² ≈ 0.930/0.930/0.920.
- **`subsample` (stochastic GB, Friedman 2002):** test R² vs fraction (ν=0.1, depth 3, 300 trees) —
  0.25 → 0.9295, **0.5 → 0.9363**, 0.75 → 0.9359, 1.0 → 0.9292. Row-sampling 0.5–0.75 **beats** full-sample
  (regularizer + speedup). `subsample<1` also exposes `oob_improvement_`/`oob_score_` — a *free* running
  estimate, but **noisier than a held-out set** (honest caveat: at ν=1 its cumulative peaks at tree 299
  while the test R² peaks at 10 — OOB did **not** find the optimum), so early stopping stays the robust tool.
- **`feature_importances_` — MDI vs permutation (friedman1's known structure):** MDI sum **x₀–x₄ = 0.988**,
  **x₅–x₉ = 0.012**; per-feature MDI / permutation agree on the ranking — x₃ 0.362/0.728, x₀ 0.224/0.493,
  x₁ 0.224/0.435, x₄ 0.097/0.170, x₂ 0.081/0.150, noise ≈ 0.003/≈0. The model recovers the 5 real features
  and ignores the 5 noise ones; MDI's known biases (ch 06) are mild here, permutation confirms.
- **Honest `GridSearchCV`** (`learning_rate × max_depth`, 5-fold on train): best = **{lr 0.1, depth 3}** =
  the **default**; CV R² 0.9279, tuned **sealed-test 0.9292 = default 0.9292** — tuning bought nothing on
  the sealed test (the sklearn defaults are already sane; echoes ch 06/07).
- **API:** GBR/GBC have **no `staged_score`** (use `staged_predict`); `loss` — regressor `'squared_error'`
  (default), `'absolute_error'`, `'huber'`, `'quantile'`; classifier `'log_loss'` (default), `'exponential'`
  (= AdaBoost's objective, NB 3); **`'deviance'`/`'ls'`/`'lad'` removed**. `HistGradientBoostingRegressor`:
  test R² 0.9280, `max_iter=100`, `early_stopping='auto'`, `max_leaf_nodes=31`, `max_bins=255`, native
  categorical/missing (named here; the speed/score teaser is NB 6).

## Cell-by-cell (~21 cells; integrative; "Read the figure" after every figure)

1. (md) **Header** — `# 05 — The estimator and its parameters`; *Chapter 08 · Notebook 5 of 6*. NB 1–4
   built and understood the machine by hand; now the real `GradientBoosting*` estimator and the dials that
   matter, gathered around **early stopping** (the cure for NB 4's overfit). **Prerequisites:** NB 1–4
   (the residual loop, the gradient view, classification, ν/depth/n_estimators), ch 06-NB 4 / ch 07-NB 4
   (reading an estimator's parameters honestly), ch 06-NB 5 (permutation importance). **What you'll be able
   to do:** use `GradientBoostingRegressor`/`Classifier`; set the `loss`; stop the right number of trees
   automatically with early stopping; regularize with `subsample`; read MDI vs permutation importance;
   tune honestly; and know when to reach for `HistGradientBoosting*`.
2. (code) **Imports & setup** — numpy, pandas, matplotlib; `make_friedman1`, `train_test_split`,
   `GradientBoostingRegressor`, `HistGradientBoostingRegressor`, `permutation_importance`, `GridSearchCV`,
   `r2_score`; `viz`/`COLORS`; `SEED=0`; the NB-4 friedman1 split; print shapes.
3. (md) **Recap & the question + the estimator named.** NB 4 ended on "how many trees, what ν?". **Parity
   recap:** the by-hand loop *is* `GradientBoosting*` — regression to 1e-16 (NB 1), classification to
   3.6e-15 with the Newton leaf (NB 3). So from here we use the real estimator and turn its knobs. Name the
   object: `GradientBoostingRegressor` / `GradientBoostingClassifier`, the shared parameters.
4. (md) **The dials & the `loss`.** The `loss` *is* the method (NB 2): regressor `'squared_error'`
   (default), `'absolute_error'`, `'huber'`, `'quantile'`; classifier `'log_loss'` (default), `'exponential'`
   (= AdaBoost's objective, NB 3). **`'deviance'`/`'ls'`/`'lad'` are removed** in current sklearn. The other
   dials: `learning_rate` & `n_estimators` (NB 4), `max_depth`/`max_features` (NB 4 — one line each, not
   re-swept here), `subsample` (below). **API trap:** unlike `AdaBoostClassifier`, GB has **no
   `staged_score`** — compute the metric over `staged_predict` yourself.
5. (md) **Intuition — early stopping.** NB 4: too many trees overfit at large ν, and even at ν=0.1 you must
   pick `n_estimators`. Instead of guessing, hold out a **validation slice** (`validation_fraction`), watch
   its score each round, and **stop when it stops improving** for `n_iter_no_change` rounds (within `tol`).
6. (code) **Fig J — early stopping in action.** Fit a full `GradientBoostingRegressor(n_estimators=2000,
   lr=0.1, max_depth=3)` and an early-stopped one (`validation_fraction=0.1, n_iter_no_change=10, tol=1e-4`).
   Plot the full model's **staged test R² vs trees (log-x)**; mark the early-stop at **142** with a vertical
   line; annotate full-2000 R² 0.927. Print `n_estimators_` (2000 vs 142) and the two test R² (0.927 vs 0.930).
7. (md) **Read the figure (J).** The curve climbs to ≈0.93 by ~150 trees, then is **flat** to 2000. Early
   stopping halts at **142** — test R² **0.930**, a touch *above* the full 2000-tree model (0.927, which has
   begun NB 4's gentle ν=0.1 overfit). ~1860 trees saved for equal-or-better accuracy, no guessing. The stop
   point wanders a little with the seed (133–163) — it tracks the validation slice, which is the point.
8. (md) **Intuition — `subsample` (stochastic gradient boosting).** Fit each tree on a random **fraction of
   the rows** (Friedman 2002). Like bagging's row-sampling (ch 06), it **decorrelates** the steps — a
   regularizer — and it is **faster** (fewer rows per tree). A bonus: `subsample<1` exposes
   `oob_improvement_`, a free out-of-bag running estimate.
9. (code) **Fig K — `subsample` as a regularizer.** Train & test R² vs `subsample ∈ {0.25, 0.5, 0.75, 1.0}`
   (ν=0.1, depth 3, 300 trees) via `viz.plot_train_test_curve`. Print the test R² row (0.9295 / 0.9363 /
   0.9359 / 0.9292).
10. (md) **Read the figure (K).** Sampling **half to three-quarters** of the rows per tree *beats*
    full-sample (test R² ≈0.936 vs 0.929) — row randomness regularizes, exactly the bagging intuition (ch 06)
    inside a booster, and it costs less per tree. About `oob_improvement_`: it is a free by-product but
    **noisier** than a held-out set (here, at ν=1 its cumulative kept rising while the test error climbed),
    so prefer the validation-based **early stopping** above for choosing the number of trees.
11. (md) **Intuition — which features did it use?** The fitted model exposes `feature_importances_` (MDI —
    impurity reduction, ch 04/06) and we cross-check with **permutation importance** (ch 06-NB 5). friedman1
    is the perfect test: we *know* x₀–x₄ matter (and x₀·x₁ interact) and x₅–x₉ are pure noise.
12. (code) **Fig L — MDI vs permutation.** Two panels via `viz.plot_feature_importances` (MDI left,
    permutation right, `permutation_importance` on the test set, `n_repeats=10`). Print MDI sum x₀–x₄=0.988
    / x₅–x₉=0.012.
13. (md) **Read the figure (L).** The model recovers the **five informative** features (MDI sum 0.988) and
    assigns ≈0 to the **five noise** features; MDI and permutation **agree on the ranking** (x₃ largest, then
    the x₀/x₁ interaction pair, then x₄/x₂). MDI's known biases (ch 06) are mild here because the noise
    features are genuinely uninformative; permutation confirms. Importance is **use, not cause** (one line).
14. (md) **Tuning honestly.** The dials interact, so search them together with cross-validation on the
    **training** set, then report **one sealed test** — never the CV-best as the headline (ch 06/07).
15. (code) **GridSearchCV.** `learning_rate × max_depth`, 5-fold on train; print `best_params_`, CV R²,
    tuned sealed-test R², and the **default**'s sealed-test R².
16. (md) **Read the result.** The grid's best is **{lr 0.1, depth 3} — the default**; tuned sealed-test
    0.929 = default 0.929. Tuning bought **nothing** here: sklearn's defaults are well-chosen for many
    tabular problems. Tune when you have reason, validate on a sealed test, and judge by the sealed number.
17. (md) **The fast modern default — `HistGradientBoosting*`.** For larger data, `HistGradientBoosting
    Regressor`/`Classifier` bin features into **histograms** (`max_bins=255`), grow **leaf-wise**
    (`max_leaf_nodes=31`), turn `early_stopping='auto'`, and handle **categoricals/missing** natively —
    much faster, similar or better accuracy. It is the bridge to **ch 09 (XGBoost)** and **ch 10 (LightGBM)**,
    which refine exactly this skeleton; the head-to-head teaser is NB 6.
18. (md) **Your turn (tiered).** *easy:* from Fig J, why does early stopping halt near 142 and not 2000?
    *medium:* re-fit with `n_iter_no_change=5` or `tol=1e-3` and report how the stop point moves and whether
    test R² changes. *harder:* `subsample=0.5` beat full-sample — explain why sampling rows can *raise* test
    R² (relate to bagging's variance cut, ch 06), and why that is not in tension with boosting being
    sequential.
19. (md) **What you built.** Bullets: drove `GradientBoosting*` and its `loss`; used **early stopping** to
    pick the number of trees automatically (2000 → 142, equal-or-better R²); regularized with `subsample`;
    read **MDI vs permutation** importance against friedman1's known structure; tuned honestly to one sealed
    test; and met `HistGradientBoosting*`. **Vocabulary:** `n_iter_no_change` / `validation_fraction` /
    early stopping · `subsample` / stochastic GB / `oob_improvement_` · MDI vs permutation importance ·
    `staged_predict` (no `staged_score`) · `HistGradientBoosting*`.
20. (md) **Going further (optional).** Early stopping spends a `validation_fraction` and depends on
    `tol`/`n_iter_no_change` (off by default). `subsample` is Friedman's *stochastic* gradient boosting
    (2002); the OOB estimate is convenient but noisy. The modern fast path (histograms, leaf-wise growth, a
    regularized objective) is ch 09–10. `loss='huber'`/`'quantile'` give robust / interval regression.
21. (md) **References** — Friedman 2001 (DOI 10.1214/aos/1013203451); Friedman 2002 — *Stochastic gradient
    boosting* (DOI 10.1016/S0167-9473(01)00065-2; `subsample`, OOB); Friedman 1991 — make_friedman1 (DOI
    10.1214/aos/1176347963); ESL §10.11–10.13 (DOI 10.1007/978-0-387-84858-7); permutation importance —
    ch 06-NB 5 (Breiman 2001, DOI 10.1023/A:1010933404324). `Previous: 04 — Shrinkage and the trees.`
    `Next: 06 — A demanding case: California housing.`

## Figures (3, each followed by "Read the figure")
- **Fig J (cell 6)** — early stopping: full model's staged test R² vs trees (log-x), early-stop marker at
  142 (vs 2000 requested). *notebook-local matplotlib (one curve + vertical line + annotation).*
- **Fig K (cell 9)** — `subsample` as a regularizer: train & test R² vs subsample fraction; 0.5–0.75 beats
  full-sample. *reuse `viz.plot_train_test_curve`.*
- **Fig L (cell 12)** — MDI vs permutation importance (two panels) on friedman1's known structure.
  *reuse `viz.plot_feature_importances` ×2.*

## `src/` & guards
**No `src/` change** (reuse `viz.use_course_style` + `viz.plot_train_test_curve` + `viz.plot_feature_importances`;
Fig J is notebook-local matplotlib with `ml_course.colors`; pytest stays **20**). Build via
`uv run python <scratchpad>/build_ch08_nb5.py`; **re-measure every anchor at build** (early stopping
`n_estimators_`, the subsample sweep incl. train R², MDI + permutation, the GridSearchCV best/sealed-test,
the API facts). nbconvert top-to-bottom **from project cwd** on a scratchpad copy (tracked file
**output-free**); **banned-word scan over JSON real cell text** = 0 (watch "just"/"simply"); hex clean;
`ruff`/`black` clean (notebook ruff-clean); `gen_llms_txt` re-run.

## Honest scoping
Integrative (not "one concept") — the ch 06/07-NB 4 precedent: the spine is **early stopping**, the other
dials are supporting acts (`max_depth`/`max_features` are one-line cross-refs to NB 4, not fresh sweeps).
Regression throughout (the classifier is *named*, its losses listed; classification was built in NB 3).
`oob_improvement_` is presented as a **noisy** free by-product, **not** an optimum-finder (measured: it
disagreed with the test optimum at ν=1). Tuning honestly = the grid's CV-best is the **default** here and
buys nothing on the sealed test (no overclaim). `HistGradientBoosting*` is **named**, not benchmarked
(its win shows on bigger data — NB 6 / ch 09–10). Both reviewers PASS (no BLOCK) + Rémy validates visually
before commit; ff-merge notebook → chapter.
