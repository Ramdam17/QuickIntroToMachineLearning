# NB plan — 10_LightGBM / 04_estimator_and_parameters — the estimator & its parameters

> Status: **APPROVED by Rémy (via ExitPlanMode, 2026-06-28)**. The integrative "method & its parameters"
> notebook. No reviewer gate at the NB-plan stage (reviewers return on the built notebook).

## Context

Chapter 10, NB 4 of 5 (integrative). The learner has built LightGBM's three distinctive mechanisms by
hand — leaf-wise growth (NB 1), GOSS (NB 2), the optimal categorical split (NB 3). NB 4 meets the real
estimator `LGBMClassifier`/`LGBMRegressor` and its knobs, each taught from the concept that owns it.
**The genuinely-new spine is the leaf-wise capacity dial — `num_leaves` + `min_child_samples` — *tuned*
here, closing NB 1's lopsided→overfit loop.** Honest arc (ch 08 NB 5 / ch 09 NB 4 precedent): defaults
are already sensible, tune on train via `GridSearchCV`, read one sealed test.

## Live anchors (measured, lightgbm 4.6.0, SEED=0 — `measure_ch10_nb4.py`; make_classification 4000×20,
8 informative / 4 redundant, class_sep 0.9, flip_y 0.04; 2800 train / 1200 test)

- **Resolved defaults:** `num_leaves=31, max_depth=-1, min_child_samples=20, learning_rate=0.1,
  n_estimators=100, reg_lambda=0, reg_alpha=0, subsample=1, colsample_bytree=1`.
- **Single tree (`n_estimators=1`) num_leaves sweep — the overfit peak (closes NB 1):** test peaks at
  **num_leaves≈64 (0.878)** then falls/plateaus (127/255 → 0.875); train 0.733→0.952; gap 0.007→0.077;
  depth 1→18; **num_leaves=255 builds only 168 leaves (a cap)**.
- **Ensemble (`n_estimators=100`, lr 0.1) num_leaves sweep — robust:** test 0.861(2) → 0.902(4) →
  0.916(8) → 0.9225(31) → 0.9267(127/255); train → 1.0; depth → 25.
- **`min_child_samples` floor (num_leaves=255):** 1 → **0.858 (gap 0.14)** / 5 → 0.923 / **20 → 0.927** /
  50 → 0.918 / 100 → 0.919 / **300 → 0.906 (underfit, depth → 7)**. The floor matters more than the ceiling.
- **Rule `num_leaves < 2^max_depth`:** a leaf-wise tree with `num_leaves=2^d` can reach depth up to
  `2^d − 1` (lopsided); num_leaves is the capacity knob, max_depth an optional cap.
- **`learning_rate × n_estimators`:** lr0.3/100 → 0.9225, lr0.1/100 → 0.9225, lr0.05/300 → 0.9250,
  lr0.03/600 → 0.9267.
- **`reg_lambda` posture (num_leaves=63, 200 trees):** 0 → 0.9192 / 1 → 0.9208 / 10 → 0.9175 / 100 →
  0.9058 — **L2 does not lift accuracy** (posture vs XGBoost's λ=1, not a lever — ch 09 NB 4 again).
- **Stochasticity knobs (200 trees):** gbdt 0.9192 / goss 0.9167 / bagging 0.5 0.9183 / feature_fraction
  0.5 0.9175 — all ≈ on dense data (NB 2).
- **Honest tuning:** default sealed **0.9225**; `GridSearchCV` best `{lr0.1, mcs10, ne400, nl31}` CV
  0.9329 → **tuned sealed 0.9233 (Δ +0.0008)**. Defaults are good.
- **Importances:** top-1 agrees (feat 5); `'split'` vs `'gain'` rankings differ — MDI caveat (honest
  reading → NB 5).
- **Early stopping (4.x):** `callbacks=[lgb.early_stopping(N)]` + `eval_set`, `best_iteration_` (re-measure
  the stop in the build).

## Cell-by-cell (~24 cells, 4 figures) — intuition → implementation → interpretation

1. **(md) Header** — estimator & params; integrative; knobs by owning concept; honest spine. Prereqs
   NB 1–3, ch 08 NB 4, ch 06/08, ch 00.
2. **(md) From mechanisms to the estimator** — recap NB 1–3; `num_leaves` (budget in NB 1) is tuned here;
   name `LGBMClassifier`/`LGBMRegressor`.
3. **(code) Setup + resolved defaults** — imports, `viz.use_course_style()`, `SEED=0`, dataset, split;
   print key default params.
4. **(md) The capacity dial: `num_leaves`** — leaf-wise grows to a budget; a single tree should overfit
   as it grows — does boosting tame it?
5. **(code) num_leaves sweep — single tree + 100-tree ensemble** (train/test/leaves/depth).
6. **(code) Fig 1 — the dial** — single-tree test (peaks ~64 then falls) vs ensemble (robust plateau),
   log x; mark the peak.
7. **(md) Read fig 1** — single tree peaks ~64 then overfits (caps at 168/255 leaves, depth→18);
   ensemble plateaus ~0.92–0.927 — boosting averages the variance. **Closes NB 1's loop.**
8. **(md) The floor: `min_child_samples`** — min rows per leaf, the brake on leaf-wise's deepest branches.
9. **(code) min_child_samples sweep** (num_leaves=255).
10. **(code) Fig 2 — the floor** — test vs min_child_samples (log x), overfit→good→underfit + gap.
11. **(md) Read fig 2** — 1→0.858 (gap 0.14), 20→0.927, 300→0.906 (depth→7); the floor matters more than
    the ceiling. Rule `num_leaves < 2^max_depth`.
12. **(md) Shrinkage: `learning_rate × n_estimators`** — recap ch 08 NB 4.
13. **(code) lr × n_estimators sweep + Fig 3.**
14. **(md) Read fig 3** — smaller lr + more trees climbs slightly (0.9225→0.9267); early stopping picks
    the count (below).
15. **(md) Knobs that don't move dense-data accuracy** — `reg_lambda`/`reg_alpha` (off by default, the
    posture vs XGBoost λ=1); `feature_fraction`/`bagging_fraction`; `data_sample_strategy='goss'` (NB 2).
16. **(code) Measure them** — reg_lambda {0,1,10,100}; gbdt/goss/bagging/feature_fraction → table.
17. **(md) Read the table** — L2 doesn't lift accuracy (posture, shrinks leaf weights); sampling knobs
    ~flat on dense data, earn their keep on large/sparse and for speed.
18. **(md) Early stopping** — the 4.x API `callbacks=[lgb.early_stopping(N)]` + `eval_set`,
    `best_iteration_` (cross-ref ch 08/09).
19. **(code) Early-stopping demo** — validation slice, n_estimators=1000, early_stopping(30); print
    `best_iteration_` + sealed test. Log visible.
20. **(md) Honest tuning** — LightGBM defaults sensible (vs XGBoost ch 09 NB 4); tune on train, touch
    test once.
21. **(code) GridSearchCV → one sealed test + Fig 4** (default vs tuned bars) + best params/CV; `verbose`
    on so folds show.
22. **(md) Read fig 4** — tuned 0.9233 vs default 0.9225 (Δ +0.0008): defaults already good; the dial
    *understands* more than it rescues on clean data — tuning bites harder on messy/large data (capstone).
23. **(code) Importances `'split'` vs `'gain'`** — two top-5 rankings; top-1 agrees, ordering differs.
24. **(md) Read + What you built + References** — MDI caveat (both train-only, biased; permutation → NB 5).
    Vocabulary list. Next: NB 5 capstone. Refs: Ke et al. 2017; LightGBM parameter docs; Friedman
    2001/2002; ch 08 NB 4, ch 09 NB 4.

## `src/` & guards
- **No `src/` change** (reuse `viz`; `LGBMClassifier` + `GridSearchCV`; `dump_model` for depth/leaves;
  pytest 20). Colours only from `ml_course.colors`; seeds fixed.
- Build from `build_ch10_nb4.py` (source of truth; rebuild right before `git add` — kernel-drift guard).
- **Never silence output** — no `verbose=-1` (LightGBM banner, early-stopping log, GridSearch fold output
  stay visible).
- Exit guards: nbconvert exit 0 (4 figures), banned scan = 0, hex clean, ruff/black clean, output-free;
  **two-reviewer gate** (no BLOCK) → fold → **Rémy visual** → end-of-NB checklist (`gen_llms_txt.py`,
  `common_errors` +rows, `course_map` §10 mark NB 4, pytest 20, STATE) → commit
  `feat(10_lightgbm): notebook 04 — the estimator and its parameters` → `git merge --ff-only` into
  `chapter/10_LightGBM`.

## Verification (end-to-end)
1. nbconvert-execute a scratchpad copy → exit 0, 4 figures, anchors reproduce (single-tree peak ~64;
   ensemble plateau ~0.92–0.927; min_child_samples 1→0.858 / 20→0.927 / 300→0.906; reg_lambda flat;
   goss/bagging/feature ≈; tuned 0.9233 vs default 0.9225). 2. hex + banned + ruff → clean. 3. pytest →
   20 passed. 4. Two-reviewer gate, then Rémy visual.
