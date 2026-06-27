# Notebook plan — 08_GradientBoosting / 06_california_housing

> Status: **APPROVED by Rémy (2026-06-27) & persisted.** No reviewer gate at the NB-plan stage (Rémy
> validates alone; both reviewers return on the *built* notebook). Next: build from
> `<scratchpad>/build_ch08_nb6.py`, re-measuring every anchor. **Last NB of the chapter — after it ships,
> close chapter 08 via PR `chapter → main` (`--no-ff`).**

## Context

NB **6 of 6** — the chapter's **visualization-first capstone**: a full, honest regression workflow on a
real dataset, mobilizing everything from NB 1–5. Predict California median house values, end to end —
look at the data, build baselines, tune a gradient booster **with early stopping** (NB 5's spine), report
**R² and MAE in real dollars** on a sealed test, do **error analysis** (where does it fail?), foil it
against a random forest and a linear model, and close with the **`HistGradientBoosting*` speed/score
teaser** that bridges to ch 09 (XGBoost) / ch 10 (LightGBM). Per the capstone-visual-first standard:
**~26 cells, 7 figures**, every figure followed by a "Read the figure".

## Dataset & anchors (measured on scikit-learn 1.9.0, seed 0 — re-measured at build)

- **Data:** `fetch_california_housing(as_frame=True)` (called directly, like ch 06's `fetch_covtype`; no
  loader, **no `src/` change**). 20640×8, all-numeric; target = median house value in **units of $100,000**
  (min 0.15, max **5.00**, mean 2.069). **Honest artifact: the target is capped at $500k — 992 rows (4.8%)
  sit exactly at 5.00.** Features: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, **Latitude,
  Longitude**. Split `test_size=0.20, random_state=0` → train 16512 / test 4128. One-time ~14 MB fetch
  (visible logging — never silenced).
- **Baselines:** LinearRegression test R² **0.594** (MAE $53.5k); DecisionTree(depth 3) **0.499** ($60.6k);
  full tree 0.586 ($46.9k).
- **Gradient boosting (the ladder, matches chapter-plan D2):** GBR default (100 trees) **0.777** ($37.4k)
  → **early-stopped** (lr 0.1, depth 3, `n_iter_no_change=15` → **453 trees**) **0.821** ($32.7k). *(A deeper
  tuned variant — depth 4, lr 0.05 → 749 trees — reaches 0.834 / $31.2k; an exercise / going-further.)*
- **Foils + fast modern:** RandomForest(100) **0.798** ($33.6k) — beats *default* GB, loses to the tuned
  GB; **HistGBR default 0.837** ($31.0k) in **1.6 s** (vs the GB's ~7 s), and HistGBR(max_iter=1000) 0.851
  ($29.4k) in 3.7 s. **No universal best** — the four are close; HistGBR leads here *and* is fastest.
- **Importances on the headline early-stop GB — the dramatic MDI-vs-permutation divergence:**
  MedInc MDI **0.576** / perm **0.490**; AveOccup 0.124 / 0.137; **Longitude 0.113 / perm 3.18**;
  **Latitude 0.104 / perm 3.38**; HouseAge 0.041 / 0.027; AveRooms 0.026 / 0.053; AveBedrms 0.009 / 0.008;
  Population 0.007 / 0.003. **MDI says MedInc dominates and location is minor (~0.1); permutation on the
  held-out set says LOCATION dominates everything (lat/lon ≈ 3.2–3.4, i.e. shuffling either collapses the
  model) with MedInc second.** The strongest MDI-vs-permutation contradiction in the course — ch 06's MDI
  bias, made vivid.
- **Residual error analysis (headline GB), MAE by true-price bucket:** [0,2) $24.8k · [2,3.5) $35.5k ·
  [3.5,4.5) $54.0k · [4.5,5.0] **$74.2k**. The model errs ~3× more on the priciest homes (and the $500k cap
  truncates them) — an honest, located limit. Residual mean −0.009 (unbiased), std 0.465.

## Cell-by-cell (~26 cells; 7 figures; "Read the figure" after every figure)

1. (md) **Header** — `# 06 — A demanding case: California housing`; *Chapter 08 · Notebook 6 of 6*. The
   capstone: a full honest regression workflow that uses everything from NB 1–5. **Prerequisites:** all of
   Chapter 08 (the engine, the gradient view, classification, ν/depth/n_estimators, the estimator + early
   stopping), Chapter 06 NB 5 (permutation importance), Chapter 00 (train/test, regression). **What you'll
   do:** look at the data → baselines → a tuned GB with early stopping → R²/MAE in dollars on a sealed test
   → error analysis → RF/linear foils → the `HistGradientBoosting*` teaser → the bridge to ch 09–10.
2. (code) **Setup & fetch** — imports (numpy, pandas, matplotlib; `fetch_california_housing`,
   `train_test_split`, `LinearRegression`, `DecisionTreeRegressor`, `GradientBoostingRegressor`,
   `RandomForestRegressor`, `HistGradientBoostingRegressor`, `permutation_importance`, `r2_score`,
   `mean_absolute_error`); `viz`/`COLORS`; `SEED=0`. Fetch (visible logging), `as_frame=True`; print shape,
   target range, the **$500k cap count (992 = 4.8%)**; 80/20 split.
3. (md) **The problem & the honest first look.** Predict median house value (in $100k) from 8 features;
   we always look before modelling. Name the **$500k cap** (a survey artifact: every block at/above $500k
   was recorded as exactly 5.0) and the two geographic features (Latitude, Longitude) we'll watch.
4. (code) **Fig 1 — the target.** Histogram of `y` (note the spike at 5.0). *notebook-local.*
5. (md) **Read Fig 1.** Most homes $100k–$300k; a hard **ceiling at $500k** holds 4.8% of rows — no model
   can predict above it, and it will distort errors at the top (we'll see it).
6. (code) **Fig 2 — the map.** Scatter Longitude (x) vs Latitude (y), coloured by median house value
   (`CMAP_*`/sequential). *notebook-local.*
7. (md) **Read Fig 2.** The shape of California emerges; value clearly tracks **location** — the coast
   (Bay Area, LA, San Diego) is dear, the inland Central Valley cheap. Hold that thought for the importance
   figure.
8. (md) **Baselines first.** Never reach for the complex model first — fit a linear model and a shallow
   tree, on the training set, scored on the sealed test, so we know what "good" must beat.
9. (code) **Baselines.** Fit LinearRegression and DecisionTree(depth 3); print test R² and MAE in dollars.
10. (md) **Read the baselines.** Linear R² **0.59** ($53.5k typical error); a depth-3 tree **0.50**. The
    relationship is non-linear with interactions (location × income) — exactly the territory boosting owns.
11. (md) **A tuned gradient booster.** Apply the chapter: a small learning rate, shallow trees (depth 3),
    and **early stopping** (NB 5) to pick the number of trees — no guessing.
12. (code) **Fig 3 — the learning curve.** Fit GBR default (100) and GBR early-stopped (lr 0.1, depth 3,
    `validation_fraction=0.1, n_iter_no_change=15`); plot the full model's staged test R² vs trees with the
    early-stop tree marked; print default 0.777 → early-stop **0.821** (used 453), MAE $37.4k → **$32.7k**.
    *notebook-local (like NB 5's Fig J).*
13. (md) **Read Fig 3.** Default 100 trees already 0.78; early stopping climbs to **0.82** at ~450 trees
    and halts — $4–5k less typical error, chosen automatically.
14. (code) **Fig 4 — cross-method comparison.** Fit RandomForest(100) and HistGBR(default); bar chart of
    test R² (and a second panel of MAE in $k) for Linear, Tree, RF, GB (early-stopped), HistGBR. Print the
    row. *notebook-local bars with `COLORS`.*
15. (md) **Read Fig 4.** The ladder: linear 0.59 → tree 0.50 → **RF 0.80** → **GB 0.82** → **HistGBR 0.84**.
    The forest beats *default* GB but the tuned GB beats the forest, and HistGBR tops both. They are close —
    **no universal best**; the winner tracks the table, and here the modern histogram booster leads. Report
    both R² and MAE (in dollars) — they rank the same here but tell a human-readable story.
16. (code) **Fig 5 — predicted vs actual.** Scatter `y_test` vs the GB's predictions with the diagonal.
    *notebook-local.*
17. (md) **Read Fig 5.** Tight around the diagonal in the bulk; at the top a **vertical wall at 5.0** — the
    capped homes the model cannot exceed — and wider spread for expensive houses. The metric hides this; the
    picture shows it.
18. (code) **Fig 6 — where it errs.** MAE by true-price bucket (bar). *notebook-local.*
19. (md) **Read Fig 6.** Error rises with price: ~**$25k** on cheap homes to ~**$74k** on the priciest
    bucket (3×). Two honest reasons: the $500k cap truncates the top, and expensive homes are rarer and more
    heterogeneous. A single MAE is an average over very unequal regions.
20. (code) **Fig 7 — importances: MDI vs permutation.** Two panels via `viz.plot_feature_importances`
    (MDI; permutation on the test set, `n_repeats=10`). Print both. *reuse the helper.*
21. (md) **Read Fig 7 — the divergence.** MDI says **MedInc** dominates (0.58) and **location is minor**
    (~0.10). Permutation on held-out data says the opposite: shuffling **Latitude or Longitude collapses the
    model** (≈ 3.2–3.4 drop), so **location is the dominant factor**, MedInc second. Why the gap? MDI counts
    training-impurity reduction and is biased toward features split on early/often in a single scale; lat/lon
    enter many small splits and MDI under-credits them. **Permutation — measured on held-out data — is the
    trustworthy read** (ch 06 NB 5), and it matches the **map** (Fig 2). A reminder: importance is **use,
    not cause** (location *proxies* for unmeasured neighbourhood quality).
22. (md) **The fast modern default & the bridge.** HistGBR reached the **best** R² (0.84) in **~1.6 s** vs
    the classic GB's ~7 s — histograms + leaf-wise growth + a regularized objective. That is exactly what
    **XGBoost (Chapter 09)** and **LightGBM (Chapter 10)** refine. The chapter's engine, scaled up.
23. (md) **Your turn (tiered).** *easy:* from Fig 4 name the best model and its dollar error, and from
    Fig 6 say which homes it struggles with. *medium:* re-tune the GB with `max_depth=4, learning_rate=0.05`
    + early stopping — does the sealed-test R² beat 0.82? (relate depth to NB 4's interaction order). *harder:*
    explain the MDI-vs-permutation divergence for Latitude/Longitude, and argue from Fig 2 whether location
    *should* be important — then say why "important" still does not mean "causal".
24. (md) **What you built — and the chapter.** You ran a full honest regression workflow (look → baseline →
    tune with early stopping → sealed R²/MAE in dollars → error analysis → foils → the fast modern default),
    and closed Chapter 08: from fitting residuals by hand (NB 1) to a competitive, well-tuned, honestly
    evaluated booster on real data. **Vocabulary:** held-out R² / MAE in units · baseline-first · error
    analysis / residual-by-segment · MDI vs permutation divergence · `HistGradientBoosting*`.
25. (md) **Going further (optional).** The $500k cap argues for honest reporting (or modelling the censoring).
    `loss='huber'`/`'quantile'` for robust / interval predictions. Spatial features (lat/lon) interact —
    boosting captures it via depth (NB 4). HistGBR's native categorical/missing handling (ch 09–10). Proper
    practice would tune on a validation split / nested CV, not a single grid.
26. (md) **References** — Pace & Barry 1997 — California housing (DOI 10.1016/S0167-7152(96)00140-X);
    Friedman 2001 (DOI 10.1214/aos/1013203451); Friedman 2002 (DOI 10.1016/S0167-9473(01)00065-2);
    Breiman 2001 — permutation importance (DOI 10.1023/A:1010933404324); ESL §10.11–10.13
    (DOI 10.1007/978-0-387-84858-7); forward: Chen & Guestrin 2016 — XGBoost (DOI 10.1145/2939672.2939785);
    Ke et al. 2017 — LightGBM. `Previous: 05 — The estimator and its parameters.`
    `Next: Chapter 09 — XGBoost.`

## Figures (7, each followed by "Read the figure")
1. **Target histogram** (cell 4) — the $500k cap spike. *notebook-local.*
2. **Geographic scatter** (cell 6) — lon/lat coloured by price; the California map / coastal premium. *notebook-local.*
3. **GB learning curve** (cell 12) — staged test R² vs trees, early-stop marked. *notebook-local.*
4. **Cross-method bars** (cell 14) — R² + MAE$ for Linear / Tree / RF / GB / HistGBR. *notebook-local.*
5. **Predicted vs actual** (cell 16) — diagonal + the $500k ceiling wall. *notebook-local.*
6. **MAE by price bucket** (cell 18) — error rises with price. *notebook-local.*
7. **MDI vs permutation** (cell 20) — the location divergence. *reuse `viz.plot_feature_importances` ×2.*

## `src/` & guards
**No `src/` change** — `fetch_california_housing` is called directly (ch 06 `fetch_covtype` precedent);
reuse `viz.use_course_style` + `viz.plot_feature_importances`; all other figures notebook-local matplotlib
with `ml_course.colors`; **pytest stays 20**. *(The candidate `viz.plot_regression_diagnostics` is NOT
added — only 2 in-notebook uses, below the 3× reuse bar; revisit if ch 09–10 repeat it.)* Build via
`uv run python <scratchpad>/build_ch08_nb6.py`; **re-measure every anchor at build**. nbconvert top-to-bottom
**from project cwd** on a scratchpad copy (tracked file **output-free**); **banned-word scan over JSON real
cell text** = 0 (watch "just"/"simply"); `check_no_hardcoded_hex` passes; `ruff`/`black` clean (notebook
ruff-clean — mind unused imports, long lines, `zip(strict=)`); `gen_llms_txt` re-run.

## Honest scoping
A real, messy dataset: **no universal best** (linear 0.59 → RF 0.80 → GB 0.82 → HistGBR 0.84, all close;
HistGBR leads here and is fastest, but that is dataset-specific). **R² and MAE both reported, MAE in
dollars.** The **$500k cap** is surfaced and shown to distort top-end errors (honest limit, not hidden).
The **MDI-vs-permutation divergence** is the headline honesty lesson — MDI misleads on location, permutation
on held-out data is trusted; **importance ≠ cause** (location proxies for neighbourhood). `HistGradientBoosting*`
is shown (one fit) as the fast modern leader and the ch 09–10 bridge, not oversold. Capstone-visual-first
(7 figures). Both reviewers PASS (no BLOCK) + Rémy validates visually before commit; ff-merge notebook →
chapter; then **close the chapter via PR `chapter → main` (`--no-ff`)**.
