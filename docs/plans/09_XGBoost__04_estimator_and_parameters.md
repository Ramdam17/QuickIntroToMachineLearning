# Notebook plan — 09_XGBoost / 04_estimator_and_parameters

> Status: **APPROVED by Rémy (2026-06-28)** (no reviewer gate at the per-NB plan stage — both reviewers
> return on the built notebook). All anchors measured live (xgboost 3.2.0 / sklearn 1.9.0 / numpy
> 2.4.6 / pandas 3.0.3, SEED=0); re-measured at build. Build via `<scratchpad>/build_ch09_nb4.py`.

## Context

NB **4 of 5**, the **integrative** notebook. It is **not** "one concept" (that rule governs NB 1–3):
it drives the real estimator `XGBClassifier` and **owns the histogram method** as a named mechanism.
**Spine:** the aggressive defaults **memorize** the training set → understand each knob *grouped by the
concept it controls* (NB 1–3, ch 04/06/08) → the **histogram** engine that makes it fast (measured) →
tune **honestly** (cross-validate on train, open the test once). It holds the chapter's honesty bar:
the defaults overfit, but XGBoost ships with **L2=1 ON**, so the default test is already strong and
disciplined tuning buys a **modest** gain — no guaranteed accuracy win.

## Anchors (measured live, xgboost 3.2.0 / sklearn 1.9.0, SEED=0 — re-measured at build)

Through-line: `make_classification(n_samples=6000, n_features=20, n_informative=8, n_redundant=4,
n_clusters_per_class=2, class_sep=0.9, flip_y=0.04, random_state=0)`; stratified 70/30 (train 4200 /
test 1800, pos 0.498). `flip_y=0.04` = 4% flipped labels (the noise ceiling, ≈0.95–0.96).

- **Resolved core defaults** (`booster.save_config()`): `eta`=0.3, `max_depth`=6, `min_child_weight`=1,
  `subsample`/`colsample_*`=1, **`lambda`(L2)=1**, `alpha`=0, `gamma`=0, `max_bin`=256,
  `grow_policy`=depthwise, `tree_method`=auto(→hist), `n_estimators`→100, `objective`=binary:logistic.
  Aggressive vs sklearn GB (depth 6≫3, eta 0.3≫0.1); **L2 on by default** (sklearn GB has none).
- **Defaults overfit:** train **1.0000** / test **0.9444** / gap 0.0556 / 2608 leaves. Tamer
  (depth3 eta0.1 n200): 0.9679 / 0.9378 / gap 0.0301 — smaller gap, *not* higher test (honest: the two
  are different goals on noise-capped data).
- **max_depth** (eta0.3 n100): d1 .921/.904(200 lv) · d2 .960/.933(392) · d3 .981/.942(708) ·
  **d4 .992/.9467(1176)=test peak** · d6 1.0/.9444(2608) · d8 1.0/.9444(3287) · d10 1.0/.9444(3603).
  Default depth 6 is *past* the peak; train pegged at 1.0 from depth 6 on.
- **gamma** (min_split_loss; depth6 eta0.3 n100): g0 1.0/.9444(2608 lv)→g0.5(1243)→g1 .986/.941(948)
  →g2(649)→g5 .965/.936(389)→g10 .956/.939(264). **Leaves 2608→264, gap 0.056→0.017.**
- **min_child_weight** (Cover=ΣH floor): mcw1 1.0/.9444(2608)→mcw20 .980/.939(882)→mcw100
  .933/.915(321).
- **reg_lambda** (L2): λ0 1.0/.9456 · λ1 1.0/.9444 · λ10 .9998/.9444 · λ100 .978/.9433 · λ1000
  .950/.9283 — mild until ≈100 (default λ=1 gentle).
- **subsample**: .5/.7/1.0 → .9361/.9417/.9444 (rows: slightly hurts here). **colsample_bytree**:
  .5/.7/1.0 → **.9472**/.9461/.9444 (columns: slightly helps). Mild/mixed — earn their keep on bigger
  noisier data.
- **eta × n_estimators** (staged via `iteration_range`, 600 trees, depth6): eta0.3 best **.9450@200**,
  flat to .9444@600 (no collapse); eta0.1 best **.9494@200**; eta0.03 best **.9511@600** (climbs).
  Lower eta → more trees → slightly higher flatter optimum (ch 08 NB 4, re-felt).
- **grow_policy**: depthwise(depth6) .9444/2608 lv · lossguide(max_leaves63,depth0) .9478/3250 lv
  (LightGBM leaf-wise — ch-10 bridge, named not adopted).
- **HISTOGRAM** — larger draw (60000×50, 42000/18000 split, 300 trees): **`exact` 4.23s / .9646**;
  **`hist` 0.85s / .9660 (~5× faster, no accuracy cost)**; `hist(max_bin=64)` 0.70s/.9651;
  `hist(max_bin=256)` 0.85s/.9660.
- **GridSearchCV → sealed test** (96 cands × 5 folds, 5.5s): best {lr0.3, depth6, n300, reg_lambda10,
  subsample0.7}, CV 0.9495; **tuned sealed 0.9472 vs default 0.9444 → +0.0028** (closed the gap more
  than it raised test).
- `feature_importances_`: (20,), sums to 1, gain-normalized MDI-style (ch 06/08 caveat → permutation in
  NB 5).

## Cell-by-cell (~23 cells; 4 figures; "Read the figure" after each)

1. (md) **Header** — `# 04 — The estimator and its parameters: XGBoost in practice`; *Ch 09 · NB 4 of 5*.
   Prereqs: NB 1 (`w*=−G/H`), NB 2 (regularized objective λ/γ, structure-score gain, `Cover=ΣH`), NB 3
   (sparsity-aware splits); ch 08 NB 4 (eta×n_estimators, depth=interaction order, overfit); ch 00
   (train/test, CV, the sealed test); ch 06/08 (MDI caveat). What you'll do.
2. (md) **Recap — the engine, and what the knobs touch.** `w*=−G/(H+λ)`, gain `½[…]−γ`, learned default
   direction — wrapped in `fit`/`predict`. The parameters are dials on pieces you built by hand:
   `reg_lambda`/`gamma` price complexity inside the objective (NB 2); `min_child_weight` floors
   `Cover=ΣH` (NB 1/2); `max_depth` caps interaction order (ch 04/08). No new math.
3. (code) **Setup & dataset.** imports (numpy, pandas, matplotlib; `XGBClassifier`; sklearn
   `make_classification`/`train_test_split`/`accuracy_score`/`GridSearchCV`); `viz.use_course_style()`/
   `COLORS`; `SEED=0`. Build the set (comment `flip_y=0.04` = 4% noise); stratified split; print shapes,
   balance (≈0.498), a 2-line `describe()` peek (pandas-first).
4. (md) **The defaults are aggressive.** Deeper/faster than sklearn GB (depth 6 vs 3, eta 0.3 vs 0.1) →
   tends to overfit; but **L2 on by default** (`lambda=1`) → ships partly defended.
5. (code) **Read the resolved defaults.** `json.loads(m.get_booster().save_config())` → small table of
   the core knobs (the one place we read the engine config).
6. (code) **Fit defaults; watch them memorize.** `XGBClassifier()` → train 1.0000 / test 0.9444 /
   gap 0.0556 / 2608 leaves; tamer contrast (depth3 eta0.1 n200 → 0.9679/0.9378).
7. (md) **Read it — memorization is the tell.** Train = exactly 1.0 (memorized 4200 rows + the flipped
   labels); the 5.6-pt gap is the price; yet test 0.944 is decent (L2 on). Goal = stop memorizing /
   shrink the gap / find the robust setting, **not** maximize a noise-capped test. (Tamer = smaller gap,
   not higher test — different goals.)
8. (md) **Knob group 1 — tree complexity: `max_depth`** (= interaction order, ch 04/08).
9. (code) **max_depth sweep + Fig 1.** depth∈{1,2,3,4,6,8,10} → train/test/leaves table. **Fig 1 — the
   depth dial:** train & test acc vs depth, test peak@4 & default@6 marked; leaf count exploding.
10. (md) **Read Fig 1.** Test peaks @4 (0.9467); default depth 6 is *past* it (0.9444), train pegged at
    1.0, leaves 1176→3603 for zero test; depth 1 underfits. Always sweep depth.
11. (md) **Knob group 2 — pricing complexity in the objective: `reg_lambda`, `gamma`,
    `min_child_weight`** (NB 2's λ/γ + the `Cover=ΣH` floor, as constructor args).
12. (code) **gamma & min_child_weight sweeps + Fig 2.** gamma∈{0,.5,1,2,5,10}, mcw∈{1,5,20,50,100} →
    leaves+train/test; a one-line `reg_lambda` note (mild until ≈100). **Fig 2 — regularizers close the
    gap:** two panels (gamma | mcw), #leaves collapsing, train/test bands converging.
13. (md) **Read Fig 2.** `gamma` prunes splits below the gain bar (NB 2's γ): leaves 2608→264, gap
    0.056→0.017. `min_child_weight` refuses thin leaves: `Cover=ΣH=Σp(1−p)`, p≈0.5 → each point ≈0.25,
    so mcw=1 ≈ "≥~4 points/leaf", mcw=100 ≈ "≥~400". Over-regularizing underfits — CV finds the sweet
    spot, not the eye.
14. (md) **Knob group 3 — stochasticity & learning rate: `subsample` (Friedman 2002), `colsample_*`
    (new vs sklearn GB — the RF idea, ch 06), `eta`×`n_estimators`** (ch 08 NB 4's trade-off).
15. (code) **eta×n_estimators (staged) + subsample/colsample + Fig 3.** eta∈{0.3,0.1,0.03}, 600 trees,
    staged test via `iteration_range`; print curves+best; subsample/colsample one-liners. **Fig 3 — eta
    × trees:** staged test acc vs #trees (log-x), one line per eta, best marked.
16. (md) **Read Fig 3.** Low eta → more trees → higher flatter optimum (eta0.03→0.951@600; eta0.3
    plateaus ~0.944). Same trade-off as ν in ch 08 — but **no collapse** here (L2 keeps each step
    modest). subsample/colsample ±0.005 — mild/mixed; earn their keep on bigger data.
17. (md) **The engine that makes it fast — histogram split finding.** Every split scan used to consider
    *every distinct value* (ch 04); `tree_method='hist'` **bins** each feature into ≤`max_bin` (256)
    buckets, scanning ≤256 edges — same gain (NB 2), a fraction of the work. *(Weighted quantile sketch,
    C&G §3.2–3.3, named & deferred to ch 10.)*
18. (code) **hist vs exact, measured + Fig 4.** Larger draw (42000×50, 300 trees): time exact vs hist,
    hist with max_bin∈{64,256}; print time+acc. **Fig 4 — the histogram engine:** (a) binning idea (one
    feature's values + a few quantile bin edges; labelled illustratively, default 256); (b) fit-time
    bars exact 4.23s vs hist 0.85s; (c) test-acc bars 0.9646 vs 0.9660.
19. (md) **Read Fig 4.** ~5× faster (4.2s→0.85s), **no accuracy cost** (0.9646→0.9660 — a hair better,
    bins also lightly regularize); `max_bin=64` faster still (0.70s) for a hair less accuracy. Why `hist`
    is the 3.x default. `grow_policy`: depthwise (default) vs lossguide (leaf-wise — LightGBM, ch 10;
    here 0.9478 vs 0.9444).
20. (md) **Tuning honestly — CV on train, open the test once** (ch 00). Search on train with
    cross-validation, pick by CV, look at test **exactly once** at the end. Tuning against test = the
    cardinal sin.
21. (code) **GridSearchCV → the sealed test.** Grid (96 cands × 5 folds, `verbose=1` so folds show) →
    best params, best CV, tuned sealed test, default sealed test, the delta. *(No figure: the +0.003 test
    delta would render as two identical bars — report as numbers + the gap story. Keeps 4 figures.)*
22. (md) **Read the tuning result.** CV chose {depth6, eta0.3, n300, reg_lambda10, subsample0.7}
    (CV 0.9495); sealed test **0.9472 vs default 0.9444 → +0.003**. Honest: the default was already near
    the ceiling, so tuning bought a **modest** gain and mostly a **more robust** model (less
    memorization, smaller gap). The chapter's truth — XGBoost's edge is speed, missing-value handling
    (NB 3), regularization you now control, *not* a big accuracy jump. `feature_importances_` here is
    gain-MDI → permutation (held-out) is the honest read (NB 5).
23. (md) **Your turn.** *easy:* refit `max_depth=3, n_estimators=300`, predict then report the new
    train/test gap. *core:* sweep `gamma` **or** `min_child_weight` (3 values), report leaves + test,
    and name the NB-2 quantity you're thresholding. *reach:* a depth-6 sklearn `GradientBoosting` has no
    L2; XGBoost's default `lambda=1` does — why does that make XGBoost's aggressive defaults overfit
    *less catastrophically*? and why can `hist` match `exact`'s accuracy while scanning far fewer
    thresholds?
24. (md) **What you built.** Drove the real estimator: read its aggressive defaults, watched them
    memorize, turned each knob from the concept that owns it (depth=interaction order;
    `reg_lambda`/`gamma`/`min_child_weight`=NB 2 + `Cover=ΣH`; `subsample`/`colsample_*`=stochasticity;
    `eta`×`n_estimators`=the step trade-off), met the **histogram** engine (~5× faster, no cost), tuned
    honestly (CV→sealed test) for a modest robust gain. **Vocabulary:** `tree_method='hist'` · `max_bin`
    · `grow_policy` · `reg_lambda`/`gamma`/`min_child_weight` · `subsample`/`colsample_bytree` ·
    `eta`×`n_estimators` · gain (MDI) importance · sealed test. Next: the demanding case.
25. (md) **References** — Chen & Guestrin 2016 (histogram §3.2–3.3; DOI 10.1145/2939672.2939785);
    Friedman 2002, Stochastic GB (DOI 10.1016/S0167-9473(01)00065-2); Ke et al. 2017, LightGBM
    (leaf-wise/`lossguide` forward ref; NeurIPS 2017); Pedregosa et al. 2011 (`GridSearchCV`); ESL §10
    (DOI 10.1007/978-0-387-84858-7). `Previous: 03 — Sparsity-aware splits.` `Next: 05 — A demanding case.`

## Figures (4, each followed by "Read the figure")
1. **The depth dial** (cell 9) — train/test acc vs max_depth; test peak@4 & default@6 marked; leaves exploding.
2. **Regularizers close the gap** (cell 12) — gamma & min_child_weight: leaves collapse, train/test converge.
3. **eta × trees** (cell 15) — staged test acc vs #trees (log-x), eta 0.3/0.1/0.03, best marked.
4. **The histogram engine** (cell 18) — binning idea (≤max_bin edges) + hist-vs-exact fit-time + test-acc bars.

(Tuning = measured numbers + prose, cells 21–22; not a misleadingly-flat bar chart.)

## `src/` & guards
**No `src/` change** — notebook-local matplotlib + `viz.use_course_style`; dataset via
`make_classification`; trees via `trees_to_dataframe`; **pytest stays 20**. Build via
`<scratchpad>/build_ch09_nb4.py`; re-measure anchors at build; nbconvert top-to-bottom from project cwd
on a scratchpad copy (output-free tracked file); banned-word JSON scan = 0 (watch "just"/"simply");
`check_no_hardcoded_hex`; ruff/black clean; `gen_llms_txt` re-run. Two-reviewer gate (no BLOCK) + Rémy
visual before commit; ff-merge `notebook → chapter`.

## Honest scoping
Integrative (not one concept): the estimator + its knobs, each tied to the concept that owns it. The
**histogram method is built as a named concept** (intuition → measured ~5× speedup at no accuracy
cost); the weighted quantile sketch (C&G §3.2–3.3) is **named & deferred to ch 10**, not built.
`grow_policy=lossguide` shown once as the ch-10 leaf-wise bridge, not adopted. Headline honesty:
**defaults overfit (train→1.0) but the default test is already strong (L2=1 on); disciplined tuning
buys a modest test gain (+0.003 here) and mostly a more robust model** — no guaranteed accuracy win,
exactly the chapter thesis. Early stopping is named as NB 5's tool, not built here. Every number
measured live, seed-pinned, re-measured at build.
