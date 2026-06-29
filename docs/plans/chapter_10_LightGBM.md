# Chapter plan — 10_LightGBM

> Status: **APPROVED by Rémy (2026-06-28)**; both gate reviewers **REVISE → NO BLOCK**, items folded
> (the speed thesis → matched-capacity-both-directions; GOSS → statistical efficiency, wall-clock
> regime-dependent; the "leaf-wise is already the modern default / HistGBR is a leaf-wise sibling" gift).
> Per-method arc over **5 notebooks**; per-NB plans drafted and Rémy-validated one at a time before each
> build. All API facts **measured live** (lightgbm 4.6.0 / xgboost 3.2.0 / sklearn 1.9.0).
>
> **RESTRUCTURED & re-gated (2026-06-28, both reviewers no BLOCK):** the standalone `num_leaves` NB was
> light → dropped (it stays the budget in NB 1 + the tuning dial in NB 4). New arc: NB 1 leaf-wise · NB 2
> GOSS+EFB · **NB 3 the optimal categorical split (built, Fisher 1958)** · NB 4 estimator · NB 5 capstone.
> All three fundamentals now build a mechanism by hand. (See "The per-method arc" below.)

## What this chapter is

LightGBM is **not a new paradigm** — it is histogram-based gradient boosting (the histogram method ch 09
NB 4 built and measured), with three distinctive choices + an optimal categorical split (Ke et al. 2017):

1. **Leaf-wise (best-first) growth** — expand the leaf with the largest loss reduction next, governed by
   **`num_leaves`** rather than `max_depth`.
2. **GOSS** (Gradient-based One-Side Sampling) — keep the large-gradient rows, subsample the small ones
   and up-weight them so the split-gain estimate stays unbiased; train on fewer rows.
3. **EFB** (Exclusive Feature Bundling) — bundle approximately-exclusive sparse features into one,
   shrinking the effective feature count.

**The shared modern skeleton (the spine).** Leaf-wise growth is **not a LightGBM exclusive** — the
learner met it *twice already*: ch 08 NB 5 named `HistGradientBoosting*` as growing **leaf-wise**
(`max_leaf_nodes=31`), and ch 09 NB 4 named XGBoost's `grow_policy='lossguide'`. Measured: sklearn's
`HistGradientBoostingClassifier` defaults to `max_leaf_nodes=31, max_depth=None` — it *is* a leaf-wise,
`num_leaves`-governed histogram booster (`max_leaf_nodes=31` is literally LightGBM's `num_leaves=31`). So
**leaf-wise + histogram is the modern default across HistGBR, XGBoost-hist, and LightGBM** — which is why
their speeds and scores are close.

**Honesty bar — the speed claim, refused as a spec sheet.** LightGBM's famous speed advantage was
dramatic against XGBoost's original `exact` method (pre-2017); against modern `hist` it is **small and
direction-dependent**. Measured *default-vs-default* (200k×50): XGBoost-hist 1.17s vs LightGBM 2.40s — but
that compares **different-shaped trees** (XGBoost depthwise depth-6 vs LightGBM leaf-wise num_leaves 31 →
depth 15). **At matched capacity the winner flips with the convention** (re-measured at NB-5 plan): both
leaf-wise num_leaves=31 → LightGBM faster (≈2.4s vs ≈3.2s, accuracy tied); both depth-6 → XGBoost faster
(≈1.2s vs ≈3.1s, tied). The honest verdict is **"no universal best — measure on your data,"** not
"XGBoost is faster now." **When to reach for LightGBM:** very large `n` and/or wide sparse categorical
data, where GOSS and EFB were designed to bite — the capstone *dials `n` up* to find the regime where
LightGBM's curve crosses XGBoost's.

Builds on ch 08 (gradient boosting; the per-row gradient as each tree's target — the GOSS prerequisite)
and ch 09 (histogram method, leaf-wise named, native categoricals, PR-AUC/threshold/permutation). Closes
the boosting family before neural nets (ch 11–12).

## Live API ground truth (measured, lightgbm 4.6.0 / xgboost 3.2.0 / sklearn 1.9.0)

`lightgbm` is in the `boosting` extra (+ `libomp` on macOS, installed).

- **Resolved defaults**: `boosting_type='gbdt'`, **`num_leaves=31`**, **`max_depth=-1` (unbounded)**,
  `learning_rate=0.1`, `n_estimators=100`, **`min_child_samples=20`**, `subsample`/`colsample_bytree`=1.0,
  **`reg_lambda=0`/`reg_alpha=0`** (no L2 by default — a *posture* difference from XGBoost's `lambda=1`,
  paired with a different capacity control: num_leaves+min_child_samples vs depth+min_child_weight, **not**
  "less regularized overall"), `max_bin=255`, `min_split_gain=0`.
- **HistGBR (the leaf-wise sibling)**: sklearn `HistGradientBoostingClassifier` defaults
  `max_leaf_nodes=31, max_depth=None` — leaf-wise, num_leaves-governed; the natural matched sibling for NB 5.
- **Leaf-wise** (1 tree, `max_depth=-1`): `num_leaves` 4/8/31/127 → leaves 4/8/31/**107** (a *cap*: built
  107 of the 127 allowed before `min_child_samples` stopped it), depth 2/4/6/**15**. Large `num_leaves` →
  deep, lopsided, overfit-prone trees.
- **GOSS**: `data_sample_strategy='goss'` (4.x; `boosting_type='goss'` legacy alias). EFB on by default.
  **GOSS buys ~no wall-clock on dense moderate data** (measured: gbdt 2.39s vs goss 2.41s @150k×50; 3.19s
  vs 3.20s @500k×50) — its real payoff is **statistical efficiency** (near-full quality on fewer rows,
  beating a uniform subsample); wall-clock only when the data-pass dominates (very large/wide data).
- **Early stopping** (4.x): `callbacks=[lgb.early_stopping(N)]` + `eval_set` (no `early_stopping_rounds`
  fit-kwarg); `best_iteration_` exposed (1000 → 99).
- **Categorical**: a pandas `category` dtype column is auto-detected, split by an optimal partition
  (categories sorted by gradient stats, Fisher 1958 / Ke §4) — confirmed.
- **Speed/accuracy** (200k×50, 300 trees): default-vs-default LightGBM 2.40s/0.9631 vs XGBoost-hist
  1.17s/0.9657; **matched** flips with the convention (above).

## The per-method arc (5 notebooks)

3 fundamentals (one concept each, by hand) → estimator & parameters → demanding case. **Histogram binning
is reused from ch 09 NB 4 (named, not rebuilt).**

### NB 1 — Leaf-wise (best-first) growth, by hand
- **One concept:** grow the leaf with the largest loss reduction *next*, not level by level. **Footing:**
  the learner met leaf-wise *named* twice — ch 08 NB 5 (`HistGradientBoosting*`, `max_leaf_nodes`) and
  ch 09 NB 4 (`grow_policy='lossguide'`); **here we build it.** On a toy, grow a tree both ways; leaf-wise
  reaches lower **training** loss for the same leaf count (state TRAINING explicitly — held-out is NB 2's
  question), while growing **lopsided** (qualitative here; the quantitative depth-vs-num_leaves sweep is
  NB 2). Match LightGBM's first tree.
- ~3 figures (the two growth orders side by side; training-loss vs #leaves, leaf-wise ≤ level-wise; one
  lopsided leaf-wise toy tree).

> **Restructured (2026-06-28, re-gated no BLOCK):** the standalone `num_leaves` NB was light (a parameter
> sweep, no by-hand build, overlapping NB 4) → dropped; `num_leaves` stays the *budget* in NB 1 and the
> *tuning dial* in NB 4. The three fundamentals are rebalanced onto by-hand-buildable mechanisms.

### NB 2 — GOSS (built) + EFB (named): how LightGBM gets light
- **Scope:** **one built concept — GOSS** + **one named companion — EFB** (the ch 09 NB 4 build-one /
  name-one precedent).
- **GOSS, by hand, led by statistical efficiency:** recall the split-gain is a **sum over rows** of `g,h`
  (ch 09 NB 2's structure score) — *name that anchor*. Rank rows by `|gradient|`, keep the top `a`, sample
  `b` of the rest and **up-weight by `(1−a)/b`** so the row-sum estimate stays ~unbiased; fit on the
  smaller reweighted set. Show **GOSS ≈ full-data quality on far fewer rows, and beats a uniform subsample
  at the same fraction** (the discriminating experiment). State plainly: **GOSS's wall-clock benefit is
  regime-dependent — ~flat on dense moderate data (measured); it bites on very large/wide data.**
- **EFB, by intuition (named, not built):** bundle **approximately**-exclusive sparse features (tolerates
  a small conflict rate — Ke §4 — so it is more than concatenating one-hot columns) into one, cutting the
  effective feature count and histogram cost.
- ~3 figures (GOSS: `|gradient|` distribution + kept-top/sampled-rest; **GOSS vs uniform subsample
  accuracy at matched fraction**; EFB bundling schematic).

### NB 3 — The optimal categorical split, by hand (NEW — replaces the light `num_leaves` NB)
- **One concept:** how to split a *categorical* feature optimally. Naively, partitioning K categories into
  two groups is `2^(K−1)−1` ways (exponential). Fisher (1958): for a convex split criterion — **the
  structure-score gain `G²/(H+λ)` built in ch 09 NB 2** — the optimal binary partition is **contiguous**
  once categories are sorted by their gradient statistic `G/H`, so only **`K−1`** candidates (linear).
- **Build it by hand** on a toy: per-category `(G,H)`, sort by `G/H`, evaluate the `K−1` contiguous splits
  with the structure-score gain, pick the max → the category set going left. **Match LightGBM exactly**
  (de-risked: by-hand LEFT={1,3,5} == LightGBM {0,2,4}, complementary = identical partition;
  brute-force-confirmed global-optimal). Contrast XGBoost's partition heuristic — ch 09 *used* native
  categoricals but never built the split (genuinely new).
- **Build-time pins (ml-expert):** exact parity needs `min_data_per_group=1` (the default 100 returns a
  non-optimal split on a small toy), plus `min_data_in_leaf=1, min_sum_hessian_in_leaf=0, cat_l2=0,
  cat_smooth=0` — OR ≥100 rows/category. **Teaching note:** with `h=1` (regression toy) `G/H` reduces to
  the per-category **target mean** (sort categories by their average target); give `G/H` as the general
  key (carries to classification / second-order). Keep the toy **binary** (multiclass is out of scope).
- ~3 figures (categories by mean/gradient; the sorted order + the `K−1` contiguous candidates; by-hand
  partition == LightGBM).

### NB 4 — The estimator `LGBMClassifier`/`LGBMRegressor` & its parameters
- **Integrative.** **`num_leaves`/`min_child_samples` — the leaf-wise capacity dial + its floor (this is
  where `num_leaves` is tuned).** Close NB 1's lopsided→overfit loop here: a single tree's test peaks ~64
  leaves then falls (train→1.0, depth→21); `num_leaves` is a *cap* (built 107 of 127); the ensemble is
  robust (test plateaus ~0.92–0.927, gap widens, depth→25) but the **floor matters** (`min_child_samples=1`
  → test 0.858 / default 20 → 0.927 / 300 → underfits 0.906); the rule `num_leaves < 2^max_depth`. Also:
  `learning_rate`×`n_estimators`; `feature_fraction`/`bagging_fraction`(+`bagging_freq`); `reg_lambda`/
  `reg_alpha` (off by default — the *posture* contrast with XGBoost); `data_sample_strategy='goss'` (NB 2);
  **native categorical** (NB 3, the optimal split); **early stopping** via `callbacks=[lgb.early_stopping]`;
  importances (`'split'` vs `'gain'`, MDI caveat). Honest tuning → one sealed test (`GridSearchCV` with
  `verbose` so folds show — never `verbose=-1`).
- ~3–4 figures (num_leaves × min_child_samples [the dial + floor]; gbdt vs goss accuracy/efficiency;
  default-vs-tuned).

### NB 5 — A demanding case (visualization-first capstone)
- A **larger** tabular problem where speed can matter. **Dataset (verification owed at NB-5 plan):**
  primary **MiniBooNE** (`fetch_openml`, ~130 064×50, binary, numeric — verified loadable, 72/28, no NaN);
  fallback a **scaled synthetic** (`make_classification` 300k–500k) that lets us **dial `n`** to draw the
  speed/accuracy curve. (Avoid reuse: covtype/spambase/California/Adult/breast_cancer.)
- **The matched-capacity comparison — pre-committed convention:** compare the three histogram boosters
  **LightGBM / XGBoost-hist / HistGBR** under **one** convention — **num_leaves (= HistGBR's
  `max_leaf_nodes`), leaf-wise, depth unbounded** (HistGBR the natural matched sibling). Report **fit time
  AND score** at matched capacity, and **reconcile in prose against the spine's unmatched default-vs-
  default 200k number** (defaults differ → the unmatched gap is tree shape, not the library). Then **dial
  synthetic `n` up** to find the regime where LightGBM crosses ahead (the positive "when to reach for
  it"). GOSS on/off shown as **efficiency** (accuracy vs fraction), not a flat speed bar.
- **Arc** (≥6 figures, **~28–30 cells**): look → baselines → tuned LightGBM + early stopping → held-out
  metrics (PR-AUC/threshold, ch 00) → the matched speed/accuracy comparison + the dial-`n` crossover →
  error analysis → honest cross-method → importances (split vs gain vs permutation). **Honesty axis:**
  speed is **measured & conditional**, the three boosters are close on accuracy, the winner depends on
  data/shape — "no universal best." Closes the boosting family.

## Honest scoping (the ml-expert bar)
- **LightGBM = histogram GBDT + leaf-wise + GOSS + EFB + optimal categorical split.** Not a new paradigm;
  histogram (ch 09) reused, not re-taught. Leaf-wise + histogram is the **shared modern skeleton** (HistGBR
  already does it) — the cleanest "no universal best."
- **Speed edge is small & flips with the matching convention** (measured both ways) — *not* "XGBoost
  faster"; the unmatched default-vs-default number is a motivating first observation, not the verdict.
- **GOSS = statistical efficiency** (near-full quality on fewer rows, beats uniform subsample); its
  **wall-clock benefit is regime-dependent and ~flat on dense moderate data (measured)** — surfaced, not
  asserted. **EFB named** (tolerates a small conflict rate), not built.
- **`num_leaves` is the leaf-wise capacity dial, not `max_depth`** (introduced as the budget in NB 1,
  *tuned* in NB 4 — measured: single tree peaks ~64 then overfits; ensemble robust; `min_child_samples`
  the floor). No standalone NB — it is a dial, not a by-hand mechanism.
- **The optimal categorical split is BUILT (NB 3, Fisher 1958)** — sort categories by `G/H`, best
  contiguous partition, matched to LightGBM exactly. Genuinely new vs ch 09 (which used native categoricals
  but never built the split). The three fundamentals are all by-hand builds: leaf-wise · GOSS · categorical.
- **`reg_lambda=0`** is a posture difference (paired with the capacity-control contrast), not
  "unregularized."
- Gain importance MDI-biased → permutation on held-out (ch 06/08/09). Every anchor re-measured at
  NB-plan/build; seeds pinned; the speed/efficiency claims measured, never quoted.

## `src/` & guards
No `src/` change expected (reuse `viz`; `dump_model`/`booster_` for trees; `fetch_openml`/
`make_classification`; sklearn baselines/CV; pytest 20). By hand before the library; colours only from
`ml_course.colors`; seeds fixed; "Read the figure" after every figure; **never silence output** (no
`verbose=-1` in notebooks — LightGBM's log stays visible; GridSearch `verbose` on so folds show);
banned-word scan = 0; hex clean; ruff/black clean; output-free; `gen_llms_txt`; two-reviewer gate + Rémy
visual before each commit; ff-merge `notebook→chapter`; **chapter→main via PR (`--no-ff`)** at close.
(At the NB-3 build, fix the categorical line in the measurement harness so that anchor actually executes.)

## References (chapter-level; per-NB DOIs at build)
- Ke, G., Meng, Q., Finley, T., et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision
  Tree.* NeurIPS 30. (Leaf-wise; GOSS §3; EFB §4; optimal categorical split.)
- Fisher, W. D. (1958). *On grouping for maximum homogeneity.* JASA 53(284). DOI
  10.1080/01621459.1958.10501479.
- Chen & Guestrin 2016 (XGBoost; DOI 10.1145/2939672.2939785); Friedman 2001 (DOI 10.1214/aos/1013203451);
  ESL §10 (DOI 10.1007/978-0-387-84858-7).

## Reviewer gate outcome
Both reviewers **REVISE → no BLOCK**, each re-running live checks. **ml-expert**: the speed comparison was
unmatched/one-sided and flips under matching → reframed to matched-both-directions; GOSS buys no
wall-clock on dense data (measured) → reframed to statistical efficiency; MINOR/NIT (EFB conflict rate;
NB 1 *training* loss; `reg_lambda=0` = posture; num_leaves a *cap*; categorical harness-bug) folded.
**pedagogy**: leaf-wise named *twice* before + HistGBR is a leaf-wise num_leaves booster → used in the
spine and as NB 5's matched sibling; NB 3 = "one built (GOSS) + one named (EFB)" with the unbiased-reweight
anchored on ch 09 NB 2's row-sum gain; NB 5 one matched convention + reconcile with the spine; positive
"when to reach for it" + dial-`n` crossover; NB 4 names the optimal categorical split; capstone ~28–30
cells. Per-notebook specifics (MiniBooNE-vs-synthetic verification; the matched-capacity numbers) assigned
to the per-NB plan stage.
