# Chapter plan — 09_XGBoost

> Status: **APPROVED by Rémy (2026-06-27)**; both gate reviewers NO BLOCK. The chapter follows the
> per-method arc (3 fundamentals → the estimator & its parameters → the demanding case) over **5
> notebooks**. Per-notebook plans (cell-by-cell) are drafted and Rémy-validated one at a time before
> each build. All API facts below were **measured live on the installed stack** (see "Live API ground
> truth"), not quoted from memory.

## What this chapter is

XGBoost is **not a new algorithm** — it is the gradient-boosting engine of Chapter 08, refined in
three concrete ways and engineered for speed:

1. a **second-order** view (use the loss's curvature, the Hessian, not only its gradient),
2. a **regularized objective** (penalize tree complexity *inside* the loss, via λ and γ),
3. **sparsity-aware** split finding (a learned default direction for missing values),

plus the engineering (histogram split finding, column subsampling, parallelism) that made it the
dominant tabular method of the 2015–2020 Kaggle era (Chen & Guestrin 2016).

The spine of the chapter is **continuity with Chapter 08**: the learner already fit the negative
gradient with trees (ch 08 NB 2) and already used the Hessian once — the Newton leaf-step for
classification (ch 08 NB 3, `γ_leaf = Σ(y−p)/Σ p(1−p)`). XGBoost generalizes that single move into a
uniform, regularized framework for **any** twice-differentiable loss. We build each new piece **by
hand** before the library, exactly as the course charter demands, and we keep the honesty bar high:
**XGBoost does not reliably beat a well-tuned `GradientBoosting`/`HistGradientBoosting` on accuracy** —
its real edges are missing-value handling, speed, and the regularization knobs. "No universal best,"
again.

This chapter builds directly on ch 04 (trees / split criteria), ch 08 (gradient boosting, the Newton
leaf), and ch 00 (honest evaluation, ROC/PR/threshold). It is the launchpad for ch 10 (LightGBM, the
leaf-wise sibling).

## Live API ground truth (measured, xgboost 3.2.0 / sklearn 1.9.0 / numpy 2.4.6 / pandas 3.0.3)

Environment fix recorded: xgboost on macOS needs the OpenMP runtime — `brew install libomp` (done;
keg-only at `/opt/homebrew/opt/libomp`). `xgboost` and `lightgbm` live in the `boosting` extra
(`uv sync --extra dev --extra boosting`).

- **Resolved core defaults** (from `booster.save_config()`, not memory): `eta`=0.3, `max_depth`=6,
  `min_child_weight`=1, `subsample`=1, `colsample_bytree`=1, **`lambda` (L2)=1**, `alpha` (L1)=0,
  `gamma`=0, `max_bin`=256, `grow_policy`=`depthwise`, `tree_method`=`auto` (→`hist`), `n_estimators`=100.
  Note the **aggressive defaults** vs sklearn GB (eta 0.3 ≫ 0.1, depth 6 ≫ 3) — they often overfit;
  and **L2=1 by default** (sklearn GB has none) — regularization is on out of the box.
- **Early stopping is a *constructor* param in 3.x**: `XGB*(early_stopping_rounds=…, eval_metric=…)`,
  with `eval_set=[(X,y)]` passed to `.fit()` (no longer a `.fit()` kwarg). Measured: requested 500 →
  `best_iteration`=195, 211 trees built, `best_score` exposed.
- **Native missing-value handling**: default `missing=np.nan`; fit with NaNs present works with no
  imputation. Confirmed across estimators: **plain `GradientBoosting*` rejects NaN** (`ValueError`),
  **`HistGradientBoosting*` accepts** it (ch 08 named it), **XGBoost accepts** it (2016, first).
- **`feature_importances_`** present (gain-type, normalized to sum 1) — same MDI-style bias as ch 06/08
  → permutation on held-out data is the honest cross-check.
- **The by-hand correctness anchor is real** (measured on a 1-tree/1-split toy, λ=1, γ=0, η=1,
  pinned `base_score`; both reviewers independently reproduced it): leaf weights `w* = −G/(H+λ)` match
  XGBoost **exactly**; XGBoost's **reported `Gain` = 2× the textbook formula**
  `½[G_L²/(H_L+λ)+G_R²/(H_R+λ)−G²/(H+λ)]−γ` (the library drops the ½ — argmax-invariant); `Cover` in
  `trees_to_dataframe` = ΣH (the quantity `min_child_weight` thresholds). Ship honestly.
- **Adult/Census (NB-5 candidate)**: `fetch_openml('adult', as_frame=True)` ≈ 48 842×14, ~23.9% positive
  (`>50K`), genuine NaN in `occupation`/`workclass`/`native-country`; **missingness is informative** —
  P(>50K | occupation missing) ≈ 0.094 vs ≈ 0.248 when present. XGBoost 3.x native categoricals via
  `enable_categorical=True` confirmed.

## The per-method arc (5 notebooks)

Three fundamentals (one concept each, by hand) → the estimator & its parameters → the demanding case.

### NB 1 — The second-order view: gradients *and* curvature (by hand)

- **One concept:** approximate any loss to **second order** around the current prediction and read off
  the optimal constant on a region. *(Regularization deliberately deferred to NB 2 — here λ=0; state
  this once so the reader doesn't expect the famous `−G/(H+λ)` yet.)*
- **Re-lay the move, don't presuppose it:** ch 08 NB 3 gave *one* Newton leaf for *one* loss as a
  recipe. NB 1 re-derives the general idea **by hand on a scalar first**:
  `L(F+w) ≈ L(F) + g·w + ½h·w²` is a parabola in `w`; its minimum is `w* = −g/h`; sum over the points
  in a leaf → **`w* = −G/H`** (`G=Σg`, `H=Σh`). Intuition: the gradient says which way, the Hessian
  says how far — flat regions (small h) take bigger steps, steep regions (large h) take cautious ones.
- **Pin the convention once:** `gᵢ = ∂L/∂Fᵢ`, `hᵢ = ∂²L/∂Fᵢ²`, leaf `w* = −G/H`. Then the minus sign is
  *what reproduces* ch 08's positive updates:
  - **Squared error** `L=½(y−F)²` → `g = F−y` (not `y−F`), `h=1` → `w* = −G/n = +mean residual`
    = ch 08 NB 1's leaf=mean (ch 08 carries the ½, so `h=1` is clean).
  - **Log-loss** → `g = p−y`, `h = p(1−p)` → `w* = −G/H = +Σ(y−p)/Σ p(1−p)` = ch 08 NB 3's Newton leaf,
    exactly. A one-line "why the signs line up" note reconciles the minus with ch 08's stated
    "negative gradient = y−F / y−p."
- **Equal billing for the two recoveries:** regression **then** classification, both as full reveals
  culminating in the "two-rules-one-rule" figure — the **classification** recovery (the direct ch 08
  NB 3 callback) is the climax, not an appendix.
- **NB 1 gets its OWN library parity anchor** (keep the by-hand↔library contract every fundamentals NB
  has held): `XGB*(reg_lambda=0, gamma=0, learning_rate=1, base_score=pinned, n_estimators=1,
  max_depth=1)` produces a leaf weight **exactly equal** to the by-hand `−G/H`. "Turn XGBoost's
  regularizer off, and its leaf is precisely our `−G/H`." Sets up NB 2 as "now turn λ on and watch the
  leaf shrink."
- ~3 figures (loss + its quadratic approximation at a point; gradient-only vs second-order step; the
  two-losses-one-rule figure). Regression + classification, both full. No `src/` change expected.

### NB 2 — The regularized objective: λ, γ, and the gain that decides splits (by hand)

- **One concept:** put complexity **into the objective**: `Obj = Σ loss + Ω(f)`, with
  `Ω(f) = γ·T + ½λ·Σⱼ wⱼ²` (T = number of leaves). Built by hand.
- **Derive the gain from the structure score — with the right sign** (a real trap): do *not* present
  `½[…]−γ` as a memorized rule. Start from the **structure score**
  `Obj(structure) = −½ Σⱼ Gⱼ²/(Hⱼ+λ) + γT` (Chen & Guestrin eq. 6) — the minimized objective for a
  fixed tree, using `w*ⱼ = −Gⱼ/(Hⱼ+λ)`. A split replaces one leaf's `−½G²/(H+λ)` with the two
  children's, so **gain = score(before) − score(after)** =
  `½[G_L²/(H_L+λ) + G_R²/(H_R+λ) − G²/(H+λ)] − γ` (eq. 7). State the sign meaning explicitly: gain > 0
  ⇔ the (negative) objective got *more* negative. (The naive "parent loss − children loss" reasoning
  flips the sign — call that out as the trap.)
- **Read the dials:** `w* = −G/(H+λ)` (L2 shrinks each leaf toward 0 — "turn λ on, watch it shrink");
  **γ** = minimum gain to bother splitting (pre-pruning). Contrast ch 04/08, where complexity was
  controlled only by `max_depth` / number of trees — XGBoost controls it **in the loss**.
- **The correctness anchor + the ½, framed kindly:** by-hand leaf weights match XGBoost exactly;
  by-hand gain matches the paper formula and **XGBoost reports 2× ours**. Pre-committed framing:
  *"A constant ½ multiplies every split's gain equally, so it never changes which split wins (the
  argmax). The textbook keeps the ½ for clean leaf-objective algebra; XGBoost drops it because it only
  ever compares gains. Same decisions, same tree — the ½ is bookkeeping, not a disagreement."* Tie it
  to the gain-bar figure (scaling all bars by 2, with γ doubled, leaves both the argmax and the prune
  decision unchanged).
- **`base_score`, honestly:** pin `base_score` only to make F₀ a *known constant* for the 1-tree
  hand-check — the same role as ch 08's `init_`. Left free, XGBoost 3.x **learns** it (a fitted
  intercept), which is itself a sensible default, not a hack. For log-loss it lives on the probability
  scale via the link → reuse ch 08 NB 3's `F₀ = log-odds`.
- ~4 figures (the Ω penalty; w* vs λ shrinkage; the worked split-gain bar with γ as the threshold line
  + the ×2 invariance; by-hand-vs-XGBoost parity table). Regression, by hand. No `src/` change.

### NB 3 — Sparsity-aware splits: a default direction for missing values (by hand)

- **One concept, kept pure** (the histogram method moves to NB 4): XGBoost handles **missing / sparse**
  entries with **no imputation** — at each split, all missing rows are routed to a learned **default
  direction**, chosen by trying both ways and keeping the higher gain (Chen & Guestrin 2016 §3.4).
  Build it by hand on a small set with `NaN`s: enumerate candidate thresholds on the non-missing
  values, for each compute the gain (NB 2's formula) sending the missing group **left vs right**, pick
  the max; read off the default direction.
- **A pinned, measured anchor** (every by-hand NB has one): at the NB-3 plan stage, pin a small
  NaN-bearing toy set, **measure XGBoost's chosen default direction and split**, and record it so the
  by-hand build checks against a concrete library number. *(Numbers measured at NB-3 plan time.)*
- **Parity & contrast:** by-hand default direction == XGBoost's learned direction (single split).
  Contrast (measured) sklearn `GradientBoosting*` (rejects NaN) vs `HistGradientBoosting*` (accepts,
  ch 08) vs XGBoost (accepts, 2016 first); LightGBM (ch 10) adopted the same idea.
- ~3 figures (a split with the missing group + the two default-direction trials; gain(left) vs
  gain(right); a small fitted tree showing the default arrows). No `src/` change expected.

### NB 4 — The estimator `XGBClassifier` / `XGBRegressor` & its parameters

- **The integrative notebook** — and it **owns the histogram method as a named concept**, not buried
  among parameters. Recap the by-hand parity (NB 2). Then:
  - **Histogram / approximate split finding (the one genuinely new mechanism NB 4 introduces):** an
    explicit *intuition → measure* beat — binning continuous features into ≤`max_bin` buckets turns the
    threshold scan from "every distinct value" into "≤256 bin edges," the engineering that makes
    boosting scale. Show `tree_method='hist'` (the 3.x default) and `max_bin`, and **measure** the
    speed (hist vs `exact`) and the negligible accuracy cost. *(What is NOT built: the weighted quantile
    sketch / approximate global-vs-local proposal, C&G §3.2–3.3 — named and motivated only; ch 10
    builds histogram growth directly.)*
  - **objective regularizers** (NB 2): `reg_lambda` (L2, default 1), `reg_alpha` (L1, default 0),
    `gamma` / min_split_loss (default 0);
  - **tree complexity:** `max_depth` (default **6** — deeper than sklearn GB's 3), `min_child_weight`
    (a floor on a leaf's `Cover`=ΣH — NB 1/2), `grow_policy` (`depthwise` vs `lossguide` — leaf-wise,
    the ch 10 bridge, named not used);
  - **stochasticity:** `subsample` (rows, Friedman 2002 — ch 08 NB 5) and **column subsampling**
    `colsample_bytree/bylevel/bynode` (new vs sklearn GB);
  - **`learning_rate`/`eta`** (default 0.3) × `n_estimators` — the ch 08 NB 4 trade-off, re-felt.
- **Honest tuning & failure modes:** the **aggressive defaults overfit** (eta 0.3 + depth 6) — show it,
  then tune with `GridSearchCV`/`RandomizedSearchCV` on train → **one sealed test**; relate every knob
  to its concept. Note `feature_importances_` is gain-type MDI (ch 06/08 caveat) — honest reading in
  NB 5.
- ~3–4 figures (hist-vs-exact timing/accuracy; a regularizer's train/test curve, e.g. λ or max_depth;
  CV grid / default-vs-tuned sealed bars). No `src/` change expected (reuse `viz` helpers; pytest 20).

### NB 5 — A demanding case (visualization-first capstone)

- **A full, honest tabular workflow** mobilizing the whole chapter. Per the capstone-visual-first
  standard: **~26 cells, ≥6 figures**, a "Read the figure" after each.
- **Dataset — recommendation + alternative (pinned at the NB-5 plan, with the verification below):**
  - **Primary: Adult / Census Income** (`fetch_openml('adult', as_frame=True)`, ~48 842×14, **binary**,
    mixed numeric+categorical, **genuine, informative missing values**, ~24% positive). It exercises
    NB 3's missing handling and native categoricals (`enable_categorical=True`, measured), and the
    imbalance brings ch 00's precision/recall/threshold/PR-AUC back honestly. Carries a real
    **ethics/limits** discussion (income predicted from demographics → proxy features, do-not-deploy) —
    a substantive section, not a throwaway.
  - **Verification owed at NB-5 plan time:** Adult's missingness is largely in *categorical* features
    (`?`), while NB 3 builds the default-direction on *numeric* splits. **Measure** how much Adult's
    PR-AUC actually depends on native NaN handling vs imputation (verify `?`→NaN), and whether native
    categoricals carry the NB-3 callback. If the missing-value lever proves negligible, lean the NB-3
    callback honestly on the *categorical* native handling, or fall back to **Ames Housing**
    (`fetch_openml`, ~1 460×80, regression, *structural numeric* missingness — the canonical
    Kaggle-XGBoost set) where NB 3 pays off on numeric splits directly.
  - (Avoid reuse: spambase = ch 07, covtype = ch 06, California housing = ch 08, breast_cancer = ch 05.)
- **Arc:** look at the data — **including a measured missingness-vs-target panel** (informative
  missingness) → baselines (logistic/linear + a shallow tree) → tuned XGBoost with **early stopping**
  (`eval_set`) → held-out metrics (acc/precision/recall/PR-AUC + threshold; or regression metrics for
  Ames) → **error analysis** → **honest cross-method comparison**: XGBoost vs ch 08
  `GradientBoosting`/`HistGradientBoosting` vs `RandomForest` vs a linear baseline.
- **The comparison's honesty axis:** plain GB/RF/linear **cannot** take NaN → they need an imputation
  step XGBoost/HistGBR skip, so part of any edge on this dataset is *missing handling*, not the booster.
  State the preprocessing each model gets, and **run XGBoost both ways (native NaN vs imputed)** to
  isolate the missing-handling contribution. Frame "native-NaN vs imputed" as a deliberate, named
  axis — and hold the line: expect the boosters **close** on accuracy; XGBoost's honest edge is
  missing-handling/speed/regularization, *not* a guaranteed win.
- Close with **gain MDI vs permutation** importance (the recurring honesty lesson) and a **LightGBM
  teaser** (ch 10 bridge, as ch 08 teased HistGBR).
- `src/` only if a diagnostic helper hits the 3× reuse bar (else notebook-local matplotlib); pytest
  likely stays 20. Anchors measured at the NB-5 plan/build.

## Honest scoping (the ml-expert bar)

- **XGBoost = gradient boosting + (2nd-order, regularized objective, sparsity-aware, histogram).** Not
  a new paradigm. The 2nd-order leaf was already built in ch 08 NB 3 — we generalize, we don't reveal
  magic.
- **No universal best.** The capstone shows XGBoost ≈ tuned GB ≈ HistGBR within noise on accuracy;
  its honest edges are missing-handling, speed, and regularization knobs — stated, not oversold; and
  the comparison names native-NaN-vs-imputed as a confound-by-design.
- **Defaults overfit** (eta 0.3, depth 6) — tuning + early stopping earn their place.
- **Gain importance is MDI-biased** → permutation on held-out is the trusted read (ch 06/08).
- **The ½-factor gain detail** and **`base_score` pinning** are surfaced, not hidden — measured parity.
- **Deferred by design:** the approximate / weighted-quantile-sketch split-finding (C&G §3.2–3.3) is
  *named and motivated*, not built by hand — it is engineering for scale, and ch 10 builds histogram
  growth directly.
- Every anchor re-measured on the live stack at NB-plan/build time; seeds pinned; nothing from memory.

## `src/` & guards

No `src/` change expected across the chapter (reuse `viz.use_course_style`,
`viz.plot_train_test_curve`, `viz.plot_feature_importances`; XGBoost trees dumped via
`trees_to_dataframe`; datasets fetched directly via `fetch_openml` — ch 06/08 precedent, no Adult/Ames
loader needed). Each notebook: built by hand before the library; colours only from `ml_course.colors`;
seeds fixed; "Read the figure" after every figure; banned-word scan over JSON cell text = 0;
`check_no_hardcoded_hex` clean; ruff/black clean; output-free in git; `gen_llms_txt` re-run;
two-reviewer gate (no BLOCK) + Rémy visual before each commit; `notebook → chapter` ff-merge;
**chapter → main via PR (`--no-ff`)** at close.

## References (chapter-level; per-NB DOIs at build)

- Chen & Guestrin 2016 — *XGBoost: A Scalable Tree Boosting System* (DOI 10.1145/2939672.2939785) —
  the regularized objective (eq. 6–7), the sparsity-aware split (§3.4), the approximate/histogram split
  (§3.2–3.3).
- Friedman 2001 — gradient boosting (DOI 10.1214/aos/1013203451); Friedman 2002 — stochastic GB
  (DOI 10.1016/S0167-9473(01)00065-2).
- ESL §10 (DOI 10.1007/978-0-387-84858-7). Forward: Ke et al. 2017 — LightGBM (ch 10).

## Reviewer gate outcome

Both reviewers **NO BLOCK**, each having independently re-run live checks (the second-order optimum,
both ch-08 recoveries, the eq.7 gain, the exact leaf-weight parity + the 2× gain + `Cover=ΣH`, the
resolved defaults, NaN behaviour across GB/HistGBR/XGBoost, native categoricals, the early-stopping
API, and Adult's shape/informative-missingness/balance). All MAJOR/MINOR items folded above; the
remaining per-notebook specifics (NB-3's pinned toy anchor; NB-5's Adult-vs-Ames missingness
verification) are explicitly assigned to the per-notebook plan stage.
