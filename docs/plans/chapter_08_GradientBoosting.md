# Chapter plan — 08_GradientBoosting

> Status: **APPROVED by Rémy (2026-06-24) & persisted.** Reviewer-gated (pedagogy **PASS**; ml-expert
> **REVISE → folded**, no BLOCK). **Regression end-to-end, PLUS an added classification notebook → SIX
> notebooks** (mirrors 03_LogisticRegression's six — a concept that earns its own notebook). Next: open
> & plan NB 1.

## Where this chapter sits

Chapter 06 built **bagging** — independent trees, in parallel, to cut variance. Chapter 07 built the
first **boosting** method, AdaBoost: weak learners trained **sequentially**, each refocused on the
previous ensemble's mistakes by **reweighting** the points, combined as a weighted additive vote,
minimizing **exponential loss**. Chapter 08 is the **general form** of that idea. Instead of
reweighting points, gradient boosting **fits each new tree to the residuals** — what the current model
still gets wrong — and the punchline is that those residuals are the **negative gradient of a loss**:
the ensemble is a point in *function space*, and each tree is one **gradient-descent step**. Swap the
loss and you swap the residual — squared error → ordinary residual (regression); log-loss → y − p
(classification); exponential loss → the objective **AdaBoost** minimizes. So AdaBoost is revealed as
the exponential-loss member (`loss='exponential'`) of a far more general machine (Friedman 2001; FHT
2000) — exactly the bridge promised at the close of ch 07.

Base learner: ch 04's **regression tree** (shallow — GB wants weak learners, the mirror of RF's deep
ones). This chapter is also the launchpad for ch 09 (XGBoost) and ch 10 (LightGBM), which add a
regularized objective, second-order steps, and histogram splits to this same skeleton.

**First regression in the course.** Every method so far has been classification. Gradient boosting's
cleanest, most honest by-hand intuition is **regression with squared error**, where the pseudo-residual
*is* the ordinary residual and the by-hand build matches sklearn **to machine precision** (1e-16). We
make the chapter regression-first to keep the course's by-hand↔library contract exact, *and* add a
dedicated classification notebook (NB 3) so the learner sees the general engine on a class label —
where an honest, named Newton leaf-step is required and the AdaBoost-as-exponential-loss-member reveal
lands.

## The per-method arc → six notebooks (one concept each on NB 1–4)

Six, like 03_LogisticRegression — the classification case "earns its own", as gradient descent did
there. NB 1–4 fundamentals (one concept each), NB 5 the estimator, NB 6 the demanding case.

1. **Boosting as fitting residuals (by hand, regression).** F₀ = mean → residual r = y − F → fit a
   regression tree to r → add a shrunken slice → repeat. The residual mechanism; exact by-hand ==
   `GradientBoostingRegressor` parity. *Does not yet name "gradient."*
2. **The residual was the gradient: gradient descent in function space (regression).** Reveal
   r = −∂(½(y−F)²)/∂F; the ensemble is a point in function space, each tree a downhill step, ν the
   step size. The central abstraction.
3. **Gradient boosting for classification: a different loss, a different residual** *(the added
   notebook)*. Log-odds + sigmoid (recap ch 03); log-loss as the differentiable surrogate;
   pseudo-residual = y − p; fit a regression tree to it even for a class; the **honest Newton
   leaf-step** (regression leaves = mean, exact; classification leaves need a one-step Newton
   correction sklearn applies for you); and the **unifying reveal**: `loss='exponential'` is the
   exponential-loss member of the family — the objective AdaBoost minimizes (shown on ch 07's
   make_moons).
4. **Shrinkage and the trees: ν, depth, n_estimators — and why GB overfits with too many trees (at
   large ν).** ν by hand → the ν × n_estimators trade-off (small ν needs more trees) → depth =
   interaction order (shallow trees; mirror of RF) → the staged test curve that **bottoms then rises at
   large ν** (the headline contrast with RF, which does not overfit with more trees).
5. **The estimator `GradientBoostingRegressor`/`Classifier` & its parameters.** Parity recap; `loss`
   (current names — `'log_loss'`, `'exponential'`, `'squared_error'`; `'deviance'`/`'ls'`/`'lad'`
   removed); `subsample` (stochastic GB + the OOB monitor); **early stopping**
   (`n_iter_no_change`/`validation_fraction` — the principled cure for NB 4's overfit);
   `max_depth`/`max_features`; `feature_importances_` (MDI + permutation caveats); honest
   `GridSearchCV` → one sealed test; **name `HistGradientBoosting*`** as the fast modern default + the
   ch 09–10 bridge.
6. **A demanding case (visualization-first capstone): tuning a competitive model honestly — California
   housing (regression).** Full honest workflow: baseline → GB tuned with early stopping → held-out
   R²/MAE in real dollars → error analysis (residual plots) → cross-method foil (RF-regressor, linear)
   → `HistGB` speed/score teaser → the bridge to ch 09–10.

## Decisions

- **D1 — Regression end-to-end + an added classification notebook (Rémy, 2026-06-24).** Six notebooks:
  regression core (NB 1–2), classification (NB 3), complexity control (NB 4), estimator (NB 5),
  regression capstone (NB 6). Rationale: GB's by-hand↔library parity is **exact only in regression**;
  classification needs log-odds + log-loss + a silent Newton leaf-step (naive mean-leaf gives a
  *different* model — verified). Regression-first preserves the course's by-hand contract and pays off
  ch 00's "regression" promise; the dedicated NB 3 gives classification an honest, un-cramped home (and
  resolves the NB-2-overload risk the concept tour flagged). Six-NB precedent: 03_LogisticRegression.
  *(Alternative not taken: a classification capstone on adult/census with HistGB's categorical/NaN
  story — Rémy chose regression end-to-end.)*
- **D2 — Capstone dataset: California housing** (`fetch_california_housing`, 20640×8, all-numeric
  regression; Pace & Barry 1997). A strong non-linear tabular regression where GB shines and HistGB
  shines more (GBR ≈0.78 → early-stop ≈0.82 → HistGBR ≈0.84; RF foil ≈0.79; seed band, re-measured at
  build), with honest evaluation in dollars (MAE ≈0.37 = $37k). Continues NB 1's regression intuition to
  the end. *(NB 6 foils against RF-regressor / linear, not the classification cross-method spine.)*
- **D3 — Teach on the transparent `GradientBoosting*`; name `HistGradientBoosting*` as the fast modern
  default (NB 5–6 only, not taught).** The classic estimator's `n_estimators` *are* the trees built by
  hand; `staged_predict` exposes the exact additive sequence; by-hand matches it to machine precision.
  For squared error the tree's leaf value (the residual mean) already equals Friedman's per-leaf
  line-search optimum, so **no leaf re-estimation is needed** — exactly why the regression parity is
  exact; swap the loss and the leaf value must be re-estimated (the Newton step of NB 3). Hist's
  histogram binning / leaf-wise growth / regularized objective are exactly what ch 09–10 teach — named
  here as the motivated handoff.
- **D4 — The Newton leaf-step stated honestly in NB 3** (regression leaves = mean, exact; classification
  leaves need a one-step Newton correction `Σr / Σp(1−p)` that sklearn applies for you; show the
  verified numeric gap; full derivation in "Going further"). This is the chapter's **correctness trap**,
  analogous to ch 07-NB 2's SAMME-α reconciliation the ml-expert reviewer caught. The robust shipped
  claim is the **direction** (Newton < mean, and Newton matches sklearn to machine precision), **not** a
  specific hard-coded log-loss pair (that gap is config-dependent — pin the config and re-measure at
  build).

## Anchors (cartographer-measured + ml-expert-verified on **scikit-learn 1.9.0**, `random_state`-pinned; re-measured at each NB build)

**API verified on the live install.** `GradientBoostingClassifier(*, loss='log_loss',
learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='deprecated', min_samples_leaf=1,
max_depth=3, max_features=None, validation_fraction=0.1, n_iter_no_change=None, tol=1e-4,
ccp_alpha=0.0, …)`; Regressor identical with `loss='squared_error'` (+ `alpha=0.9`). **`loss='deviance'`
is REMOVED** (→ `'log_loss'`); regressor `'ls'`/`'lad'` removed (→ `'squared_error'`/`'absolute_error'`).
`'exponential'` is a valid classifier loss (**AdaBoost's objective** — the exponential-loss member; see
NB 3). **API trap: GB has NO `staged_score`** — only `staged_predict`/`staged_decision_function`/
`staged_predict_proba` (compute `1 − accuracy_score` or R² over `staged_predict`; note `AdaBoostClassifier`
*does* have `staged_score`, so a learner moving from ch 07 will reach for it and hit an AttributeError).
`subsample<1` enables `oob_improvement_`. Default early stopping is **OFF** (`n_iter_no_change=None`).
`init_` = `DummyRegressor(mean)` / `DummyClassifier(prior)`.

- **NB 1/2 (regression by hand):** by-hand GB (F₀=mean, regression tree on residuals, shrunken step)
  == `GradientBoostingRegressor` **to 1.11e-16** (verified) — the chapter's exact-parity crown jewel.
  Pseudo-residual = ordinary residual (squared error); the tree's mean-leaf already is the squared-error
  line-search optimum, so no leaf re-estimation is needed (why the parity is exact).
- **NB 3 (classification):** F₀ = log-odds of the **training** base rate (= sklearn's `init_`); on the
  balanced moons-0.20 (p₀ = 0.5 exactly → F₀ = 0) the round-1 residuals are **±0.5000**. The **Newton
  leaf `Σr/Σp(1−p)` matches `GradientBoostingClassifier` to machine precision (~0)**; the naive mean-leaf
  gives a demonstrably *different* model (Newton < mean train log-loss at **every** config). The specific
  log-loss gap is **config-dependent** — pin (n_estimators, max_depth, ν, seed) at NB-3 build, do **not**
  hard-code a pair. `loss='exponential'` **shares AdaBoost's objective**: identical test accuracy
  (0.9417) and ~95% identical predictions on moons-0.20, **not** bit-identical (different optimizer).
- **NB 4 (shrinkage/overfit):** GB **overfits with too many trees at large ν** — ν=1.0 best R² ≈0.80 @
  ~130 trees → **0.748 @ 1000** (bottoms then rises); **at ν=0.1 it does NOT turn up within 1000 trees**
  → 0.837 @ 1000; ν=0.01 still climbing. The overfit is **ν-dependent**; the RF contrast is *mechanistic*
  (each RF tree is an independent variance-reduction draw and cannot raise complexity — ch 06 NB 4 —
  whereas each GB tree fits the current residual and adds capacity). Noisy-set illustration: test err
  0.100 @ 30 → 0.167 @ 800 while train→0.
- **NB 5 (estimator):** early stopping `n_iter_no_change=10` stops a 2000-tree request early
  (**≈365–390, seed-dependent**), equal R²; `subsample<1` → `oob_improvement_` monitor (Friedman 2002).
- **NB 6 (capstone, California):** GBR default **R²≈0.78 / MAE ≈0.37** ($37k) → early-stop **R²≈0.82** →
  HistGBR **R²≈0.84**; RF foil **≈0.79** (seed band; re-measured at build). No universal best.
- **Honest-limits anchors:** clean spambase GB 0.952 ≈ AdaBoost 0.949 ≈ RF 0.959 (no universal best);
  spam @40% label noise AdaBoost(stumps) **0.823 > GB(depth-3) 0.719**, GB(stumps) recovers 0.815 — "GB
  beats AdaBoost under noise" is **depth/loss-dependent, NOT a law** (cited, not the chapter's headline).

## Notebook-by-notebook detail

### NB 1 — Boosting as fitting residuals (by hand, regression)
- **One concept:** GB builds an additive model by repeatedly **fitting a regression tree to the
  residuals** of the current model and adding a shrunken slice — contrasted with AdaBoost's *reweighting*
  (same goal, focus on the mistakes; different mechanism).
- **By hand on a small 1-D synthetic regression** (a smooth target + noise — the clearest canvas to
  *see* residuals and the stepwise fit): F₀ = mean → r = y − F → fit `DecisionTreeRegressor` to r →
  F ← F + ν·tree → repeat; watch residuals shrink and the staircase fit emerge. Close with **exact**
  by-hand == `GradientBoostingRegressor` parity (1e-16).
- **Re-lay regression honestly** (ch 00 named it, the course never *did* it): target is continuous; we
  predict a number; the error is the residual; the loss is squared error. **And re-lay that a regression
  tree predicts the mean of the targets in each leaf** (the value minimizing squared error there — ch 04's
  trees were classification-framed); this mean-leaf already equals the per-leaf line-search optimum, so
  **no leaf re-estimation is needed for squared error** — exactly why the parity is exact (and the hinge
  to NB 3's Newton leaf, where a *different* loss does need re-estimation). Brief but real.
- **Figures:** (A) the residual-fitting story — panels (data + current fit; the residuals; the tree fit
  to the residuals; the updated fit) across a few rounds; (B) train MSE vs number of trees. + "Read the
  figure".
- **Honest scoping:** this is *regression* (first in the course); "gradient" is not named yet (NB 2's
  reveal); greedy stagewise ≠ global optimum; the exact parity is a squared-error gift (other losses, and
  classification, need leaf re-estimation — flagged forward to NB 3). Cite Friedman 2001 §4.1.

### NB 2 — The residual was the gradient: gradient descent in function space (regression)
- **One concept:** the residual we fit is the **negative gradient of the loss**; the ensemble F is a
  point in function space and each tree is a **gradient-descent step**, ν the learning rate. Re-lay ch 03
  NB 4's gradient descent (in *parameter* space) → here it is in *function* space.
- **By hand:** show numerically that y − F = −∂(½(y−F)²)/∂F per point; recompute NB 1's update as "fit
  the negative gradient"; verify the identical sequence. Then the generalisation in words: *a different
  loss → a different gradient → a different residual* (sets up NB 3).
- **Figures:** (C) the step picture — F as a point, the negative-gradient step (schematic + the per-point
  gradient = residual scatter); (D) train loss vs rounds with the gradient-step annotation. + "Read the
  figure".
- **Honest scoping:** function-space gradient descent is the unifying idea but the steps are *trees*
  (not free functions) → a greedy approximation; the full functional-gradient derivation for arbitrary
  losses → "Going further" (Friedman 2001).

### NB 3 — Gradient boosting for classification: a different loss, a different residual *(added notebook)*
- **One concept:** swap the loss and GB does classification — **log-loss** gives pseudo-residual
  **y − p**, fit a regression tree to it (in log-odds space, squashed by the sigmoid). The general engine
  on a class label.
- **By hand on make_moons-0.20** (the ch 07 through-line — continuity + a head-to-head with AdaBoost):
  **F₀ = log(p₀/(1−p₀))** with p₀ the *training* base rate (= sklearn's `init_`, `DummyClassifier(prior)`;
  balanced moons → F₀ = 0) → p = σ(F) → r = y − p → regression tree on r → **Newton leaf value
  `Σr/Σp(1−p)`** (the denominator Σp(1−p) is the loss *curvature* — what makes it a Newton step, not a
  mean) → shrunken step → repeat. Build the **mechanism by hand first and make it work**; then show the
  honest leaf step: **regression leaves = mean (exact, NB 1); classification leaves need the Newton
  correction sklearn applies for you** (mean-leaf gives a *different* model — the verified gap, re-measured
  at build). Parity-in-direction with `GradientBoostingClassifier` (Newton matches to ~0).
- **The unifying reveal (objective-level, stated honestly):** `GradientBoostingClassifier(loss='exponential')`
  optimizes the **same exponential loss** AdaBoost minimizes → AdaBoost is the exponential-loss member of
  the gradient-boosting family. The two are **not bit-identical** (different optimizers: GB's
  functional-gradient step + Newton leaves + shrinkage vs AdaBoost's reweighting + α = ln((1−ε)/ε)) — on
  ch 07's make_moons they reach **identical test accuracy and ~95% identical predictions**. Figure (F)
  shows the *agreement* (overlaid boundaries / accuracy match), never asserts a parity number. (FHT 2000;
  Friedman 2001.)
- **Figures:** (E) classification boundary sharpening with rounds on moons; (F) the agreement of
  `loss='exponential'` with AdaBoost (overlaid boundaries / matching accuracy) and/or the
  mean-leaf-vs-Newton-leaf gap. + "Read the figure".
- **Honest scoping:** the Newton leaf-step is the chapter's correctness trap — built by hand, stated
  plainly, full derivation in "Going further" (a learner who finds the algebra hard keeps the method);
  the exponential-loss=AdaBoost reveal is **objective-level, not predictor-level** (share the loss, differ
  in the optimizer); multiclass = one-tree-per-class (named, not built); log-loss is gentler than
  exponential on outliers (a forward hook to noise behaviour, **not** shipped as "GB beats AdaBoost" —
  depth/loss-dependent, see honest-limits). Phrase sklearn's Newton leaf as "the library does this for
  you", never as the library being sneaky.

### NB 4 — Shrinkage and the trees: ν, depth, n_estimators; why GB overfits with too many trees (at large ν)
- **One concept (declared):** how GB controls its complexity — the **ν × n_estimators** trade-off,
  **depth** as interaction order, and the headline that **GB overfits with too many trees at large ν**
  (unlike RF). *(Richer-scope single concept, the ch 07-NB 3 precedent: keep depth subordinate — "the
  other complexity dial, held shallow" — with the bottoms-then-rises curve as the unmistakable headline.)*
- **Built/measured:** establish ν by hand (a shrunken step; what ν=0.1 does to one tree's contribution)
  → ν × n_estimators (small ν needs more trees; large ν + many trees overfits) → depth sweep (stumps =
  additive, deeper = interactions; GB keeps trees shallow — mirror of ch 06's deep RF, ch 07's weak base)
  → **the staged train/test curve that bottoms then rises at large ν** (the RF contrast).
- **Figures:** (G) ν × n_estimators (test R²/error vs trees for ν ∈ {1.0, 0.1, 0.01}); (H) staged
  train/test showing the overfit at large ν (bottoms then climbs); (I) depth sweep. + "Read the figure".
- **Honest scoping:** shrinkage is a regulariser (smaller ν, more trees, better generalisation — not
  "free", it costs trees); the overfit is real, **ν-dependent** (sharp at large ν, gentle/unseen within
  budget at small ν), and is *the* reason early stopping (NB 5) exists. State it **mechanistically**:
  each RF tree is an independent variance-reduction draw that **cannot** raise complexity, whereas each GB
  tree fits the current residual and adds capacity — so "more trees always help" is false for GB at large
  ν, true for RF. Never overgeneralise to "GB always overfits with more trees".

### NB 5 — The estimator `GradientBoostingRegressor`/`Classifier` & its parameters
- **Integrative — anchor on the early-stopping story** (the principled cure for NB 4's overfit, the
  chapter's strongest through-line). Parity recap (regression exact; classification with the Newton leaf);
  then the dials — **`loss`** (current names; `'deviance'`/`'ls'`/`'lad'` removed; `'exponential'` =
  AdaBoost's objective, NB 3), **`learning_rate` × `n_estimators`**, **`subsample`** (stochastic GB + the
  free `oob_improvement_` monitor, Friedman 2002), **early stopping** (`n_iter_no_change`/
  `validation_fraction`; stops 2000→≈365–390), `feature_importances_` (MDI; permutation cross-check;
  ch 04/06 caveats in one line). Treat **`max_depth`/`max_features`** as one-line cross-references back to
  NB 4 (not fresh sweeps). Honest **`GridSearchCV`** on train → one sealed test. State the **API trap (no
  `staged_score`)**. **Name `HistGradientBoosting*`** as the fast modern default (`early_stopping='auto'`,
  `max_leaf_nodes`, `max_bins=255` histogram, native categorical/missing) + the ch 09–10 bridge.
- **Figures (pick ~3–4):** (J) subsample / OOB-improvement curve; (K) early-stopping curve (requested vs
  stopped); (L) a `learning_rate × n_estimators` or tuning heatmap; (M) importances MDI vs permutation.
  + "Read the figure".
- **Honest scoping:** subsample is a regulariser/speedup, not a magic accuracy knob; early stopping
  spends a `validation_fraction` and depends on `tol`/`n_iter_no_change` (off by default); importances not
  causal; no `'deviance'`.

### NB 6 — A demanding case (visualization-first capstone): California housing (regression)
- **Tuning a competitive model honestly.** Full workflow on `fetch_california_housing(as_frame=True)`
  (20640×8, named columns): look at the data → a baseline (linear / a shallow tree) → a tuned GB regressor
  **with early stopping** → honest held-out **R² and MAE in dollars** → **error analysis** (residual
  plots; where does it err — high-value / coastal homes?) → cross-method foil (RF-regressor, linear) → a
  **`HistGradientBoostingRegressor` speed/score teaser** (R² ≈0.84 vs GB ≈0.78, faster) → the bridge to
  **ch 09 XGBoost / ch 10 LightGBM**.
- **Visualization-first:** ~24–26 cells (a *floor*), ≥6 figures (target distribution; predicted-vs-actual;
  residual plot; staged learning curve with the early-stop point marked; ν × trees or tuning heatmap;
  cross-method bar; importances MDI vs permutation). Per the capstone-visual-first standard.
- **Your turn (tiered):** *easy* — from the learning curve, name the round early stopping fires and why
  running further would not help; *medium* — change ν and re-find the early-stop point / best R², relate
  to NB 4; *harder* — swap in `HistGradientBoostingRegressor`, compare speed and R², and explain (from
  NB 5 / the ch 10 bridge) what the histogram approach buys.
- **Honest scoping:** no universal best (GB ≈ / ≷ RF / Hist depending on the table); R² and MAE tell
  different stories (report both, in units); early stopping is the lever; importances reflect this
  dataset, not cause; the model is strong but its *speed* and *regularisation* are exactly where ch 09–10
  improve.

## `src/` plan

Mostly reuse (`viz.use_course_style`; `plot_train_test_curve` — verified metric-agnostic on y, so it
works for regression staged curves with x = n_estimators and y = MSE or 1−R², **passing an explicit
`ylabel`** ("test MSE" / "1 − R²") so the axis never silently reads "error"; `plot_decision_boundary`
for NB 3's moons; `plot_feature_importances`; `ml_course.colors`). The residual-fitting visuals (NB 1),
the function-space step schematic (NB 2), and the ν×trees / overfit curves (NB 4) are notebook-local
matplotlib. **One candidate new helper:** `viz.plot_regression_diagnostics` (predicted-vs-actual +
residual plot) for NB 6 — the existing `viz` is classification-oriented (inventory confirmed: no
predicted-vs-actual / residual helper). Add it **only if a 3× reuse emerges** (NB 1 residual view? NB 6
capstone?) at NB-plan time, per the project's helper discipline; if added, **+1 smoke test → pytest
20→21**. Otherwise pytest stays 20.

## References (with DOIs)
- Friedman 2001 — *Greedy function approximation: a gradient boosting machine*, Ann. Statist.
  29(5):1189–1232. DOI 10.1214/aos/1013203451. **(the GBM — core reference; §4.1 regression leaf, §4.5–4.6
  TreeBoost / classification Newton leaf)**
- Friedman 2002 — *Stochastic gradient boosting*, CSDA 38(4):367–378. DOI
  10.1016/S0167-9473(01)00065-2. (`subsample`; OOB improvement)
- Friedman, Hastie & Tibshirani 2000 — *Additive logistic regression: a statistical view of boosting*,
  Ann. Statist. 28(2):337–407. DOI 10.1214/aos/1016218223. (AdaBoost = forward-stagewise + exp loss; the
  bridge)
- Hastie, Tibshirani & Friedman, *ESL* §10.9–10.13 (boosting trees, gradient boosting, shrinkage,
  interaction order). DOI 10.1007/978-0-387-84858-7.
- James et al., *ISLR* §8.2.3 (boosting). DOI 10.1007/978-1-0716-1418-1.
- Pace & Barry 1997 — California housing, Stat. & Prob. Letters 33(3):291–297. DOI
  10.1016/S0167-7152(96)00140-X. (NB 6 dataset)
- *Forward (name only, ch 09–10):* Chen & Guestrin 2016 — XGBoost, DOI 10.1145/2939672.2939785; Ke et
  al. 2017 — LightGBM (NeurIPS).

## Verification / guards (every NB)

Build via `uv run python <scratchpad>/build_*.py`; **re-measure every anchor on sklearn 1.9.0 at build**
(esp. the regression 1e-16 parity and the classification Newton-leaf *direction* — Newton matches sklearn
to ~0, mean-leaf differs; do **not** hard-code a log-loss pair, pin the config); run top-to-bottom via
nbconvert **from project cwd** on a scratchpad copy (tracked file **output-free**); **banned-word scan
over JSON real cell text** = 0; `check_no_hardcoded_hex` passes; `ruff`/`black` clean; `gen_llms_txt`
re-run; `pytest` 20 (→21 only if the regression-diagnostics helper is added). **No `staged_score`** —
use `staged_predict`. Learner-facing cells phrase sklearn's Newton leaf as "the library does this for
you", never as the library being sneaky. Both reviewers PASS (no BLOCK) + Rémy validates each NB
visually. One NB at a time; commit per NB; ff-merge notebook → chapter; close the chapter via PR into
`main` (`--no-ff`).
