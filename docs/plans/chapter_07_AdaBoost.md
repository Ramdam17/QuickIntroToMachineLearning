# Chapter plan — 07_AdaBoost

> Status: **APPROVED by Rémy (2026-06-23) & persisted.** Reviewer-gated (ml-expert + pedagogy, both
> REVISE→folded, no BLOCK). The first **boosting** method. Decisions resolved: **NB 3 = richer scope**
> (one declared concept), **NB 5 = spambase**. Next: open NB 1.

## Where this chapter sits

Chapter 06 built the first **ensemble**: a random forest averages many *independent*, high-variance
trees in **parallel** to cut variance. AdaBoost is the first **boosting** method and the opposite
move: weak learners are trained **sequentially**, each one focused on the mistakes the running
ensemble still makes, then combined as a **weighted additive vote**. Bagging asks many independent
opinions and averages; boosting builds one opinion, sees where it errs, and corrects — round after
round. This is the pivot into the boosting family (07 AdaBoost → 08 Gradient Boosting → 09 XGBoost →
10 LightGBM); ch 08 will reveal AdaBoost as the special case of a far more general idea (gradient
descent in function space on *any* differentiable loss). Base learner: ch 04's **decision stump**
(a depth-1 tree) — the canonical weak learner.

## The per-method arc → five notebooks (one concept each on 1–3)

1. **Boosting intuition: focus on the mistakes (reweighting, by hand).** The sequential loop; sample
   reweighting; the weak-learner weight α. Built entirely by hand.
2. **Weak learners and the additive model.** The weighted additive vote `F(x)=sign(Σ αₜ hₜ(x))`; the
   reveal that the *same* α is the vote weight; the statistical view — AdaBoost minimizes
   **exponential loss** by forward stagewise additive modelling (where α comes from), taught from scratch.
3. **Learning rate vs number of rounds; overfitting behaviour.** Shrinkage; the famous
   resistance-to-overfitting on clean data (margins) **and** its honest limit — overfitting under
   label noise. *(Richer scope — see Decisions.)*
4. **The estimator `AdaBoostClassifier` & its parameters.** Parity with the by-hand model;
   `estimator` / `n_estimators` / `learning_rate`; the base-learner-strength knob; the removed
   `algorithm` param (SAMME only now); multiclass SAMME; honest `GridSearchCV`.
5. **A demanding case (visualization-first capstone): where AdaBoost shines, and where noise hurts
   it.** A strong real-data baseline on **spambase** + the honest, measured noise story.

## Decisions (resolved by Rémy)

- **Decision A — NB 3 scope: RICHER NB 3.** One *declared* concept — "how boosting controls its own
  complexity: the rounds × learning-rate trade-off, and what it does to generalization." Teach
  `learning_rate` by hand first, then show **both** faces on the 2D set: clean-data resistance (more
  rounds keep helping after train→0) **and** the noise overfit (more rounds hurt). Map-aligned
  (course_map §07 NB 3 = "overfitting behaviour"); prevents the "AdaBoost never overfits"
  misconception; NB 5 does the real-data noise story *at scale* without re-teaching. *(Alternative not
  taken: a lean NB 3 = control knobs + clean resistance only, deferring all noise to NB 5.)*
- **Decision B — NB 5 dataset: SPAMBASE.** 4601×57, UCI, **ESL ch 10's canonical boosting dataset** —
  fresh to the course. AdaBoost shines (test err ≈ 0.051) and the staged curve shows clean-data
  resistance on *real* data. Leaves the breast_cancer cross-method spine (KNN 0.942 · LogReg 0.953 ·
  tree 0.906 · SVM 0.965 · RF ≈0.95); NB 5 will *reference* the spine numbers for continuity, as ch 06
  did with covtype. *(Alternatives not taken: breast_cancer — keeps the spine but 4th reuse + an
  anticlimactic headline; make_moons+flip — max visual clarity but synthetic.)*

## Anchors (re-measured at plan time on **sklearn 1.9.0**; every estimator `random_state`-pinned; both reviewers reproduced)

**API verified on the live install** — `AdaBoostClassifier(estimator=None, *, n_estimators=50,
learning_rate=1.0, random_state=None)`. `base_estimator` is **gone** (now `estimator`); the
**`algorithm` parameter is REMOVED** — only **SAMME** remains (SAMME.R deleted). Default base =
`DecisionTreeClassifier(max_depth=1)` (a stump).

**Chapter 2D through-line:** `make_moons(n_samples=400, noise=0.20, random_state=0)`, 70/30 stratified
(seed 0) → n_train=280, n_test=120. (Parity is exact arithmetic; re-measured at each NB build.)

- **NB 1 — by-hand SAMME == sklearn, exactly.** Single stump on moons-0.20 is weak (train error ≈
  0.157, test acc ≈ 0.87). The by-hand α = ln((1−ε)/ε) reproduces sklearn `estimator_weights_` to 4+ dp
  and by-hand staged predictions == sklearn test acc — **the NB 1 build runs this parity end-to-end on
  the through-line set (moons-0.20)**; the exact triple [1.4547, 0.7254, 1.3773] == `estimator_weights_[:3]`
  was a second-witness probe on moons-0.30 (ε₁=0.189). Round-2 ε can exceed round-1 ε — reweighting
  makes the next problem deliberately harder. Running-ensemble **train error falls monotonically**.
- **NB 2 — train error → 0; α from exponential loss.** On moons-0.20, AdaBoost(stumps) drives **train
  error to 0 by ≈ round 114**; the boundary sharpens with rounds. α_t is the per-round minimizer of
  Σ wᵢ·exp(−α yᵢ hᵢ) (verified on a grid). SAMME multiclass adds **ln(K−1)** (binary: 0; K=3 by-hand
  term verified == sklearn).
- **NB 3 — resistance on clean data, overfitting on noise (both measured).**
  - *Clean:* moons-0.20 — train→0 @≈114; **test error stays in ≈0.04–0.06 with no upward drift** out to
    400 rounds. spam — test error 0.218 → **bottoms ≈0.049 @ round 280, then plateaus to ≈0.051 @ 400
    with no upward drift** (the ESL ch 10 resistance curve; note spam *train* error never reaches 0,
    floor ≈0.045 — so "train→0" is a moons-only phenomenon).
  - *Noisy:* breast_cancer + 20% train-label flip (test clean) — test error bottoms **0.082 @ round 17,
    then climbs to 0.170 @ 400**: more rounds make it worse.
  - *Learning rate:* moons lr∈{1.0, 0.5, 0.1} — smaller lr learns slower (lr=0.1 reaches its ≈0.90
    plateau only by ~25 rounds) and is a touch steadier; lr trades rounds for smoothness.
- **NB 4 — parity + the base-strength knob.** sklearn AdaBoost == by-hand. Base depth sweep on moons:
  depth-1 train 0.921 / test 0.892; depth-2 train 1.000 / test 0.892; depth-3 train 1.000 / **test
  0.875** — a *stronger* base learner overfits faster (boosting needs the base to stay weak). `GridSearchCV`
  over {n_estimators, learning_rate} on train → one sealed test (re-measured at NB-plan).
- **NB 5 — the honest noise signature (dataset-independent), not an RF horse-race.** The robust
  statement is *internal*: under label noise AdaBoost piles weight on the mislabeled points (by-hand:
  20% of points hold **~45% of all weight**, 0.198 → 0.449 over rounds) and its **test error rises with
  rounds** — exponential loss's non-robustness (Friedman et al. 2000). The naive "RF beats AdaBoost
  under noise" is **dataset-dependent and not robustly true** (holds on breast_cancer — 25% flip:
  AdaBoost 0.819 vs RF 0.912 — but reverses on spam — 40% flip: AdaBoost 0.839 vs RF 0.711), so the
  chapter will **not** ship it as a law. On clean spam: AdaBoost(stumps) ≈ 0.949, RF ≈ 0.956, stump 0.782.

## Notebook-by-notebook detail

### NB 1 — Boosting intuition: focus on the mistakes (reweighting, by hand)
- **One concept:** boosting trains weak learners *sequentially*, each re-focused on the current
  ensemble's errors via **sample reweighting** — explicitly contrasted with ch 06 bagging (parallel,
  independent, equal vote).
- **By hand on moons-0.20:** uniform weights wᵢ=1/n → fit a stump on weighted data → weighted error ε
  → weak-learner weight **α = ln((1−ε)/ε)** → reweight misclassified points up: wᵢ ← wᵢ·exp(α·𝟙[miss]),
  renormalise → repeat. Watch the weight migrate onto the hard points; watch the running ensemble's
  training error fall. Close with the by-hand↔sklearn parity (estimator_weights_ + staged preds).
- **Figures:** (A) the reweighting story — 2–3 panels, point size ∝ weight, each round's stump split
  drawn (you *see* the focus shift); (B) running-ensemble train error vs rounds. + "Read the figure".
- **Honest scoping:** α here is the **SAMME** form `ln((1−ε)/ε)` (what sklearn uses); the classic
  Freund–Schapire form carries a ½ and gives the *same* classifier — one honest sentence here ("we'll
  see exactly why in NB 2"), the scale-invariance argument deferred to NB 2 (the additive vote isn't
  defined yet). Boosting attacks **both bias and variance** by chasing the hard cases — *which is also
  exactly why it is sensitive to mislabeled hard cases* (flagged here, demonstrated NB 3/5): claim and
  caveat in one breath.

### NB 2 — Weak learners and the additive model
- **One concept:** the final model is a **weighted additive vote** `F(x)=sign(Σ αₜ hₜ(x))`, and the
  *same* α that drove reweighting is the vote weight (the reveal). "Weak learner" = better than chance
  (ε<0.5); why a sum of stumps is expressive.
- **The statistical view — taught from scratch (intuition → implementation → interpretation):**
  1. **Re-lay ch 03's loss idea (2 sentences):** a loss is a number we drive down; log-loss punishes
     a confident-wrong prediction *without bound* (unlike squared error). 2. **Exponential loss as a
     picture first:** plot `exp(−margin)` against the 0–1 step — a smooth stand-in that punishes
     negative margins ever harder. 3. **Forward stagewise in words:** "freeze all earlier rounds; ask
     which single (h, α) most reduces the loss *this* round" (Friedman, Hastie & Tibshirani 2000). 4.
     **Then the short derivative** → α_t = ½·ln((1−ε)/ε); **grid-verify** the minimiser numerically
     ("see it's really the bottom"). The heaviest algebra goes in a clearly-marked **"Going further
     (optional)"** — the mechanism (reweight + α from NB 1) already works; NB 2 explains *where α comes
     from*, so a learner who finds the algebra hard keeps the method.
- **Reconcile the conventions (honest):** SAMME uses `ln((1−ε)/ε)` (+ `ln(K−1)` multiclass), the classic
  exp-loss derivation gives `½·ln(…)` — a constant factor 2 on **every** α; since α enters **both** the
  vote and the reweighting **with the same sign**, the two define the same classifier up to that global
  scale — bookkeeping, not a different algorithm.
- **Figures:** (C) decision boundary at T=1 / 10 / 50 sharpening (`plot_decision_boundary`); (D)
  exponential loss vs α for one round (the minimiser) *or* train error → 0 vs rounds. + "Read the figure".
- **Honest scoping:** exponential loss is a smooth **surrogate** for 0–1 loss; its exponential penalty
  on large negative margins is exactly *why* mislabeled points come to dominate (the seed of NB 3/5's
  noise story). Greedy stagewise fitting ≠ global optimum.

### NB 3 — Learning rate vs number of rounds; overfitting behaviour  *(RICHER scope — Decision A)*
- **One concept (declared):** how boosting controls its own complexity — the **rounds × learning-rate**
  trade-off and what it does to generalization (resistant on clean data, not immune under noise).
- **Built/measured:** **establish `learning_rate` ν by hand first** (`F = Σ ν·αₜ hₜ`; what ν=0.1 does
  to one round's contribution) *before* any sweep. Then the staged train+test curves: (i) **clean**
  (moons-0.20 & spam) — test keeps dropping / plateaus after train→0 (margin maximisation continues,
  Schapire et al. 1998); (ii) **noisy** (label-flip) — test error **rises** with rounds; (iii) lr sweep
  {1.0, 0.5, 0.1}.
- **Figures:** (E) staged train/test error, clean — resistance (`plot_train_test_curve`); (F) test
  error vs rounds for 3 learning rates — *notebook-local plot* (3 test curves, not a train/test pair);
  (G) noisy data — test error rising with rounds. + "Read the figure".
- **Honest scoping:** "resistance to overfitting" is **not immunity** — it is a low-noise,
  margin-driven phenomenon; on mislabeled data more rounds *hurt*. ν is a regulariser (smaller ν, more
  rounds, smoother fit). Cite Schapire 1998 (margins), Dietterich 2000 (noise).

### NB 4 — The estimator `AdaBoostClassifier` & its parameters
- **Integrative notebook:** parity (sklearn `estimator_weights_` & staged predictions == by-hand);
  then the dials — **`estimator`** (the base learner; default stump; *strength* knob: deeper base →
  less weak → overfits faster, measured depth 1/2/3), **`n_estimators`** & **`learning_rate`** (their
  interplay), `random_state`. State the current-API fact: **`algorithm` is removed (SAMME only)**.
  Multiclass via SAMME. `feature_importances_` available (MDI; restate ch 06's bias/dilution caveat in
  one line, not a bare back-ref). Honest **`GridSearchCV`** on train → one sealed test.
- **Figures:** (H) boundary stump-base vs depth-3-base (overfit); (I) the `n_estimators × learning_rate`
  CV heatmap (low lr needs more rounds); (J) staged test, tuned vs default. + "Read the figure".
- **Honest scoping:** the base learner must stay **weak** — boosting deep trees defeats the purpose;
  no `SAMME.R`/`algorithm` choice anymore; importances are MDI (not causal), same caveats.

### NB 5 — A demanding case (visualization-first capstone): spambase  *(Decision B)*
- **Shines:** AdaBoost(stumps) test error ≈ **0.051** (acc ≈ 0.949), competitive with RF ≈ 0.956, far
  above a single stump 0.782; the staged test curve shows clean-data resistance (bottoms ≈round 280,
  then plateaus — no overfit). Honest cross-method: AdaBoost ≈ RF — **not** universally best; *figure
  captions and "Read the figure" stay measured ("competitive with", never triumphal)*. References the
  breast_cancer spine numbers for continuity.
- **Where noise hurts (honest, internal framing):** inject train-label noise → show (a) test error
  **rises with rounds**, (b) the mislabeled points hoard a disproportionate share of the weight, (c)
  degradation vs noise level — framed as **exponential-loss non-robustness**, with the RF comparison
  shown but explicitly *not* generalised into a law (dataset-dependent, per the anchors).
- **Visualization-first:** ~24–26 cells (a *floor*), ~7 figures (class balance; cross-method accuracy;
  staged resistance curve; confusion matrix; top features (MDI); noise-degradation curve AdaBoost vs
  RF; weight-on-noise concentration *or* a 2D moons inset showing the boundary contorting). All reusing
  existing `viz` helpers.
- **Your turn (tiered, answerable from NB 1–4 + NB 5's figures):** *easy* — from the staged-test curve,
  name the round where test error bottoms on the noisy run; *medium* — raise the injected-noise level,
  re-measure where AdaBoost's test error turns upward, and relate it to the weight-concentration figure;
  *harder* — swap the stump base for a depth-3 base and explain (from NB 3/4) why the stronger base
  overfits the noise faster.
- **Boosting bridge:** AdaBoost = exponential-loss special case; **ch 08 Gradient Boosting** generalises
  to any differentiable loss (and more noise-robust losses) via gradient descent in function space.

## `src/` plan

**No new helper forced.** Reuse `viz.use_course_style`, `plot_decision_boundary` (2D moons, NB 1–4),
`plot_train_test_curve` (train/test staged-error *pairs*, NB 2–3 figs E/G), `plot_confusion_matrix`,
`plot_class_balance`, `plot_feature_importances` (NB 5), `ml_course.colors`. The NB 1 reweighting visual
(point size ∝ weight) and NB 3 figure F (3 learning-rate test curves) are **notebook-local matplotlib**,
not reusable helpers. **pytest stays 20.** (A small staged/multi-curve helper added only if a clear 3×
reuse emerges at NB-plan time.)

## References (with DOIs)

- Freund & Schapire 1997 — *A decision-theoretic generalization of on-line learning…*, JCSS
  55(1):119–139. DOI 10.1006/jcss.1997.1504. (AdaBoost)
- Schapire, Freund, Bartlett & Lee 1998 — *Boosting the margin*, Ann. Statist. 26(5):1651–1686.
  DOI 10.1214/aos/1024691352. (why it resists overfitting)
- Friedman, Hastie & Tibshirani 2000 — *Additive logistic regression: a statistical view of boosting*,
  Ann. Statist. 28(2):337–407. DOI 10.1214/aos/1016218223. (exponential loss / forward stagewise)
- Schapire & Singer 1999 — *Improved boosting algorithms using confidence-rated predictions*, Mach.
  Learn. 37(3):297–336. DOI 10.1023/A:1007614523901.
- Zhu, Zou, Rosset & Hastie 2009 — *Multi-class AdaBoost* (SAMME), Stat. Interface 2(3):349–360.
  DOI 10.4310/SII.2009.v2.n3.a8.
- Dietterich 2000 — *An experimental comparison … bagging, boosting, and randomization*, Mach. Learn.
  40(2):139–157. DOI 10.1023/A:1007607513941. (noise sensitivity)
- Hastie, Tibshirani & Friedman, *ESL* §10 (boosting; spam example). DOI 10.1007/978-0-387-84858-7.
- James et al., *ISLR* §8.2 (boosting). DOI 10.1007/978-1-0716-1418-1.
- spambase: UCI Machine Learning Repository (Hopkins, Reeber, Forman & Suermondt, HP Labs).
  DOI 10.24432/C53G6X.

## Verification / guards (every NB)

Build via `uv run python - < <scratchpad>/build_*.py` (stdin); re-measure every anchor on sklearn 1.9.0
at build; run top-to-bottom via nbconvert **from project cwd** on a scratchpad copy (tracked file
**output-free**); **banned-word scan over JSON real cell text** = 0; `check_no_hardcoded_hex` passes;
`ruff`/`black` clean; `gen_llms_txt` re-run; `pytest` 20. Both reviewers PASS (no BLOCK) + Rémy
validates each notebook visually before commit. One notebook at a time; commit per notebook; ff-merge
notebook → chapter; close the chapter via PR into `main` (`--no-ff`).
