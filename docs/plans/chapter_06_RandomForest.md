# Chapter plan — 06_RandomForest (Random Forests)

> Status: **APPROVED** (2026-06-23, by Rémy; reviewer-gated — both `@ml-expert-reviewer` +
> `@pedagogy-reviewer` **REVISE → all MAJOR/MINOR folded**, see the reviewer notes at the end).
> **NB 5 dataset = covtype** (Rémy's choice). Drives the notebook loop in `docs/WORKFLOW.md`. Numbers
> measured at plan time on sklearn **1.9.0**; **re-measured at each notebook build** and reconciled
> into prose, with every random forest's `random_state` pinned.
>
> **Five notebooks** (the standard per-method arc). The course's **sixth method** and its **first
> ensemble**: many decision trees (ch 04) averaged into one low-variance model. The direct answer to
> the weakness ch 04 ended on — a single tree is high-variance — and the gateway to the boosting
> family (ch 07–10).

## Context

Chapter 04 built a decision tree and ended by *feeling its defining weakness*: a single tree is
**high-variance**. It showed this twice. On `make_moons`, an unlimited tree scored **test 0.878** but
with bootstrap **std 0.032** — wobbly. And on the breast_cancer capstone, the closing exercise had the
learner **hand-bag 25 bootstrap trees and majority-vote** them, lifting test accuracy to **0.930** and
steadying it — "a random forest in miniature," in ch 04's own words. **Chapter 06 picks up that
idea** and explains *why* it works, then makes it the real method.

A small but deliberate dataset choice frames the chapter: **NB 1 re-runs the bagging story on
`make_moons`, not on breast_cancer.** moons is 2-D, so we can *draw* a dozen single-tree boundaries
(each jagged and different) and watch the averaged forest boundary come out smooth — the variance and
its cancellation made *visible*. NB 2 then moves to the 30-feature breast_cancer set, because the
second idea (feature subsampling) only bites when there are many features to subsample. So the ch 04 →
NB 1 bridge is a genuine continuation of the *idea* (bag trees, vote, variance falls), on a dataset
chosen so the picture is visible — stated plainly, not a silent switch.

A random forest is two ideas stacked on the decision tree:

1. **Bagging (Bootstrap AGGregating, Breiman 1996).** Train each tree on its own bootstrap resample
   of the data and **average** their votes. Averaging many noisy-but-unbiased predictors **cancels
   their variance** while leaving the bias untouched — the wisdom-of-crowds effect, made precise.
2. **Random feature subsampling (the "random" in random forest, Ho 1998; Breiman 2001).** At *each
   split*, let the tree choose only among a random subset of features. This **decorrelates** the
   trees — without it, every tree latches onto the same few dominant features and they all make the
   same mistakes, so averaging helps less. Decorrelation lowers the variance floor that bagging alone
   cannot pass.

Why this chapter is pivotal:

- It is the **first ensemble** — the course pivots from single models (KNN → NB → LogReg → tree →
  SVM) to *committees* of models. The mental move ("average many weak-ish models into a strong one")
  is the foundation for **all of ch 07–10** (AdaBoost, gradient boosting, XGBoost, LightGBM).
- It is the method most practitioners reach for **first** on tabular data: a **strong baseline** that
  needs almost no tuning, no scaling (inherited from trees), gives a **free validation estimate**
  (out-of-bag) and **feature importances**, and is **embarrassingly parallel**.
- It teaches, by contrast, the **honest limits** of "averaging fixes everything": importances on
  correlated features must be read carefully, the model is no longer one readable rule set, and on a
  **near-linear** problem (breast_cancer, the ch 04 spine) a forest does *not* beat a linear/margin
  model — the right tool depends on the problem's shape.

The chapter's honest spine, all measured: bagging lifts moons from **0.878 → 0.933** and shrinks the
ensemble's run-to-run std **~9×** (0.0465 → 0.0053 from 1 → 100 trees); feature subsampling lifts
breast_cancer from **0.924 → 0.95** purely by decorrelation (individual trees are no better); the
bootstrap leaves out a steady **~37 %** of points per tree, giving an OOB estimate ≈ test; and on a
genuinely non-linear problem the forest **wins** (covtype RF 0.846 ≫ LogReg 0.728) where on a
near-linear one it does **not** (breast_cancer RF ≈ 0.94–0.95 < SVM 0.965).

## Prerequisites (re-established briefly; first-contacts flagged, never mislabelled as recaps)

- **Module 00 (genuine recaps):** train/test split & leakage (NB 04); accuracy + baseline (NB 06);
  **confusion matrix, precision/recall, macro vs weighted** (NB 07) — central to NB 5's honest eval
  under class imbalance, and **re-laid briefly in NB 5** (macro = unweighted mean over classes, so
  rare classes count fully; accuracy/weighted is dominated by the big classes), not presupposed;
  **cross-validation** (NB 10) — model selection on train, one sealed test. **Note the absence:**
  standardization (NB 11) is *not* needed — a forest of trees is scale-invariant (inherited from
  ch 04), a deliberate, stated contrast with KNN/LogReg/SVM.
- **Chapter 04 (Decision Trees) — the load-bearing prerequisite, genuine recaps:** the **tree as the
  base learner** (one split → grow → read); **a single tree is high-variance** (the root feature flips
  across resamples) — *the exact weakness this chapter fixes*; **Gini/MDI feature importance and its
  bias** toward continuous/high-cardinality features (Strobl 2007), with permutation importance named
  as the honest cross-check; **scale-invariance**; the **hand-bagged-25 taste** (breast_cancer → 0.930)
  and the **moons variance** (std 0.032) — NB 1 resumes the *idea* on moons (so the variance is
  drawable).
- **Chapter 01 (KNN) / 03 (LogReg) / 05 (SVM) — the cross-method demanding-case spine** on the pinned
  breast_cancer 70/30 seed-0 split: KNN 0.942 · LogReg 0.953 · single tree 0.906 · SVM **0.965**. NB 5
  places the forest on this spine (or its covtype analogue) and reads the result honestly. **Ch 05's
  measured scaling wall** (SVM RBF fit-time **≈ n^1.6** there, worst-case O(n³)) is the foil for RF's
  graceful near-linear growth.
- **Re-established from scratch (assumed from nobody) — the FIRST CONTACTS, each budgeted as a
  taught-from-scratch concept, never called a recap:**
  - **bootstrap resampling** — sample n points *with replacement* from n; what changes, what repeats —
    **NB 1**;
  - **aggregation / majority vote** of many models, and **why averaging unbiased estimators reduces
    variance** (the σ²/B intuition; ESL §15.2) — **NB 1**;
  - **pairwise correlation ρ between trees**, and the variance-of-an-average law
    **Var = ρσ² + (1−ρ)σ²/B** — **derived from scratch in NB 2** (expand the variance of a mean of B
    equally-correlated predictors: B variance terms + B(B−1) covariance terms → the two-term law; let
    B→∞ to expose the ρσ² floor) — why decorrelation lowers the floor bagging cannot pass — **NB 2**;
  - **random feature subsampling at each split** (`max_features`) as the decorrelation lever — **NB 2**;
  - **out-of-bag (OOB) estimation** — the ~1/e of points each tree never saw form a free held-out set
    — **NB 3**;
  - **MDI importance averaged over a forest** (more stable than one tree, still biased) and
    **permutation importance** as the honest read, including **correlated-/one-hot-feature dilution** —
    **NB 4 (intro) / NB 5 (honest reading)**.

## Datasets (measured at plan time on sklearn **1.9.0**; seeds fixed; re-measured at each build)

**RandomForestClassifier defaults verified on the live 1.9.0 install:** `n_estimators=100`,
**`max_features='sqrt'`**, `bootstrap=True`, `oob_score=False` (opt-in), `max_depth=None` (trees grown
deep), `criterion='gini'`, `class_weight=None`, `n_jobs=None`. **Every RF in the chapter pins
`random_state`** (the ensemble's own RNG) so anchors reproduce; test-set numbers on the 171-point
breast_cancer split carry ±0.01 seed wobble, so the chapter rests headlines on the **robust** signals
(ρ trends, OOB, the cross-method gaps), not on fragile per-setting test rankings.

- **NB 1 — `make_moons(n_samples=300, noise=0.30, random_state=0)`, 210/90 stratified split (seed 0)**
  — the same non-linear set ch 04 carved, so the single-tree number is a *continuity anchor*:
  - **Single unlimited tree:** train 1.000 / **test 0.8778** (= ch 04's 0.878, confirmed); single-tree
    **bootstrap std 0.0314** (≈ ch 04's 0.032) — wobbly.
  - **Hand-bagging (majority vote of B unlimited trees, each on its own bootstrap):** test 0.900 (B=1)
    → 0.922 (B=5) → **0.933** (B≥10, plateau). **Ensemble run-to-run variance** (std of test accuracy
    across 20 different bagging seeds): **0.0465 (B=1) → 0.0310 (B=5) → 0.0089 (B=25) → 0.0053
    (B=100)** — the σ²/B curve, variance shrinking ~9×.
  - **Honest parity:** hand-bag(200) = **0.9333** = `RandomForestClassifier(n_estimators=200,
    max_features=None)` **exactly** — *bagging is a random forest with feature subsampling switched
    off*. (No bit-exact single-tree parity is claimed: `RF(n=1)` ≠ a lone `DecisionTree` because the
    forest's splitter randomises tie-breaking — stated, not hidden.)
- **NB 2 — breast_cancer (`datasets.load_breast_cancer`, 569 × 30, malignant = 1), 70/30 stratified
  split (seed 0), `StratifiedKFold(5, shuffle, random_state=0)`** — 30 (correlated) features, so
  feature subsampling can bite (it cannot on moons' 2 features):
  - **Bagging (`max_features=None`, all 30) vs RF (`max_features='sqrt'` ≈ 5):** ensemble test
    **0.924 → ~0.95**; CV-on-train **0.945 → 0.955**; mean pairwise tree correlation **ρ 0.82 → 0.80**.
    **The gem:** individual trees are *no better* (mean acc ≈ 0.91 either way) — the entire ensemble
    gain comes from **decorrelation**, not stronger trees.
  - **`max_features` is the decorrelation dial** (`n_estimators=80`): ρ rises **monotonically** with it
    — `max_features` 1 → 2 → 5 → 10 → 15 → 20 → 30 gives **ρ 0.70 → 0.76 → 0.80 → 0.81 → 0.82 → 0.82 →
    0.82**. This monotone ρ trend is the robust headline; test accuracy across these settings is
    seed-fragile on 171 points (do **not** rank `max_features` by test accuracy in the prose). The
    teaching beat: all-features = highest ρ = least decorrelation benefit, and the **ensemble beats
    bagging** (bagging 0.924 < RF ≈ 0.95). The "random" earns its keep.
  - **Why moons can't show this (stated honestly):** on 2 features, `'sqrt'` → 1 feature per split
    *starves* the trees (RF 0.900 < bagging 0.933) — feature subsampling pays only when there are many,
    ideally correlated, features. This is why NB 1's variance story lives on moons and NB 2's
    decorrelation story lives on breast_cancer.
- **NB 3 — breast_cancer (same pinned split), out-of-bag:**
  - **The ~1/e left-out fraction:** theory `(1 − 1/n)^n` = **0.3674** → `1/e` = 0.3679 (n = 398);
    empirical mean OOB fraction per bootstrap **0.3673**. ~37 % of points sit out each tree.
  - **OOB ≈ test, for free:** `RF(500, oob_score=True)` → **OOB ≈ 0.96 vs sealed test ≈ 0.94** (close;
    here mildly *optimistic* — stated). At very few trees OOB is unreliable (sklearn warns when some
    points are never OOB) — the honest caveat, surfaced not silenced.
- **NB 4 — breast_cancer (curves) + moons (boundary), the estimator & parameters:**
  - **`n_estimators`** (OOB & test error): n 1 → 5 → 25 → 100 → 500 gives OOB err 0.271 → 0.090 →
    0.040 → 0.045 → 0.040 — **diminishing returns, monotone-ish, does not systematically overfit** by
    adding trees (more trees only steady the estimate; the cost is compute/memory).
  - **`max_features`** — the central dial (NB 2's ρ trend, now as the hyperparameter); **`'sqrt'`** the
    robust default.
  - **`max_depth` / `min_samples_leaf`** — RF grows **deep** trees and tolerates it (max_depth None /
    5 / 10 ≈ 0.94–0.95): averaging controls the variance a lone deep tree could not.
  - **`bootstrap`, `class_weight`, `n_jobs`** — named/shown lightly.
  - **Feature importance — introduced here (one figure):** MDI over the forest **spreads** across the
    correlated group (peak ≈ 0.15–0.17, several features in the 0.09–0.17 band) where the **single
    tree** (ch 04) concentrated **≈ 0.8** on one — averaging stabilises the read. *Which* feature leads
    MDI is itself seed/criterion-sensitive across the correlated radius/perimeter/concavity group (read
    the actual top features off the fitted forest at build, do not hard-code a name/value). The **bias
    caveat** restated; **permutation importance named**, its honest reading deferred to NB 5.
  - **Honest tuning:** `GridSearchCV` over `max_features` / `min_samples_leaf` / `max_depth` on TRAIN
    → CV ≈ 0.957 → **sealed test ≈ 0.95** (pinned seed; exact value read at build).
- **NB 5 — demanding case (recommended: covtype; alternatives below).** Forest cover type
  (`fetch_covtype`, 581 012 × 54, 7 classes), a **30 000-row stratified subsample**, 70/30 split
  (seed 0) — thematically apt (a random *forest* on *forest* cover) and genuinely **non-linear**:
  - **The forest wins:** `RF(300)` **test 0.846** (OOB 0.839) ≫ single tree 0.775 ≫ **LogReg 0.728** —
    the opposite of breast_cancer, where the linear/margin models won. *The right tool tracks the
    problem's shape.*
  - **Honest eval under imbalance:** accuracy 0.846 hides it; **macro-F1 0.733** reveals it (re-lay
    macro vs weighted first). Per-class recall: Lodgepole 0.903 / Spruce-Fir 0.823 / Ponderosa 0.865
    but **Aspen ≈ 0.28** (n ≈ 145) / Cottonwood ≈ 0.48 / Douglas-fir ≈ 0.62 — the rare classes are
    hard, and accuracy alone would lie (exact per-class counts read off the NB 5 split at build).
  - **Reading importance honestly:** **Elevation** dominates (MDI 0.231; permutation 0.286 — here the
    two *agree*, because one feature genuinely dominates and is not correlation-diluted), while the
    **40 one-hot Soil_* columns combined score only 0.140** (each ≈ 0.0035) — the **one-hot / dilution
    caveat** made vivid (a single dominant continuous feature outweighs forty diluted dummies).
  - **OOB at scale, scaling, and the boosting bridge:** OOB 0.839 ≈ test 0.846; **RF fit-time grows
    roughly linearly in n** (measured exponent ≈ 1.0–1.2 over 1 000 → 64 000 single-thread; theory
    O(n log n · trees · √p)) and is embarrassingly parallel — the graceful counterpoint to **ch 05's
    SVM ≈ n^1.6 super-linear wall**. Honest close: a forest is a strong, low-effort baseline but
    **boosting often edges it with more tuning** (ch 07–10).
- All datasets offline (sklearn-bundled / generated) except covtype's one-time fetch (≈ 11 MB download,
  ≈ 14 MB cached on disk; the established `load_newsgroups` pattern, visible INFO logging); colours
  from `ml_course.colors`; "Read the figure" after every figure; a tiered **"Your turn"** in each
  notebook; the charter close (**Your turn → What you built → optional Going further → References**); a
  running random-forest vocabulary box.

## Primordial concepts → notebooks 1–3 (one concept each; by hand before any library)

| NB | Title | The one concept | Done by hand | Key figure(s) → "Read the figure" | Your turn (tiered) |
|----|-------|-----------------|--------------|-----------------------------------|--------------------|
| 1 | The wisdom of trees: **averaging cuts variance** | **Bagging.** Many high-variance, low-bias trees, each on a **bootstrap** resample, **majority-voted**, give one **low-variance** model — averaging unbiased predictors cancels their noise (≈ σ²/B) without adding bias | resume ch 04's wobbly moons tree (test 0.878); draw **bootstrap** samples by hand; grow a tree on each; **majority-vote**; watch test rise to **0.933** and run-to-run std fall **9×**; confirm **hand-bag == `RF(max_features=None)`** | (a) **12 single bootstrap-tree boundaries vs the averaged forest boundary** (jagged-and-varied → one smooth) — *"each tree is confident and different; the vote is the steady consensus"*; (b) **test accuracy & run-to-run std vs number of trees** — *"more trees buy stability, with diminishing returns"* | easy — majority-vote three given trees on a point; medium — bootstrap a small set by hand, note repeats/omissions; harder — sweep B and plot test accuracy + its std |
| 2 | The "random" in the forest: **decorrelating the trees** | **Random feature subsampling.** If every tree may use **all** features they correlate (same dominant splits, same mistakes); letting each split see only a **random subset** (`max_features`) **decorrelates** them. **Derive** Var = ρσ² + (1−ρ)σ²/B from scratch → lower ρ lowers the variance floor bagging alone cannot pass | derive the variance-of-a-mean law by hand (B variance + B(B−1) covariance terms; B→∞ → ρσ² floor); on breast_cancer measure **pairwise tree correlation ρ** for bagging (all 30 feats, ρ 0.82) vs RF (`'sqrt'`≈5, ρ 0.80); show ensemble test **0.924 → 0.95** while **individual trees stay equal** (gain = decorrelation); sweep `max_features` → ρ rises monotonically | (a) **ρ vs `max_features`** (rising) with the all-features = bagging end marked highest-ρ — *"the more features each split may see, the more alike the trees, the less averaging helps"*; (b) **ensemble vs mean-individual-tree accuracy** — *"the committee beats its members, and most when they disagree"* | easy — from ρ pick which setting averages best; medium — explain why all-features bagging has the highest ρ; harder — use the **derived** law to argue why ρ↓ matters more than B↑ past a point |
| 3 | What the bootstrap gives for free: **out-of-bag estimation** | **OOB error.** Each bootstrap omits ~**1/e ≈ 37 %** of points; for any point, the trees that *didn't* see it form a held-out mini-forest, so the forest **scores itself on data it never trained on** — a validation estimate for free, no split set aside | derive `(1−1/n)^n → 1/e` and **measure** the left-out fraction (0.367); build the OOB vote for a point by hand from the trees that missed it; compare **OOB (≈0.96) ≈ sealed test (≈0.94)**; show OOB is unreliable with too few trees | (a) **bootstrap/OOB schematic** — which points each of a few trees saw vs missed — *"the grey points trained this tree; the coloured ones grade it"*; (b) **OOB error vs n_estimators converging toward the test error** — *"the free estimate tracks the honest one once there are enough trees"* | easy — from a small in-bag table, name a point's OOB graders; medium — compute the OOB fraction for a given n; harder — compare OOB to a held-out test and discuss when to trust it |

## Notebook 4 — the estimator & its parameters (~22–24 cells; the integrative notebook)

`sklearn.ensemble.RandomForestClassifier` for real (1.9.0). **Honest parity first:** hand-bagging (NB 1)
== `RF(max_features=None)`; a random forest = that **plus** per-split feature subsampling (NB 2). Then
walk the knobs, keeping a **soft ~24-cell ceiling** (ch 04's lesson) so the chapter's ideas breathe:
- **`n_estimators`** — the OOB/test-error curve: **diminishing returns, does not systematically
  overfit** (more trees only steady the estimate; cost is compute & memory). Pick "enough", not "as
  many as possible".
- **`max_features`** — the **central** dial, NB 2's ρ trend as the hyperparameter; **`'sqrt'`** the
  robust default; the bias/variance picture it controls.
- **`max_depth` / `min_samples_leaf`** — RF grows **deep** and tolerates it (averaging tames the
  variance a lone deep tree could not) — measured (depth None/5/10 ≈ 0.94–0.95).
- **`bootstrap` / `class_weight` / `n_jobs`** — shown/named lightly (`bootstrap=False` removes the
  bagging; `class_weight` for imbalance, used in NB 5; `n_jobs` = the embarrassingly-parallel win).
- **Feature importance — introduced (one figure):** MDI over the forest **spreads** across correlated
  features (peak ≈ 0.15–0.17) vs the single tree's ≈ 0.8 spike — averaging stabilises the read; the
  leader is seed/criterion-sensitive (read at build); the **bias caveat** restated; **permutation
  importance named** (honest reading → NB 5).
- **Honest tuning:** `GridSearchCV` on TRAIN (`max_features`/`min_samples_leaf`/`max_depth`), one
  sealed test (CV ≈ 0.957 → test ≈ 0.95) — the module-00 discipline.
- **Your turn:** turn `n_estimators` and read the OOB curve; set `max_features` to the bagging end and
  watch ρ/accuracy; confirm raw == standardized (scale-invariance inherited from ch 04).

## Notebook 5 — the demanding practical case (recommended: **covtype**)

A strong tabular baseline on **forest cover type** (30 000-row subsample, 7 classes, 54 features) — the
full honest workflow, **visualization-first**. Per the capstone rule, **~24–26 cells is a floor, not a
ceiling**, and **figures may exceed six** if the imbalance and importance reads each need their own
"Read the figure" (do not let NB 4's soft ceiling leak onto the capstone):
- **Look → fit → (lightly) tune → sealed test:** class balance (imbalanced); **no standardization**
  (a forest point); `RandomForestClassifier`; OOB as the cheap in-train estimate; CV-light tuning; one
  held-out read.
- **The forest wins (its identity):** RF **0.846** ≫ single tree 0.775 ≫ LogReg 0.728 — a genuinely
  non-linear problem where the committee of trees beats the linear model decisively, **the reverse of
  breast_cancer** (referenced: there RF ≈ 0.94–0.95 < SVM 0.965). *The right tool tracks the problem's
  shape.*
- **Honest evaluation under imbalance:** re-lay macro vs weighted (macro = unweighted mean over
  classes, so rare classes count fully); accuracy 0.846 vs **macro-F1 0.733**; **per-class recall**
  (Aspen ≈ 0.28 / Cottonwood ≈ 0.48 vs Lodgepole 0.903) — accuracy alone would lie; the 7×7 confusion
  matrix shows where.
- **Reading importance honestly:** **Elevation** dominates (MDI 0.231 ≈ permutation 0.286 — they
  *agree* when one feature truly dominates), while the **40 one-hot Soil_* columns combine to only
  0.140** — the **dilution caveat** (MDI spreads across one-hot dummies / correlated groups;
  permutation under-credits correlated twins) — ch 04's importance lesson, now at forest scale.
- **Scaling & the boosting bridge:** OOB ≈ test; **RF fit-time grows roughly linearly in n** (measured
  ≈ n^1.0–1.2 over 1k–64k single-thread; theory O(n log n · trees · √p); parallel) — the counterpoint
  to ch 05's SVM ≈ n^1.6 wall; honest close — a forest is a **strong, low-effort baseline**, but it is
  no longer one readable tree, and **boosting (ch 07–10) often edges it** with more tuning.
- **Your turn:** read OOB vs the sealed test; compare RF to the single tree and the linear baseline and
  explain the gap from the problem's shape; (harder) rank features by MDI and by permutation, and
  explain why the Soil dummies look unimportant either way.

## Notebook count — five (the standard arc)

Fundamentals split cleanly into **three one-concept notebooks** (averaging↓variance / decorrelation /
OOB), then the estimator (NB 4) and the demanding case (NB 5). **Refinement of `course_map.md` §06
noted:** the map paired "OOB estimation; feature importance" in NB 3; this plan keeps **NB 3 = OOB
only** (one concept) and distributes **feature importance across NB 4 (intro) + NB 5 (honest
reading)** — mirroring ch 04's proven importance arc and keeping NB 4 from overloading. All of §06's
concept coverage (variance reduction, bagging, subsampling, decorrelation, OOB, importance + caveats,
the parameters, the demanding case) is preserved. No 6th notebook is needed.

## Resolved decision — NB 5 dataset = **covtype**

**Rémy chose covtype** (2026-06-23) over the alternatives below — the one capstone where the forest is
the *right* tool and wins decisively on a non-linear problem (RF 0.846 ≫ LogReg 0.728), teaches honest
evaluation under imbalance (macro-F1 0.733; per-class recall), and where "reading importances honestly"
bites hardest (Elevation dominates; 40 one-hot Soil columns dilute). The breast_cancer spine is
*referenced* for the instructive reversal (there RF ≈ 0.94–0.95 < SVM 0.965 on near-linear data).

| Option | Forest result | Why | Cost |
|--------|---------------|-----|------|
| **covtype** *(chosen)* | RF **0.846** ≫ tree 0.775 ≫ LogReg 0.728 | thematic; the forest **genuinely wins** (non-linear = RF's identity); imbalance → honest macro-F1/recall; named features + one-hot dilution lesson | one-time ≈ 14 MB cached fetch; severe class imbalance (rich but heavy) |
| digits | RF **0.974** (tree 0.837, SVM 0.982) | fully **offline**; the most dramatic single-tree→forest lift in the course (+14 pts); balanced 10-class; importance-as-8×8-image | RF does not "win" (SVM edges it); pixel importances are weaker "honest-reading" material |
| breast_cancer | RF **≈ 0.94–0.95** (LogReg 0.953, SVM 0.965) | **completes the 5-method spine** on one dataset | RF *loses* to linear/margin (honest but anticlimactic in RF's own chapter); 4th reuse |

## Library additions (decided per-notebook, with tests; none forced now)

- **Likely one helper: `viz.plot_feature_importances(names, importances, *, std=None, top=10,
  ax=None)`** — a sorted horizontal bar (charter colours), reused in **NB 4 and NB 5** (clears the
  "≥ 2×" bar), able to overlay/contrast MDI vs permutation. If promoted, add a smoke test
  (`pytest` 19 → 20). Decided at NB 4 build.
- **Reused as-is from `ml_course.viz`:** `use_course_style`, `plot_decision_boundary` (single tree vs
  forest on moons — NB 1/4), `plot_train_test_curve` (variance/error vs B; OOB vs test — NB 1/3/4),
  `plot_confusion_matrix` (NB 5), `plot_class_balance` (NB 5).
- **One-off in-notebook (charter colours):** the bootstrap/OOB schematic, the overlaid-boundaries
  variance figure, the ρ-vs-`max_features` curve, the per-class-recall bar, the scaling curve.

## Honest limits stated in the notebooks

- **Averaging cancels variance, not bias.** A forest of biased trees stays biased; bagging fixes
  variance, not a wrong model class (NB 1).
- **Feature subsampling only pays with enough features.** On moons (2 feats) `'sqrt'` *hurts*
  (0.900 < 0.933) — stated, not hidden (NB 2).
- **OOB is approximate and needs enough trees** — mildly optimistic here (≈ 0.96 vs test ≈ 0.94), and
  unreliable at very few trees (sklearn warns) (NB 3).
- **Importance is not causal and is biased.** MDI favours continuous/high-cardinality features and
  **dilutes across correlated / one-hot groups**; permutation under-credits correlated twins — read at
  the *group* level, cross-check, never as ground truth (Strobl 2007/2008) (NB 4/5).
- **A forest is not one readable tree.** The ch 04 payoff (a flowchart a clinician reads) is **gone** —
  hundreds of trees trade interpretability for accuracy and stability (NB 5).
- **Not always the most accurate.** On near-linear data a forest *loses* to linear/margin models
  (breast_cancer); boosting often edges it on tabular data (ch 07–10). RF's claim is **strong with
  little effort**, not **best** (NB 5).
- **A genuine strength, stated:** scales roughly linearly in n and is parallel — the opposite of
  ch 05's SVM wall — and needs no scaling.

## References (with DOI; dereferenced & pinned by the ml-expert reviewer at build)

- Breiman L (2001). *Random Forests.* Machine Learning 45:5–32. DOI 10.1023/A:1010933404324 — the
  method.
- Breiman L (1996). *Bagging predictors.* Machine Learning 24:123–140. DOI 10.1007/BF00058655 —
  bagging / variance reduction.
- Ho TK (1998). *The random subspace method for constructing decision forests.* IEEE TPAMI
  20(8):832–844. DOI 10.1109/34.709601 — random feature subspaces.
- Strobl C, Boulesteix A-L, Zeileis A, Hothorn T (2007). *Bias in random forest variable importance
  measures.* BMC Bioinformatics 8:25. DOI 10.1186/1471-2105-8-25 — MDI bias.
- Strobl C, Boulesteix A-L, Kneib T, Augustin T, Zeileis A (2008). *Conditional variable importance
  for random forests.* BMC Bioinformatics 9:307. DOI 10.1186/1471-2105-9-307 — correlated-feature
  importance.
- Hastie T, Tibshirani R, Friedman J (2009). *ESL*, §15 (Random Forests) — incl. the
  Var = ρσ² + (1−ρ)σ²/B law. DOI 10.1007/978-0-387-84858-7.
- James G, Witten D, Hastie T, Tibshirani R (2021). *ISLR*, §8.2 (bagging, random forests). DOI
  10.1007/978-1-0716-1418-1.
- Blackard JA, Dean DJ (1999). *Comparative accuracies … forest cover types ….* Computers and
  Electronics in Agriculture 24(3):131–151. DOI 10.1016/S0168-1699(99)00046-0 — the covtype dataset.

## Verification (per notebook, at its commit)

Every number re-run and reconciled into prose at build on sklearn **1.9.0**, with each RF's
`random_state` pinned (NB 1: single tree test 0.878 / bootstrap std 0.031, hand-bag 0.933, run-to-run
std 0.0465→0.0053, hand-bag == RF(mf=None); NB 2: derive the ρ-law, bagging→RF test 0.924→~0.95, ρ
0.82→0.80, indiv trees ≈ equal, ρ-vs-max_features 0.70→0.82 monotone (no test-acc ranking of mf),
moons sqrt 0.900 < 0.933; NB 3: (1−1/n)^n 0.367 → 1/e, empirical 0.367, OOB ≈ 0.96 vs test ≈ 0.94;
NB 4: n_estimators OOB-err 0.271→0.040 diminishing, max_features dial, depth None/5/10 ≈ 0.94–0.95,
MDI spread (peak ≈ 0.15–0.17, leader read at build) vs single-tree ≈ 0.8, GridSearch CV ≈ 0.957 →
test ≈ 0.95; NB 5: RF 0.846 ≫ tree 0.775 ≫ LogReg 0.728, macro-F1 0.733, per-class recall incl. Aspen
≈ 0.28, MDI Elevation 0.231 / perm 0.286 / Soil-40-combined 0.140, fit-time ≈ n^1.0–1.2). Runs
top-to-bottom (nbconvert to scratchpad; tracked file **output-free**); **banned-word scan over the
JSON real cell text** = 0; `check_no_hardcoded_hex` passes; `gen_llms_txt` re-run; `pytest` green (19,
→ 20 only if `viz.plot_feature_importances` lands); `ruff`/`black` clean; `course_map.md` §06 aligned;
`common_errors.md` extended per new trap. Both reviewers pass (no BLOCK) on each notebook; Rémy
validates visually; commit per notebook; chapter close via PR into `main` (protected).

## Per-notebook status (update as the chapter is built)

| NB | Branch | Status |
|----|--------|--------|
| 1 | `notebook/06_RandomForest__01_averaging_cuts_variance` | planned |
| 2 | `notebook/06_RandomForest__02_decorrelating_trees` | planned |
| 3 | `notebook/06_RandomForest__03_out_of_bag` | planned |
| 4 | `notebook/06_RandomForest__04_estimator_and_parameters` | planned |
| 5 | `notebook/06_RandomForest__05_covtype_strong_baseline` | planned (covtype, per Rémy) |

## Reviewer notes (chapter-plan gate — both reviewers REVISE → all folded; re-verified at plan time)

- **ML expert — REVISE (no BLOCK; load-bearing anchors reproduced on 1.9.0; covtype section exact to
  three decimals):** MAJOR — the SVM scaling foil "n^1.67" contradicts the shipped ch 05 plan's
  measured **n^1.6** → corrected to ≈ n^1.6 everywhere. MAJOR — RF "n^1.18" not reproducible
  (re-measured ≈ 1.0, linear) → reworded "roughly linear, ≈ n^1.0–1.2, theory O(n log n · trees · √p)".
  MAJOR — "max_features=30 is the worst ensemble by test acc (0.930)" is seed-fragile on 171 points →
  the headline is now the **monotone ρ trend** + the **ensemble-beats-bagging** gap, not a per-mf
  test ranking. MINOR — the specific MDI triplet drifts (the leader is seed/criterion-sensitive) →
  prose now states "spread, peak ≈ 0.15–0.17 vs single-tree ≈ 0.8", read off the fitted forest at
  build. MINOR — RF test anchors sit in a ±0.01 seed band → `random_state` pinned for every RF; the
  reversal framed "RF ≈ 0.94–0.95 < SVM 0.965". MINOR — Aspen n re-measured 147 → "n ≈ 145, read at
  build". MINOR — covtype cache ≈ 14 MB on disk (not ~70 MB) → corrected. Praised: ρ-law verified by
  Monte-Carlo, the decorrelation gem, exact OOB fraction, the covtype capstone reproducing to three
  decimals, the honest reversal, complete honest-limits, correct citations, the sound NB 3 refinement.
- **Pedagogy — REVISE (no BLOCK):** MAJOR — the ch 04 → NB 1 bridge conflated two datasets (the ch 04
  hand-bag-25 was on breast_cancer → 0.930; NB 1 resumes the **moons** tree → 0.933) → Context + the
  prerequisite line now state the switch plainly (ch 04 felt the idea twice; NB 1 re-runs bagging on
  moons because it is 2-D and the boundaries are *drawable*). MAJOR — the variance law was leaned on
  (incl. by the NB 2 "harder" exercise) without a from-scratch budget → NB 2 now **derives**
  Var = ρσ² + (1−ρ)σ²/B (B variance + B(B−1) covariance terms; B→∞ floor) before the exercise uses it.
  MINOR — NB 5 cell count → stated as a **floor** (capstone rule), figures may exceed six. MINOR —
  "clearly wins" softened to "wins decisively on a non-linear problem" with the honest per-class
  caveat. MINOR — macro-vs-weighted gets a short re-laying beat in NB 5 before the per-class figure.
  Praised: exemplary first-contact fencing, NB 1 vs NB 2 genuinely distinct (NB 1 proves subsampling
  off), the sound NB 3 refinement, the complete honest-limits framed as growth, the candid NB 5
  dataset table, every figure carrying a "Read the figure", and a clean banned-word scan.
