# Notebook plan — 06_RandomForest / 04_estimator_and_parameters

> Status: **APPROVED** (2026-06-23, by Rémy). Built next; both reviewers (`@ml-expert-reviewer` +
> `@pedagogy-reviewer`) gate the **built** notebook (no gate at the plan stage). Anchors re-measured
> at build on sklearn **1.9.0**, every random forest `random_state`-pinned. **`viz.plot_feature_
> importances` helper: APPROVED to add (the recommended option) + smoke test → pytest 19 → 20.**

## Context

NB **4 of 5** of chapter 06 — the **integrative** notebook: the real
`sklearn.ensemble.RandomForestClassifier`, after NB 1–3 built every piece by hand (bagging → the
"random" / decorrelation → out-of-bag). One honest move opens it — turn the "random" **off**
(`max_features=None`) and the library **is** our NB-1 hand-bag — then we walk each knob and learn what
it does, how it fails, how to tune it. Per ch 04's de-overload lesson this holds a **soft ~24-cell
ceiling** and **3 figures**, so the ideas breathe. The honest headlines (all re-measured below): more
trees never overfit (diminishing returns); `max_features` is the decorrelation dial (NB 2's ρ) but a
forest is **forgiving** (`'sqrt'` a safe default); a forest **grows deep trees on purpose** and
tolerates it (averaging cancels the variance that sinks a lone deep tree); forest MDI **spreads**
across the correlated group where a single tree **spiked**; and honest `GridSearchCV` barely beats the
default — RF's real selling point is **strong with almost no tuning**.

## Anchors (re-measured at plan time, sklearn **1.9.0**; every RF `random_state`-pinned)

breast_cancer (569 × 30, **malignant = 1**), 70/30 stratified split (seed 0), n_train = 398;
`StratifiedKFold(5, shuffle, random_state=0)` for CV-on-train. **RF defaults confirmed live:**
`n_estimators=100`, `max_features='sqrt'`, `bootstrap=True`, `oob_score=False`, `max_depth=None`,
`criterion='gini'`, `class_weight=None`, `n_jobs=None`.

- **Parity (the bridge):** hand-bag (B=200 unlimited trees, majority vote) test **0.9357** ==
  `RF(n_estimators=200, max_features=None)` **0.9357** (exact accuracy match); the default
  `RF(max_features='sqrt')` **0.9415** — the "random" helps with 30 features. *One-line moons recall
  (NB 1/2):* `RF(mf=None)` 0.9333 == NB 1's hand-bag; `RF(sqrt)` 0.9000 *starves* 2 features. (No
  bit-exact per-tree parity claimed — accuracy match at B=200; `RF(n=1)` ≠ a lone tree, NB 1.)
- **`n_estimators` (OOB & test error, `oob_score=True`):** n = 1 / 5 / 10 / 25 / 50 / 100 / 300 / 500
  → OOB err **0.271 / 0.090 / 0.063 / 0.040 / 0.040 / 0.045 / 0.043 / 0.040**; test err noisy
  **~0.05–0.10** (no upward drift). sklearn **warns** at n ≤ 10 (some points never OOB) — shown, not
  silenced. Diminishing returns, **does not systematically overfit**; more trees only steady the
  estimate (cost = compute/memory).
- **`max_features` (OOB & CV-on-train, n=200):** mf = 1 / 2 / 'sqrt'(5) / 'log2'(4) / 10 / 20 / 30 →
  OOB **0.960 / 0.955 / 0.955 / 0.962 / 0.960 / 0.957 / 0.952**; CV **0.960 / 0.962 / 0.957 / 0.957 /
  0.947 / 0.950 / 0.947**. **Accuracy is flat across the whole range** — a forest is *forgiving*;
  `'sqrt'` is the robust default. The dial's *mechanism* is decorrelation (NB 2's ρ rose 0.70→0.82
  with mf); all-features (mf=30) = plain bagging = highest ρ = least decorrelation. **No test-acc
  ranking of mf in prose** (171 pts seed-fragile, NB 2).
- **`max_depth` — single tree vs forest (train/test):** md 1 → 2 → 3 → 5 → 8 → None: single tree
  **0.935/0.860 → 0.967/0.865 → 0.975/0.918 → 0.995/0.895 → 1.000/0.906 → 1.000/0.906**; forest
  (n=200) **0.940/0.918 → 0.970/0.936 → 0.985/0.936 → 0.997/0.942 → 1.000/0.942 → 1.000/0.942**. Both
  memorise train at depth; the **lone tree's test wobbles ~0.86–0.918**, the **forest climbs to 0.942
  and plateaus**. Run-to-run std (20 seeds): single deep tree **0.0163** (mean 0.917) vs forest
  **0.0043** (mean 0.947) — ≈4× steadier *and* more accurate. (`min_samples_leaf` same story: leave it
  at 1; over-restricting — msl 10/20 — slightly hurts CV 0.957→0.940.)
- **Remaining knobs (named):** `bootstrap=False` removes the bagging (trees then differ only by the
  per-split feature RNG; test 0.947 here, within seed noise); `class_weight` for imbalance (used in
  NB 5); `n_jobs=-1` trains trees across cores (the embarrassingly-parallel win, no accuracy change).
- **Feature importance — MDI spread vs single-tree spike:** **single tree** puts **0.740** on
  `mean concave points` (one spike); the **forest** (n=300) spreads — `mean concave points` **0.146** /
  `worst concave points` 0.136 / `worst perimeter` 0.107 / `worst radius` 0.105 / `worst area` 0.100 /
  `mean concavity` 0.076 … (peak **0.146**, 6 features ≥ 0.05). Leader seed/criterion-sensitive (read
  at build; here `mean concave points`). **Bias caveat** (Strobl 2007): MDI favours
  continuous/high-cardinality features and **dilutes across correlated groups**; **permutation
  importance** *named* as the honest cross-check — full reading deferred to **NB 5**.
- **`GridSearchCV` (n=300, on TRAIN):** best `{max_features 'log2', min_samples_leaf 1, max_depth
  None}` CV **0.957** → **sealed test 0.947**; default RF CV 0.955 → test 0.942. Tuning barely beats
  the default; the grid keeps **deep** trees (None, leaf 1).
- **Scale-invariance (Your turn):** raw vs standardized predictions **identical (1.000)** — inherited
  from ch 04 (trees split on thresholds).

## Library / figures

- **`src/` add (APPROVED): `viz.plot_feature_importances(names, importances, *, std=None, top=10,
  ax=None)`** — a sorted horizontal bar (charter colours), able to contrast two series (single-tree
  spike vs forest spread; MDI vs permutation in NB 5). Reused in **NB 4 + NB 5** → clears the "≥ 2×
  reuse" bar; add a smoke test → **pytest 19 → 20**.
- **Reused as-is:** `viz.use_course_style`; `plot_train_test_curve` (Fig A, two error lines vs n;
  Fig B, single-tree vs forest test vs depth — override the train/test legend labels);
  `ml_course.colors`. Sklearn: `RandomForestClassifier`, `DecisionTreeClassifier` (single-tree foil +
  hand-bag), `GridSearchCV`, `StratifiedKFold`, `cross_val_score`, `permutation_importance` (named),
  `train_test_split`, `accuracy_score`.
- **Three figures** (integrative NB, not the capstone): **A** OOB & test error vs `n_estimators`;
  **B** single-tree vs forest test accuracy vs `max_depth` (the "why RF grows deep" contrast);
  **C** single-tree MDI spike vs forest MDI spread. Each + "Read the figure".

## Cell-by-cell (~24 cells; soft ceiling; "Read the figure" after every figure)

1. (md) **Header** — `# 04 — The estimator RandomForestClassifier and its parameters`; *notebook 4 of
   5*; warm welcome. **Prerequisites:** NB 1 (bagging: bootstrap + majority vote; hand-bag ==
   `RF(max_features=None)`); NB 2 (the "random": per-split feature subsampling, the ρ decorrelation
   dial); NB 3 (OOB = the free in-training estimate); ch 04 (the single tree — high variance; MDI
   importance & its bias); module 00 — train/test split & leakage (NB 04), cross-validation (NB 10).
   **What you'll be able to do:** drive the real `RandomForestClassifier`; set `n_estimators` from the
   OOB curve; use `max_features` (the decorrelation dial, `'sqrt'` a robust default); say why a forest
   grows deep trees and tolerates it; read forest importances and their bias; tune honestly with
   `GridSearchCV` and report one sealed test.
2. (code) **Imports + seed + style + data** — breast_cancer (malignant = 1), 70/30 split (seed 0),
   `StratifiedKFold(5, shuffle, 0)`; print shapes & class counts; **print the RF defaults** (n=100,
   `max_features='sqrt'`, bootstrap True, oob_score False, max_depth None, gini).
3. (md) **Intuition — the estimator is exactly the two ideas we built.** NB 1–3 built the forest by
   hand; ch 04 gave the base tree and its variance. RF = many bootstrap trees (NB 1) + per-split
   feature subsampling (NB 2). Turn subsampling **off** (`max_features=None`) and it should reproduce
   plain bagging — let's confirm against our own hand-bag, then turn the knobs.
4. (code) **Parity: `RF(max_features=None)` == hand-bagging.** Hand-bag (B=200 unlimited trees,
   majority vote) **0.9357** == `RF(n_estimators=200, max_features=None)` **0.9357**; default
   `RF('sqrt')` **0.9415**. One-line moons recall: `RF(mf=None)` 0.9333 == NB 1's hand-bag, `RF(sqrt)`
   0.9000 starved 2 features.
5. (md) **Read the result.** Strip the randomness and the library is our hand-bag *exactly*; add
   per-split subsampling and, with 30 features to choose among, the trees decorrelate (NB 2) and the
   forest improves. From here we tune the real estimator.
6. (md) **Intuition — `n_estimators`: how many trees?** Each tree is a vote; more votes steady the
   average (NB 1's σ²/B). OOB (NB 3) reads the error for free as trees accumulate — watch it.
7. (code) **Fig A — OOB error & test error vs `n_estimators`.** Sweep n = 1,5,10,25,50,100,300,500
   (`oob_score=True`); sklearn **warns** at small n (shown). OOB err 0.271 → 0.040; test err noisy.
8. (md) **Read the figure (A).** Steep gains to ~25 trees, then **diminishing returns**: the error
   flattens (~0.04 OOB) and does **not** climb — more trees never overfit, they steady the estimate
   (cost = compute/memory). Pick "enough" (~100–300). The small-n warning is honest (NB 3: too few
   trees → some points have no OOB grader).
9. (md) **Intuition — `max_features`: the decorrelation dial (NB 2's ρ, now a knob).** At each split a
   tree may consider only `max_features` inputs. Small → more decorrelation (lower ρ); all → plain
   bagging. `'sqrt'` (≈ 5 of 30) is the classification default.
10. (code) **`max_features` sweep — OOB & CV-on-train** (n=200): mf = 1, 2, 'sqrt', 'log2', 10, 20, 30
    → print the small table (OOB 0.952–0.962, CV 0.947–0.962). No test-acc ranking.
11. (md) **Read the result.** Two honest points. (1) The dial's *mechanism* is decorrelation (NB 2's ρ
    rose 0.70 → 0.82 with mf); here accuracy is **flat across the whole range** — a forest is
    **forgiving**, `'sqrt'` is safe and rarely needs touching. (2) All-features (mf=30) = plain
    bagging = highest ρ = least decorrelation (the *weakest* setting in principle; here the gap is
    within noise). The knob bites most with many correlated features — decisively in NB 5.
12. (md) **Intuition — `max_depth` / `min_samples_leaf`: why a forest grows deep.** A lone deep tree
    overfits (ch 04's U-curve). A forest grows its trees **full depth on purpose** — each a low-bias,
    high-variance learner — and averaging cancels the variance (NB 1). Hence `max_depth=None`,
    `min_samples_leaf=1` by default.
13. (code) **Fig B — single tree vs forest test accuracy vs `max_depth`.** md = 1,2,3,5,8,None; both
    train → 1.000; single-tree test wobbles 0.86 → 0.918 → 0.906, forest test rises to 0.942 and
    plateaus. Print run-to-run std: single deep tree **0.0163** vs forest **0.0043** (≈4× steadier),
    forest mean 0.947 > tree 0.917.
14. (md) **Read the figure (B).** The lone tree overfits as it deepens — train hits 1.0 while test
    sags and wobbles. The forest grows the *same* full-depth trees yet its test accuracy climbs and
    **stays flat and high**: averaging cancels the variance that sinks a single deep tree (ch 04's
    weakness, fixed). That is why a forest tolerates — even wants — deep trees; `min_samples_leaf`
    behaves the same (leave it at 1).
15. (md) **The remaining knobs, named.** `bootstrap` (True = the bagging; `False` removes it — trees
    then differ only by the feature RNG); `class_weight` (re-weight rare classes for imbalance — used
    in NB 5); `n_jobs=-1` (trains trees across cores — the embarrassingly-parallel win, no accuracy
    change). A forest's defaults are already sensible.
16. (md) **Intuition — feature importance over a forest.** A single tree's MDI (ch 04) spikes on
    whichever feature it split first. A forest averages MDI over hundreds of trees on different
    bootstraps and feature subsets — the read **stabilises and spreads** across the useful group.
17. (code) **Fig C — single-tree MDI spike vs forest MDI spread.** Single tree: `mean concave points`
    **0.740** (one spike). Forest (n=300): peak **0.146** spread across `concave points / perimeter /
    radius / area`. Via `viz.plot_feature_importances` (the approved helper).
18. (md) **Read the figure (C).** The single tree put **0.74** on one feature — an artefact of which
    correlated twin it split first. The forest spreads the credit across the whole correlated group
    (peak 0.146): a more honest picture. **Caveat (Strobl 2007):** MDI favours continuous /
    high-cardinality features and **dilutes across correlated groups** — read at the group level,
    never as ground truth. **Permutation importance** (shuffle a column, measure the accuracy drop) is
    the honest cross-check; we put it to work in **NB 5**.
19. (md) **Intuition — honest tuning.** Module 00's discipline: search hyperparameters by
    **cross-validation on TRAIN**, then read the **sealed test once**. (OOB is a cheap shortcut, but
    for a defensible headline use CV + a held-out test.)
20. (code) **`GridSearchCV` on TRAIN → one sealed test.** Grid over `max_features` / `min_samples_leaf`
    / `max_depth`; best `{'log2', 1, None}` CV **0.957** → sealed test **0.947**; default RF CV 0.955 →
    test 0.942.
21. (md) **Read the result.** Tuning nudged CV **0.955 → 0.957** and test **0.942 → 0.947** — barely.
    The default forest was already strong: that is RF's real selling point — **strong with almost no
    tuning** (and no scaling — confirm in *Your turn*). The grid kept deep trees (`max_depth=None`,
    `min_samples_leaf=1`): it agrees a forest wants deep trees.
22. (md) **Your turn** (3 tiered) — *easy:* re-run the `n_estimators` sweep and name the smallest
    forest whose OOB has stopped improving; *medium:* set `max_features` to the all-features (bagging)
    end vs `'sqrt'` and explain via NB 2's ρ which decorrelates more (check OOB); *harder:* standardize
    `X`, confirm the forest's predictions are **identical** to raw (scale-invariance, ch 04), and
    explain why a tree-based model needs no scaling.
23. (md) **What you built** — drove `RandomForestClassifier`; confirmed `RF(max_features=None)` ==
    hand-bagging; read the `n_estimators` diminishing-returns curve (never overfits); `max_features`
    the decorrelation dial (`'sqrt'` robust, forgiving); why RF grows deep and tolerates it; forest MDI
    **spreads** (vs the single-tree spike) with the bias caveat + permutation named; honest
    `GridSearchCV` → sealed test (barely beats the default). **Vocabulary:** `n_estimators` ·
    `max_features` · `max_depth`/`min_samples_leaf` · `bootstrap`/`class_weight`/`n_jobs` · MDI vs
    permutation importance · OOB-guided selection.
24. (md) **Going further (optional) + References.** Going further: `ExtraTrees` (extra randomness —
    random thresholds); `warm_start` to grow a forest incrementally; importance the honest way
    (permutation, NB 5). **References:** Breiman 2001 (Random Forests, DOI 10.1023/A:1010933404324);
    Breiman 1996 (bagging, DOI 10.1007/BF00058655); Ho 1998 (random subspaces, DOI 10.1109/34.709601);
    Strobl 2007 (MDI bias, DOI 10.1186/1471-2105-8-25); ESL §15 (DOI 10.1007/978-0-387-84858-7);
    ISLR §8.2 (DOI 10.1007/978-1-0716-1418-1). `Previous: 03 — Out-of-bag estimation` · `Next: 05 — A
    demanding case: covtype`.

## Honest scoping (stated in the notebook)

- **`max_features` barely moves accuracy here** — a forest is forgiving; the dial's *mechanism* is
  decorrelation (NB 2's ρ), and its decisive accuracy payoff is NB 5, not this near-flat sweep.
- **Tuning barely beats the default** — RF's value is *strong with little effort*, not squeezing the
  last point. Stated with the numbers (CV 0.955 → 0.957; test 0.942 → 0.947).
- **MDI is biased and dilutes across correlated groups** (Strobl 2007); permutation importance is only
  *named* here — its honest reading is NB 5.
- **More trees never overfit** but cost compute/memory — pick "enough", not "as many as possible".
- **Parity is an accuracy match at B=200**, not a bit-exact per-tree clone (`RF(n=1)` ≠ a lone tree —
  NB 1's RNG-tie-breaking caveat carries over).
- **Scale-invariance inherited** from ch 04 (raw == standardized predictions).

## Verification

Build via `uv run python - < <scratchpad>/build_ch06_nb4.py` (stdin). Re-measure at build: parity
hand-bag 0.9357 == RF(mf=None) 0.9357, RF(sqrt) 0.9415 (moons mf=None 0.9333 / sqrt 0.9000);
n_estimators OOB-err 0.271→0.040 (warns ≤10, no overfit); max_features OOB/CV flat 0.947–0.962;
max_depth single-tree 0.86–0.918 wobble vs forest 0.918→0.942 plateau, std 0.0163 vs 0.0043; MDI
forest peak 0.146 (mean concave points) vs single-tree 0.740; GridSearch best {'log2',1,None} CV 0.957
→ test 0.947 (default 0.955/0.942); scale-invariance raw==std 1.000. Runs top-to-bottom (nbconvert
from project cwd to scratchpad; tracked file **output-free**); **banned-word scan over the JSON real
cell text** = 0; `check_no_hardcoded_hex` passes; `gen_llms_txt` re-run; `pytest` 20 (the new
`viz.plot_feature_importances` smoke test); `ruff`/`black` clean; `course_map.md` §06 + `common_errors`
updated. Both reviewers PASS (no BLOCK); Rémy validates visually; commit `feat(06_random_forest):
notebook 04 — the estimator & its parameters`; merge `notebook → chapter`.
