# Notebook plan — 04_DecisionTree / 04_estimator_and_parameters

> Status: **APPROVED** (2026-06-21, by Rémy; notebook plan validated by Rémy alone — the two reviewers
> gate the *built* notebook). Numbers re-measured at build. Drives the NB-4 build in `docs/WORKFLOW.md`.

## Context

NB **4 of 5** — the **integrative** notebook (per the per-method arc): the real
`sklearn.tree.DecisionTreeClassifier` and the dials that control it. NB 1–3 built the mechanism by
hand (impurity, growth, pruning); here we meet the estimator, confirm it is the same mechanism
(parity), walk its **knobs**, and surface the one property that drives the rest of the course — a
single tree is **high-variance**. We also show what trees handle that KNN and logistic regression
needed machinery for (scale-invariance, multiclass, missing values), and read **feature importance**
with its caveat. **Soft ~24-cell ceiling**: the headline is the variance section, not knob-completeness.

## Anchors (measured at plan time, sklearn **1.9.0**; re-measured at build)

- **Parity:** the by-hand depth-2 tree (NB 2) == `DecisionTreeClassifier(max_depth=2)` on penguins,
  train **0.9964** (the mechanism, confirmed once more).
- **Four shown dials** (on `make_moons(300, 0.30, 0)`, 210/90 seed 0; CV = `StratifiedKFold(5,
  shuffle, seed 0)` on train):
  - **`min_samples_leaf`** 1 / 5 / 20 / 50 → leaves 23 / 16 / 7 / 3, test 0.878 / **0.933** / 0.800 /
    0.744 (a pre-pruning dial; floors the leaf size).
  - **`criterion`** gini / entropy / log_loss → default-depth CV **0.9095 / 0.9143 / 0.9143** (a
    near-tie; entropy edges Gini *here* but it is noise-level — Gini is the default for being **cheaper
    to compute**, no logarithm).
  - **`max_depth`** and **`ccp_alpha`** — recapped from NB 3 (the complexity dial / cost-complexity
    pruning), named as knobs here, not re-taught.
- **Two named dials (one line each, no figure):** **`max_features`** — random split-feature subsets, a
  deliberate **foreshadow of random forests** (single-tree CV drops None **0.9095** → 1 **0.8857**, but
  that injected randomness is exactly what an ensemble averages over); **`class_weight`** — reweight
  under class imbalance (used lightly in NB 5).
- **Instability & variance — THE headline:** 20 bootstrap resamples of the moons train set, under the
  **pinned recipe** (`np.random.default_rng(0)`, 20 resamples, `random_state=0` per tree,
  decision-region disagreement on a 150×150 grid) → **full trees std 0.032 / pairwise disagreement
  6.3 %**, depth-3 **0.022 / 5.6 %**. Two trees fit on two resamples draw **visibly different
  boundaries** — the weakness ensembles fix (the explicit bridge to ch 06).
- **Trees' native strengths:** **scale-invariance** — `DecisionTreeClassifier` on **raw** penguins vs
  **standardized** penguins gives **identical predictions** (verified True; train 1.000 == 1.000, CV
  0.9818 == 0.9818): the ch 01 scale trap does not arise, and the ch 03 `StandardScaler` step is
  unnecessary. **Native multiclass + missing values:** `penguins_full` (3 species, **2 numeric-NaN
  rows kept**) trees to **CV 0.9535** untouched — no OvR/softmax, no imputation. Honest limit: sklearn
  still needs **numeric encoding for string categoricals**.
- **Gini feature importance** (on `penguins_full`, 4 numeric features): `flipper_length_mm` **0.549**,
  `bill_length_mm` **0.363**, `bill_depth_mm` 0.062, `body_mass_g` 0.026 — size dominates, sensibly.
  Stated **caveat**: Gini importance is **biased toward continuous / high-cardinality features**
  (Strobl 2007); **permutation importance** is the honest cross-check (named here, shown in NB 5).
- **Honest tuning:** `GridSearchCV` over `max_depth` / `min_samples_leaf` / `criterion` / `ccp_alpha`
  on the moons **train** set → best `{max_depth 6, criterion gini, min_samples_leaf 1, ccp_alpha 0}`,
  CV **0.919**, **sealed test 0.889** (consistent with NB 3's CV-best depth 6).

## Library / figures

- **No `src/` change** (pytest stays 17). Reuse `viz.use_course_style`, **`viz.plot_decision_boundary`**
  (Figs A, B), `ml_course.colors`. `DecisionTreeClassifier`, `GridSearchCV`, `StratifiedKFold`,
  `cross_val_score`, `make_moons`, `train_test_split` from sklearn; `datasets.penguins_xy` /
  `load_penguins_full`.
- **Three figures:** **A** `min_samples_leaf` = 1 (jagged, overfit) vs 5 (smooth) boundaries on moons —
  a pre-pruning dial, shown; **B (headline)** **two bootstrap trees' boundaries side by side** — visibly
  different (the variance); **C** the **Gini feature-importance bar** on `penguins_full`. Each + "Read
  the figure". Knobs not given figures (criterion / max_features / scale-invariance / tuning) are shown
  via printed numbers, to hold the cell budget and avoid re-drawing NB 3's boundaries.

## Cell-by-cell (~23 cells; "Read the figure" after every figure)

1. (md) **Header** — `# 04 — The estimator & its parameters`; *notebook 4 of 5*; warm welcome.
   **Prerequisites:** NB 1–3 (impurity, growth, overfitting & pruning); module 00 — cross-validation
   (NB 10), the train/test split (NB 04); chapter 01 (the scale trap, NB 2) and chapter 03
   (standardization) — the contrast trees escape. **What you'll be able to do:** drive
   `DecisionTreeClassifier`'s main dials; see why a single tree is high-variance; use trees' native
   handling of scale, multiclass, and missing values; read feature importance (and its caveat); tune
   honestly with `GridSearchCV`.
2. (code) **Imports + seed + style + data** — sklearn pieces; `np.random.seed(0)`; `use_course_style()`;
   penguins (`X, y_bin`), moons (`Xtr/Xte/ytr/yte`, seed 0 split), `penguins_full` numeric (3 species).
3. (md) **Recap — from by-hand to the estimator.** NB 1–3 built impurity, greedy growth, and pruning
   by hand. `DecisionTreeClassifier` is that mechanism wrapped in an API; this notebook walks its dials
   and surfaces the property — high variance — that the rest of the course is built to fix.
4. (code) **Parity** — by-hand depth-2 (NB 2's four-leaf rule) `==`
   `DecisionTreeClassifier(max_depth=2)` on penguins, train 0.9964 (identical predictions).
5. (md) **Read the result** — same splits, same predictions: the library is the mechanism we built,
   not a black box. Now the dials.
6. (md) **Intuition — the complexity dials.** `max_depth` and `ccp_alpha` (NB 3's pre- and post-prune
   handles) cap complexity; **`min_samples_leaf`** is a third — refuse any leaf smaller than *m* points;
   **`criterion`** chooses the impurity measure. All move the same bias/variance trade.
7. (code) **Fig A — `min_samples_leaf` boundaries** (`plot_decision_boundary` ×2): `min_samples_leaf` = 1
   (jagged, 23 leaves, test 0.878) vs 5 (smooth, 16 leaves, test 0.933); print the 1 / 5 / 20 / 50 table.
8. (md) **Read the figure (A)** — flooring the leaf size forbids the tree from carving a box around one
   point; at 5 the boundary smooths and test rises to 0.933, at 20 it starts to underfit (0.800). Like
   `max_depth` and `ccp_alpha`, it is a complexity handle, tuned by CV.
9. (code) **`criterion` + the two named dials** — `criterion` gini / entropy / log_loss CV
   (0.910 / 0.914 / 0.914); `max_features` None vs 1 CV (0.910 vs 0.886); one print line each.
10. (md) **Read the result** — Gini, entropy and log-loss tie within noise; Gini is the default because
    it skips the logarithm (cheaper), not because it scores higher. **`max_features`** (try only a random
    subset of features per split) makes *one* tree a touch worse — but that injected randomness is the
    seed of **random forests** (ch 06). **`class_weight`** reweights classes under imbalance (NB 5).
    `max_depth` and `ccp_alpha` are NB 3's dials.
11. (md) **Intuition — a single tree is high-variance.** Change the training data a little and the whole
    tree can change — different splits, a different boundary. This instability is the decision tree's
    defining weakness, and the reason the next chapters average or add trees.
12. (code) **Fig B — two bootstrap trees** (`plot_decision_boundary` ×2 on two `default_rng(0)`
    resamples of the moons train set); print the 20-bootstrap numbers (full std 0.032 / disagreement
    6.3 %; depth-3 0.022 / 5.6 %).
13. (md) **Read the figure (B)** — same data-generating process, two resamples, two visibly different
    boundaries; across 20 resamples the unpruned trees disagree on 6.3 % of the plane. A shallow tree is
    steadier but weaker. **The fix is to average many trees (bagging → random forests, ch 06) or add
    them in sequence (boosting, ch 07–10)** — the whole back half of the course.
14. (md) **Intuition — what trees handle natively.** Three things KNN and logistic regression needed
    extra machinery for come free with a tree: **scale**, **more than two classes**, and **missing
    values**.
15. (code) **Native strengths** — fit on **raw** vs **standardized** penguins → identical predictions
    (`True`), identical CV (0.9818); then `penguins_full` (3 species, 2 numeric-NaN rows kept) → 3-class
    CV 0.9535 with no scaling, no one-vs-rest, no imputation.
16. (md) **Read the result** — a split asks "is x ≤ t?", so rescaling only moves the threshold: the
    ch 01 scale trap and the ch 03 `StandardScaler` step **do not apply** to a tree. Multiclass and
    `NaN` are handled directly (sklearn ≥ 1.3 routes missing values down the split). The one caveat:
    sklearn still wants **numbers** — string categories must be encoded first (the theory handles them;
    the implementation does not).
17. (code) **Fig C — Gini feature importance** (`penguins_full`, 4 numeric features): a sorted bar
    (`flipper_length_mm` 0.55, `bill_length_mm` 0.36, `bill_depth_mm` 0.06, `body_mass_g` 0.03).
18. (md) **Read the figure (C)** — importance = how much each feature cut impurity across the tree;
    size measurements dominate, sensibly. **Caveat:** Gini importance is **biased toward continuous /
    high-cardinality features** (Strobl 2007) — a feature with many distinct values gets more chances to
    look useful. **Permutation importance** is the honest cross-check (we use it in NB 5).
19. (code) **Honest tuning** — `GridSearchCV` over `max_depth` / `min_samples_leaf` / `criterion` /
    `ccp_alpha` on the moons **train** set; print best params (`max_depth 6`, CV 0.919) and the **sealed
    test** (0.889).
20. (md) **Read the result** — CV on the training set picks the dials; the test set is read once. The
    grid lands on `max_depth 6` — the same depth NB 3 chose — for a sealed-test 0.889. Honest tuning, the
    module-00 NB 10 discipline, now across several knobs at once.
21. (md) **Your turn** (3 tiered) — *easy*: set `min_samples_leaf=10` on moons and report leaves and
    test; *medium*: refit penguins with the two features in different units (e.g. flipper in cm) and
    confirm the tree is unchanged (scale-invariance); *harder*: fit `max_features=1` five times with
    different `random_state` and describe how much the boundary moves — the seed of a random forest.
22. (md) **What you built** — drove `DecisionTreeClassifier`'s dials (`max_depth`, `min_samples_leaf`,
    `ccp_alpha`, `criterion`; `max_features` / `class_weight` named); saw a single tree's **high
    variance** (the ensemble motivation); used trees' **native** handling of scale, multiclass and
    missing values; read **Gini importance** and its bias; **tuned with `GridSearchCV`**. **Vocabulary:**
    `min_samples_leaf` · `criterion` · `max_features` · `class_weight` · variance / instability ·
    scale-invariance · feature importance · permutation importance · `GridSearchCV`.
23. (md) **Going further (optional) + References** — `max_features` & the bias–variance of randomized
    trees (the RF preview, ch 06); permutation vs impurity importance; trees handle categoricals in
    theory (CART) but sklearn needs encoding. **References:** Breiman et al. 1984 (CART); Strobl et al.
    2007 (importance bias, DOI 10.1186/1471-2105-8-25); Breiman 1996 (bagging, DOI 10.1007/BF00058655);
    ESL §9.2 (DOI 10.1007/978-0-387-84858-7); ISLR §8.1 (DOI 10.1007/978-1-0716-1418-1). `Previous: 03 —
    Overfitting & pruning` · `Next: 05 — A demanding case: breast cancer`.

## Honest scoping (stated in the notebook)

- **Variance is the headline**, framed as the precise motivation for ensembles (ch 06+), not a vague
  flaw; the knobs serve that story rather than knob-completeness.
- **Gini importance is biased** (Strobl 2007) — stated, with permutation importance named as the honest
  cross-check (shown in NB 5), not silently trusted.
- **Scale-invariance is exact** (identical predictions raw vs standardized), and the categorical-encoding
  limit is stated (implementation, not CART theory).
- **`max_features` makes one tree worse** here — said plainly; its value is as the randomness an
  ensemble averages (ch 06), not as a single-tree win.
- **Tuning is CV-on-train, test read once** — the grid's sealed test (0.889) reuses NB 3's discipline.
- `max_depth` / `ccp_alpha` are **recapped, not re-taught** (NB 3); `min_samples_leaf` is the new dial.

## Verification

Build via `uv run python - < <scratchpad>/build_ch04_nb4.py` (stdin). Re-measure at build: parity train
0.9964; `min_samples_leaf` 1/5/20/50 → 0.878/0.933/0.800/0.744; criterion 0.910/0.914/0.914;
`max_features` None 0.910 / 1 0.886; bootstrap variance full 0.032 / 6.3 %, depth-3 0.022 / 5.6 %;
scale-invariance identical (True), CV 0.9818==0.9818; penguins_full 3-class CV 0.9535 (2 NaN rows);
importances flipper 0.55 / bill_length 0.36 / bill_depth 0.06 / body_mass 0.03; GridSearchCV best
max_depth 6 / CV 0.919 / test 0.889. Runs top-to-bottom (nbconvert to scratchpad; tracked file
**output-free**, `--clear-output --inplace`); **banned-word scan over the JSON real text** = 0;
`check_no_hardcoded_hex` passes; `gen_llms_txt` re-run; `pytest` 17 (no `src/` change); `ruff` clean.
Both reviewers PASS (no BLOCK); Rémy validates visually; commit `feat(04_decision_tree): notebook 04 —
the estimator & its parameters`; merge `notebook → chapter`.
