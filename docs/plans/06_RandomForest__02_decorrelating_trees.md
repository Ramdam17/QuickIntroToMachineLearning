# Notebook plan — 06_RandomForest / 02_decorrelating_trees

> Status: **APPROVED** (2026-06-23, by Rémy). Built next; both reviewers (`@ml-expert-reviewer` +
> `@pedagogy-reviewer`) gate the **built** notebook (no gate at the plan stage). Anchors re-measured
> at build on sklearn **1.9.0**, every random forest `random_state`-pinned.

## Context

NB **2 of 5**. One concept: **random feature subsampling decorrelates the trees** — the "random" in
random forest. NB 1 built bagging and saw variance fall *roughly* like σ²/B; the hedge was
"roughly", because bagged trees are **correlated** (and on moons the default `RF(sqrt)` even scored
*below* bagging). This notebook names the correlation **ρ**, derives the exact variance law
**Var = ρσ² + (1−ρ)σ²/B** from scratch (exposing the **ρσ² floor** bagging cannot pass), and shows
that letting each split choose among only a random subset of features **lowers ρ**, lifting the
ensemble even though the individual trees are no better. We work on **breast_cancer** (30 correlated
features), because feature subsampling can only bite when there are many features to subsample — which
also resolves NB 1's moons puzzle.

## Anchors (measured at plan time, sklearn **1.9.0**; `random_state` pinned; re-measured at build)

breast_cancer (`datasets.load_breast_cancer`, 569 × 30, **malignant = 1**), 70/30 stratified split
(seed 0), `StratifiedKFold(5, shuffle, random_state=0)`. ρ = mean pairwise Pearson correlation of the
trees' test-set predictions.

- **Bagging (`max_features=None`, all 30) vs RF (`max_features='sqrt'` ≈ 5), n=200, seed 0:**
  CV-on-train **0.9472 → 0.9572** (the robust headline); ρ **0.8221 → 0.7968**; **individual trees
  essentially equal** (mean acc 0.9096 vs 0.9092). The ensemble gain is **pure decorrelation**, not
  stronger members. (Test 0.9357 → 0.9415; OOB 0.9523 → 0.9548 — secondary, ±0.01 seed band.)
- **`max_features` is the decorrelation dial (n=120, seed 0):** ρ rises monotonically with it — mf
  1 / 2 / 3 / 5 / 7 / 10 / 15 / 20 / 30 → ρ **0.702 / 0.759 / 0.779 / 0.799 / 0.800 / 0.811 / 0.819 /
  0.821 / 0.820**. CV is best at the **most decorrelated** end (mf=1 → **0.962**), where individual
  trees are the **weakest** (0.887) — decorrelation beats individual strength. The all-features end
  (mf=30 = bagging) sits at high ρ (0.82).
- **The variance law, Monte-Carlo-verified:** for B unit-variance variables with pairwise correlation
  ρ, Var(mean) = ρ + (1−ρ)/B — empirical vs formula: ρ=0 → 0.010 vs 0.010 at B=100 (→ σ²/B vanishes);
  ρ=0.8 → 0.84 (B=5) → 0.80 (B=100) vs formula 0.84 → 0.802 (→ the **ρσ² floor**, here 0.8).
- **NB 1's moons puzzle resolved:** on moons (2 features) `RF(sqrt)`→1 feature/split **0.900** <
  bagging **0.933** (subsampling starves 2-feature trees); on breast_cancer (30 features) `RF(sqrt)`
  **0.957** > bagging **0.947**. Feature subsampling pays only with many features.

## Library / figures

- **No `src/` change** (`pytest` stays 19). Reuse `viz.use_course_style`; `ml_course.colors`. Two
  one-off charter figures: **A** ρ vs `max_features` (rising, sqrt and bagging-end marked); **B**
  ensemble CV vs mean-individual-tree accuracy across `max_features`. Sklearn: `RandomForestClassifier`,
  `cross_val_score`, `StratifiedKFold`, `train_test_split`, `accuracy_score`; numpy for ρ and the
  variance-law Monte-Carlo.

## Cell-by-cell (~22 cells)

1. (md) **Header** — `# 02 — The "random" in the forest: decorrelating the trees`; *notebook 2 of 5*;
   warm welcome. **Prerequisites:** NB 1 (bagging; the σ²/B intuition and its "roughly"); ch 04 (a deep
   tree, scale-invariant); module 00 — cross-validation (NB 10). **What you'll be able to do:** measure
   the pairwise correlation ρ between trees; derive Var = ρσ² + (1−ρ)σ²/B and read its floor; see
   feature subsampling lower ρ and lift the ensemble; use `max_features` as the decorrelation dial;
   explain why a forest beats a plain bag (and why that needs many features).
2. (code) **Imports + seed + style + data** — breast_cancer, `y = (target == 0)` (malignant = 1),
   70/30 stratified split (seed 0), `StratifiedKFold(5, shuffle, random_state=0)`; print shapes,
   `sqrt(30) ≈ 5.5`.
3. (md) **Recap / footing** — NB 1: bagging averaged trees and variance fell *roughly* like σ²/B; two
   loose ends were left. (i) "roughly" — because the trees are **correlated**. (ii) on moons the default
   `RF(sqrt)` scored *below* our bag. Both come from the same missing idea, which we build here.
4. (md) **Intuition — why bagging has a ceiling.** Bagged trees see overlapping data and all latch onto
   the same few strong features, so they tend to make the **same** mistakes. Averaging cancels mistakes
   only when they *differ*; correlated mistakes survive the vote. Name it: **ρ**, the average pairwise
   correlation between trees' predictions.
5. (code) **Measure ρ for bagging.** Fit bagging (`max_features=None`, n=200); collect each tree's test
   predictions; compute mean pairwise correlation → **ρ ≈ 0.82**; individual-tree mean acc ≈ 0.91;
   ensemble CV ≈ 0.947.
6. (md) **Read the result.** ρ ≈ 0.82 is high: the trees largely agree because they keep splitting on
   the same dominant measurements, so they err together — and averaging correlated errors buys less
   than the σ²/B of NB 1 promised.
7. (md) **Intuition — the variance law, from scratch.** For B predictors each of variance σ² and
   pairwise correlation ρ, expand the variance of their mean: B variance terms + B(B−1) covariance
   terms → **Var(mean) = ρσ² + (1−ρ)σ²/B**. As B→∞ the second term vanishes and the variance hits a
   **floor of ρσ²**. Two levers: more trees (B↑, NB 1) kills the second term; **lower correlation
   (ρ↓) lowers the floor** — the only way past it.
8. (code) **Verify the law (Monte-Carlo).** Simulate B unit-variance variables with pairwise
   correlation ρ; compare empirical Var(mean) to ρ + (1−ρ)/B for ρ = 0, 0.5, 0.8 and B = 5, 25, 100
   → matches; at ρ=0.8 the variance sticks near 0.8 (the floor) however large B.
9. (md) **Read it.** The algebra holds: with ρ = 0 the variance melts to σ²/B (NB 1's ideal), but with
   ρ = 0.8 it is stuck near 0.8 no matter how many trees. Bagging alone (ρ ≈ 0.82 here) is parked just
   above its floor. To go lower we must make the trees **disagree** more.
10. (md) **Intuition — feature subsampling (the "random").** At **each split**, let the tree pick the
    best feature from only a **random subset** (`max_features`) instead of all of them. Forced down
    different features, the trees grow different shapes and make different mistakes — **lower ρ**, lower
    floor. This single twist turns a bag of trees into a **random forest** (Ho 1998; Breiman 2001).
11. (code) **Measure RF (`max_features='sqrt'`).** Same as cell 5 but `max_features='sqrt'` → ρ
    **0.82 → 0.80**; **individual trees essentially equal** (0.909 vs 0.909); ensemble CV
    **0.947 → 0.957**.
12. (md) **Read the result — the gem.** The individual trees are *no better* (0.909 either way), yet
    the ensemble improves — **entirely from decorrelation**. The committee gains because its members
    disagree more, not because any member got smarter.
13. (code) **Figure A — ρ vs `max_features`.** Sweep mf = 1 … 30; plot ρ (rising 0.70 → 0.82); mark
    `'sqrt'` (≈5) and the all-features end (= bagging).
14. (md) **Read the figure (A).** The more features each split may see, the more alike the trees become
    (ρ climbs) and the less averaging can help; the all-features end — plain bagging — sits at the
    high-ρ top. `max_features` is the **decorrelation dial**: turn it down to push ρ (and the variance
    floor) down.
15. (code) **Figure B — committee vs members.** Across `max_features`, plot ensemble CV accuracy and
    the mean individual-tree accuracy. At low mf the members are **weaker** (mf=1: 0.887) yet the
    ensemble is **strongest** (CV 0.962).
16. (md) **Read the figure (B).** The striking part: the *weakest* individual trees (mf = 1) make the
    *strongest* committee, because they are the most decorrelated — decorrelation can outweigh
    individual strength. (Honest caveat: push mf too low and trees can get too weak to help; **`'sqrt'`
    is the robust default**, not always the literal optimum.)
17. (md) **Intuition — resolve NB 1's moons puzzle.** On moons there are only **2** features, so
    `'sqrt'` → 1 feature per split starves each tree with nothing to gain from decorrelation — hence
    `RF(sqrt)` 0.900 < bagging 0.933 there. On breast_cancer's **30** features, `'sqrt'` → 5
    decorrelates productively (0.957 > 0.947). Feature subsampling **pays only with many features**.
18. (code) **The contrast, side by side.** Print moons (`RF(sqrt)` 0.900 vs bagging 0.933) next to
    breast_cancer (`RF(sqrt)` CV 0.957 vs bagging 0.947).
19. (md) **You have built the random forest.** Bagging (NB 1) + feature subsampling (NB 2) = a random
    forest. `'sqrt'` is the robust default for `max_features`. Next (NB 3): what the bootstrap hands us
    for free — an honest accuracy estimate with no test set set aside (out-of-bag).
20. (md) **Your turn** (3 tiered) — *easy:* given two trees' 0/1 predictions on five points, compute
    their agreement (and sense of ρ); *medium:* from the derived law with ρ = 0.8, σ² = 1, give the
    variance at B = 10 vs B = 1000, and the floor; *harder:* sweep `max_features`, plot ρ and ensemble
    accuracy together, and pick the setting you would ship — and say why it is not always mf = 1.
21. (md) **What you built** — the pairwise correlation **ρ**; the **variance law** Var = ρσ² +
    (1−ρ)σ²/B and its **floor**; **feature subsampling** as the decorrelation lever; the **gem**
    (ensemble gain from decorrelation, not stronger trees); **`max_features`** the dial; NB 1's moons
    puzzle resolved. **Vocabulary:** pairwise correlation ρ · decorrelation · feature subsampling ·
    `max_features` · the variance floor ρσ² · random forest.
22. (md) **Going further (optional)** — even more randomness: **Extremely Randomized Trees**
    (`ExtraTrees`) randomize the split *threshold* too, pushing ρ lower still; and `max_features` has
    its own bias–variance trade (too low → trees too weak/biased). **References:** Ho TK (1998), random
    subspaces (DOI 10.1109/34.709601); Breiman L (2001), Random Forests (DOI 10.1023/A:1010933404324);
    ESL §15.2 (the ρσ² + (1−ρ)σ²/B law); ISLR §8.2. `Previous: 01 — averaging cuts variance` ·
    `Next: 03 — out-of-bag estimation`.

## Honest scoping (stated in the notebook)

- **ρ is the floor-setter** — bagging parks just above ρσ²; only ρ↓ (decorrelation) lowers it. Stated
  via the derived law + Monte-Carlo.
- **Feature subsampling needs many features** — on 2-feature moons it *hurts* (0.900 < 0.933); the
  moons puzzle is resolved, not hidden.
- **`'sqrt'` is a robust default, not the literal optimum** — here mf=1 wins on CV, but pushing mf too
  low risks too-weak trees; choose by CV (NB 4).
- **Individual-tree strength is not the goal** — the ensemble can improve while members stay equal (or
  even weaken); decorrelation is what counts.

## Verification

Build via `uv run python - < <scratchpad>/build_ch06_nb2.py` (stdin). Re-measure at build: bagging vs
RF(sqrt) CV 0.947→0.957, ρ 0.822→0.797, indiv 0.910≈0.909; max_features sweep ρ 0.70→0.82 monotone,
CV best at mf=1 (0.962); variance-law Monte-Carlo (ρ=0 → σ²/B, ρ=0.8 → floor 0.8); moons RF(sqrt)
0.900 < bagging 0.933 vs breast_cancer RF(sqrt) 0.957 > bagging 0.947. Runs top-to-bottom (nbconvert
from project cwd to scratchpad; tracked file **output-free**); **banned-word scan over the JSON real
cell text** = 0; `check_no_hardcoded_hex` passes; `gen_llms_txt` re-run; `pytest` 19 (no `src/`
change); `ruff` clean. Both reviewers PASS (no BLOCK); Rémy validates visually; commit
`feat(06_random_forest): notebook 02 — the "random" in the forest: decorrelating the trees`; merge
`notebook → chapter`.
