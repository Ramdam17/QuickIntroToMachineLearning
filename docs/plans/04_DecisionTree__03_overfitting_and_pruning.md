# Notebook plan — 04_DecisionTree / 03_overfitting_and_pruning

> Status: **APPROVED** (2026-06-21, by Rémy; notebook plan validated by Rémy alone — the two reviewers
> gate the *built* notebook). Numbers re-measured at build. Drives the NB-3 build in `docs/WORKFLOW.md`.

## Context

NB **3 of 5** — one concept: **overfitting, and pruning it back; depth is the complexity dial**. NB 2
ended on a hint — the *unpruned* tree cross-validated slightly worse than the depth-2 one. Here we make
that precise. We move to **`make_moons`** (the curved, noisy set logistic regression could *not* fit —
its first appearance in this chapter) so depth visibly under- and over-fits, replay module 00 NB 09's
**train/test U-curve** with depth on the x-axis, then learn to **prune**: pre-prune with `max_depth` and
post-prune with **cost-complexity** (`ccp_alpha`), choosing the dial honestly by **cross-validation**
(module 00 NB 10). `min_samples_leaf` and the full estimator API are NB 4.

## Anchors (measured at plan time, sklearn **1.9.0**; re-measured at build)

- **`make_moons(n_samples=300, noise=0.30, random_state=0)`**, 210/90 **stratified** split (seed 0);
  CV = `StratifiedKFold(5, shuffle=True, random_state=0)` on the **train** set.
- **Depth dial (train / test / CV):** depth 1 → 0.833 / 0.744 / 0.795 (2 leaves, **underfit**); depth 6
  → 0.981 / **0.889** / **0.919** (17 leaves, **CV-best**); depth 7 → 0.990 / **0.900** / 0.910 (the
  raw test peak); depth ≥ 8 → **1.000** / 0.878 / ~0.905 (23 leaves, **memorizes**; the unlimited tree
  is depth 8). **Training error falls monotonically to 0; test error is U-shaped (min at depth 6–7);
  the gap widens** — the overfitting signature.
- **CV picks depth 6 without touching test** (CV 0.919); its **sealed-test** accuracy is **0.889**
  (note: the raw test optimum sits at depth 7 / 0.900 — but we never tune on test).
- **Cost-complexity pruning** (`cost_complexity_pruning_path` on train; CV over the alphas): the
  unpruned tree (α=0) has **23 leaves / test 0.878**; **CV-best α ≈ 0.0087 → 8 leaves / test 0.900**
  (CV 0.914); α=0.020 → 5 leaves / 0.844; α=0.063 → 2 leaves / 0.744. **Pruning the memorizing tree
  *raises* held-out accuracy** (0.878 → 0.900) while cutting 23 leaves to 8.
- Two routes, one idea: **pre-prune** (cap `max_depth` at 6) and **post-prune** (grow full, cut by
  `ccp_alpha` to 8 leaves) both land near the sweet spot; both chosen by CV.

## Library / figures

- **No `src/` change** (pytest stays 17). Reuse `viz.use_course_style`, **`viz.plot_decision_boundary`**
  (Figs A, D — the boxes), **`viz.plot_train_test_curve`** (Fig B — the U-curve; pass `ylabel="error"`),
  `ml_course.colors`. `DecisionTreeClassifier`, `cost_complexity_pruning_path`, `StratifiedKFold`,
  `cross_val_score`, `make_moons`, `train_test_split` from sklearn.
- **Four figures:** **A** three decision boundaries (depth 1 / 6 / unlimited) — underfit → good →
  jagged; **B** the **train/test error U-curve vs depth** with a vertical line at the CV-best depth;
  **C** the **cost-complexity pruning path** (test accuracy and #leaves vs `ccp_alpha`, CV-best marked);
  **D** the **unpruned (23-leaf) vs CV-pruned (8-leaf) boundary** side by side — the jagged boundary
  smoothed. Each + "Read the figure".

## Cell-by-cell (~21 cells; "Read the figure" after every figure)

1. (md) **Header** — `# 03 — Overfitting & pruning: depth is the complexity dial`; *notebook 3 of 5*;
   warm welcome. **Prerequisites:** NB 1–2 (impurity, growth, the depth-2 tree & its CV hint); module
   00 — over/under-fitting & the generalization gap on `make_moons` (NB 09), cross-validation (NB 10),
   the train/test split (NB 04). **What you'll be able to do:** recognise a tree under- and
   over-fitting from its boundary and its train/test curve; read depth as the complexity dial; prune a
   tree (pre- and post-) and choose the strength by cross-validation.
2. (code) **Imports + seed + style + data** — sklearn pieces; `np.random.seed(0)`; `use_course_style()`;
   `Xm, ym = make_moons(n_samples=300, noise=0.30, random_state=0)`; 210/90 stratified split (seed 0).
3. (md) **Recap & the move to moons.** NB 2 left a hint: the unpruned tree cross-validated a touch
   *worse* than the depth-2 one. To see why clearly we need data a tree can over-fit — the two
   interleaving **moons** (noise 0.30), the curved set logistic regression could not fit. A tree
   carves it; the question is *how hard to carve*.
4. (md) **Intuition — depth is the complexity dial.** Shallow = a few big boxes (may **underfit**, too
   coarse to follow the curve); very deep = a box around nearly every point (**overfits**, chasing
   noise). Somewhere between is right.
5. (code) **Fig A — three boundaries** (`plot_decision_boundary` ×3): `max_depth` 1, 6, and unlimited.
6. (md) **Read the figure (A)** — depth 1 is one stripe (far too coarse); depth 6 follows the two
   crescents; the unlimited tree grows little islands around stray points — memorizing the noise.
7. (md) **Intuition — the generalization gap returns (module 00 NB 09).** As depth grows, **training**
   error keeps falling (a deep tree can memorize to 0), but **test** error falls, bottoms out, then
   rises. The growing gap between them is overfitting, now drawn for a tree.
8. (code) **Fig B — train/test error vs depth** (`plot_train_test_curve`, `ylabel="error"`) over depth
   1…12, with a vertical line at the **CV-best depth**; print the per-depth train / test / CV table.
9. (md) **Read the figure (B)** — training error slides to 0 by depth 8; test error dips to its low
   around depth 6–7 then creeps back up; the widening gap is the overfitting. The U is shallow here
   (clean data), but the shape and the train→0 collapse are unmistakable.
10. (md) **Intuition — choose the dial by CV, never by the test set (module 00 NB 10).** The test curve
    above is for *understanding*; we must not pick a depth by peeking at it. Cross-validate on the
    training set, pick the depth, then read the sealed test **once**.
11. (code) **CV picks the depth** — `cross_val_score` over depths on **train** → best depth **6** (CV
    0.919); fit at depth 6, read the **sealed test** (0.889). Note the raw test max was depth 7 (0.900)
    — which we did not, and should not, chase.
12. (md) **Read the result** — CV chose depth 6 without ever seeing the test set; the honest test
    estimate is 0.889. It is not the test maximum (0.900 at depth 7), and that is exactly right — tuning
    to the test set would have inflated the number.
13. (md) **Intuition — pruning: grow, then cut back.** Capping depth is **pre-pruning** (stop early).
    The alternative is **post-pruning**: grow the full tree, then snip the splits that buy too little —
    **cost-complexity pruning**, governed by `ccp_alpha` (bigger α → more snipping → fewer leaves).
14. (code) **Fig C — the cost-complexity pruning path** — `cost_complexity_pruning_path` on train; for
    each α plot **test accuracy** and **#leaves** vs α (two panels), mark the **CV-best α**; print the
    α / leaves / test / CV table.
15. (md) **Read the figure (C)** — as α rises, leaves drop from 23 toward 2; test accuracy rises to a
    plateau around **8 leaves (α ≈ 0.009)** then falls when we over-prune. CV picks that α — and its
    test accuracy, **0.900**, *beats* the unpruned tree's 0.878. Pruning a memorizing tree helps.
16. (code) **Fig D — unpruned vs CV-pruned boundary** (`plot_decision_boundary` ×2): the 23-leaf
    unpruned tree (jagged, test 0.878) beside the 8-leaf CV-pruned tree (smooth, test 0.900).
17. (md) **Read the figure (D)** — pruning erases the little noise-islands; the boundary becomes a
    clean pair of crescents that generalizes better. Fewer, larger boxes — a simpler model that is also
    more accurate on held-out data.
18. (md) **What this means + honest scope** — depth (pre-prune) and `ccp_alpha` (post-prune) are two
    handles on **one** dial, model complexity; both are tuned by CV, never on the test set. Honest: the
    test gap here is *small* (0.889/0.900 vs 0.878) because the data is fairly clean — the *shape*
    (train→0, the U, the jagged boundary) is the lesson, not the decimals; on messier data the gap is
    larger. `min_samples_leaf` and the rest of the API are NB 4.
19. (md) **Your turn** (3 tiered) — *easy*: from the U-curve, name the depth you would ship and say
    why not depth 1 or depth 12; *medium*: refit at `ccp_alpha=0.02` (5 leaves) and report train/test —
    is it under- or over-pruned? *harder*: write the one-line reason a tree's **training** error can
    always reach 0 on `make_moons` but its **test** error cannot.
20. (md) **What you built** — read under/over-fitting from a boundary and a train/test curve; depth as
    the complexity dial; **pre-pruning** (`max_depth`) and **post-pruning** (`ccp_alpha`); choosing the
    strength by **cross-validation**; pruning a memorizing tree to generalize better. **Vocabulary:**
    overfitting · underfitting · complexity dial · generalization gap · pre-pruning · post-pruning ·
    cost-complexity pruning · `ccp_alpha`.
21. (md) **Going further (optional) + References** — minimal cost-complexity pruning (ESL §9.2.3);
    pre- vs post-pruning trade-offs; `min_samples_leaf`/`min_samples_split` preview (NB 4).
    **References:** Breiman et al. 1984 (CART — cost-complexity pruning); ESL §9.2 / §9.2.3 (DOI
    10.1007/978-0-387-84858-7); ISLR §8.1 (DOI 10.1007/978-1-0716-1418-1). `Previous: 02 — Growing a
    tree, and reading it` · `Next: 04 — The estimator & its parameters`.

## Honest scoping (stated in the notebook)

- **The test gap is small here** (clean data) — the *shape* (train→0, the U, the jagged boundary) is the
  lesson, not the exact decimals; messier data widens it. Stated, not hidden.
- **CV chooses the dial; the test set is read once** — the depth-6 sealed test (0.889) is deliberately
  *not* the test maximum (0.900 at depth 7), to model honest selection.
- **Two handles, one dial** — `max_depth` (pre) and `ccp_alpha` (post) both control complexity; not two
  unrelated knobs.
- **`min_samples_leaf` and the full API are NB 4** — flagged forward, not crammed here.
- `make_moons` only (the set LogReg couldn't fit); no standardization (scale-invariance, NB 1).

## Verification

Build via `uv run python - < <scratchpad>/build_ch04_nb3.py` (stdin). Re-measure at build: depth sweep
0.744 → 0.889 (CV-best 6, CV 0.919) → 0.900 (depth 7) → 0.878 (unlimited, train 1.000, 23 leaves);
ccp path 23→8→2 leaves, CV-best α ≈ 0.0087 → 8 leaves / test 0.900 vs unpruned 0.878. Runs top-to-bottom
(nbconvert to scratchpad; tracked file **output-free**, `--clear-output --inplace`); **banned-word scan
over the JSON real text** = 0; `check_no_hardcoded_hex` passes; `gen_llms_txt` re-run; `pytest` 17 (no
`src/` change); `ruff` clean. Both reviewers PASS (no BLOCK); Rémy validates visually; commit
`feat(04_decision_tree): notebook 03 — overfitting & pruning: depth is the complexity dial`; merge
`notebook → chapter`.
