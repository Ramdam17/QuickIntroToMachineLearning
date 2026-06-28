# Notebook plan — 10_LightGBM / 01_leaf_wise_growth

> Status: **APPROVED by Rémy (2026-06-28)** (no reviewer gate at the per-NB plan stage — both reviewers
> return on the built notebook). All anchors measured live (lightgbm 4.6.0 / sklearn 1.9.0, SEED=0);
> re-measured at build. Build via `<scratchpad>/build_ch10_nb1.py`.

## Context

NB **1 of 5**, the first fundamental. **One concept:** **leaf-wise (best-first) tree growth** — grow the
leaf whose best split reduces the loss most, wherever it sits, instead of splitting level by level
(ch 04/09 depthwise). Built by hand on a toy with ch 09 NB 2's structure-score gain, shown to reach a
lower **training** loss for the same number of leaves (while growing lopsided), and matched **exactly**
to LightGBM's single tree. Footing: the learner met leaf-wise *named* twice (ch 08 NB 5 HistGBR
`max_leaf_nodes`; ch 09 NB 4 `grow_policy='lossguide'`) — here it is **built**.

## Anchors (measured live, lightgbm 4.6.0, SEED=0 — re-measured at build)

Toy (asymmetric so one branch deserves more splits): n=240, x0,x1 ~ U(0,1);
`y = staircase(x1, 5 steps) if x0 > 0.5 else 0`, + N(0,0.05). `F0 = mean(y) = 1.1277`, root SSE 523.92.
Gain = ch 09 NB 2 structure score on `g = F0−y, h = 1, λ = 0`.

- **Leaf-wise vs level-wise TRAIN SSE per #leaves:** 2→268.3/268.3 (same first split); 3→**65.4**/268.3;
  4→**25.8**/268.3; 5→**12.2**/268.3; 6→**0.57**/65.4; 8→**0.53**/25.7. Leaf-wise pours its budget into the
  structured x0>0.5 branch; level-wise spends early splits breadth-first on the flat branch.
- **Lopsidedness (6 leaves):** leaf-wise leaf sizes [23,26,26,28,28,**109**] (one big flat leaf + fine
  cuts in the structured branch); level-wise [1,1,4,56,75,103]. leaf-wise order: x0<0.5, then four x1
  splits (all in the dear branch).
- **Exact LightGBM parity:** `LGBMRegressor(n_estimators=1, num_leaves=6, learning_rate=1,
  min_child_samples=1, min_split_gain=0, reg_lambda=0, min_child_weight=0, max_bin=100000,
  min_data_in_bin=1)` → its single tree == by-hand leaf-wise: **max|pred_byhand − pred_lgb| = 0.000000**,
  SSE 0.5660 == 0.5660, same feature order (tiny threshold diffs = LightGBM's bin-boundary convention;
  partition identical).

## Cell-by-cell (~19 cells; 3 figures; "Read the figure" after each)

1. (md) **Header** — `# 01 — Leaf-wise growth: grow the best leaf first`; *Ch 10 · NB 1 of 5*. Prereqs:
   ch 04 (a tree splits to reduce impurity), ch 09 NB 2 (the structure-score gain `½G²/(H+λ)`), ch 08/09
   (the per-row gradient `g=F−y`). **Footing:** you met leaf-wise *named* twice (ch 08 NB 5 HistGBR
   `max_leaf_nodes`; ch 09 NB 4 `lossguide`) — here you **build** it. What you'll do.
2. (md) **Level-wise vs leaf-wise — the question.** A tree grows by splitting leaves; the question is
   *which leaf next*. **Level-wise** (ch 04/09 depthwise): split every leaf at the current depth before
   going deeper — balanced. **Leaf-wise** (LightGBM): split the leaf whose best split reduces the loss
   most, wherever it is — greedy, lopsided. Same candidate splits; different **order**.
3. (code) **Setup & the toy.** imports (numpy, matplotlib; `LGBMRegressor`; `viz`/`COLORS`; `SEED=0`).
   The toy (x0>0.5 → 5-step staircase in x1, else flat — the asymmetry is the point). `F0=mean(y)`,
   `g=F0−y`, `h=1`. Print shape, F0, root SSE.
4. (md) **The gain we'll use (reused, not rebuilt).** A node's structure score is `−½G²/(H+λ)` (ch 09
   NB 2); a split's **gain** = score(parent) − score(left) − score(right), `λ=0`, `g=F0−y`, `h=1` here.
   Maximising it = most reducing training SSE. (We reuse ch 09's gain; the new idea is the growth
   **order**.)
5. (code) **`best_split(node)` by hand.** Try every feature × midpoint threshold, compute the gain,
   return the best `(feature, threshold, gain, left, right)`. Print the root's best split (x0<0.5) + gain.
6. (md) **The two strategies, as code.** **Level-wise** = a queue (split leaves in creation order,
   breadth-first). **Leaf-wise** = a frontier (always expand the max-gain leaf). Both stop at `num_leaves`
   leaves — the one-line difference is *which leaf you pop*.
7. (code) **Build both + Fig 1.** `grow('level')` / `grow('leaf')` to a budget; budgets 2–8 → train SSE.
   **Fig 1 — training loss vs #leaves:** SSE vs #leaves, leaf-wise (low) vs level-wise (high); mark the
   gap at 6 leaves (0.57 vs 65.4). *local.*
8. (md) **Read Fig 1.** For the same budget, leaf-wise reaches far lower **training** loss — it spends
   each split where the gain is largest (the x0>0.5 branch), while level-wise spends early splits
   breadth-first on the flat branch. (Whether lower *training* loss helps held-out is **NB 2's**
   question — leaf-wise's freedom is also its overfitting risk.)
9. (code) **Fig 2 — the partitions / lopsidedness.** Scatter `(x0,x1)` coloured by leaf, leaf-wise vs
   level-wise side by side; print leaf sizes. Leaf-wise carves x0>0.5 finely, leaves x0≤0.5 as one big
   leaf; level-wise cuts both evenly. *local.* *(Tune the budget for a legible level-wise panel at build.)*
10. (md) **Read Fig 2.** Leaf-wise grew **lopsided**: one big flat leaf (≈109 pts) + fine cuts where the
    signal is; level-wise split both regions regardless of payoff. Best-first follows the **gain**;
    level-wise follows **position**.
11. (md) **Matching LightGBM.** LightGBM grows leaf-wise. Turn off its extras (`reg_lambda=0`,
    `min_split_gain=0`, `min_child_samples=1`, `max_bin` huge + `min_data_in_bin=1` so binning is
    lossless, one tree, `learning_rate=1`) and its single tree should be **exactly** our by-hand
    leaf-wise tree. (Those knobs preview NB 2/4; here we set them to see the bare mechanism.) **We do not
    silence LightGBM's log** — its info banner stays visible (never-silence rule).
12. (code) **Parity + Fig 3.** Fit that `LGBMRegressor`; `dump_model` → its split order (by `split_index`
    = best-first creation order) printed beside the by-hand order; **Fig 3 — by-hand vs LightGBM
    predictions** on the diagonal (identical, max|Δ|=0). Print SSE 0.566 == 0.566. *local.*
13. (md) **Read the parity.** By-hand leaf-wise reproduces LightGBM's single tree **exactly** — same
    feature order, identical partition and predictions. LightGBM's "best-first" growth *is* the frontier
    you built. (Tiny threshold diffs = LightGBM's bin-boundary convention; partition identical.)
14. (md) **Where this sits — and it is not LightGBM-only.** Leaf-wise reaches a target training loss with
    **fewer leaves** (efficient) but grows **deep and lopsided** — the overfitting risk **NB 2** tames
    with `num_leaves`. And it is the **modern default**: sklearn's `HistGradientBoosting*` (ch 08 NB 5,
    `max_leaf_nodes=31`) and XGBoost's `lossguide` (ch 09 NB 4) grow the same way.
15. (md) **Your turn.** *easy:* change the `num_leaves` budget and re-read the leaf-wise vs level-wise SSE
    gap. *core:* write the one-line difference between the strategies (pop the max-gain leaf vs the first
    queued leaf) and confirm both reduce to the same first split. *reach:* construct a toy where leaf-wise
    and level-wise build the **same** tree — when does the order not matter?
16. (md) **What you built.** Leaf-wise (best-first) growth, by hand: a frontier expanding the max-gain
    leaf; lower training loss per leaf than level-wise, but lopsided; **exact** LightGBM parity.
    **Vocabulary:** leaf-wise / best-first · level-wise / depthwise · the leaf frontier · `num_leaves`
    (the budget — NB 2). Next: **`num_leaves`, the central dial** and the overfitting it can cause.
17. (md) **References** — Ke et al. 2017 (LightGBM; leaf-wise growth; NeurIPS 30); Shi, H. (2007),
    *Best-first decision tree learning* (best-first induction); ch 09 NB 2 (the structure-score gain).
    `Previous: Chapter 09 — XGBoost.` `Next: 02 — num_leaves, the central dial.`

## Figures (3, each followed by "Read the figure")
1. **Training loss vs #leaves** (cell 7) — leaf-wise (low) vs level-wise (high); the gap.
2. **The partitions** (cell 9) — `(x0,x1)` by leaf, leaf-wise (lopsided) vs level-wise (even).
3. **By-hand vs LightGBM** (cell 12) — predictions on the diagonal, identical (exact parity).

## `src/` & guards
**No `src/` change** — notebook-local matplotlib + `viz.use_course_style`; toy inline; LightGBM via
`dump_model`; **pytest stays 20**. Build via `<scratchpad>/build_ch10_nb1.py`; re-measure anchors at
build; nbconvert top-to-bottom from project cwd on a scratchpad copy (output-free tracked file);
**never silence output** (no `verbose=-1` in the notebook — the parity fit's LightGBM banner stays
visible); banned-word JSON scan = 0; `check_no_hardcoded_hex`; ruff/black clean; `gen_llms_txt` re-run.
Two-reviewer gate (no BLOCK) + Rémy visual before commit; ff-merge `notebook → chapter`.

## Honest scoping
One concept (leaf-wise / best-first growth), built by hand on ch 09's gain and matched to LightGBM
exactly. Level-wise is the familiar contrast (ch 04/09). The lower-training-loss-per-leaf is a
**training** claim — the held-out consequence (overfitting via large `num_leaves`) is **named and
deferred to NB 2**. Histogram binning is *reused* (ch 09 NB 4), not re-taught. Leaf-wise framed as the
modern default (HistGBR, XGBoost-lossguide), not a LightGBM exclusive. The parity fit's LightGBM log is
left visible (no `verbose=-1`), per the never-silence rule.
