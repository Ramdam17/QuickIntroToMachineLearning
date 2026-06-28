# NB plan — 10_LightGBM / 03_categorical_split — the optimal categorical split (Fisher 1958)

> Status: **APPROVED by Rémy (via ExitPlanMode, 2026-06-28)**. One concept built by hand. No reviewer
> gate at the NB-plan stage (reviewers return on the built notebook).

## Context

Chapter 10, NB 3 of 5 (a fundamental, **NEW** — it replaced the dropped standalone `num_leaves` NB in
the restructure). One concept, built by hand: **how to split a categorical feature optimally.** A feature
with K categories has `2^(K−1)−1` ways to partition its categories into two groups — exponential. Fisher
(1958) proved that for a **convex** split criterion — here the structure-score gain `G²/(H+λ)` built in
ch 09 NB 2 — the optimal binary partition is **contiguous** once the categories are sorted by their
gradient statistic `G/H`, leaving only **`K−1`** candidates (linear). Genuinely new vs ch 09, which
*used* XGBoost's native categoricals but never built the split. Builds on ch 09 NB 2 (the gain), ch 08
(the per-row gradient), and NB 1–2.

## The one concept

Sort the K categories by their gradient statistic `G/H` (with squared-error `h=1`, `G/H` = the
per-category **mean** gradient = `F₀ − mean(y_c)`, i.e. sort by mean target); the optimal two-group split
is one of the `K−1` **contiguous** cuts of that sorted order. Build by hand, brute-force-confirm it is the
global optimum, and match LightGBM's categorical split **exactly**.

## Live anchors (measured, lightgbm 4.6.0, SEED=0 — `measure_ch10_nb3_categorical.py`)

- Toy: 1 categorical feature, **K=6** categories, 120 rows each (n=720), distinct mean targets in
  scrambled index order `{0:2, 1:5, 2:1, 3:4, 4:0, 5:3}`, `y = mean[cat] + N(0, 0.05)`; `F₀=2.499`,
  `g=F₀−y`, `h=1`, `λ=0`.
- Per-category `G/H` = `F₀ − mean(y_c)`: cat0 +0.495 / cat1 −2.495 / cat2 +1.502 / cat3 −1.501 /
  cat4 +2.500 / cat5 −0.501. **Sorted by `G/H` (low→high): `[1, 3, 5, 0, 2, 4]`.**
- The `K−1=5` contiguous candidate gains: cut-after-1 448.08 · -2 718.45 · **-3 808.78 (best)** · -4
  720.59 · -5 449.99 → **best contiguous split LEFT = {1,3,5}** (high-mean categories), gain 808.78.
- **Brute force all `2^(K−1)−1 = 31` partitions → global best LEFT = {0,2,4}, gain 808.78** — the
  complement of {1,3,5}, i.e. the **identical partition**. So **contiguous-after-sorting == global
  optimum** (5 candidates vs 31 → 6.2× fewer; K=30 → 29 vs ~5×10⁸).
- **LightGBM single tree (pins below) partition = `[[0,2,4],[1,3,5]]` == by-hand, exactly.**
- **Build-time pins:** `n_estimators=1, num_leaves=2, learning_rate=1, min_data_per_group=1,
  min_data_in_leaf=1, min_sum_hessian_in_leaf=0, cat_l2=0, cat_smooth=0, min_split_gain=0, reg_lambda=0,
  max_bin=100000, min_data_in_bin=1`; fit with `categorical_feature=["cat"]` on a pandas `category`
  column. (LightGBM prints `min_data_in_leaf`/`min_sum_hessian_in_leaf` override warnings — left visible.)

## Cell-by-cell (~22 cells, 3 figures) — intuition → implementation → interpretation

1. **(md) Header** — title; one concept (the optimal categorical split); arc (NB 1 grew trees, NB 2 made
   them light, NB 3 = split a *categorical* feature without trying every partition); prereqs ch 09 NB 2
   gain, ch 08 gradient, NB 1–2.
2. **(md) The problem** — a categorical feature has no order; `2^(K−1)−1` two-group partitions (31 for
   K=6, ~5×10⁸ for K=30); brute force hopeless. Contrast ch 04 one-hot/ordinal; ch 09 used native
   categoricals but did not build the split.
3. **(code) Setup + toy** — imports, `viz.use_course_style()`, `SEED=0`; K=6 scrambled means; `g=F₀−y`,
   `h=1`; print per-category `mean(y)`.
4. **(md) The gradient statistic `G/H`** — ch 09 NB 2: group summarised by `G,H`, leaf `−G/H`; one number
   per category. With `h=1`, `G/H = F₀ − mean(y_c)` (sort by mean target); state `G/H` as the general key
   (carries to classification, `h=p(1−p)`).
5. **(code) Per-category `(G,H,G/H)`** — compute and print the table.
6. **(code) Fig 1 — categories by `G/H`** — bar chart in index order, coloured to show not sorted.
7. **(md) Read fig 1** — arbitrary index order; `G/H` gives a 1-D position; next we order by it.
8. **(md) Fisher's result (1958)** — for a convex criterion the optimal partition is **contiguous** in the
   `G/H`-sorted order → `K−1` cuts not `2^(K−1)−1`. Intuition (not full proof): best boundary is a
   threshold on the 1-D summary; mixing high/low `G/H` across it never helps a convex objective. Cite
   Fisher; note Breiman/CART for binary targets.
9. **(code) Build it** — sort by `G/H`; structure-score `gain(left_cats)` (λ=0); evaluate the `K−1`
   contiguous cuts; print each + best (LEFT={1,3,5}, gain 808.78).
10. **(code) Fig 2 — the K−1 contiguous candidates (centerpiece)** — gain vs cut position, peak at cut-3;
    annotate `K−1=5` vs `2^(K−1)−1=31`.
11. **(md) Read fig 2** — single peak at the high/low-mean boundary; one scan finds it.
12. **(md) Is contiguous the *global* optimum?** — honest check: enumerate all `2^(K−1)−1`.
13. **(code) Brute force** — all 31 partitions (fix cat 0 left); global best {0,2,4} gain 808.78 ==
    contiguous winner (complementary = identical); print `K−1` vs `2^(K−1)−1` + K=30 blow-up.
14. **(md) The payoff** — exponential → linear search with no loss of optimality (convex, 1-D target);
    why LightGBM splits high-cardinality categoricals natively and fast.
15. **(md) Matching LightGBM** — it sorts by the gradient statistic, best contiguous split (Ke §4 /
    Fisher); turn off extras (pins), one tree.
16. **(code) LightGBM parity** — fit `LGBMRegressor` (pins) with `categorical_feature`; recover partition
    by grouping categories by predicted leaf value; print `[[0,2,4],[1,3,5]]` == by-hand. Log/warnings
    visible.
17. **(code) Fig 3 — by-hand == LightGBM** — categories coloured by side, LightGBM's two leaf values
    overlaid/annotated identical.
18. **(md) Read fig 3** — same two groups; the by-hand Fisher sort *is* what LightGBM does.
19. **(md) Scope & honesty** — exactness holds for a **convex** criterion + **1-D** target (regression,
    binary classification via `G/H`); **multiclass** = heuristic, not provably optimal (out of scope).
    Small/low-noise toy → clean partition; on noisy data `min_data_per_group`/`cat_l2`/`cat_smooth` (off
    here) guard rare categories.
20. **(md) Your turn** — (a) change means, re-sort, new best cut; (b) raise K, print `2^(K−1)−1` vs `K−1`;
    (c) add a rare 7th category, reason/measure how `cat_smooth`/`min_data_per_group` protects it.
21. **(md) What you built** — optimal categorical split by hand: sort by `G/H`, best of `K−1` contiguous
    cuts, == global optimum == LightGBM. Vocabulary: categorical split · `G/H` · contiguous partition ·
    `2^(K−1)−1`→`K−1` · `min_data_per_group`/`cat_l2`/`cat_smooth`. Next: NB 4 — the estimator & params.
22. **(md) References** — Fisher 1958 JASA 53(284) DOI 10.1080/01621459.1958.10501479; Ke et al. 2017
    NeurIPS (§4); Breiman et al. 1984 CART (binary-target sort); Chen & Guestrin 2016 (the gain, DOI
    10.1145/2939672.2939785).

## `src/` & guards
- **No `src/` change** (reuse `viz`; numpy + `itertools`; `LGBMRegressor` + `predict`/`dump_model`;
  pytest 20). Colours only from `ml_course.colors`; seeds fixed.
- Build from `build_ch10_nb3.py` (source of truth; rebuild right before `git add` — kernel-drift guard).
  Cell ids `cell-NN`; kernelspec `ml-course (3.12.12)`.
- **Never silence output** — no `verbose=-1` (LightGBM banner + override warnings stay visible).
- Exit guards: nbconvert exit 0 (3 figures), banned scan = 0, hex clean, ruff/black clean, output-free;
  **two-reviewer gate** (no BLOCK) → fold → **Rémy visual** → end-of-NB checklist (`gen_llms_txt.py`,
  `common_errors` +rows, `course_map` §10 mark NB 3, pytest 20, STATE) → commit
  `feat(10_lightgbm): notebook 03 — the optimal categorical split` → `git merge --ff-only` into
  `chapter/10_LightGBM`.

## Verification (end-to-end)
1. nbconvert-execute a scratchpad copy → exit 0, 3 figures, anchors reproduce (sorted [1,3,5,0,2,4];
   contiguous best {1,3,5} gain 808.78; brute-force global == contiguous; LightGBM [[0,2,4],[1,3,5]] ==
   by-hand). 2. hex + banned + ruff → clean. 3. pytest → 20 passed. 4. Two-reviewer gate, then Rémy visual.
