# Notebook plan — 09_XGBoost / 03_sparsity_aware_splits

> Status: **APPROVED by Rémy (2026-06-27)** (no reviewer gate at the per-NB plan stage — both reviewers
> return on the built notebook). All anchors measured live (xgboost 3.2.0 / sklearn 1.9.0, SEED=0);
> re-measured at build. Build via `<scratchpad>/build_ch09_nb3.py`.

## Context

NB **3 of 5**, the last fundamental. **One concept:** XGBoost handles **missing / sparse** values with
**no imputation** — at each split, every missing row is sent down a single **learned default
direction**, chosen by trying both ways and keeping the higher gain (Chen & Guestrin 2016 §3.4). We
build that split by hand on NB 2's gain, watch the missing rows route themselves to where they belong,
match XGBoost's learned direction exactly, and close with the measured contrast: plain
`GradientBoosting*` cannot take a NaN at all, while `HistGradientBoosting*` (ch 08) and XGBoost can —
XGBoost first (2016). This is XGBoost's third ingredient, after the second-order view (NB 1) and the
regularized objective (NB 2).

## Anchors (measured live, xgboost 3.2.0 / sklearn 1.9.0, SEED=0 — re-measured at build)

Toy: 1 feature with missing values that **carry signal**. `x = [1,2,3,7,8,9,nan,nan]`,
`y = [1.0,1.2,0.9, 3.0,3.2,2.9, 3.1,3.0]`, `F0 = mean(y) = 2.2875`; squared error (`g=F0−y`, `h=1`).
Non-missing split into a low group (x=1,2,3 → y≈1) and a high group (x=7,8,9 → y≈3); the two missing
rows have y≈3, so they **belong with the high group** (the right child).

- **By-hand sparsity-aware search** (λ=1, half-gain; candidate thresholds = midpoints of the sorted
  non-missing x): at the winning threshold `x < 5`, **missing→left gain = 1.043** vs **missing→right
  gain = 2.949** → global best = **(x<5, missing→right, half-gain 2.949)**. (missing→right dominates at
  every threshold.)
- **XGBoost agrees exactly:** `trees_to_dataframe` → split `x < 5.0`, the **`Missing` column points to
  the right child** (`No`); reported `Gain = 5.8985 = 2 × 2.949` (the no-½ convention of NB 2). Leaf
  means: left −0.94 (low group), right +0.63 (high group + the missing rows).
- **Who accepts NaN?** (measured) `GradientBoostingRegressor` → **REJECTS** (`ValueError: Input X
  contains NaN`); `HistGradientBoostingRegressor` → **ACCEPTS**; `XGBRegressor` → **ACCEPTS**.

## Cell-by-cell (~19 cells; 3 figures; "Read the figure" after each)

1. (md) **Header** — `# 03 — Sparsity-aware splits: a default direction for missing values`;
   *Ch 09 · NB 3 of 5*. Prereqs: NB 2 (the split gain `½[…]−γ`, and `Cover = Σh`); ch 08 NB 5/6
   (`HistGradientBoosting*` was named as the fast booster that takes NaN). What you'll do: build the
   sparsity-aware split by hand → watch missing rows route themselves → match XGBoost's learned default
   direction → see who can take a NaN and who cannot.
2. (md) **Recap — missing values are everywhere.** Real tabular data has gaps. Chapter 08's
   `GradientBoosting` cannot fit a matrix with a `NaN` in it — you must **impute** first (fill the gap
   with a mean, a median, a flag). XGBoost takes the `NaN` directly. How? At every split it learns a
   **default direction** for the missing rows. No imputation — and, as we'll see, the *fact* of being
   missing can itself carry signal.
3. (code) **Setup & the toy.** imports (numpy, matplotlib; `XGBRegressor`; `GradientBoostingRegressor`,
   `HistGradientBoostingRegressor`); `viz`/`COLORS`; `SEED=0`. The toy `x` (with two `NaN`) and `y`;
   `F0=mean(y)`, `g=F0−y`, `h=1`, the missing mask; print the missing rows' `y`.
4. (md) **The idea — a learned default direction.** At a split on feature `x`, the non-missing rows go
   left or right by the threshold. The **missing** rows have no value to compare — so XGBoost sends
   them *all* to one side, the **default direction**, and **learns which side** by computing the split
   gain (NB 2) **both ways** and keeping the larger (Chen & Guestrin §3.4). The split's threshold and
   its default direction are chosen *together*, to maximise the gain.
5. (code) **Fig 1 — the data and the question.** Scatter the non-missing `(x, y)` (low vs high group);
   draw the two missing rows in a shaded "x missing" band at their `y` values; a caption-arrow posing
   "which way should these go?". *notebook-local.*
6. (md) **Read Fig 1.** The non-missing points fall into a cheap group (y≈1) and a dear group (y≈3).
   The two missing rows sit at y≈3 — by eye they belong with the dear group, on the right. The split
   does not get to see their `x`; can it still send them the right way? Yes — by trying both.
7. (md) **The recipe, by hand.** For each candidate threshold (midpoints of the sorted non-missing
   `x`), compute the NB-2 gain **twice** — once sending the missing rows left, once right — and keep
   the `(threshold, direction)` with the highest gain. (We use `λ=1, γ=0`, as in NB 2.)
8. (code) **The search.** Enumerate thresholds; for each print `gain(missing→left)` and
   `gain(missing→right)` with the resulting `(n_left/n_right)`; report the global best
   (**x<5, missing→right, half-gain 2.949**).
9. (code) **Fig 2 — the search, drawn.** Gain vs threshold, two curves (missing→left, missing→right);
   the right curve dominates and peaks at `x<5`; mark the winner. *notebook-local.*
10. (md) **Read Fig 2.** Sending the missing rows **right** beats sending them left at *every*
    threshold here, and the best split overall is `x<5` with missing→right (gain 2.949). The missing
    rows are being used as evidence — their being missing pushes them toward the group they match.
11. (code) **Fig 3 — by hand vs XGBoost.** Fit `XGBRegressor(…, missing=np.nan, tree_method='exact')`;
    print `trees_to_dataframe` (note the `Missing` column → the right child); a bar of the two trials
    at `x<5` (left 1.043 vs right 2.949) with XGBoost's learned direction marked, and the gain parity
    (`2 × 2.949 = 5.8985 =` reported `Gain`). *notebook-local.*
12. (md) **Read Fig 3.** XGBoost independently learned **missing→right** — the same direction we found
    by hand — and its reported gain is `2×` our half-gain (the no-½ convention from NB 2). The missing
    rows landed in the dear-group leaf, with no imputation anywhere.
13. (md) **Who can take a NaN?** Native handling is not universal — let us check three boosters.
14. (code) **The contrast.** Fit `GradientBoostingRegressor` (raises `ValueError: Input X contains
    NaN`), `HistGradientBoostingRegressor` (fits), `XGBRegressor` (fits); print each outcome.
15. (md) **Read the contrast.** Plain `GradientBoosting` **rejects** the NaN — you would impute first,
    which *erases* the signal in the missing-ness. `HistGradientBoosting` (Chapter 08) and XGBoost
    **accept** it natively; XGBoost introduced the sparsity-aware split in 2016. When missingness is
    informative, the learned default direction can beat impute-then-fit — and it is simpler. *(Aside:
    CART's classic answer was **surrogate splits**, ESL §9.2.4 — a different, heavier mechanism;
    XGBoost's single default direction is cheaper and scales.)*
16. (md) **Where this sits.** Three XGBoost ingredients now: the second-order view (NB 1), the
    regularized objective (NB 2), and sparsity-aware splits (NB 3). What remains is **engineering** —
    the histogram split-finding that makes all this fast — which we meet in NB 4 alongside the
    estimator and its parameters.
17. (md) **Your turn.** *easy:* change the missing rows' `y` from ≈3 to ≈1 and predict (then check)
    the new default direction. *core:* at threshold `x<5`, compute `gain(missing→left)` and
    `gain(missing→right)` by hand from `G,H` and confirm right wins. *reach:* explain why imputing the
    column mean before a NaN-blind model can do *worse* than XGBoost's native handling when the
    missing-ness is informative — what information does the mean-impute throw away?
18. (md) **What you built.** The sparsity-aware split: missing rows take a **learned default
    direction**, found by computing the gain both ways and keeping the larger — built by hand, matched
    to XGBoost exactly. Plain `GradientBoosting` rejects NaN; `HistGradientBoosting` and XGBoost accept
    it. **Vocabulary:** missing / sparse · default direction · sparsity-aware split · native NaN
    handling. Next: the estimator `XGBRegressor`/`XGBClassifier`, its parameters, and the histogram
    method that makes it fast.
19. (md) **References** — Chen & Guestrin 2016 (§3.4, the sparsity-aware split; DOI
    10.1145/2939672.2939785); Hastie, Tibshirani & Friedman, ESL §9.2.4 (surrogate splits — CART's
    classic missing-value mechanism; DOI 10.1007/978-0-387-84858-7); Friedman 2001
    (DOI 10.1214/aos/1013203451). `Previous: 02 — The regularized objective.`
    `Next: 04 — The estimator and its parameters.`

## Figures (3, each followed by "Read the figure")
1. **The data and the question** (cell 5) — low/high groups + the missing rows in a shaded band at y≈3.
2. **The search** (cell 9) — gain vs threshold for missing→left and missing→right; right dominates, peaks at x<5.
3. **By hand vs XGBoost** (cell 11) — the two trials at x<5 (1.043 vs 2.949) + XGBoost's learned direction & gain parity.

## `src/` & guards
**No `src/` change** — notebook-local matplotlib + `viz.use_course_style`; the toy built inline; XGBoost
via `trees_to_dataframe`; **pytest stays 20**. Build via `uv run python <scratchpad>/build_ch09_nb3.py`;
**re-measure every anchor at build**; nbconvert top-to-bottom from project cwd on a scratchpad copy
(tracked file **output-free**); **banned-word scan over JSON cell text** = 0 (watch "just"/"simply");
`check_no_hardcoded_hex` passes; ruff clean; `gen_llms_txt` re-run. Two-reviewer gate (no BLOCK) + Rémy
visual before commit; ff-merge `notebook → chapter`.

## Honest scoping
One concept (the learned default direction for missing values), built by hand on NB 2's gain and
matched to XGBoost exactly (direction + the 2× gain). The toy is constructed so missingness is
**informative** (the honest motivation for native handling over imputation) — stated, not oversold:
when missingness is *not* informative, impute-then-fit and native handling are close. The
histogram/approximate split-finding (the engineering that scales this) is **named and deferred to
NB 4**, not built here. CART surrogate splits named as the classic alternative, not built.
