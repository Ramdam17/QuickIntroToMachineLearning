# Notebook plan — 09_XGBoost / 02_regularized_objective

> Status: **APPROVED by Rémy (2026-06-27)** (no reviewer gate at the per-NB plan stage — both reviewers
> return on the built notebook). All anchors measured live (xgboost 3.2.0 / sklearn 1.9.0, SEED=0);
> re-measured at build. Build via `<scratchpad>/build_ch09_nb2.py`.

## Context

NB **2 of 5**. **One concept:** put complexity control **inside the objective**. Chapter 08 (and ch 04)
controlled complexity only from the outside — `max_depth`, number of trees. XGBoost adds a penalty
`Ω(f) = γ·T + ½λ·Σⱼ wⱼ²` to the loss itself. Two consequences, both built by hand on NB 1's
second-order machinery: the leaf weight becomes `w* = −G/(H+λ)` (L2 shrinks it toward 0), and a
principled **split gain** — derived from the *structure score* — decides whether a split is worth
making, with `γ` as a per-split price (pre-pruning). Checked against XGBoost: leaf weights match
exactly, and we surface the **measured 2×/½** convention honestly.

## Anchors (measured live, xgboost 3.2.0, SEED=0 — re-measured at build)

Toy: 1-D regression `x=[0..5]`, `y=[1.0,1.2,0.9,5.0,5.3,4.8]`, `F0=mean(y)=3.0333`; squared error
(`g=F0−y`, `h=1`). Split `x<2.5` → left `{0,1,2}` (`G_L=6, H_L=3`), right `{3,4,5}` (`G_R=−6, H_R=3`),
root `G=0, H=6`.

- **λ shrinks the leaf** `w*=−G/(H+λ)` (right leaf `G_R=−6,H_R=3`): λ=0 → +2.0, λ=1 → +1.5, λ=3 → +1.0,
  λ=10 → +0.4615, λ=100 → +0.0583. (Left leaf is the mirror, −.)
- **Split gain** (λ=1, γ=0): textbook ½-version `½[G_L²/(H_L+λ)+G_R²/(H_R+λ)−G²/(H+λ)] = 9.0`; **XGBoost
  reports `Gain = 18.0` (exactly 2×)**; leaf weights `[−1.5, +1.5]` == by-hand.
- **γ is a prune threshold in XGBoost's no-½ units** (measured sweep, λ=1): split survives for γ ≤ 18,
  **pruned at γ = 19**. So XGBoost uses the no-½ value (18) consistently for *both* the reported Gain and
  the γ comparison — the ½ scales gain and the implied γ together, leaving the decision unchanged.
- **`Cover = ΣH`**: root Cover 6.0, each leaf 3.0 (= the count here, since `h=1`) — the quantity
  `min_child_weight` will threshold (NB 4).
- **`base_score` is learned when unpinned**: default fit → `base_score = 3.0333 = mean(y)` (the ch 08
  `init_` role); we pin it only to make `F₀` a known constant for the hand-check.

## Cell-by-cell (~20 cells; 3 figures; "Read the figure" after each)

1. (md) **Header** — `# 02 — The regularized objective: penalising complexity in the loss`;
   *Ch 09 · NB 2 of 5*. Prereqs: NB 1 (`w*=−G/H`), ch 04 (a split scored by its gain), ch 08 (trees as
   the base learner; `init_`). What you'll do: add `Ω = γT + ½λΣw²` → leaf `w*=−G/(H+λ)` (λ shrinks) →
   derive the split gain from the structure score (γ pre-prunes) → check vs XGBoost (leaf weights exact;
   the measured 2× gain).
2. (md) **Recap.** NB 1 gave the best leaf for any loss, `w*=−G/H`, and XGBoost matched it once we
   switched its regularizer **off**. Now we switch it on. XGBoost's defining move is to charge for
   complexity **inside** the objective — not only through `max_depth` and the number of trees (ch 04/08).
3. (code) **Setup & the toy.** imports (numpy, matplotlib, `XGBRegressor`); `viz`/`COLORS`; `SEED=0`.
   The toy; `F0=mean(y)`; `g=F0−y`, `h=1`; the `x<2.5` split; print `G_L,H_L,G_R,H_R,G,H`.
4. (md) **The regularized objective.** `Obj = Σ loss + Ω(f)`, `Ω(f) = γ·T + ½λ·Σⱼ wⱼ²` (`T` = number of
   leaves). `λ` penalises large leaf weights (an L2 penalty, like ridge / ch 03's `C`); `γ` charges a
   flat cost per leaf. A simpler tree is preferred unless the data earns the complexity.
5. (md) **Effect 1 — λ shrinks the leaf.** Adding `½λw²` to a leaf's objective gives
   `G·w + ½(H+λ)w²`, minimised at **`w* = −G/(H+λ)`**. The `+λ` in the denominator pulls every leaf
   toward 0 — the second-order leaf of NB 1, now reined in.
6. (code) **Fig 1 — the regularized parabola.** Plot the right leaf's objective `G_R·w + ½(H_R+λ)w²`
   for `λ ∈ {0,1,10}`; mark each minimum `−G_R/(H_R+λ)`; print the shrinkage values. *local.*
7. (md) **Read Fig 1.** The penalty steepens the bowl and slides its minimum toward 0: +2.0 (λ=0) →
   +1.5 (λ=1) → +0.46 (λ=10). The leaf still points the right way, just less far — a safer step. `λ=0`
   recovers NB 1's leaf; large `λ` gives a timid tree.
8. (md) **Effect 2 — the structure score.** To score a whole tree, plug `w*=−G/(H+λ)` back into the
   objective: `Obj(structure) = −½ Σ_leaves G²/(H+λ) + γT` (Chen & Guestrin **eq. 6**). Lower is better;
   it rewards leaves with a large `|G|` relative to `H+λ`.
9. (md) **The split gain (derive it — mind the sign).** A split replaces one leaf
   (score `−½G²/(H+λ)`) with two children. **gain = score(before) − score(after)** =
   `½[G_L²/(H_L+λ) + G_R²/(H_R+λ) − G²/(H+λ)] − γ` (**eq. 7**). A *positive* gain means the (negative)
   objective got more negative — the split helped. *(Trap: the score is negative, so the intuitive
   "parent − children" flips the sign; derive it from eq. 6, do not memorise `½[…]−γ`.)*
10. (code) **The gain by hand.** With λ=1, γ=0: compute and print `G_L²/(H_L+1)`, `G_R²/(H_R+1)`,
    `G²/(H+1)`, and the ½-gain (= **9.0**).
11. (md) **γ — the price of a split.** `γ` is subtracted from every split's gain: a split is kept only
    if its gain clears `γ`. So `γ` is **pre-pruning** — it blocks splits that do not earn their keep;
    larger `γ` → shallower trees.
12. (code) **Fig 2 — the gain and the γ threshold.** Bar of the split's gain with a horizontal `γ`
    line; overlay the measured sweep (split survives for γ ≤ 18, pruned above). *local.*
13. (md) **Read Fig 2 — and the honest 2×/½.** Our textbook ½-gain is 9; **XGBoost reports 18 and
    prunes only once γ exceeds 18**. XGBoost works in **no-½ units consistently** — both its reported
    `Gain` and its `γ` threshold are the no-½ value (18). The ½ in the paper is a unit convention: it
    scales the gain *and* the implied `γ` together, so the winning split and the prune decision are
    identical. Practical takeaway: **`γ` lives in the units XGBoost prints as `Gain`** — to suppress a
    split whose `Gain` is 18, set `γ` above 18.
14. (code) **Fig 3 — by-hand vs XGBoost (parity).** Fit `XGBRegressor(n_estimators=1, max_depth=1,
    reg_lambda=1, gamma=0, base_score=mean(y), tree_method='exact')`; two small panels — (left) leaf
    weights, by-hand `±1.5` vs XGBoost (identical bars); (right) the gain, by-hand `9` vs XGBoost `18`,
    annotated "×2 — the ½ XGBoost drops." Print `trees_to_dataframe` (leaves, Gain, Cover). *local.*
15. (md) **Read Fig 3.** Leaf weights match exactly; the gain differs only by the documented ½;
    **`Cover = ΣH`** (here 3 per leaf, = the count since `h=1`) — the quantity `min_child_weight`
    thresholds (NB 4). `base_score` was pinned to the mean for the check; left free, XGBoost **learns**
    it (measured: `= mean(y)`), the ch 08 `init_` role.
16. (md) **Where this sits.** `λ` and `γ` move complexity control **into the loss** — a structural step
    beyond ch 04/08's `max_depth` and tree count. The engine is unchanged (NB 1's second-order leaf),
    now regularized. Next (NB 3): how XGBoost splits when features are **missing**.
17. (md) **Your turn.** *easy:* compute `w*=−G/(H+λ)` for the given `G,H` at λ=0,1,10 and confirm the
    shrinkage. *core:* compute the ½-gain for the toy's split by hand, verify XGBoost reports 2×, and
    find the `γ` that just prunes it. *reach:* show that raising `λ` lowers *every* split's gain (so
    `λ` also discourages splitting, indirectly), and explain how that differs from `γ`'s flat per-leaf
    charge.
18. (md) **What you built.** `Ω = γT + ½λΣw²` → leaf `w*=−G/(H+λ)` (λ shrinks) + the split gain
    `½[…]−γ` (γ pre-prunes), both from the structure score. XGBoost matches the leaf exactly and reports
    the gain in no-½ units (2× the textbook), pruning when `γ` exceeds it. **Vocabulary:** regularized
    objective · L2 leaf penalty `λ` (`reg_lambda`) · per-leaf cost `γ` (`gamma`) · structure score ·
    split gain · `Cover = ΣH`. Next: sparsity-aware splits for missing values.
19. (md) **References** — Chen & Guestrin 2016 (eq. 6–7: the structure score, the optimal leaf weight,
    and the split gain; DOI 10.1145/2939672.2939785); Friedman 2001 (DOI 10.1214/aos/1013203451);
    ESL §10.10 (DOI 10.1007/978-0-387-84858-7). `Previous: 01 — The second-order view.`
    `Next: 03 — Sparsity-aware splits.`

## Figures (3, each followed by "Read the figure")
1. **The regularized parabola** (cell 6) — leaf objective for λ∈{0,1,10}, minima sliding to 0. *local.*
2. **Gain and the γ threshold** (cell 12) — the split's gain vs a γ line; split-vs-prune sweep. *local.*
3. **By-hand vs XGBoost** (cell 14) — leaf weights identical; gain 9 vs 18 (the ×2 / ½). *local.*

## `src/` & guards
**No `src/` change** — notebook-local matplotlib + `viz.use_course_style`; the toy built inline; XGBoost
via `trees_to_dataframe`; **pytest stays 20**. Build via `uv run python <scratchpad>/build_ch09_nb2.py`;
**re-measure every anchor at build**; nbconvert top-to-bottom from project cwd on a scratchpad copy
(tracked file **output-free**); **banned-word scan over JSON cell text** = 0 (watch "just"/"simply");
`check_no_hardcoded_hex` passes; ruff clean; `gen_llms_txt` re-run. Two-reviewer gate (no BLOCK) + Rémy
visual before commit; ff-merge `notebook → chapter`.

## Honest scoping
One concept (complexity inside the objective), built by hand on NB 1's `(g,h)`. The **2×/½** gain
convention is surfaced and measured, not hidden — and the honest resolution is that XGBoost is
internally consistent (no-½ for both gain and γ), so the decision matches the textbook. The sign of the
gain is derived from the structure score (not asserted). `base_score` is pinned only for the hand-check
(learned when free). Regularization is the *only* new idea here; the histogram/approximate split-finding
and missing-value handling are deferred (NB 3 / NB 4).
