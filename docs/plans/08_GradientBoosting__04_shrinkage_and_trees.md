# Notebook plan — 08_GradientBoosting / 04_shrinkage_and_trees

> Status: **APPROVED by Rémy (2026-06-26) & persisted.** No reviewer gate at the NB-plan stage (Rémy
> validates alone; both reviewers return on the *built* notebook). Next: build from
> `<scratchpad>/build_ch08_nb4.py`, re-measuring every anchor.

## Context

NB **4 of 6** — the last fundamentals notebook, and the one that earns NB 5's early stopping. NB 1–3
built the *engine* (fit the residual = the negative gradient of any loss). NB 4 is about **control**:
the three dials that decide how well that engine generalizes — the learning rate **ν**, the tree
**depth**, and the **number of trees** — and the headline that, unlike a random forest, **adding more
trees can make gradient boosting worse**. Richer-scope single concept (the ch 07-NB 3 precedent): depth
is kept subordinate ("the other dial, held shallow"), with the bottoms-then-rises test curve as the
unmistakable headline. **Back to regression.**

## Anchors re-measured vs the chapter-plan estimates (approved 2026-06-26)

The chapter plan's NB 4 anchors were cartographer **estimates** (ν=1.0 best R²≈0.80@~130→0.748@1000;
ν=0.1→0.837@1000). Re-measured on the live install they land differently — and **cleaner**. The
*direction* of every claim holds; only the exact values move. Notably the ν=1.0 peak is at **~18 trees,
not ~130** — a stronger teaching point (at full step you overfit almost at once). Rémy approved these
re-measured numbers.

## Dataset & anchors (measured on scikit-learn 1.9.0, seed 0 — re-measured at build)

- **Data:** `make_friedman1(n_samples=2000, noise=1.0, random_state=0)`,
  `train_test_split(test_size=0.30, random_state=0)` → **train 1400 / test 600**, 10 features. Chosen
  because its structure is *known*: `y = 10·sin(π·x₀·x₁) + 20·(x₂−0.5)² + 10·x₃ + 5·x₄ + noise`
  (Friedman 1991) — **5 informative features + 5 pure noise**, and a genuine **2-way interaction x₀·x₁**
  that makes "depth = interaction order" visible. *(NB 1–2 used a 1-D sine — too small to show
  overfit-with-trees. California is reserved for the NB 6 capstone, so a different set keeps NB 6 fresh.)*
- **ν × n_estimators (depth=3, n_estimators=1000), staged test R²:**

  | ν | best test R² | @ trees | R²@1000 | train R²@1000 | reading |
  |---|---|---|---|---|---|
  | **1.0** | 0.8637 | **18** | **0.8130** | 1.0000 | overfits: peaks early, then sags (test MSE 3.39→4.65) |
  | **0.1** | 0.9300 | 308 | 0.9282 | 0.9970 | lower floor, **no turn-up within budget** |
  | **0.01** | 0.9213 | **1000** | 0.9213 | 0.9623 | **still climbing** (underfit within budget) |

- **By-hand ν (NB 1 recap):** F₀ = mean(y_train) = **14.13** (train MSE 25.28); one depth-3 tree drops
  train MSE to **9.63 at ν=1.0** vs **22.31 at ν=0.1** (a tenth of the way). Trees to reach train
  MSE ≤ 2.0: **13 / 48 / 496** for ν=1.0 / 0.1 / 0.01 — the trade-off in single numbers (honest nuance:
  ν=1.0 and ν=0.5 both ≈12–13, so it bites as ν *shrinks*, not at the top).
- **depth = interaction order (ν=0.1, n_estimators=300), test / train R²:** depth 1 **0.873 / 0.905**
  (stumps: additive only, can't represent x₀·x₁ — underfit even train) → depth 2 **0.931 / 0.966** (the
  pairwise interaction captured — the jump) → depth 3 0.929 / 0.983 → depth 5 **0.923 / 0.998**
  (memorizing: train↑, test↓). sklearn's default depth = 3.
- **RF contrast (RandomForestRegressor, default depth):** test R² **0.858 / 0.862 / 0.862** at
  B = 50 / 200 / 1000 — flat. More trees never hurts a forest (each an independent variance-reduction
  draw, ch 06 NB 4); GB at large ν is the opposite.

## Cell-by-cell (~20 cells; one declared concept; "Read the figure" after every figure)

1. (md) **Header** — `# 04 — Shrinkage and the trees: ν, depth, and the number of trees`;
   *Chapter 08 · Notebook 4 of 6*. Warm welcome: NB 1–3 built the engine; now the three dials that decide
   generalization, and the headline that more trees can make GB *worse*. **Prerequisites:** NB 1 (the
   residual loop, F += ν·tree, leaf = mean), NB 2 (residual = negative gradient, ν = step size), ch 06
   NB 4 (RF: more trees never hurts; deep trees), ch 07 NB 3 (boosting's overfitting behaviour).
   **What you'll be able to do:** read a staged test curve; set ν vs n_estimators knowingly; choose depth
   as interaction order; explain *mechanistically* why GB overfits with too many trees at large ν while
   RF does not.
2. (code) **Imports & setup** — numpy, pandas, matplotlib; `make_friedman1`, `train_test_split`,
   `GradientBoostingRegressor`, `RandomForestRegressor`, `DecisionTreeRegressor`, `r2_score`,
   `mean_squared_error`; `from ml_course import viz` + `viz.use_course_style()`; `from ml_course.colors
   import COLORS`; `SEED=0`. Build the friedman1 split; wrap X in a DataFrame `x0..x9`; print shapes,
   `.head()`, and the target's `describe()`.
3. (md) **Recap & meet the data.** Recap NB 1/2: F₀ = mean → fit a regression tree to the residual
   (= negative gradient) → add **ν × tree** → repeat. Three knobs decide generalization: **ν** (step
   size), **depth** (tree expressiveness), **n_estimators** (number of steps). Introduce friedman1
   honestly: the known formula, 5 informative + 5 noise features, the real x₀·x₁ interaction. A dataset
   whose structure we *know*, so we can watch each dial work.
4. (md) **Intuition — ν is a step-size brake.** ν multiplies every tree before it's added (F += ν·tree).
   ν=1 takes the whole step the tree suggests; ν=0.1 a tenth. Small steps move slowly but aim finely; big
   steps move fast but overshoot. ν trades **speed for precision** — and is coupled to the number of trees.
5. (code) **By hand — ν shrinks each step.** F₀ = ytr.mean() (14.13); fit one depth-3 tree to the
   residual; print train MSE after the step at ν=1.0 (→9.63) vs ν=0.1 (→22.31). Then the by-hand loop's
   "trees to reach train MSE ≤ 2": 13 / 48 / 496 for ν=1.0 / 0.1 / 0.01.
6. (md) **Read the result.** One tree at ν=0.1 covers a tenth of the ground a full step would, so it needs
   roughly ten times the trees to get as far — the **ν × n_estimators trade-off**. (Honest nuance: at large
   ν the step size isn't the bottleneck — ν=1.0 and ν=0.5 reach the target in ≈12–13 trees; the trade-off
   bites as ν shrinks.) Why shrink at all? Because where you *end up* differs — next figure.
7. (md) **Intuition — why small steps generalize better.** Full-strength trees chase the training residual
   hard, committing to whatever it says, noise included; small steps average many gentle corrections,
   leaning on signal consistent across rounds. Shrinkage is a **regularizer** — not free (it costs trees),
   but it buys a lower test error and protection from overfitting.
8. (code) **Fig G — the ν × n_estimators trade-off.** GBR(n_estimators=1000, max_depth=3) for
   ν ∈ {1.0, 0.1, 0.01}; staged **test R²** via `staged_predict` + `r2_score`. Plot test R² vs number of
   trees, **log-x**, three curves (semantic colours: large-ν error/coral, best-ν highlight, tiny-ν muted),
   peak markers. Print each best R²@trees and R²@1000. *(notebook-local matplotlib — three curves, the
   helper does two.)*
9. (md) **Read the figure (G).** ν=1.0 shoots up, peaks at **18 trees (0.864)**, then **sags to 0.813** —
   extra full-strength trees now fit noise. ν=0.1 climbs slower to a **higher** peak (0.930 @ ~308) then
   stays flat — more trees don't hurt it within budget. ν=0.01 is **still climbing** at 1000 (0.921) — too
   timid. Smaller ν reaches a **better** place AND is **safer**, but needs more trees: the cost of
   shrinkage is trees, the reward is generalization.
10. (md) **Intuition — the headline: more trees can make GB worse.** The sharp break from ch 06. A forest
    averages independent trees — more only sharpens the average, error flattens, never climbs. GB is
    **sequential**: each tree fits the *current* residual and **adds capacity**. Past the point the
    residual is signal, new trees fit noise and test error **rises**. "More trees is always safe" is true
    for RF, **false for GB at large ν**.
11. (code) **Fig H — the overfit, made unmistakable (+ the RF contrast).** ν=1.0, depth=3:
    `viz.plot_train_test_curve(trees, train_MSE, test_MSE, xlabel="number of trees", ylabel="MSE")`; set
    the ax to **log-x**; overlay a flat dashed RF reference (RandomForestRegressor test MSE, ~constant
    across B, in `COLORS["muted"]`). Train MSE → ~0; test MSE bottoms 3.39@18 then climbs to 4.65@1000.
    Print RF test R² 0.858/0.862/0.862 (B=50/200/1000).
12. (md) **Read the figure (H).** Train MSE marches to zero — the ensemble memorizes 1400 points. Test
    MSE turns at 18 trees and climbs: the overfit is real and, at ν=1, fast. The RF line barely moves from
    B=50 to 1000 — adding trees to a forest can't raise its complexity (independent variance-reduction
    draws, ch 06 NB 4), so it can't overfit this way. Same "more trees", opposite outcome — boosting builds
    *dependence*, a forest builds *independence*. (This rising curve is exactly why NB 5 brings **early
    stopping**.)
13. (md) **Intuition — depth is the interaction order.** The third dial. A stump (depth 1) splits on
    **one** feature — it can only sum single-feature effects (an additive model). Depth 2 lets a leaf
    depend on **two** features at once — a 2-way interaction like x₀·x₁. Depth d captures up-to-d-way
    interactions. friedman1 *has* a sin(π·x₀·x₁) term, so we expect stumps to fall short and depth ≥ 2 to
    capture it.
14. (code) **Fig I — depth sweep.** GBR(n_estimators=300, ν=0.1) for max_depth ∈ {1,2,3,5};
    `viz.plot_train_test_curve(depths, train_R2, test_R2, xlabel="max_depth (interaction order)",
    ylabel="R²")`. Print depth1 0.873/0.905, depth2 0.931/0.966, depth3 0.929/0.983, depth5 0.923/0.998.
15. (md) **Read the figure (I).** Stumps plateau at test R² 0.873 — single-feature effects only, x₀·x₁ out
    of reach (train R² 0.905: they underfit even train). Depth 2 jumps to 0.931 — the interaction captured.
    Depth 3 is about the same on test (0.929) while train keeps rising; depth 5 pushes train to 0.998 but
    test slips to 0.923 — deeper trees memorize. GB keeps trees **shallow** (default depth 3): enough
    interaction order, not enough to memorize — the **mirror image** of a random forest's deep trees
    (ch 06). Boosting wants weak learners; bagging wants strong ones.
16. (md) **The three dials together.** Synthesis + a small table (dial · what it does · which way it
    overfits · our friedman1 reading): ν and n_estimators trade off (small ν + more trees ≈ the sweet spot,
    best generalization); depth = interaction order (shallow, matched to the problem's real interactions —
    here 2). The danger unique to boosting: too many trees at large ν. The cure (NB 5): small ν, shallow
    depth, and **stop adding trees when validation error stops improving**.
17. (md) **Your turn (tiered).** *easy:* from Fig G, name the round ν=1.0 peaks and say in one sentence why
    every later tree hurts. *medium:* re-run the ν-sweep with ν=0.05 — does it beat ν=0.1's best test R²
    within 1000 trees, and about how many trees does it need? Relate to the trade-off. *harder:* friedman1's
    only interaction is the 2-way x₀·x₁ — predict what max_depth=2 vs 1 does to test R², check it, then
    argue why no depth beyond 2 is *needed* for this target (even though 3 is the default).
18. (md) **What you built.** Bullets: felt ν as a step-size brake and measured the ν × n_estimators
    trade-off (small ν needs more trees, generalizes better); saw **more trees can make GB worse** (test
    bottoms then rises at large ν) with the mechanistic RF contrast; learned **depth = interaction order**
    (shallow trees, the mirror of RF's deep ones). **Vocabulary:** learning rate / shrinkage ν · the
    ν × n_estimators trade-off · staged test curve · overfitting with too many trees · interaction order ·
    weak (shallow) learner.
19. (md) **Going further (optional).** Shrinkage as regularization (Friedman 2001 §5; ESL §10.12 —
    empirically ν ≤ 0.1 with more trees generalizes best). Tree depth ≈ interaction order (ESL §10.11).
    Stochastic GB (`subsample`, Friedman 2002) is another regularizer (NB 5). The principled cure for the
    rising curve is **early stopping** (NB 5). A forest's test error is ~monotone in B (ch 06 NB 4); a
    boosting ensemble's is U-shaped at large ν — same word "trees", different mathematics.
20. (md) **References** — Friedman 2001 (DOI 10.1214/aos/1013203451, §5 shrinkage); Friedman 1991 —
    MARS / the friedman1 function (DOI 10.1214/aos/1176347963); Friedman 2002 — Stochastic GB
    (DOI 10.1016/S0167-9473(01)00065-2); ESL §10.11–10.12 (DOI 10.1007/978-0-387-84858-7); ch 06 NB 4;
    ch 07 NB 3. `Previous: 03 — Gradient boosting for classification.`
    `Next: 05 — The estimator and its parameters.`

## Figures (3, each followed by "Read the figure")
- **Fig G (cell 8)** — ν × n_estimators: test R² vs number of trees (log-x) for ν ∈ {1.0, 0.1, 0.01};
  large-ν overfits (peak@18 then sag), best-ν highest+flat, tiny-ν still climbing. *notebook-local
  matplotlib (3 curves).*
- **Fig H (cell 11)** — the overfit + RF contrast: ν=1.0 train vs test MSE vs trees (log-x), train→0 /
  test bottoms@18 then climbs, with a flat dashed RF reference. *reuse `viz.plot_train_test_curve`, add
  the RF line + log-x on its ax.*
- **Fig I (cell 14)** — depth = interaction order: test & train R² vs max_depth ∈ {1,2,3,5}; stumps
  underfit (no x₀·x₁), depth 2 jumps, depth 5 memorizes. *reuse `viz.plot_train_test_curve`.*

## `src/` & guards
**No `src/` change** (reuse `viz.use_course_style` + `viz.plot_train_test_curve`; Fig G is notebook-local
matplotlib with `ml_course.colors`; pytest stays **20**). Build via
`uv run python <scratchpad>/build_ch08_nb4.py`; **re-measure every anchor at build** (the ν-sweep staged
R², the by-hand ν steps, the depth sweep, the RF flat line — do not hard-code beyond the seed). nbconvert
top-to-bottom **from project cwd** on a scratchpad copy (tracked file **output-free**); **banned-word scan
over JSON real cell text** = 0 (watch "just"/"simply"); `check_no_hardcoded_hex` passes; `ruff`/`black`
clean; `gen_llms_txt` re-run.

## Honest scoping
Richer-scope single concept (how GB controls complexity), depth kept subordinate — the ch 07-NB 3
precedent. The overfit is **ν-dependent** (sharp at ν=1, unseen within budget at ν=0.1) — never
overgeneralized to "GB always overfits with more trees". The RF contrast is **mechanistic** (independence
vs dependence), not a performance claim (GB at ν=0.1 actually beats RF here — irrelevant to the point).
Still a synthetic teaching set with known structure (the real demanding case is NB 6). Early stopping is
*named* as the cure, *built* in NB 5. Both reviewers PASS (no BLOCK) + Rémy validates visually before
commit; ff-merge notebook → chapter.
