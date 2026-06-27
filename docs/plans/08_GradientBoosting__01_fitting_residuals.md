# Notebook plan — 08_GradientBoosting / 01_fitting_residuals

> Status: **APPROVED by Rémy (2026-06-25) & persisted.** No reviewer gate at NB-plan stage; both
> reviewers return on the *built* notebook. Next: build from `build_ch08_nb1.py`.

## Context

NB **1 of 6** — the first fundamentals notebook of chapter 08. **ONE concept:** gradient boosting builds
a model by **fitting each new (regression) tree to the residuals of the current model, adding a shrunken
slice, and repeating** — the opposite mechanism to AdaBoost's reweighting, the same goal (focus on the
mistakes). Built entirely **by hand** on a 1-D synthetic regression, closing with **exact parity** to
`GradientBoostingRegressor`. This is the **course's first regression** problem (re-laid honestly).
**"Gradient" is deliberately NOT named** — NB 2 reveals that the residual is the negative gradient.

## Anchors (measured on sklearn 1.9.0, seed 0 — re-measured at build)

- **Data:** `y = sin(x) + N(0, 0.25²)`, `x ~ Uniform[0, 2π]`, n=120, seed 0; `y.mean() = −0.1199`.
- **Config:** by-hand GB — F₀ = mean → `DecisionTreeRegressor(max_depth=2, random_state=0)` on the
  residual → `F ← F + ν·tree`, **ν = 0.3**, M = 100.
- **Parity:** by-hand **== `GradientBoostingRegressor(loss='squared_error', n_estimators=100,
  learning_rate=0.3, max_depth=2, subsample=1.0, random_state=0)` to 2.22e-16** (final & every staged
  round); F₀ = −0.1199 = `init_.constant_` (`DummyRegressor`, mean).
- **Train MSE:** 0.502 (F₀) → 0.299 @1 → 0.191 @2 → 0.135 @3 → 0.081 @5 → 0.050 @10 → 0.039 @20 →
  0.018 @50 → 0.0073 @100. Residual std 0.7085 (F₀) → shrinks with rounds.
- **Single `DecisionTreeRegressor(max_depth=2)` on y:** MSE 0.105 — the ensemble passes it ≈ round 5.
- **API:** `staged_predict` present; **`staged_score` ABSENT**; `init_` = `DummyRegressor`; default loss
  `squared_error`.

## Cell-by-cell (~21 cells; one concept; "Read the figure" after every figure)

1. (md) **Header** — `# 01 — Boosting as fitting residuals (by hand)`; *Chapter 08 · Notebook 1 of 6*;
   warm welcome (the start of gradient boosting — the general form of the boosting family; built, like
   AdaBoost, by hand first). **Prerequisites:** ch 07 (sequential boosting; the additive model F = Σ of
   weak learners; "weak learner"); ch 04 (decision trees — here *regression* trees); ch 00 NB 1/4
   (regression named; train/test; MSE). **What you'll be able to do:** explain how GB improves a model by
   fitting its residuals; build the loop by hand; match `GradientBoostingRegressor` exactly; read the
   residual-shrinking and MSE-vs-trees curves.
2. (code) **Imports & setup** — numpy, matplotlib, pandas; sklearn `DecisionTreeRegressor`,
   `GradientBoostingRegressor`, `mean_squared_error`; `from ml_course import viz`, `viz.use_course_style()`;
   colours from `ml_course.colors`; `rng = np.random.default_rng(0)` (documented seed).
3. (md) **Recap & footing** — ch 07: boosting trains weak learners *sequentially*, each focused on the
   running ensemble's mistakes, combined additively; AdaBoost did it by **reweighting points**. Chapter 08
   does it a different way. **Re-lay regression (the course's first):** the target is a continuous number
   (not a class); we predict a number; the error is the **residual** r = y − prediction; we score with
   **squared error / MSE**. **And re-lay the regression tree:** a regression tree predicts the **mean of
   the training targets in each leaf** — the constant minimizing squared error there (ch 04's trees voted
   a class; same splits, a numeric leaf). We lean on this "leaf = mean" fact.
4. (md) **Intuition — fit the leftovers.** Start with the simplest model: a constant, the **mean** of y.
   It is wrong almost everywhere — look at what is *left over* (the residuals). Fit a small tree to those
   residuals: it captures structure the constant missed. Add a **shrunken slice** of it (a fraction ν).
   Now there are new, smaller leftovers — fit another tree to *those*. Repeat: each tree cleans up the
   previous model's residuals. (Contrast AdaBoost: reweight the hard points; here we fit the residuals
   directly.)
5. (code) **The data + the starting model.** Build the 1-D synthetic as a DataFrame (`x`, `y`) + the true
   sin curve for reference. Plot: scatter of (x, y), the true curve (muted), and the flat line F₀ = mean.
   Print MSE(F₀) = 0.502.
6. (md) **Read the figure** — the mean is a flat, poor model; the points sit in a clear sine pattern above
   and below it; that pattern *is* the residual we will attack. MSE 0.502.
7. (md) **Intuition — round 1.** Compute residuals r = y − F₀, fit a depth-2 tree to r (not to y), and add
   ν = 0.3 of its prediction. ν is a **small step** ("a shrunken slice"); *why* a fraction rather than the
   whole tree is studied in NB 4 — here we just take careful steps.
8. (code) **By hand — round 1.** `F0 = y.mean()`; `r = y - F0`; `tree1 =
   DecisionTreeRegressor(max_depth=2, random_state=0).fit(X, r)`; `F1 = F0 + 0.3 * tree1.predict(X)`.
   Print MSE 0.502 → 0.299.
9. (code) **Fig A — round-1 mechanics** (2 panels): (left) the residuals r as points with `tree1`'s step
   prediction overlaid (the structure the tree grabbed); (right) the data with the updated fit F₁ (no
   longer flat — it bends toward the sine). Charter colours.
10. (md) **Read the figure (A)** — the tree found residuals negative on the left, positive in the middle,
    etc., and drew a coarse step through them; adding 0.3× that step bent the flat line toward the data.
    One weak tree, one small improvement: MSE 0.502 → 0.299.
11. (md) **Intuition — repeat.** Round 2 fits a tree to the *new* residuals y − F₁, and so on. The loop:
    `F_m = F_{m−1} + ν · tree_m(x)`, with `tree_m` fit on `r = y − F_{m−1}`.
12. (code) **By hand — the loop.** `for m in range(100)`: `r = y − F`; `tree =
    DecisionTreeRegressor(max_depth=2, random_state=0).fit(X, r)`; `F += 0.3 * tree.predict(X)`; record
    MSE + snapshots. Print MSE at a few rounds (0.299@1 … 0.0073@100).
13. (code) **Fig B — the fit builds up + the MSE curve** (2 panels): (left) data with cumulative F_m
    overlaid for m ∈ {0, 1, 3, 10, 60} (flat → sine, a colour gradient); (right) train MSE vs number of
    trees (0.502 → 0.0073), with the single-tree MSE (0.105) as a horizontal reference line.
14. (md) **Read the figure (B)** — as trees accumulate, the staircase fit tightens onto the sine and the
    MSE falls steeply, then more slowly; a **single** depth-2 tree alone reaches only 0.105 (the reference
    line), but a sum of shrunken depth-2 trees sails past it by ≈ round 5 and keeps improving — weak
    learners, stacked on each other's leftovers, become a strong one.
15. (md) **Intuition — did we get it right?** We built this by hand. The real `GradientBoostingRegressor`
    does exactly this — let us check, the course's by-hand↔library contract.
16. (code) **Parity with the library.** `GradientBoostingRegressor(n_estimators=100, learning_rate=0.3,
    max_depth=2, loss='squared_error', subsample=1.0, random_state=0)`; compare `.predict(X)` to the
    by-hand F (`max|Δ| = 2.22e-16`); compare `staged_predict` to the by-hand stages (same to ~1e-16).
    Note `init_` is the mean (our F₀), and that staged predictions come from **`staged_predict`** — there
    is **no `staged_score`** (a change from AdaBoost; compute MSE over `staged_predict` yourself).
17. (md) **Read the result / celebrate** — by-hand and library agree to ~1e-16 (machine precision): you
    built gradient boosting's regression core by hand. (We have not used the word "gradient" once — that
    reveal, and *why* the residual is exactly the right thing to fit, is NB 2.)
18. (md) **Your turn** (tiered) — *easy:* set ν = 0.1 and ν = 1.0, re-run the loop, describe how the
    MSE-vs-trees curve changes (slower vs faster — a forward look at NB 4). *medium:* change `max_depth` to
    1 (stumps) and to 3; how do the single-tree baseline and the ensemble curve move? *harder:* start
    F₀ = 0 instead of the mean; show it costs only an extra round or two to catch up, and explain (from the
    leaf = mean fact) why the mean is the best possible constant start.
19. (md) **What you built** — bullets: started from the mean and *fit the residuals* with shallow trees,
    adding a shrunken slice each round; watched MSE fall 0.502 → 0.007 and the fit grow from a flat line
    into the sine; matched `GradientBoostingRegressor` to machine precision. **Vocabulary:** residual ·
    additive model · shrinkage / learning rate ν · weak learner (a shallow regression tree) · staged
    prediction · regression tree leaf = mean.
20. (md) **Going further (optional)** — why the mean is F₀ (the constant minimizing squared error); why
    depth-2 leaves need no extra work — for squared error the tree's leaf (the residual mean) already is
    the optimal per-leaf value (Friedman 2001 §4.1), a fact **NB 3 will contrast** with classification,
    where the leaf needs a correction. Forward: **NB 2** reveals the residual y − F is the **negative
    gradient** of the squared-error loss, so this whole loop is gradient descent — in the space of
    functions.
21. (md) **References** — Friedman 2001 (DOI 10.1214/aos/1013203451, §4.1); ESL §10.9–10.10 (DOI
    10.1007/978-0-387-84858-7); ISLR §8.2.3 (DOI 10.1007/978-1-0716-1418-1). `Previous: Chapter 07 —
    AdaBoost.` `Next: 02 — The residual was the gradient.`

## Figures (3, each followed by "Read the figure")
- **cell 5** — data + true curve + the flat F₀ = mean line.
- **Fig A (cell 9)** — round-1 mechanics: residuals + the tree's step | the updated fit F₁.
- **Fig B (cell 13)** — the fit builds up (cumulative F_m, m ∈ {0,1,3,10,60}) | MSE-vs-trees with the
  single-tree reference line.
All notebook-local matplotlib with `ml_course.colors` (no new `viz` helper — these residual-fitting
visuals are bespoke to NB 1).

## `src/` & guards
**No `src/` change** (pytest stays 20). Build via `uv run python <scratchpad>/build_ch08_nb1.py`;
**re-measure parity + the MSE curve at build**; run nbconvert top-to-bottom **from project cwd** on a
scratchpad copy (tracked file **output-free**); **banned-word scan over JSON real cell text** = 0;
`check_no_hardcoded_hex` passes; `ruff`/`black` clean; `gen_llms_txt` re-run. Honest scoping: this is
*regression* (first in the course, re-laid); "gradient" not named (NB 2); greedy stagewise ≠ global
optimum; exact parity is a squared-error gift (classification needs leaf re-estimation — flagged forward
to NB 3). Both reviewers PASS (no BLOCK) + Rémy validates visually before commit; ff-merge notebook →
chapter.
