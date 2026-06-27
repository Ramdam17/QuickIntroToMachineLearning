# Notebook plan — 09_XGBoost / 01_second_order_view

> Status: **APPROVED by Rémy (2026-06-27)** (no reviewer gate at the per-NB plan stage — both reviewers
> return on the built notebook). All anchors measured live (xgboost 3.2.0 / sklearn 1.9.0, SEED=0);
> re-measured at build. Build via `<scratchpad>/build_ch09_nb1.py`.

## Context

NB **1 of 5**, the chapter's first fundamental. **One concept:** approximate any loss to **second
order** around the current prediction and read off the optimal leaf constant — `w* = −G/H`. The payoff
is a unification: Chapter 08's *two* leaf rules (squared-error leaf = mean residual, NB 1; log-loss
Newton leaf `Σ(y−p)/Σp(1−p)`, NB 3) are revealed to be the **same** second-order rule, one loss apart.
That rule is the engine XGBoost generalizes. Built by hand, then checked against XGBoost with its
regularizer turned off (λ deferred to NB 2).

## Anchors (measured live, xgboost 3.2.0 / sklearn 1.9.0, SEED=0 — re-measured at build)

- **Scalar 2nd-order step** (one point, log-loss, y=1, F=0): `g=p−y=−0.5`, `h=p(1−p)=0.25` →
  `w*=−g/h=+2.0`; gradient-only step `−g=+0.5` (too short). The quadratic approximation's argmin = −g/h.
- **Mixed leaf for the figures** `{y=1, y=1, y=0}` at F=0: `G=−0.5`, `H=0.75` → `w*=−G/H=+0.6667`; the
  true leaf optimum is `log(2/1)=+0.6931` — one Newton step lands close, not exact (so boosting
  iterates). *(Finite min — unlike a single point, whose log-loss min is at +∞; use a mixed leaf.)*
- **Regression recovery** (ch 08 NB 1 sine: `y=sin(x)+N(0,0.25²)`, n=120): `F0=mean(y)=−0.1199`
  (matches ch 08). First `max_depth=2` tree on residuals — per leaf `−G/H` **==** mean residual **==**
  tree leaf value: `+0.29451 / +0.75892 / −0.68149 / −0.22921`. (h=1 → the mean IS the 2nd-order optimum.)
- **Classification recovery** (ch 08 NB 3 moons: `make_moons(400, noise=0.20)`, split 280/120):
  `p0=0.500`, `F0=log-odds=0.0000`. First `max_depth=2` tree on pseudo-residuals — per leaf `−G/H`
  **==** Newton `Σ(y−p)/Σp(1−p)`: `−2.0 / +1.72881 / −1.62590 / +2.0`. (h=p(1−p) → the Newton leaf.)
- **λ=0 XGBoost parity** (toy `x=0..5`, `y=[1,1.2,0.9,5,5.3,4.8]`, `XGBRegressor(n_estimators=1,
  max_depth=1, learning_rate=1, reg_lambda=0, gamma=0, base_score=mean(y)=3.033, tree_method='exact')`):
  split at x<2.5, leaf weights `[−2.0, +2.0]` **==** by-hand `−G/H` exactly.

## Cell-by-cell (~22 cells; 3 figures; "Read the figure" after each)

1. (md) **Header** — `# 01 — The second-order view: gradients and curvature`; *Chapter 09 · Notebook 1
   of 5*. The chapter's promise: XGBoost is Chapter 08's engine, sharpened — and the first sharpening is
   to use the loss's **curvature**, not only its slope. **Prerequisites:** ch 08 NB 2 (the residual is
   the negative gradient; boosting = gradient descent in function space), ch 08 NB 3 (the Newton leaf
   for classification), ch 04 (regression trees, leaf = mean). **What you'll do:** re-derive the
   second-order step by hand → `w*=−G/H` → watch it *become* ch 08's two leaf rules → check it against
   XGBoost with regularization off.
2. (md) **Recap — where Chapter 08 left us.** Boosting fits the negative gradient with trees. For
   squared error the leaf is the mean residual (NB 1). For log-loss, NB 3 needed a *correction* — the
   Newton leaf `Σ(y−p)/Σ p(1−p)` — applied silently by the library. Two different-looking leaf rules.
   This notebook shows they are **one** rule, and it is the rule XGBoost is built on.
3. (code) **Setup** — imports (`numpy`, `matplotlib`; `DecisionTreeRegressor`; `make_moons`,
   `train_test_split`; `XGBRegressor`); `viz.use_course_style()`, `COLORS`; `SEED=0`; a `sigmoid`
   helper. (Visible output; no silencing.)
4. (md) **The idea: approximate the loss to second order.** Around the current prediction F, a smooth
   loss obeys `L(F+w) ≈ L(F) + g·w + ½h·w²`, with `g=∂L/∂F` (slope) and `h=∂²L/∂F²` (curvature) — a
   parabola in the step `w`. Set the derivative `g + h·w = 0` → the minimum is **`w*=−g/h`**. The
   gradient says *which way*; the curvature says *how far*.
5. (code) **Fig 1 — the parabola.** Take a small mixed leaf `{1,1,0}` at F=0 (log-loss); plot the
   **true** leaf objective `Σ loss(F+w)` vs `w` and its second-order approximation (the parabola); mark
   `w*=−g/h≈0.667` and the true optimum `log2≈0.693`. *notebook-local.*
6. (md) **Read Fig 1.** The parabola hugs the true loss near `w=0`; its minimum `−G/H≈0.667` is one good
   step toward the true leaf optimum `0.693` — close, not exact in a single step, which is exactly why
   boosting takes many. Curvature sets the step length: small `h` (flat) → a long step, large `h`
   (steep) → a cautious one.
7. (code) **Fig 2 — gradient-only vs second-order step.** On the same true-loss curve, mark where a
   pure gradient step `−g` (or `−η·g`) lands vs where `−g/h` lands. *notebook-local.*
8. (md) **Read Fig 2.** The gradient step ignores how fast the slope changes, so it lands short;
   dividing by the curvature `h` rescales it onto the minimum. This is **Newton's method** — re-laid
   from scratch, the move ch 08 NB 3 used once for log-loss.
9. (md) **From one point to a leaf.** A leaf holds many points; its objective is their sum. Summing the
   approximations gives `Σᵢ[gᵢ·w + ½hᵢ·w²]`, minimized at **`w*=−G/H`** with `G=Σgᵢ`, `H=Σhᵢ`. One
   formula for the best constant on a leaf — for **any** twice-differentiable loss.
10. (md) **Pin the convention (once).** `gᵢ=∂L/∂Fᵢ`, `hᵢ=∂²L/∂Fᵢ²`, leaf `w*=−G/H`. The minus sign is
    what turns ch 08's "negative gradient" into a positive update. Now watch two losses fall out.
11. (md) **Loss 1 — squared error.** `L=½(y−F)²` → `g = F−y` (= −residual), `h = 1`. So
    `w* = −G/H = Σ(y−F)/n` = the **mean residual**. Constant curvature (h=1) → the second-order step is
    just the mean. *This is ch 08 NB 1's leaf = mean — now you see why.*
12. (code) **Regression recovery (ch 08 NB 1 sine).** Rebuild the sine data; `F0=mean(y)=−0.1199`; fit
    the first `max_depth=2` tree to the residuals; for each leaf print `−G/H`, the mean residual, and
    the tree's own leaf value — all identical (`+0.29451 / +0.75892 / −0.68149 / −0.22921`).
13. (md) **Read it.** Three columns, one number. For squared error the regression tree's native leaf
    (the mean) already **is** the second-order optimum, because `h=1`. No correction needed — which is
    why ch 08 NB 1 never had to mention curvature.
14. (md) **Loss 2 — log-loss.** `g = p−y`, `h = p(1−p)`. So `w* = −G/H = Σ(y−p)/Σ p(1−p)`. The curvature
    now **varies** with `p`: confident points (p near 0 or 1) have tiny `h`; uncertain points (p≈0.5)
    have `h≈0.25`. *This is exactly ch 08 NB 3's Newton leaf.*
15. (code) **Classification recovery (ch 08 NB 3 moons).** Rebuild moons; `F0=log-odds(0.5)=0`; fit the
    first `max_depth=2` tree to the pseudo-residuals; per leaf print `−G/H` and `Σ(y−p)/Σ p(1−p)` —
    identical (`−2.0 / +1.72881 / −1.62590 / +2.0`).
16. (code) **Fig 3 — two losses, one rule.** Two panels: (left) the curvature `h` vs the margin — flat
    `h=1` for squared error, the bell `p(1−p)` for log-loss; (right) the leaf formula annotated for each
    (`−G/H` = mean vs Newton). *notebook-local.*
17. (md) **Read Fig 3 — the unification.** Chapter 08's two leaf rules were never two rules. Both are
    `w*=−G/H`. Squared error has constant curvature (`h=1`), so `−G/H` is the plain mean. Log-loss has
    curvature `p(1−p)`, so `−G/H` weights each point by its certainty — the Newton leaf. **One**
    second-order rule; the loss supplies `(g,h)`. This is the engine XGBoost generalizes to any
    differentiable loss.
18. (code) **The library anchor — XGBoost with its regularizer off.** Tiny toy; `XGBRegressor(
    n_estimators=1, max_depth=1, learning_rate=1, reg_lambda=0, gamma=0, base_score=mean(y),
    tree_method='exact')`; dump `trees_to_dataframe`; the leaf weights `[−2.0, +2.0]` equal the by-hand
    `−G/H` exactly.
19. (md) **Read it.** With its regularizer switched off, XGBoost's leaf is precisely our `−G/H`. (NB 2
    turns `λ` on and watches the leaf shrink — the "regularized" in XGBoost.) We pinned `base_score` so
    `F₀` is a known constant for the hand-check; left free, XGBoost *learns* it — the role ch 08's
    `init_` played.
20. (md) **Your turn.** *easy:* for a leaf of 5 points with given `(g,h)`, compute `−G/H` by hand.
    *medium:* show that for squared error `−G/H` is the mean residual for any leaf size; then derive
    `(g,h)` for absolute-error loss and say what its `h` does to the rule. *harder:* using Fig 1,
    explain why one Newton step does not reach the leaf optimum for log-loss, and why boosting's many
    shrunken steps (the learning rate `ν`, ch 08 NB 4) do.
21. (md) **What you built.** You replaced Chapter 08's two leaf recipes with one second-order rule
    `w*=−G/H` — gradient for direction, curvature for step length. Squared error (`h=1`) → the mean;
    log-loss (`h=p(1−p)`) → the Newton leaf. And XGBoost, with its regularizer off, lands exactly there.
    **Vocabulary:** gradient `g` · Hessian `h` · second-order / Newton step · leaf weight `−G/H`. Next:
    turn on the regularizer (`λ`, `γ`) and the split-gain it implies.
22. (md) **References** — Chen & Guestrin 2016 (eq. 6 — the structure-score / leaf weight,
    DOI 10.1145/2939672.2939785); Friedman 2001 (TreeBoost — the Newton leaf, DOI 10.1214/aos/1013203451);
    ESL §10.10 (DOI 10.1007/978-0-387-84858-7). `Previous: Chapter 08 — Gradient Boosting (capstone).`
    `Next: 02 — The regularized objective.`

## Figures (3, each followed by "Read the figure")
1. **The parabola** (cell 5) — true leaf loss vs its 2nd-order approximation; `w*=−g/h` marked. *local.*
2. **Gradient vs second-order step** (cell 7) — `−g` lands short, `−g/h` lands at the min. *local.*
3. **Two losses, one rule** (cell 16) — curvature `h` (flat 1 vs bell `p(1−p)`) + the leaf formula each. *local.*

## `src/` & guards
**No `src/` change** — notebook-local matplotlib + `viz.use_course_style`; datasets rebuilt inline
(ch 08 sine + moons; a 6-point toy); XGBoost via `trees_to_dataframe`; **pytest stays 20**. Build via
`uv run python <scratchpad>/build_ch09_nb1.py`; **re-measure every anchor at build**; nbconvert
top-to-bottom from project cwd on a scratchpad copy (tracked file **output-free**); **banned-word scan
over JSON cell text** = 0 (watch "just"/"simply"); `check_no_hardcoded_hex` passes; ruff/black clean;
`gen_llms_txt` re-run. Two-reviewer gate (no BLOCK) + Rémy visual before commit; ff-merge
`notebook → chapter`.

## Honest scoping
One concept (the second-order view) built by hand before the library. The honest nuance is surfaced,
not hidden: a single Newton step does **not** reach the leaf optimum (Fig 1) — it is the best *local*
quadratic step, and boosting iterates. `base_score` is pinned only to make `F₀` a known constant for
the hand-check (it is learned when left free). The λ=0 XGBoost parity is exact on XGBoost's own split;
regularization (the `H+λ` denominator, the split gain) is deliberately deferred to NB 2.
