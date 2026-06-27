# Notebook plan — 08_GradientBoosting / 02_residual_is_gradient

> Status: **APPROVED by Rémy (2026-06-25) & persisted.** No reviewer gate at NB-plan stage; both
> reviewers return on the *built* notebook. Next: build from `build_ch08_nb2.py`.

## Context

NB **2 of 6** — the second fundamentals notebook. **ONE concept:** the residual we fit in NB 1 **is the
negative gradient** of the squared-error loss, so the whole boosting loop is **gradient descent in
function space** — the ensemble F is a point in ℝⁿ (one coordinate per training prediction), and each
tree is an approximate step downhill, ν the step size. This is the reveal NB 1 deliberately withheld
("we have not used the word *gradient* once"). It stays **regression** on NB 1's 1-D sine — no new data,
no new estimator — and ends by generalising: a *different loss* gives a *different gradient* gives a
*different residual* (absolute error → sign; sets up NB 3's classification). Re-lays ch 03 NB 4's
gradient descent (which lived in **parameter** space) and lifts it to **function** space.

## Anchors (measured on sklearn 1.9.0, seed 0 — re-measured at build; same data/loop as NB 1)

- **Data/loop:** NB 1's `y = sin(x) + N(0, 0.25²)`, n=120, x∈[0,2π]; F₀=mean, `DecisionTreeRegressor(max_depth=2)`
  on the residual, `F += ν·tree`, ν=0.3, M=100.
- **Negative gradient == residual:** for `L = ½·Σ(yᵢ − Fᵢ)²`, `−∂L/∂Fᵢ = yᵢ − Fᵢ`. Finite-difference
  check at round 3: `max|(−∂L/∂F) − (y − F)| = 8.7e-11` (numerically exact; analytically exact). Example
  point: residual 0.0797 = −∂L/∂F 0.0797.
- **The loss descends:** `L = ½·Σ(y − F)²` = **30.12 (F₀) → 17.96 @1 → 3.02 @10 → 0.44 @100**; and
  `L = (n/2)·MSE` exactly (30.12 = 60·0.502).
- **Two-point function-space slice:** the points nearest the peak (`x ≈ π/2`, y = 0.908) and the trough
  (`x ≈ 3π/2`, y = −1.152); the trajectory `(F_i, F_j)` runs from `(−0.12, −0.12)` (both start at the
  mean) to `(0.91, −1.16)` — into the slice's loss minimum at `(y_i, y_j) = (0.908, −1.152)`.
- **A different loss → a different gradient:** squared error → `y − F` (smooth; range at round 3
  [−0.79, 0.83]); absolute error `|y − F|` → `sign(y − F) ∈ {−1, +1}` (robust). Forward to NB 3:
  log-loss → `y − p`.
- **Parity (unchanged from NB 1):** the "fit the negative gradient" loop == `GradientBoostingRegressor`
  to **2.22e-16** (the reframe changes the words, not the arithmetic).

## Cell-by-cell (~21 cells; one concept; "Read the figure" after every figure)

1. (md) **Header** — `# 02 — The residual was the gradient`; *Chapter 08 · Notebook 2 of 6*; warm
   welcome (the reveal — where the name "gradient boosting" comes from). **Prerequisites:** NB 1 (the
   residual-fitting loop; exact parity with `GradientBoostingRegressor`); ch 03 NB 4 (gradient descent —
   stepping downhill on a loss); ch 00 (MSE). **What you'll be able to do:** show that the residual is
   the negative gradient of the squared-error loss; explain boosting as gradient descent in function
   space; see the loss descend on a 2-D slice; explain why a different loss changes what you fit.
2. (code) **Imports & setup** — numpy, matplotlib, pandas; `DecisionTreeRegressor`,
   `GradientBoostingRegressor`, `mean_squared_error`; `from ml_course import viz` + `viz.use_course_style()`;
   `COLORS`; `rng = np.random.default_rng(0)`. Rebuild NB 1's data and the by-hand loop (store `stages`,
   `trees`, the loss curve) — NB 2 re-illuminates the *same* loop.
3. (md) **Recap & a question.** NB 1: we started at the mean, fit a tree to the **residuals**, added a
   shrunken slice, repeated — and matched the library exactly. **Re-lay ch 03 NB 4:** there, training
   meant *gradient descent* — write the loss as a function of the weights, compute its gradient, step
   downhill. But in NB 1 we never wrote a gradient; we fit residuals. **Where is the gradient?**
4. (md) **Intuition — the loss as a function of the predictions.** Write `L(F) = ½·Σ(yᵢ − Fᵢ)²`. Think of
   the n predictions `F = (F₁, …, Fₙ)` as the variables — a single point in n-dimensional space. `L` is a
   bowl, smallest (zero) when `F = y`. Training = walking this point downhill toward `y`.
5. (code) **The negative gradient is the residual.** Compute `−∂L/∂Fᵢ` numerically (finite differences)
   at a chosen round; show it equals `y − F` (the residual) to ~1e-10. Print the identity and one point.
6. (md) **Read the result.** `−∂L/∂Fᵢ = yᵢ − Fᵢ` — the per-point negative gradient *is* the residual. So
   "fit a tree to the residual" (NB 1) = "fit a tree to the negative gradient" = take a step downhill on
   `L`. Because the variables are the function's values at the data, this is **gradient descent in
   function space**.
7. (md) **Intuition — an approximate, tree-shaped step.** Each round: `F ← F + ν·(tree ≈ negative
   gradient)`. A tree cannot set a free value per point — points sharing a leaf share one value — so the
   tree is a **piecewise-constant approximation** of the gradient. Boosting takes approximate downhill
   steps, each shaped like a tree, scaled by ν.
8. (code) **Fig C — the step picture.** At round 10, plot the per-point negative gradient `gᵢ = yᵢ − Fᵢ`
   (points) vs x, with the next tree's prediction overlaid (the piecewise-constant step actually taken).
9. (md) **Read the figure (C)** — the gradient says "go up here, down there"; the tree summarises that
   into a few flat moves (the best a depth-2 tree can do); ν scales the step. Boosting = approximate
   gradient descent, one tree-shaped step at a time.
10. (md) **Intuition — watch it descend.** The total loss `L = ½·Σ(y − F)²` (which equals `(n/2)·MSE`)
    should fall every round. On a 2-coordinate slice of function space we can *see* the path go downhill.
11. (code) **Fig D — gradient descent in function space** (2 panels): (left) the 2-point loss bowl over
    `(F_i, F_j)` for the points nearest the peak and trough — circular contours centred at `(y_i, y_j)` —
    with the boosting trajectory `(F_i^(m), F_j^(m))` stepping from `(mean, mean)` into the minimum;
    (right) the total loss `L` vs number of trees (30.1 → 0.44).
12. (md) **Read the figure (D)** — *left:* the path walks from the centre-of-mass start down into the
    bowl's minimum at `(y_i, y_j)` — literally gradient descent in (a 2-D slice of) function space;
    *right:* the total loss falls monotonically, each tree one step down. (The steps are not exactly
    steepest-descent — the tree constraint bends them — which is why it takes many small steps.)
13. (md) **Intuition — a different loss, a different gradient.** The residual = negative gradient identity
    is special to **squared error**. Change the loss and the thing you fit changes. What is the negative
    gradient of the absolute-error loss `|y − F|`?
14. (code) **A different loss.** Show `−∂/∂F |y − F| = sign(y − F) ∈ {−1, +1}` (the absolute-error
    "pseudo-residual"), printed beside squared error's `y − F` for a few points.
15. (md) **Read the result.** Squared error → fit the residual `y − F` (large where the miss is large —
    sensitive to outliers); absolute error → fit only its **sign** (robust). Classification (NB 3) → fit
    `y − p`. The general recipe: **fit the negative gradient of whatever loss you choose** — that is the
    "gradient" in gradient boosting, and why the family generalises AdaBoost's one fixed loss.
16. (code) **Parity, reframed.** The "fit the negative gradient" loop is byte-for-byte NB 1's loop, so it
    still equals `GradientBoostingRegressor` to 2.22e-16. A one-line confirmation (same arithmetic, new
    name).
17. (md) **Read / name it.** Boosting is gradient descent in function space; `GradientBoostingRegressor`
    fits the negative gradient of its `loss` each round. The name now makes sense — and NB 3 will swap the
    loss to do classification.
18. (md) **Your turn** (tiered) — *easy:* from the loss-vs-rounds, confirm `L = (n/2)·MSE` and describe the
    descent. *medium:* compute the absolute-error pseudo-residual (sign) for a few points including a noisy
    outlier, and say why it is more robust than `y − F`. *harder:* on the 2-point bowl, explain why the
    steps are not exactly along the steepest-descent direction (the tree groups points into leaves) and
    what that approximation costs.
19. (md) **What you built** — bullets: showed the residual is the negative gradient of squared-error loss;
    reframed NB 1's loop as gradient descent in function space; watched the loss descend on a 2-D slice;
    saw that a different loss changes the gradient you fit. **Vocabulary:** negative gradient ·
    pseudo-residual · function space · gradient-descent step · loss bowl.
20. (md) **Going further (optional)** — the functional-gradient view (Friedman 2001): treat F as a
    function, the negative gradient is itself a function (its pointwise values), and the tree **projects**
    it onto the space of trees; the per-leaf value is a 1-D line search (trivial = the mean for squared
    error — NB 1). NB 3 meets a loss whose leaf line search is **not** trivial (the Newton step).
21. (md) **References** — Friedman 2001 (DOI 10.1214/aos/1013203451); ESL §10.10 (DOI
    10.1007/978-0-387-84858-7); ch 03 NB 4 (gradient descent in parameter space). `Previous: 01 —
    Boosting as fitting residuals.` `Next: 03 — Gradient boosting for classification.`

## Figures (2, each followed by "Read the figure")
- **Fig C (cell 8)** — the step picture: the per-point negative gradient at round 10 + the tree's
  piecewise-constant approximation (the step actually taken).
- **Fig D (cell 11)** — gradient descent in function space: the 2-point loss bowl + the boosting
  trajectory into the minimum | the total loss vs number of trees.
Both notebook-local matplotlib with `ml_course.colors` (no new `viz` helper).

## `src/` & guards
**No `src/` change** (pytest stays 20). Build via `uv run python <scratchpad>/build_ch08_nb2.py`;
**re-measure the gradient identity + loss curve + trajectory at build**; nbconvert top-to-bottom **from
project cwd** on a scratchpad copy (tracked file **output-free**); **banned-word scan over JSON real cell
text** = 0; `check_no_hardcoded_hex` passes; `ruff`/`black` clean; `gen_llms_txt` re-run. Honest scoping:
the residual = negative gradient identity is **specific to squared error** (other losses differ — NB 3);
the tree step is an **approximation** of the gradient (not exact steepest descent); function space here is
the n training predictions (the general functional view is "Going further"); still *training*-loss
descent (over/underfitting is NB 4). Both reviewers PASS (no BLOCK) + Rémy validates visually before
commit; ff-merge notebook → chapter.
