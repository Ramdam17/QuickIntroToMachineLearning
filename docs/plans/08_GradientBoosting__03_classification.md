# Notebook plan — 08_GradientBoosting / 03_classification

> Status: **APPROVED by Rémy (2026-06-25) & persisted.** No reviewer gate at NB-plan stage; both
> reviewers return on the *built* notebook. Next: build from `build_ch08_nb3.py`. **The added
> classification notebook (Rémy's call) and the chapter's pivotal / densest NB.**

## Context

NB **3 of 6** — the third fundamentals notebook, and the chapter's pivot. **ONE concept:** swap the
loss and gradient boosting does **classification** — log-loss gives the pseudo-residual **y − p**, fit a
*regression* tree to it (in log-odds space), with the **honest Newton leaf-step** sklearn applies for
you. Built **by hand** on ch 07's `make_moons(0.20)` (continuity + a head-to-head with AdaBoost), closing
with the **unifying reveal**: `loss='exponential'` is AdaBoost's objective, so AdaBoost is the
exponential-loss member of the gradient-boosting family — the bridge ch 07 promised. Delivers on NB 2's
"a different loss → a different gradient → a different residual." Carries the chapter's **correctness
trap (D4)**: the Newton leaf is *not* the regression mean; using the mean gives a different model.

## Anchors (measured on sklearn 1.9.0, seed 0 — re-measured at build)

- **Data:** ch 07 through-line `make_moons(n_samples=400, noise=0.20, random_state=0)`, 70/30 stratified
  (seed 0) → train 280 / test 120; **balanced**, so base rate p₀ = 0.5 → **F₀ = log-odds(0.5) = 0**,
  round-1 p = 0.5, round-1 pseudo-residuals **y − p ∈ {−0.5, +0.5}**.
- **By-hand log-loss GB (Newton leaf):** F₀ = log-odds(p₀); per round p = σ(F), r = y − p, fit
  `DecisionTreeRegressor(max_depth=2)` to r, **override each leaf with the Newton value
  γ = Σr / Σ p(1−p)**, `F += ν·γ`. **Config: max_depth=2, ν=0.3, M=50** (continuity with NB 1–2; reaches
  AdaBoost's accuracy). **Parity: by-hand decision_function == `GradientBoostingClassifier(loss='log_loss',
  …)` to 3.55e-15** (machine precision; also 1–5e-15 at depth-3/lr-0.1 and depth-1/lr-1.0). Test acc
  **0.9417** (== ch 07 AdaBoost on this split).
- **The Newton-vs-mean trap (config-dependent gap — pin & re-measure):** naive **mean-leaf** (γ = mean of
  r) gives a *different* model — train log-loss **Newton 0.0347 vs mean 0.2185** at the chosen config (the
  mean under-steps because it ignores the loss curvature p(1−p) ≤ 0.25). Robust shipped facts: Newton
  matches sklearn to ~0; Newton < mean train log-loss at **every** config. **Do not hard-code the pair.**
- **Unifying reveal — `loss='exponential'` shares AdaBoost's objective:**
  `GradientBoostingClassifier(loss='exponential', max_depth=1, learning_rate=1.0, n_estimators=50)` vs
  `AdaBoostClassifier(n_estimators=50, learning_rate=1.0)` on moons: **test acc both 0.9417**, prediction
  **agreement ~98%** (train 0.982, test 0.983) — near-identical but **NOT bit-identical** (different
  optimizers: GB's gradient step + Newton leaves + shrinkage vs AdaBoost's reweighting + α=ln((1−ε)/ε)).
- **Newton leaf formula:** γ = Σ_leaf (y − p) / Σ_leaf p(1−p) — the one-step Newton update for log-loss
  (Friedman 2001 §4.5–4.6; equals Friedman's binomial leaf Σỹ / Σ|ỹ|(1−|ỹ|)).

## Cell-by-cell (~21 cells; one concept; "Read the figure" after every figure)

1. (md) **Header** — `# 03 — Gradient boosting for classification`; *Chapter 08 · Notebook 3 of 6*;
   warm welcome (the payoff of NB 2 — change the loss, and the same machine classifies; and where
   AdaBoost fits in). **Prerequisites:** NB 1 (the residual loop), NB 2 (the residual = negative gradient;
   fit the negative gradient of *any* loss); Chapter 03 (the sigmoid, log-odds, log-loss); Chapter 07
   (AdaBoost). **What you'll be able to do:** do gradient-boosting classification by hand (log-loss →
   y − p); apply the honest Newton leaf-step; match `GradientBoostingClassifier` exactly; see that
   AdaBoost is the exponential-loss member of the family.
2. (code) **Imports & setup** — numpy, matplotlib, pandas; `make_moons`, `train_test_split`,
   `GradientBoostingClassifier`, `AdaBoostClassifier`, `DecisionTreeRegressor`, `log_loss`; `from ml_course
   import viz` + `viz.use_course_style()`; `COLORS`, `CLASS_CYCLE`; `SEED=0`; a `sigmoid` helper; build the
   moons split (train 280 / test 120); config `NU=0.3, MAX_DEPTH=2, N_TREES=50`.
3. (md) **Recap & the plan.** NB 2: gradient boosting fits the **negative gradient of whatever loss you
   choose**. For classification, that loss is **log-loss**. **Re-lay ch 03:** a classifier produces a real
   score `F(x)` (the **log-odds**), squashed to a probability `p = σ(F) = 1/(1+e^{−F})`; **log-loss**
   punishes confident-and-wrong probabilities. The negative gradient of log-loss with respect to the score
   is `y − p` — "how wrong the current probability is."
4. (md) **Intuition — boost in log-odds space.** Start from the best constant score, `F₀ = log-odds of the
   base rate` (for our balanced moons, `F₀ = 0`, so every point starts at `p = 0.5`). Each round: form the
   pseudo-residual `y − p` (positive for class-1 points — push the score up; negative for class-0 — push
   it down), fit a **regression** tree to it, add a shrunken slice. The target is a class, but the thing we
   fit is the continuous `y − p`.
5. (code) **By hand — round 1.** `p0 = y_train.mean()`; `F0 = log(p0/(1−p0))`; `p = sigmoid(F0)`;
   `r = y_train − p`. Print `F0`, `p` (0.5), and the unique residual values (±0.5). Fit
   `DecisionTreeRegressor(max_depth=2)` to `r`.
6. (md) **Read the result.** With a balanced start every point sits at `p = 0.5`, so the round-1
   pseudo-residual is `+0.5` for class 1 and `−0.5` for class 0 — a clean "push up / push down" signal. We
   fit a regression tree to it, exactly as in Notebook 1, but now in score space.
7. (md) **Intuition — the leaf value: an honest Newton step.** Notebook 1's regression tree puts the
   **mean** of a leaf's residuals. For squared error that was the optimal step. For **log-loss it is not**:
   the loss curves (its second derivative at a point is `p(1−p)`), so the best leaf value is the one-step
   **Newton** update `γ = Σ(y−p) / Σ p(1−p)`. Because `p(1−p) ≤ 0.25`, this is *larger* than the mean — a
   better-scaled step. scikit-learn applies it for you; use the naive mean and you get a different model.
8. (code) **Build the classifier by hand (Newton leaf), and check it.** Loop: `p=σ(F)`; `r=y−p`; fit a
   depth-2 regression tree to `r`; for each leaf (`tree.apply`) set `γ = r[leaf].sum() / (p(1−p))[leaf].sum()`;
   `F += NU·γ`. Compare `F` (decision_function) to `GradientBoostingClassifier(loss="log_loss",
   n_estimators=50, learning_rate=0.3, max_depth=2, random_state=0)` → `max|Δ| = 3.6e-15`. Build a
   **mean-leaf** variant too; print **train log-loss: Newton 0.035 vs mean 0.219**, and test accuracy 0.942.
9. (md) **Read the result.** By-hand Newton == `GradientBoostingClassifier` to ~1e-15 (machine precision):
   you built gradient-boosting classification correctly, Newton leaf and all. The **mean-leaf** model is a
   *different* one — here its train log-loss is several times higher (it under-steps, ignoring the curvature
   `p(1−p)`). The exact gap depends on the settings; the robust facts are that the Newton leaf matches the
   library and beats the mean. This is the one place classification needs more than Notebook 1's recipe —
   and sklearn does it silently, so we did it openly.
10. (code) **Fig E — the boundary sharpens.** Fit `GradientBoostingClassifier` at `n_estimators ∈ {1, 10,
    50}` (same model we matched) and plot the moons decision boundary at each (3 panels; reuse
    `viz.plot_decision_boundary` if it fits, else notebook-local `contourf`).
11. (md) **Read the figure (E)** — at one tree the boundary is a coarse step; by 10 it follows the two
    crescents; by 50 it traces them cleanly (test accuracy 0.942). The score `F` grows sharper each round,
    the probabilities pull toward 0 and 1.
12. (code) **Fig F — the Newton leaf matters (the trap, made visible).** Train log-loss vs number of trees
    for: by-hand **Newton** (overlapping sklearn's `staged` log-loss) and by-hand **mean-leaf** (lagging
    above). Two/three curves.
13. (md) **Read the figure (F)** — the Newton curve drops fast and sits *exactly* on scikit-learn's; the
    mean-leaf curve lags above it — a different, slower model that needs many more trees to catch up. The
    leaf step is not a detail: it is the difference between the library's model and another one.
14. (md) **Intuition — the unifying reveal.** We chose log-loss. Chapter 07's AdaBoost minimized the
    **exponential loss**. By NB 2's logic, that is just another loss to plug in — so AdaBoost should be
    gradient boosting with `loss='exponential'`. Let us check, on the very same moons.
15. (code) **Exponential loss == AdaBoost's objective.** Fit `GradientBoostingClassifier(loss="exponential",
    max_depth=1, learning_rate=1.0, n_estimators=50)` and `AdaBoostClassifier(n_estimators=50,
    learning_rate=1.0)`; print both test accuracies (0.942) and the fraction of points where they agree
    (~0.98 train, ~0.98 test).
16. (code) **Fig G — two routes, one objective.** Plot the exp-GB decision boundary and the AdaBoost
    boundary on moons (overlaid or side by side).
17. (md) **Read the figure (G)** — the boundaries are nearly the same and the test accuracies are
    identical, yet the two models agree on ~98% of points, **not 100%**: they share the **objective**
    (exponential loss) but run **different optimizers** (GB takes functional-gradient steps with Newton
    leaves and shrinkage; AdaBoost reweights points and votes with α = ln((1−ε)/ε)). So AdaBoost is the
    **exponential-loss member** of the gradient-boosting family — the bridge Chapter 07 promised, now
    crossed. Gradient boosting is the general method; AdaBoost is one choice of loss.
18. (md) **Your turn** (tiered) — *easy:* from Fig F, explain in one sentence why the mean-leaf curve lags
    (the missing curvature `p(1−p)`). *medium:* raise the exp-GB / AdaBoost comparison to `n_estimators=100`
    and report whether the agreement rises. *harder:* show that the Newton leaf `Σr/Σp(1−p)` is always at
    least as large in magnitude as the mean `Σr/count` (because `p(1−p) ≤ 0.25`), and say what that means
    for the step size and for why log-loss needs it.
19. (md) **What you built** — bullets: did gradient-boosting **classification** by hand (log-loss → fit
    `y − p`); applied the **Newton leaf-step** and matched `GradientBoostingClassifier` to machine
    precision; saw the **mean leaf** is a different model; and showed **AdaBoost is the exponential-loss
    member** of the family. **Vocabulary:** log-odds score · log-loss · pseudo-residual `y − p` · Newton
    leaf-step · loss curvature `p(1−p)` · exponential-loss member.
20. (md) **Going further (optional)** — the leaf line-search for log-loss: a one-step Newton minimization
    of the leaf's loss gives `γ = Σ(y−p)/Σ p(1−p)` (the second-order Taylor term is the curvature `p(1−p)`),
    which is Friedman's TreeBoost update (2001, §4.5–4.6). Multiclass GB fits **one tree per class** each
    round (softmax). The general "line search" of NB 2 here is a Newton step; for squared error (NB 1) it
    collapsed to the mean.
21. (md) **References** — Friedman 2001 (DOI 10.1214/aos/1013203451, §4.5–4.6 TreeBoost); Friedman, Hastie
    & Tibshirani 2000 — *Additive logistic regression* (DOI 10.1214/aos/1016218223; AdaBoost = exponential
    loss); Chapter 03 (sigmoid / log-odds / log-loss); Chapter 07 (AdaBoost). `Previous: 02 — The residual
    was the gradient.` `Next: 04 — Shrinkage and the trees.`

## Figures (3, each followed by "Read the figure")
- **Fig E (cell 10)** — decision-boundary sharpening on moons at `n_estimators ∈ {1, 10, 50}`.
- **Fig F (cell 12)** — train log-loss vs trees: by-hand Newton (== sklearn) vs naive mean-leaf (lagging) —
  the correctness trap, made visible.
- **Fig G (cell 16)** — exp-GB vs AdaBoost decision boundaries on moons (near-identical; the unifying
  reveal).
Reuse `viz.plot_decision_boundary` where it fits; otherwise notebook-local `contourf` with `ml_course.colors`.

## `src/` & guards
**No `src/` change** (pytest stays 20). Build via `uv run python <scratchpad>/build_ch08_nb3.py`;
**re-measure the Newton parity (~1e-15), the Newton-vs-mean log-loss gap, and the exp/AdaBoost agreement
at build** (do **not** hard-code the log-loss pair — pin the config); nbconvert top-to-bottom **from
project cwd** on a scratchpad copy (tracked file **output-free**); **banned-word scan over JSON real cell
text** = 0; `check_no_hardcoded_hex` passes; `ruff`/`black` clean; `gen_llms_txt` re-run. Honest scoping:
the Newton leaf is the **chapter's correctness trap** — built by hand and stated plainly, full derivation
in "Going further" (a learner who finds it hard keeps the method); the exp-loss = AdaBoost reveal is
**objective-level, not predictor-level** (~98% agreement, identical test acc, not bit-identical);
multiclass = one-tree-per-class (named, not built); still a 2-D teaching set (the demanding real case is
NB 6). Both reviewers PASS (no BLOCK) + Rémy validates visually before commit; ff-merge notebook → chapter.
