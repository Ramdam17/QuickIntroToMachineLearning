# Notebook plan — 03_LogisticRegression / 04_gradient_descent

> Status: **APPROVED** (2026-06-20, by Rémy — validated alone; reviewers gate the *built* notebook).
> Build via `uv run python - < <scratchpad>/build_nb4.py` (stdin — avoids the `/tmp/struct.py` shadow).

## Context

NB **4 of 6** — the chapter's fourth fundamental and its centrepiece: **one concept — gradient descent,
the optimizer.** NB 3 built the **convex bowl** (log-loss) and said "find the weights at the bottom"; NB
4 *reaches* it. This is the **first model in the course trained by iterative optimization** — and the
engine the whole back half (MLP, neural networks) runs on, so it is taught from scratch, by hand, and
*watched* as it converges. The idea: the **gradient** points in the direction the loss rises fastest;
take a step the **opposite** way, scaled by a **learning rate**; repeat, and the weights roll downhill to
the bottom of NB 3's bowl. The gradient of the log-loss is **(P − y)·x** (the sigmoid's derivative
cancels — NB 3's teaser made real): we **state it and verify it numerically**, pointing the full
derivation to the references rather than grinding the chain rule. Convergence is **shown** (a path, a
falling loss curve), not proved; we lean on NB 3's convexity for the guarantee. By hand, 1-D, maximally
visual; the by-hand weights match `LogisticRegression(C=np.inf)`.

## Dataset & measured anchors (penguins, 1-D; re-measured at build)

- `datasets.load_penguins()` → `bill_length_mm`, **standardized by hand** (z-score; mean **42.70**, std
  **5.19**, ddof=0 as in NB 3). `y=1` for Gentoo (123 of 274). We now fit **both** `w` and `b` (NB 3 held
  `b=0`).
- **Gradient (verified, not asserted):** ∂L/∂w = mean[(P − y)·x], ∂L/∂b = mean[P − y]. Checked against a
  central finite difference: error **≈ 2e−11** — machine-precision agreement.
- **By-hand full-batch gradient descent (start w=b=0) reaches the MLE.** Reference
  `LogisticRegression(C=np.inf)`: **w\*=6.297, b\*=−0.561, loss 0.1396**. By-hand GD: lr=0.5 → within
  1e−4 of the MLE loss in ≈**1040** iterations; lr=1.0 → gap to MLE **4e−4** (2000 it); lr=2.0 → gap
  **1e−7**. The descent **rolls to the same weights sklearn's solver finds**. (Parity is against
  **`C=np.inf`** — the *unregularized* MLE — NOT the default `C=1`, which is regularized → NB 5.)
- **Learning-rate regimes — shown on the standardized 1-D loss (start w=b=0):** lr=**0.1** crawls (loss
  still ≈0.31 after 60 steps, far from the 0.14 floor); lr=**2** glides to the floor; lr=**90**
  **oscillates** (overshoots, the loss bounces above the floor). The surface is gently curved here
  (Hessian eigenvalues at the MLE ≈ **0.004, 0.041**, so steps stay stable up to lr ≈ **48**) — so the
  usable range is wide; that is itself honest and worth a sentence.
- **The knife-edge on raw (un-standardized) `bill_length`** — the "why we standardize" tie-in (a prose
  note, not a separate figure): raw bill makes the loss a stretched, steep ravine — lr=**0.003** barely
  moves, lr=**0.005** **explodes**; almost no learning rate works. Standardizing (NB 2) is what gives the
  comfortable wide range above.
- Loss-surface view for the contour figure: `w ∈ [−1, 10]`, `b ∈ [−4, 3]`; start (0,0) loss 0.693 → MLE
  loss 0.140.

## Library / figures

Reuse `viz.use_course_style`, `ml_course.colors` (`model` for the loss/curve, `highlight` for the
descent path / the "good" run, `grid`/`text`, `CMAP_PROBA` or `CMAP_COUNT` for the surface contour). All
figures **one-off in-notebook** (charter colours; numpy under the hood). `sklearn.linear_model.Logistic
Regression` is imported **only** for the one parity check (cell 14). **No `src/` change → `pytest` stays
16.**

- **Figure (a) — gradient as slope (1-param bowl):** NB 3's log-loss bowl $L(w)$ (b held at 0), a point on
  it, the **tangent** (the slope = the gradient), and a **downhill-step arrow** ($w \leftarrow w - \eta\,
  L'(w)$).
- **Figure (b) — the loss surface + the descent path:** filled contour of $L(w,b)$ with the GD **path**
  from (0,0) curving to the bottom (w\*, b\*); start and minimum marked.
- **Figure (c) — loss vs iteration:** the loss falling steeply then flattening to a floor (convergence).
- **Figure (d) — the learning-rate panel:** loss-vs-iteration for **lr = 0.1 (crawls), 2 (glides), 90
  (oscillates)** on the standardized loss.

## Cell-by-cell (~22 cells; one concept; "Read the figure" after every figure)

1. (md) **Header** — `# 04 — Fitting II: gradient descent`; *notebook 4 of 6*; purpose; warm welcome.
   **Prerequisites:** NB 3 (the log-loss **bowl** $L(w)$, convex, one bottom), NB 1 (sigmoid; score =
   log-odds), NB 2 (standardization & why it matters). **What you'll be able to do:** read the **gradient**
   as the slope of the loss; apply the **update rule** $w \leftarrow w - \eta\,\nabla L$; use the
   log-loss gradient **(P − y)·x**; run gradient descent by hand until the weights reach the bottom of the
   bowl; explain what the **learning rate** does (crawl / converge / diverge).
2. (code) **Imports + seed + style + data** — matplotlib, numpy, pandas; `from ml_course import colors,
   datasets, viz`; `use_course_style()`; `np.random.seed(0)`. Standardized 1-D `bill_length`, `y`; define
   `sigmoid`, `log_loss(w, b)`, `grad(w, b)` returning (∂w, ∂b).
3. (md) **Recap & footing** — NB 3 gave us the **bowl**: a convex loss $L$ whose bottom is the best
   weights. We *chose* weights by hand in NB 1–2; now we **reach** the bottom automatically. The recipe is
   the oldest trick for getting downhill in fog: feel which way the ground slopes, step the other way,
   repeat. The course's **first optimizer** — and the one neural networks (ch 11–12) are trained with.
4. (md) **Intuition — the gradient is the slope** — on a 1-D bowl, the derivative $L'(w)$ is the **slope**:
   positive ⇒ loss climbs to the right ⇒ go **left**; negative ⇒ go **right**. So step **against** the
   slope. How far? a small fraction of it — the **learning rate** $\eta$: $w \leftarrow w - \eta\,L'(w)$.
5. (code) **Figure (a) — gradient as slope** — NB 3's bowl $L(w)$ (b=0); mark a point (e.g. w=1), draw the
   **tangent** (slope), and an **arrow** for one downhill step $w \leftarrow w - \eta L'(w)$.
6. (md) **Read the figure (a)** — the tangent points uphill; we step the opposite way. Far from the bottom
   the slope is steep, so steps are big; near the bottom it flattens, so steps shrink — the descent
   **slows as it arrives**, all on its own.
7. (md) **Intuition — the gradient of the log-loss is (P − y)·x** — with two weights $(w, b)$ the gradient
   is a 2-vector: $\partial L/\partial w = \text{mean}[(P-y)\,x]$, $\partial L/\partial b =
   \text{mean}[P-y]$. It is just **error × input**: $(P-y)$ is how wrong the probability is, $x$ is the
   feature. The sigmoid's derivative **cancels** (NB 3's teaser) — that clean form is why this is stable.
   (Full derivation → references; here we *state it and check it.*)
8. (code) **Verify the gradient** — compare the analytic $(P-y)x$ to a central finite difference of
   `log_loss` at a test point; print both and the error (≈1e−11). "We can trust it."
9. (md) **Intuition — the algorithm** — gradient descent in three lines: compute the gradient; step
   $(w,b) \leftarrow (w,b) - \eta\,\nabla L$; repeat. Over the loss **surface** $L(w,b)$ that traces a
   **path** downhill to the single bottom (NB 3's convexity guarantees there is just one).
10. (code) **Figure (b) — loss surface + descent path** — filled contour of $L(w,b)$ over $w\in[-1,10],
    b\in[-4,3]$; run GD from (0,0) at a good $\eta$; plot the **path** to (w\*, b\*); mark start and min.
11. (md) **Read the figure (b)** — the path steps **downhill across the contours**, curving to the one
    bottom; each dot is one update. The steps are large where the bowl is steep and shrink as the gradient
    fades near the minimum.
12. (code) **Figure (c) — loss vs iteration** — the loss values along that run, falling steeply then
    flattening to a floor.
13. (md) **Read the figure (c)** — the loss drops fast at first (steep slope, big steps), then levels off
    at the floor — the bottom of the bowl. That levelling-off is what we mean by **convergence**: more
    steps stop helping.
14. (code) **Parity with sklearn** — run GD to convergence; print by-hand $(w,b)$ next to
    `LogisticRegression(C=np.inf).coef_/.intercept_`. They match (w≈6.30, b≈−0.56). (Use **`C=np.inf`** —
    the unregularized fit — not the default `C=1`, which is regularized; that knob is NB 5.)
15. (md) **Read the result** — by-hand gradient descent lands on the **same weights** the library's solver
    finds. The library is not magic: it walks the same convex bowl, with cleverer step rules. We have now
    *fitted* a logistic-regression model end to end, by hand.
16. (md) **Intuition — the learning rate is the dial** — too **small**: tiny steps, the descent **crawls**;
    **just right**: it **glides** to the floor; too **big**: each step overshoots the bottom and the loss
    **bounces** (oscillates), and larger still it **diverges**.
17. (code) **Figure (d) — the learning-rate panel** — loss-vs-iteration for $\eta = 0.1$ (crawls), $2$
    (glides), $90$ (oscillates), on the standardized loss.
18. (md) **Read the figure (d)** — small $\eta$ inches down and is nowhere near the floor when the others
    have arrived; the good $\eta$ snaps to it; the big $\eta$ jumps past the bottom and bounces. (The
    usable range is wide here because the loss is gently curved; on **raw**, un-standardized `bill_length`
    it collapses to a knife-edge — $\eta=0.003$ barely moves, $0.005$ explodes — which is another reason
    NB 2 standardized.)
19. (md) **Your turn** (3 tiered) — *easy*: change $\eta$ in the descent and describe the path (faster?
    bouncier?); *medium*: do **one** update step by hand from given $w, b, P, y, x$ using $(P-y)x$;
    *harder*: start from a bad initial $w$ (say $-5$) and predict whether GD still converges — then check.
    (Convex bowl ⇒ yes, from anywhere.)
20. (md) **What you built** — the **gradient** (steepest-ascent direction; for log-loss, **(P − y)·x**);
    the **update rule** $w \leftarrow w - \eta\nabla L$; **gradient descent** rolling to the convex bottom;
    the **learning rate** dial; and **parity** with sklearn — a model fitted by hand. **Vocabulary box:**
    gradient · learning rate $\eta$ · gradient descent · iteration / step · convergence · overshoot.
21. (md) **Going further (optional)** — we used the whole dataset each step (**batch** gradient descent);
    on large data we'd sample a **mini-batch** (stochastic GD) — same idea, noisier steps. And this *is*
    how neural networks learn: each unit a logistic-style neuron, the gradient found by the chain rule
    (**backpropagation**, ch 11–12). Forward pointer: **NB 5** meets the real `LogisticRegression` — its
    parameters `C`, `l1_ratio`, multinomial/softmax — and how to tune them honestly.
22. (md) **References** — Cox DR (1958) DOI 10.1111/j.2517-6161.1958.tb00292.x; Bishop CM (2006), *PRML*
    §4.3.3 (logistic regression, the gradient & iterative fitting); Hastie/Tibshirani/Friedman (2009),
    *ESL* §4.4 DOI 10.1007/978-0-387-84858-7. `Previous: 03 — Fitting I: what we optimize (log-loss)` ·
    `Next: 05 — The estimator & its parameters`.

## Honest scoping (stated in the notebook)

- The gradient **(P − y)·x is stated and verified numerically**, not derived line by line (derivation →
  Bishop §4.3.3 / ESL §4.4). Honest about what is shown vs. cited.
- **Convergence is shown, not proved** — a path and a falling loss curve; the *guarantee* rests on NB 3's
  convexity (one bottom, reachable from anywhere), named not re-derived.
- **Parity is against `C=np.inf`** (the unregularized MLE), explicitly — matching the default `C=1` would
  show a spurious mismatch; regularization is NB 5.
- The **learning-rate range is wide here because the standardized loss is gently curved**; the knife-edge
  on raw features is the honest contrast (and the optimization reason to standardize, tying back to NB 2).
- **Batch** gradient descent (whole dataset per step); stochastic/mini-batch named, not taught.

## Verification

Build via `uv run python - < <scratchpad>/build_nb4.py` (stdin). Re-measure at build: gradient vs
finite-diff (≈1e−11); GD → `C=np.inf` (w≈6.297, b≈−0.561; gap 4e−4 at lr=1, 1e−7 at lr=2); lr regimes
(0.1 crawls / 2 glides / 90 oscillates; raw 0.003 vs 0.005 knife-edge). Runs top-to-bottom (nbconvert to
scratchpad; tracked file **output-free**, `--clear-output --inplace`); **banned-word scan over the JSON
real text** (the robust check) = 0; `check_no_hardcoded_hex` passes; **no `src/` change** (`pytest` stays
16); `gen_llms_txt` re-run; `ruff`/`black` clean. Both reviewers PASS (no BLOCK); Rémy validates visually;
commit `feat(03_logistic_regression): notebook 04 — fitting II: gradient descent`; merge `notebook →
chapter`.
