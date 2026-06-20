# Notebook plan — 03_LogisticRegression / 03_logloss_objective

> Status: **APPROVED** (2026-06-20, by Rémy — validated alone; reviewers gate the *built* notebook).
> Build via `uv run python - < <scratchpad>/build_nb3.py` (stdin — avoids the `/tmp/struct.py` shadow).

## Context

NB **3 of 6** — the chapter's third fundamental, **one concept: the training objective — log-loss.**
NB 1–2 *chose* the weights by hand and eyeballed the result. To let a computer *find* them, we need a
single number that says **how wrong** a choice of weights is — a **loss** to minimize. This notebook
builds that number from ch 02's **likelihood**: per example, **log-loss = cross-entropy = the negative
log-likelihood of the Bernoulli model**, $-[y\log P + (1-y)\log(1-P)]$. Two properties are the lesson:
it **punishes confident-and-wrong without bound** ($-\log P \to \infty$), and as a function of the
weights it is a **convex bowl** (one bottom) — unlike **squared error through the sigmoid**, which is
**non-convex and flattens into plateaus** where a descent stalls. **Nothing is fitted** (we hold the
bias $b=0$, a chosen constant, and read the loss as a function of the weight; NB 4 *minimizes* it with
gradient descent, fitting $w$ and $b$ together). 1-D, by hand, maximally visual.

## Honest refinement to flag (deviates slightly from the chapter plan's wording)

The chapter plan called squared-error-on-the-sigmoid "**bumpy** … bumps a search could get stuck in."
**Measured** on the real 1-D data, squared error is genuinely **non-convex** (its second difference goes
negative) but has a **single** minimum with **flat plateaus** at both ends — not multiple bumps. So I
frame it honestly as **"non-convex, with stalling plateaus (vanishing gradient)"**, and make the
load-bearing point the **convexity** of log-loss (one guaranteed bottom) plus the **live-gradient vs
dead-plateau** contrast. This is more faithful than "bumps" and the ml-expert reviewer will verify it.

## Dataset & measured anchors (penguins, 1-D; re-measured at build)

- `datasets.load_penguins()` → `bill_length_mm`, **standardized by hand** (z-score: subtract mean
  **42.70**, divide by std **5.19**; equals NB 2's `StandardScaler`). Label $y=1$ for Gentoo (123 of 274).
  **No sklearn, no fitting** in this notebook.
- **Bias held at $b=0$** (a chosen constant, stated; NB 4 fits $w$ and $b$). Loss read as a function of
  the single weight $w$.
- **log-loss $L(w)$ is convex**: over $w\in[-6,20]$, second difference $\ge 0$ (min $\approx +1.8\times
  10^{-8}$), a **single** minimum at $w\approx 6.2$, $L\approx 0.146$.
- **squared-error $L(w)$ is non-convex**: second difference dips **$<0$** ($\approx -1.6\times10^{-5}$),
  min at $w\approx 7.6$, $L\approx 0.041$; **flat plateaus** at both ends — slope at $w=20$ is
  $\approx 3\times10^{-4}$ (essentially zero) versus log-loss still rising at $\approx 1.1\times10^{-2}$;
  on the wrong side ($w=-6$) squared-error slope $\approx-0.02$ versus log-loss $\approx-0.83$.
- **Per-example penalty (true $y=1$), confident-and-wrong:** $P=0.5\to$ log-loss **0.69** / sq-err 0.25;
  $P=0.1\to$ **2.30** / 0.81; $P=0.01\to$ **4.61** / 0.98; $P=0.001\to$ **6.91** / 0.998. Log-loss is
  **unbounded**, squared error capped at 1. (Mirror for $y=0$.)
- **Three hand weights** ($b=0$, for the ranking exercise): $w=1,\,3,\,6.2$ → log-loss $\approx
  0.41,\,0.20,\,0.146$ (re-measured exactly at build) — monotone, so the ranking is unambiguous.
- Convexity of the logistic log-likelihood is a **known theoretical result** (ESL §4.4; Bishop §4.3.2);
  the notebook *shows* it numerically (second-difference $\ge 0$), it does not prove it.

## Library / figures

Reuse `viz.use_course_style`, `ml_course.colors` (`model` for log-loss, `highlight` for squared error,
`grid`/`text`, `CLASS_CYCLE` if needed). All figures **one-off in-notebook** (charter colours, numpy
under the hood). **No `src/` change → `pytest` stays 16.**

- **Figure A — the per-example penalty:** for a true $y=1$, plot log-loss $-\log P$ (explodes as
  $P\to0$) beside squared error $(P-1)^2$ (bounded $\le1$) over $P\in(0,1)$; mark a confident-wrong point.
- **Figure B — the loss as a function of the weight (2 panels):** left = **log-loss $L(w)$**, a convex
  bowl, with the minimum and the three hand weights ($w=1,3,6.2$) marked as dots; right = **squared-error
  $L(w)$**, non-convex, with its minimum and the **flat plateaus** at both ends annotated. Wide $w$ range
  so the plateaus and the still-rising log-loss wall are both visible.

## Cell-by-cell (~19 cells; one concept; "Read the figure/table" after every figure/table)

1. (md) **Header** — `# 03 — Fitting I: what we optimize (log-loss)`; *notebook 3 of 6*; purpose; warm
   welcome. **Prerequisites:** NB 1 (sigmoid; the score *is* the log-odds), NB 2 (the weighted score
   $z$, set by hand); chapter 02 (the **likelihood** — maximized by counting — and **log-space** sums,
   NB 03). **What you'll be able to do:** define **log-loss = cross-entropy = negative log-likelihood**;
   explain why it punishes confident-and-wrong; say why squared error is a poor training objective here
   (non-convex, stalling plateaus); read a loss curve and rank weight choices by loss.
2. (code) **Imports + seed + style + data** — matplotlib, numpy, pandas; `from ml_course import colors,
   datasets, viz`; `use_course_style()`; `np.random.seed(0)`. Load `bill_length_mm`, standardize by hand
   (z-score), `y = (species=="Gentoo")`. Define `sigmoid`.
3. (md) **Recap & footing** — in NB 1–2 we **chose** the weights by hand and eyeballed them. To let the
   computer **find** them we need one number for **how wrong** a choice is: a **loss** (lower = better).
   Fitting = minimize the loss. This notebook builds the number; NB 4 minimizes it. (We hold $b=0$ here so
   the loss is a curve over the single weight $w$ — nothing is fitted yet.)
4. (md) **Intuition — from likelihood (ch 02) to log-loss** — the model gives each penguin a probability
   of its **true** label ($P$ if Gentoo, $1-P$ if Adélie); good weights make those high. Their **product**
   is the **likelihood** (ch 02 maximized exactly this). Two moves: take the **log** (product → sum, the
   log-space trick of ch 02 NB 3) and the **negative** (so smaller = better). Per penguin that is the
   **log-loss** $=-[y\log P+(1-y)\log(1-P)]=$ **cross-entropy** $=$ the negative log-likelihood of the
   Bernoulli model.
5. (code) **log-loss from scratch** — `def log_loss_one(y, p)`; a small table for $(y,P)$ pairs:
   $(1,0.9)\to0.105$, $(1,0.5)\to0.693$, $(1,0.1)\to2.303$, $(0,0.1)\to0.105$, $(0,0.9)\to2.303$.
6. (md) **Read the table** — near 0 when confident **and** right; large when confident **and** wrong;
   "the more the truth should surprise the model, the higher the loss" (surprisal). Symmetric in the two
   classes.
7. (md) **Intuition — confident-and-wrong is punished hardest** — $-\log P$ grows **without bound** as
   $P\to0$: predicting Gentoo at $P=0.99$ for a true Adélie costs far more than $P=0.55$. Squared error
   $(P-y)^2$ caps at 1, so it shrugs at confident mistakes. Unbounded punishment is what forces honest
   probabilities.
8. (code) **Figure A** — per-example penalty vs $P$ (true $y=1$): log-loss $-\log P$ (explodes) vs
   squared error $(P-1)^2$ (bounded); mark a confident-wrong point ($P=0.05$).
9. (md) **Read the figure (A)** — as a confident prediction turns out wrong ($P\to0$), log-loss shoots up
   without limit while squared error barely passes 1. The steep left wall is the model being held
   accountable for false confidence.
10. (md) **Intuition — the loss over all the data, as a function of the weight** — average the per-penguin
    log-loss over the 274 birds → the total loss $L(w)$ (bias held at $b=0$, weight $w$ varied). Fitting =
    find the $w$ at the bottom. Let's plot $L(w)$, and beside it what squared error would give.
11. (code) **Figure B (2 panels)** — over a wide $w$ range ($b=0$): *left* log-loss $L(w)$ (convex bowl;
    mark the minimum and the three hand weights $w=1,3,6.2$ as dots); *right* squared-error $L(w)$
    (non-convex; mark its minimum and the flat plateaus). Print the convexity check (log-loss second
    difference $\ge0$; squared-error $<0$) and the near-zero squared-error slope far out vs log-loss's
    still-rising slope.
12. (md) **Read the figure (B)** — *left:* log-loss is a single smooth **bowl** — one bottom, walls that
    keep rising (steeply on the wrong side), so from any starting weight "downhill" reaches the one best
    value; the lowest of the three dots is the best of those choices. *right:* squared error is
    **non-convex** — it sags to a minimum but **flattens into plateaus** at both ends (where $\sigma$
    saturates), so its slope nearly vanishes far from the optimum and a downhill search there **stalls**.
    (Honest: on this 1-D data each curve has a single minimum; the difference that matters is the
    **shape** — a convex bowl with live gradients everywhere vs a non-convex curve with dead plateaus.)
13. (md) **Intuition — why convexity is the prize** — a **convex** loss has exactly one minimum and no
    flat traps, so gradient descent (NB 4) is guaranteed to reach the best weights from any start.
    **Log-loss is convex** for logistic regression (a known result — ESL §4.4, Bishop §4.3.2); squared
    error through the sigmoid is not. That single fact is why classification trains with log-loss.
14. (code) **Score weight choices by loss** — compute log-loss for $w=1,3,6.2$ ($b=0$); a tiny ranked
    table. Tie back: in NB 2 we set weights by hand with no way to say which was better; now one number
    ranks them, and NB 4 drives it to the bottom.
15. (md) **Read the result** — lower loss = better weights, full stop. The hand weights from NB 2 each
    carry a loss; the fit (NB 4) finds the weight at the bottom of the bowl. We can now compare any two
    weight choices objectively, by one number.
16. (md) **Your turn** (3 tiered) — *easy*: given three weights and their losses, rank them and say which
    the fit will prefer; *medium*: explain, using $-\log P$, why a confident wrong prediction costs more
    than an unsure one; *harder*: by hand, write the log-loss of one penguin for $y=1$ and for $y=0$ at
    $P=0.8$, and check against `log_loss_one`.
17. (md) **What you built** — log-loss = cross-entropy = negative log-likelihood (the bridge from ch 02's
    likelihood); punishes confident-and-wrong without bound; **convex** (one bottom) unlike squared error
    (non-convex, stalling plateaus); a single number to rank weight choices. **Vocabulary box:** loss /
    objective, likelihood, negative log-likelihood, cross-entropy / log-loss, convex, plateau. Restate the
    new abilities.
18. (md) **Going further (optional)** — the gradient teaser: log-loss has a strikingly simple gradient,
    $\propto (P-y)\,x$ — the sigmoid's derivative cancels — which is exactly what makes NB 4's descent
    clean and stable; squared error's gradient carries an extra $\sigma'(z)$ factor that dies on the
    plateaus. Forward pointer: **NB 4** walks this bowl downhill with gradient descent.
19. (md) **References** — Cox DR (1958) DOI 10.1111/j.2517-6161.1958.tb00292.x; Bishop CM (2006), *PRML*
    §4.3.2 (logistic regression, cross-entropy & its gradient); Hastie/Tibshirani/Friedman (2009), *ESL*
    §4.4 DOI 10.1007/978-0-387-84858-7. `Previous: 02 — The decision boundary & reading the weights` ·
    `Next: 04 — Fitting II: gradient descent`.

## Honest scoping (stated in the notebook)

- **Nothing is fitted.** The bias is **held at $b=0$** (a chosen constant) so the loss is a curve over
  one weight; NB 4 fits $w$ and $b$ together. The marked "minimum" is the bowl's bottom **with $b=0$**,
  stated as such.
- Squared error is **non-convex with stalling plateaus** (single minimum on this data) — **not** multiple
  bumps; the load-bearing point is log-loss's **convexity** and its **live gradient** where squared
  error's dies. (Refinement from the chapter plan's "bumpy", flagged above.)
- Convexity is shown **numerically** (second difference $\ge0$), named as a known theoretical result —
  not proved.
- log-loss = the **negative log-likelihood** — the explicit bridge from ch 02's likelihood (no new
  probability theory smuggled in).

## Verification

Build via `uv run python - < <scratchpad>/build_nb3.py` (stdin). Re-measure at build: log-loss convex
(2nd-diff $\ge0$), single min $\approx0.146$ at $w\approx6.2$; squared-error 2nd-diff $<0$, min
$\approx0.041$ at $w\approx7.6$, plateau slope $\approx3\times10^{-4}$ at $w=20$; per-example penalties
(0.69/2.30/4.61 vs 0.25/0.81/0.98); three hand-weight losses ($\approx0.41/0.20/0.146$). Runs
top-to-bottom (nbconvert to scratchpad; tracked file **output-free**, `--clear-output --inplace`);
**banned-word grep** ("just/simply/obviously/trivially" + FR) = 0; `check_no_hardcoded_hex` passes; **no
`src/` change** (`pytest` stays 16); `gen_llms_txt` re-run; `ruff`/`black` clean. Both reviewers PASS (no
BLOCK); Rémy validates visually; commit `feat(03_logistic_regression): notebook 03 — fitting I: the
log-loss objective`; merge `notebook → chapter`.
