# Notebook plan — 03_LogisticRegression / 01_score_to_probability

> Status: **APPROVED** (2026-06-18, by Rémy — validated alone; reviewers gate the *built* notebook).
> Build via `uv run python - < /tmp/build_nb1.py` (stdin).

## Context

NB **1 of 6** — the chapter's first fundamental, **one concept: the sigmoid & log-odds** — how a
linear *score* (any real number) becomes a *probability* in [0,1], and the mirror fact that the score
**is the log-odds**. This is the on-ramp to the whole chapter: NB 2 reads the weighted line
geometrically, NB 3–4 *find* the weights. Built **by hand**, **maximally visual**, on the course
fil-rouge (penguins). No fitting here — the weights are **hand-chosen** so the learner meets σ and the
logit cleanly *before* optimization (the honest "we'll learn to find these in NB 3–4" promise).

## Dataset & measured anchors (penguins; sklearn 1.9.0; re-measured at build)

- `datasets.load_penguins()` — Adélie vs Gentoo, `bill_length_mm` (raw mm; range **32.1–59.6**). Gentoo = the
  positive class (P = P(Gentoo)).
- **Hand-chosen** illustrative weights for the preview sigmoid: **w = 1.0 per mm, b = −43.0** → ½-crossing at
  **bill_length = 43 mm** (close to, and deliberately a touch gentler than, the fitted w=1.15/b=−49.8,
  crossing 43.19 mm — gentler slope makes the transition band more visible). Accuracy of the **hand** rule
  re-measured at build (≈0.95, near the 1-D fitted 0.947 — stated as "already strong", never as the optimum;
  NB 3–4 *find* weights that do at least as well).
- p→odds→log-odds table values: p 0.10→odds 0.111→logit −2.197; 0.25→0.333→−1.099; 0.50→1.0→**0**;
  0.75→3.0→+1.099; 0.90→9.0→+2.197 (symmetric around p=0.5 ⇔ logit 0).
- ~16 % of the subset sits in the transition band P∈[0.1,0.9] (the honest "unsure" zone ≈ 40–46 mm).

## Library / figures

Reuse `viz.use_course_style`, `ml_course.colors` (CLASS_CYCLE for Adélie/Gentoo, `model` for the σ curve,
`highlight` for the ½-crossing). All figures here are **one-off in-notebook** (charter colours): the σ(z)
curve; the σ-over-`bill_length` curve with the species at P=0/1 and the ½ line. (The reusable
`viz.plot_logistic_curve` helper stays **deferred** — promote it only in NB 4/NB 6 if reused ≥2×, with a
test; NB 1 keeps the figure inline so the mechanism is visible.) No `src/` change → `pytest` stays 16.

## Cell-by-cell (~19 cells; one concept; "Read the figure/table" after every figure/table)

1. (md) **Header** — `# 01 — From a linear score to a probability`; *notebook 1 of 6 — Logistic
   Regression*; one-line purpose; warm one-line welcome. **Prerequisites:** module 00 (features & the
   feature space NB 02; the idea of a probability/score NB 06–08); chapter 02 (probability as a number in
   [0,1]). **What you'll be able to do:** explain what the sigmoid does; convert between probability, odds
   and log-odds; turn a feature into a class probability with a linear score + sigmoid; read the ½ decision
   point.
2. (code) **Imports + seed + style** — numpy, pandas, matplotlib.pyplot; `from ml_course import datasets,
   viz`; `from ml_course import colors`; `viz.use_course_style()`; `np.random.seed(0)` (documented). Load
   `df = datasets.load_penguins()`; show `df.head()` and the two classes.
3. (md) **Recap / footing** — KNN voted by distance; Naive Bayes multiplied probabilities. A new idea
   starts here: draw **one weighted line** and read a probability off it. But a line gives a *score* that
   can be any number — we first need to turn a score into a probability. (Re-establish: a probability is a
   number in [0,1]; we have only ever *counted* them.)
4. (md) **Intuition — a score is unbounded, a probability is not** — z = w·x + b ranges over (−∞,+∞);
   P must live in [0,1]. We need a smooth, monotone squashing function. One line on **e** (≈2.718, the
   natural exponential) → the **sigmoid / logistic function** σ(z) = 1/(1+e⁻ᶻ).
5. (code) **Code σ(z) from scratch & plot** — define `sigmoid(z)`; plot over z∈[−6,6]; mark σ(0)=0.5 and the
   0/1 asymptotes (charter `model` colour; `highlight` dot at (0, 0.5)).
6. (md) **Read the figure** — the S-curve: very negative z → ~0, very positive z → ~1, z=0 → exactly 0.5;
   smooth and monotone. This is how *any* score becomes a probability.
7. (md) **Intuition — odds and log-odds** — probabilities are bounded; **odds** = p/(1−p) ∈ [0,∞);
   **log-odds (logit)** = log(odds) ∈ (−∞,∞) — unbounded, like a score. Invert σ: if p=σ(z) then
   z = log(p/(1−p)). So **the score z IS the log-odds** of the positive class.
8. (code) **p → odds → log-odds table** — a small pandas DataFrame for p ∈ {0.1, 0.25, 0.5, 0.75, 0.9} with
   `odds` and `log_odds` columns; `display` it.
9. (md) **Read the table** — odds turn "9 in 10" into "9 to 1"; the logit is **symmetric around 0** (p=0.5
   → 0; p>0.5 → positive; p<0.5 → negative). σ and the logit are inverses: score→probability and
   probability→score.
10. (md) **Intuition — apply it to a real feature** — penguins' `bill_length`: longer bills lean Gentoo.
    Choose a linear score z = w·bill_length + b with weights **picked by hand** (we *find* the best ones in
    NB 3–4), push through σ → P(Gentoo).
11. (code) **The sigmoid over `bill_length` (raw mm)** — set `w, b = 1.0, -43.0` (hand-chosen; **not
    fitted** — say so in a comment); compute `P = sigmoid(w*bill_length + b)` across the range; plot the σ
    curve over bill_length with Adélie at y=0 / Gentoo at y=1 (CLASS_CYCLE), and the ½ line at 43 mm
    (`highlight`).
12. (md) **Read the figure** — short bills → P(Gentoo)≈0 (Adélie), long bills → ≈1 (Gentoo); the **½-crossing
    at ≈43 mm** is the decision point; the curve is **steep through the overlap band (~40–46 mm)** where the
    classes mix and flat outside — ~16 % of penguins sit in that band, where the model is *appropriately
    unsure*.
13. (md) **Intuition — predict by thresholding at ½** — call it Gentoo when P≥0.5 (⇔ z≥0 ⇔ bill_length ≥ the
    crossing). The ½ threshold is the natural default; it is a *choice* we revisit under asymmetric costs in
    NB 6.
14. (code) **Predict & check** — apply the ½ rule; print accuracy of the **hand** rule on the subset
    (re-measured at build, ≈0.95); show three penguins — a clear Adélie, a clear Gentoo, and a **borderline
    one near 43 mm with P≈0.5**.
15. (md) **Read the result** — the hand-chosen line already classifies most penguins; the borderline penguin
    is genuinely ~50/50, and the probability *says so* — honest uncertainty a bare label would hide. (We did
    not fit anything yet; NB 3–4 will *find* weights that do at least as well.)
16. (md) **Your turn** (3 tiered) — easy: read P(Gentoo) off the curve at bill_length = 45 mm; medium: make w
    **negative** and predict what happens to the curve and the decision; harder: from w, b solve for the
    bill_length where P=0.5 (it is −b/w), then for where P=0.9 (use the logit: z=log(9)).
17. (md) **What you built** — bullets: the **sigmoid** (score→probability), **odds & log-odds** (the score
    *is* the log-odds), turning a feature into a probability with a threshold; a **vocabulary box**
    (linear score, sigmoid/logistic function, odds, log-odds/logit, decision threshold). Restate the new
    abilities.
18. (md) **Going further (optional)** — the **logit link** and generalized linear models (one paragraph,
    clearly optional); forward pointer: NB 2 reads the weighted line geometrically, NB 3–4 *find* the
    weights by minimizing a loss.
19. (md) **References** — Cox DR (1958) DOI 10.1111/j.2517-6161.1958.tb00292.x; Berkson J (1944) DOI
    10.1080/01621459.1944.10500699; ISLR §4.3 (DOI 10.1007/978-1-0716-1418-1). `Previous: Module 02 — Naive
    Bayes` · `Next: 02 — The decision boundary & reading the weights`.

## Honest scoping (stated in the notebook)

- **Nothing is fitted in NB 1.** The weights are hand-chosen to meet σ and the logit cleanly; the "≈0.95"
  is the hand rule's accuracy, **not** an optimum — NB 3–4 *find* the weights. Stated plainly so the
  preview never reads as fitting.
- σ is introduced as a **squashing function**, the logit as its **inverse**; "the score is the log-odds" is
  the one idea, shown both directions (table + curve).
- The ½ threshold is a **default and a choice** (asymmetric-cost threshold tuning → NB 6), flagged here.
- The transition band (~16 % of points) is named as **honest uncertainty**, not model failure.

## Verification

Build via `uv run python - < /tmp/build_nb1.py` (stdin — avoids the `/tmp/struct.py` stdlib shadow).
Re-measure at build: hand-rule accuracy (≈0.95), the ≈43 mm crossing, the odds/log-odds table, the ~16 %
band. Runs top-to-bottom (nbconvert to /tmp; **output-free**, `--clear-output --inplace`); **banned-word
grep** ("just/simply/obviously/trivially" + FR) = 0; `check_no_hardcoded_hex` passes; no `src/` change
(`pytest` stays 16); `gen_llms_txt` re-run; `ruff`/`black` clean. Both reviewers PASS (no BLOCK); Rémy
validates visually; commit `feat(03_logistic_regression): notebook 01 — from a linear score to a
probability`; merge `notebook → chapter`.
