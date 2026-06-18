# Notebook plan — 02_NaiveBayes / 03_gaussian_likelihood_logspace

> Status: **APPROVED** (2026-06-18, by Rémy — validated alone; reviewers gate the *built* notebook;
> chapter stays at **5 notebooks**, split-trigger not pulled). Build via
> `uv run python - < /tmp/build_nb3.py` (stdin).

## Context

NB 3 of 5. **One concept: model P(feature ∣ class) with a continuous *density* (the Gaussian), and
compute it safely in *log-space*.** NB 1 estimated likelihoods by **binning** (counts); NB 2 multiplied
them. Bins are crude — arbitrary edges, wasted within-bin detail, and the empty-bin **zero-frequency**
trap. NB 3 replaces the histogram with a smooth **Gaussian** fitted per class per feature — which *is*
exactly `GaussianNB`. Two beats, one idea: (a) **density** (the *what* — first genuine contact with a
continuous distribution: area, not height, is probability), and (b) **log-space** (the *how* — a product
of many small densities underflows to 0.0, so we sum logs). Multinomial/Bernoulli are **named, not
built** (→ NB 5).

## Why one notebook, not two (the split-trigger decision)

The chapter plan flagged a watch: if NB 3 reads as more than one concept, split density-NB vs
log-space-NB. It does **not** need splitting: density is *what* a continuous likelihood is, log-space is
*how* to compute that same likelihood without underflow — one arc ("the Gaussian likelihood, computed
safely"). Front-loading density as its own taught beat (cells 4–6) keeps it from crowding. Chapter stays
**5 notebooks** (Rémy confirmed at the gate).

## What the learner does by hand (all measured; seed 0; `datasets.load_penguins()`)

- **Per-class Gaussian fit** on `bill_length_mm` (mean, std, ddof=0 to match sklearn): Adélie
  **μ=38.79, σ=2.65** (peak density 0.150); Gentoo **μ=47.50, σ=3.07** (peak density 0.130).
- **Density first-contact:** a `density=True` histogram (bars integrate to 1) → the smooth Gaussian
  laid over it. Honest point: the y-axis is a **density**, not a probability; **area** under the curve
  is the probability; a density's height *can* exceed 1 (here the peaks are ~0.13–0.15 because bills
  spread over ~25 mm — stated, with the general fact named).
- **By-hand Gaussian naive Bayes** (2 features): per-class per-feature (μ, σ); predict by the **argmax
  of prior × the product of the two Gaussian likelihoods**, computed as a **sum of log-pdf**. **Parity:
  matches `sklearn.naive_bayes.GaussianNB` on 100 % of rows** (train acc 0.9927) — the by-hand model
  *is* the library estimator.
- **Underflow, measured:** multiplying N identical small likelihoods (0.1) the raw float64 product hits
  **exactly 0.0 at N = 324** (sum-of-logs there ≈ −748, perfectly finite). With 2 penguin features the
  product is safe; with the hundreds–thousands of features in text (NB 5) it is not — so log-space is
  not a nicety, it is required.

## Library / figures

**No `src/` additions.** Reuse `viz.use_course_style()`, `viz.plot_decision_boundary` (the by-hand
Gaussian-NB boundary), `ml_course.colors` (`CLASS_CYCLE`, `COLORS`), `datasets.load_penguins`/
`penguins_xy`. `scipy.stats.norm` for the pdf/log-pdf; `sklearn.naive_bayes.GaussianNB` for the parity
check only. Three in-notebook figures (charter colours), all generated & eyeballed at plan time:
- **F1** density histogram (`density=True`) of `bill_length` per species **+ the fitted Gaussian bells**
  overlaid — mass → density.
- **F2** the **by-hand Gaussian-NB decision boundary** on the two features (curved, since per-class σ
  differ) via `plot_decision_boundary`; parity with `GaussianNB` printed (100 %).
- **F3** the **underflow cliff**: the true log-probability declining linearly vs N, with the point
  (N=324) where the raw float64 product collapses to 0.0 marked.
`pytest` stays 14.

## Cell-by-cell (~20 cells; one concept; "Read the figure" after every figure)

1. (md) **Header** — `# 03 — The Gaussian likelihood, computed safely`; *notebook 3 of 5*; purpose;
   **Prerequisites:** NB 01 (likelihood by binning), NB 02 (the naive product). **What you'll be able to
   do** (fit a Gaussian per class; explain density vs mass; build Gaussian naive Bayes by hand; work in
   log-space to dodge underflow).
2. (code) imports (`numpy`, `matplotlib`, `scipy.stats.norm`, `GaussianNB`, `datasets`, `viz`, colours);
   `np.random.seed(0)`; `viz.use_course_style()`; load `df`, `X, y = penguins_xy`.
3. (md) **Recap: bins were crude** — NB 1's binned likelihood threw away within-bin detail, leaned on
   arbitrary edges, and produced the empty-bin **zero-frequency** zeros. What if a smooth curve replaced
   the bars?
4. (md) **First contact: probability density** (intuition, front-loaded) — every distribution so far was
   a **count** (mass). A continuous measurement has no "how many at exactly 45.8 mm"; we use a
   **density** — a smooth curve whose **area** (not height) is the probability; total area = 1; the
   height can even exceed 1.
5. (code) `density=True` histogram of `bill_length` per species (no Gaussian yet).
6. (md) **Read the figure** — the y-axis is now **density**: each species' bars integrate to 1, and an
   area gives a probability. (The tallest bar is ~0.14, not a probability — an *area* is.)
7. (md) **The Gaussian: a smooth density** — model each class's feature as a bell with the class's mean
   and standard deviation: P(x ∣ class) = a Gaussian(μ_c, σ_c). Fit = compute μ and σ by hand. This is
   **Gaussian naive Bayes**.
8. (code) **F1** — fit per-class (μ, σ) on `bill_length`; overlay the Gaussian bells on the density
   histogram; print the four numbers.
9. (md) **Read the figure** — each bell tracks its histogram; one smooth curve replaces every bin, with
   no arbitrary edges and **no empty-bin zeros** (the NB 1 zero-frequency trap dissolves). The overlap
   near 43 mm is where the two species genuinely confuse.
10. (md) **Two features → Gaussian naive Bayes** — reuse NB 2's naive product, now with Gaussians:
    P(bill, flipper ∣ y) ≈ Gaussian(bill) · Gaussian(flipper); predict by the argmax of prior ×
    that product.
11. (code) **F2** — by-hand Gaussian NB: per-class per-feature (μ, σ); `predict` via the **sum of
    `norm.logpdf`**; plot the decision boundary (`plot_decision_boundary`); print **parity vs
    `GaussianNB`** (100 % of rows) and train accuracy 0.9927.
12. (md) **Read the figure** — a gently **curved** boundary (the per-class spreads differ), and the
    by-hand model agrees with the library's `GaussianNB` on every penguin: we just built the estimator
    notebook 4 will dial in.
13. (md) **The likelihood is a modelling choice** — Gaussian suits continuous measurements; **counts**
    (how many times each word appears) call for the **multinomial**, presence/absence for **Bernoulli**.
    Named here, built on text in **notebook 5**.
14. (md) **The numeric snag** (intuition) — a Gaussian density is often a small number; multiply a few
    and all is well, but multiply **hundreds** (a vocabulary of words, NB 5) and the product shrinks past
    what a computer can store.
15. (code) **F3** — multiply N identical small likelihoods: the raw product vs the sum of logs; show the
    product hits **exactly 0.0 at N=324** while the log-sum stays finite; the underflow-cliff figure.
16. (md) **Read the figure** — the product is not *wrong*, it is **unrepresentable** (float64 rounds it
    to 0.0); taking logs turns the fragile product into a safe **sum of log-likelihoods**, and since log
    is increasing the **argmax is unchanged**. This is why every real Naive Bayes works in log-space.
17. (md) **Your turn** (2–3, tiered) — (a) fit a Gaussian to a *third* feature and overlay it on its
    density histogram — does the bell fit?; (b) compute one penguin's log-posterior for both species by
    hand (sum of `logpdf` + log-prior) and take the argmax; (c) *harder* — name a feature where a single
    Gaussian fits badly (skewed or bimodal) and say what you'd do (transform, or a different likelihood).
18. (md) **What you built** — bullets: density vs mass; the Gaussian likelihood (μ, σ per class); Gaussian
    naive Bayes by hand = `GaussianNB`; log-space against underflow. Vocabulary (probability density;
    Gaussian likelihood; Gaussian naive Bayes; log-probability / underflow) added to the glossary.
19. (md) **Going further** (optional) — the Gaussian pdf in full; the normality assumption and how to
    check it (a quick QQ-plot idea); `var_smoothing` as a teaser for NB 4.
20. (md) **References** — ISLR §4.4.4 (DOI 10.1007/978-1-0716-1418-1); ESL §6.6.3
    (DOI 10.1007/978-0-387-84858-7). `Previous: 02 — The "naive" assumption` ·
    `Next: 04 — The estimators & their parameters`.

## Honest scoping (stated in the notebook)

- A **density** is not a probability (area is); its height can exceed 1 — said plainly, even though the
  penguin peaks (~0.13–0.15) do not, because the feature is spread over ~25 mm.
- Gaussian NB assumes each feature is **Gaussian within each class** — false for skewed/multimodal
  features; named with the fix pointed to (transform, or a different likelihood), not taught (ex. 17c).
- The by-hand model **is** `GaussianNB` (verified 100 % parity) — `var_smoothing` and the other knobs
  are NB 4; multinomial/Bernoulli are NB 5. Binning (NB 1) is now retired in favour of the density.
- Log-space is **required at scale** (text, NB 5), not a stylistic choice; with 2 features the raw
  product is fine — we show the cliff to justify the habit before it bites.

## Verification

Build via `uv run python - < /tmp/build_nb3.py`. Numbers re-run & reconciled (Adélie μ38.79/σ2.65,
Gentoo μ47.50/σ3.07; by-hand vs `GaussianNB` 100 % match, acc 0.9927; product → 0.0 at N=324). Runs
top-to-bottom (nbconvert to /tmp; **output-free**, `--clear-output --inplace` before commit);
`check_no_hardcoded_hex` passes; `gen_llms_txt` re-run; `pytest` green (14, no `src/` change); both
reviewers PASS (no BLOCK); Rémy validates visually; commit `feat(02_naive_bayes): notebook 03 — the
Gaussian likelihood & log-space`; merge `notebook → chapter`.
