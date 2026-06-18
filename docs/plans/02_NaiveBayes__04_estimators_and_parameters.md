# Notebook plan — 02_NaiveBayes / 04_estimators_and_parameters

> Status: **APPROVED** (2026-06-18, by Rémy — validated alone; reviewers gate the *built* notebook).
> Calibration deferred to NB 5 (penguins too separable; measured Brier 0.0006) — confirmed at the gate.
> Build via `uv run python - < /tmp/build_nb4.py` (stdin).

## Context

NB 4 of 5 — the chapter's **integration notebook**: the real `sklearn.naive_bayes` family and the
**dials** that control it. NB 1–3 built the mechanism by hand (and NB 3 proved the by-hand Gaussian NB
*is* `GaussianNB`). Now we take that estimator and turn each knob, **showing** what it does rather than
asserting it: `var_smoothing` (Gaussian), `alpha` (the count models — and the cure for NB 1's
zero-frequency), and the `priors`. Then the honest workflow: choose a parameter by **cross-validation on
train**, spend the test set **once**. **Calibration is named and deferred to NB 5** — penguins is too
separable to exhibit NB's over-confidence (measured Brier 0.0006), so the real reliability-diagram +
Brier treatment lands on text where it bites.

## What the learner sees, each dial *measured* (seed 0; `datasets.load_penguins()`)

- **`var_smoothing`** (GaussianNB): a variance floor added to every feature. 5-fold CV: flat at
  **0.9927** from 1e-9 to 0.1, then **1.0 → 0.989**, **10 → 0.711** — too-much inflates every variance
  until the classes look identical and the feature stops discriminating. (Too-*little* needs a
  near-constant feature to bite, absent here — named, not forced.)
- **`alpha`** (Laplace/Lidstone, the count models): **heals NB 1's penguins zero-frequency by hand.**
  On NB 1's `bill_length` contingency, P(long ∣ Adélie) and the long-bin posterior for Adélie:
  α=0 → **0.000 / 0.000** (NB 1's absurd certainty), α=0.5 → 0.0033 / 0.0093, α=1 → 0.0065 / 0.0183,
  α=5 → 0.030 / 0.081. α lifts the impossible zero off the floor; α→0 is the literal log(0). This is
  `MultinomialNB`'s `alpha` — **named**, built on text in NB 5 (no `CountVectorizer` here; NB 5 owns the
  by-hand vectorization on-ramp).
- **`priors` / `fit_prior`**: the prior is the starting belief — learn it from class frequencies
  (`fit_prior=True`) or set it. It **tilts the boundary**: sweeping P(Gentoo) 0.15 → 0.85 moves the
  Gaussian-NB boundary down-left (124 → 127 predicted Gentoo of 274; two boundary panels — *confirmed
  visibly distinct*), and the borderline penguin x=[40.8, 208] (true Adélie, P(Gentoo)=0.59 at default)
  **flips** Adélie→Gentoo as the prior rises. On separable penguins the headline accuracy barely moves —
  the honest lesson is the *mechanism* (prior as a multiplier at the margin), and that it matters under
  overlap or asymmetric costs.
- **Honest tuning:** `GridSearchCV`/`cross_val_score` over a parameter on **train**, then **one** sealed
  test score (module 00 NB 04/10). On this clean data the defaults are already near-optimal — the
  *discipline* is the lesson, not the gain.
- **Calibration:** penguins held-out Brier **0.0006** (well-calibrated *here*); the over-confidence is a
  high-dimensional/correlated-feature phenomenon → measured on text in **NB 5**.

## Library / figures

**No `src/` additions** (the `viz.plot_calibration_curve` helper lands in NB 5 where calibration is
taught). Reuse `viz.use_course_style`, `viz.plot_decision_boundary` (prior panels), `ml_course.colors`,
`datasets.load_penguins`/`penguins_xy`. sklearn: `GaussianNB`, `MultinomialNB`/`BernoulliNB` (named),
`cross_val_score`, `GridSearchCV`, `StratifiedKFold`, `train_test_split`. **Fit `GaussianNB` on
`X.to_numpy()`** so `plot_decision_boundary`'s grid predict raises no feature-name warning (clean
output). Three figures: var_smoothing CV curve; alpha-heals-the-zero (P(long∣Adélie) & posterior vs α);
prior boundary panels (P(Gentoo)=0.15 vs 0.85). `pytest` stays 14.

## Cell-by-cell (~21 cells; "Read the figure" after every figure)

1. (md) **Header** — `# 04 — The estimators & their parameters`; *notebook 4 of 5*; purpose;
   **Prerequisites:** NB 01–03 (by-hand NB; NB 3 showed by-hand == `GaussianNB`); module 00 NB 04/10
   (train/test, cross-validation). **What you'll be able to do** (name the NB family; turn
   `var_smoothing`/`alpha`/`priors` and predict their effect; choose a parameter honestly by CV).
2. (code) imports + seed + style; load `df`, `X, y`; `Xnp = X.to_numpy()`; quick parity recap line
   (`GaussianNB().fit(Xnp, y)` accuracy = the NB 3 number).
3. (md) **The Naive Bayes family** — same Bayes' rule, same naive product; the *likelihood* is the
   choice: **GaussianNB** (continuous — NB 3), **MultinomialNB** (counts — NB 5 text), **BernoulliNB**
   (presence/absence). This notebook tunes the Gaussian here and *names* the counts' main dial.
4. (md) **Dial 1 — `var_smoothing`** (intuition): a tiny floor added to every feature's variance. Too
   **little** → a near-constant feature becomes a spike and one outlier dominates; too **much** → every
   class looks equally broad and the feature stops telling classes apart.
5. (code) sweep `var_smoothing` ∈ {1e-9 … 10}; 5-fold CV; the accuracy curve (log-x).
6. (md) **Read the figure** — flat at 0.9927 up to ~0.1, then **1.0 → 0.989**, **10 → 0.711**: piling on
   smoothing washes out every variance until the classes are indistinguishable. The default (1e-9) is a
   safe floor, not a tuning knob you reach for first.
7. (md) **Dial 2 — `alpha` and the zero-frequency cure** (intuition): remember NB 1's
   P(long ∣ Adélie) = 0, which made an Adélie with a long bill *impossible* and zeroed the whole product?
   **Laplace smoothing** adds a pseudo-count α to every bin so nothing is ever exactly zero.
8. (code) on NB 1's `bill_length` contingency, sweep α ∈ {0, 0.5, 1, 5}; show P(long∣Adélie) and the
   long-bin posterior for Adélie climbing off zero; a small bar/line figure.
9. (md) **Read the figure** — α=0 reproduces NB 1's absurd certainty (0.000); any α>0 lifts it to a
   small, honest value. This α is exactly `MultinomialNB(alpha=...)`, the count model we run on real
   word-counts in NB 5; α→0 is the literal log(0) we avoided in NB 3.
10. (md) **Dial 3 — `priors` / `fit_prior`** (intuition): the prior is the starting belief. Learn it from
    the class frequencies (`fit_prior=True`, the default) or set it by hand — under imbalance or
    asymmetric error costs, where you put it tilts the decision.
11. (code) two `GaussianNB` boundary panels at P(Gentoo)=0.15 vs 0.85 (`plot_decision_boundary`); print
    the predicted-Gentoo counts (124 vs 127) and the borderline-point flip (x=[40.8, 208]).
12. (md) **Read the figure** — the boundary slides down-left as the Gentoo prior rises; the borderline
    penguin flips Adélie→Gentoo. On these well-separated penguins the headline accuracy barely moves —
    the prior bites at the **margin**, and matters most when classes overlap or mistakes cost unequally.
13. (md) **Choosing a parameter honestly** (intuition) — recap module 00: tune on the **training** data
    by cross-validation, and spend the **test** set once. Never pick a setting by its test score.
14. (code) `GridSearchCV` (or `cross_val_score`) over `var_smoothing` on **train**; report the CV-chosen
    value and **one** sealed test accuracy.
15. (md) **Read the output** — the honest workflow in miniature. On this clean data the default was
    already near-best, so tuning gains little; the **discipline** (CV to choose, test once) is the point,
    and it pays off on harder data (NB 5).
16. (md) **A limit we will measure later: calibration** — NB's *decisions* are strong, but its
    *probabilities* can be **over-confident** (it double-counts correlated evidence). On these separable
    penguins it is actually well-calibrated (held-out Brier ≈ 0.0006). The over-confidence is a
    high-dimensional, correlated-feature effect — we measure it honestly on text in **notebook 5**.
17. (md) **Your turn** (2–3, tiered) — (a) sweep `var_smoothing` yourself and find where accuracy
    collapses; (b) re-derive the Laplace-healed long-bin posterior for a chosen α; (c) *harder* — name a
    situation where you would **set** a non-uniform prior rather than learn it (a known base rate, or an
    asymmetric error cost).
18. (md) **What you built** — bullets: the NB family; `var_smoothing`; `alpha` as the zero-frequency
    cure; `priors`/`fit_prior` tilting the decision; tune-on-train / test-once. Vocabulary added.
19. (md) **Going further** (optional) — `MultinomialNB`/`BernoulliNB` for text (NB 5); `CategoricalNB`
    for discrete features; calibration and `CalibratedClassifierCV` (NB 5).
20. (md) **References** — Pedregosa et al. (2011), *Scikit-learn*, JMLR 12:2825–2830 (the `naive_bayes`
    API & default `var_smoothing`/`alpha`); ISLR §4.4.4 (DOI 10.1007/978-1-0716-1418-1). `Previous:
    03 — The Gaussian likelihood` · `Next: 05 — Text classification (the demanding case)`.

## Honest scoping (stated in the notebook)

- On separable penguins the dials barely move the **headline accuracy** — said plainly; we show each
  dial's **mechanism** (the var_smoothing collapse at 10, the α healing the zero, the prior tilting the
  boundary) and note they bite harder on the noisy, high-dimensional text of NB 5.
- `alpha` is demonstrated on NB 1's binned penguins counts (the promised zero-frequency payoff), **not**
  on text — NB 5 owns the by-hand vectorization on-ramp; `MultinomialNB`/`BernoulliNB` are named here.
- **Calibration is named, not figured** in NB 4 (penguins too separable; Brier 0.0006) — the full
  treatment is NB 5.
- Tuning on this clean data gains little; the notebook is explicit that the **discipline** is the lesson,
  not the (tiny) accuracy gain.

## Verification

Build via `uv run python - < /tmp/build_nb4.py`. Numbers re-run & reconciled (var_smoothing CV
0.9927→0.989→0.711; α heals 0.000→0.0065→…; prior 124↔127 + the borderline flip; Brier 0.0006). Runs
top-to-bottom (nbconvert to /tmp; **output-free**, `--clear-output --inplace` before commit), **no
feature-name warning** (fit on `X.to_numpy()`); `check_no_hardcoded_hex` passes; `gen_llms_txt` re-run;
`pytest` green (14, no `src/` change); both reviewers PASS (no BLOCK); Rémy validates visually; commit
`feat(02_naive_bayes): notebook 04 — the estimators & their parameters`; merge `notebook → chapter`.
