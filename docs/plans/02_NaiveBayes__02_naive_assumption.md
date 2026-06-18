# Notebook plan — 02_NaiveBayes / 02_naive_assumption

> Status: **APPROVED** (2026-06-18, by Rémy — validated alone; both reviewers gate the *built*
> notebook). Build via `uv run python - < /tmp/build_nb2.py` (stdin).

## Context

NB 2 of 5. One concept: the **"naive" assumption — conditional independence**. NB 1 did Bayes on a
*single* feature; real penguins carry several. To use two, we need the **joint** likelihood
P(bill, flipper ∣ species). Estimating it directly is expensive (a 2-D grid, data spread thin — a
curse-of-dimensionality echo from KNN). Naive Bayes takes a shortcut: **assume the features are
independent *given the class*** so the joint factorises into a product of the 1-D likelihoods we
already built in NB 1. We show what that buys (few parameters, tractable, scales), where it breaks
(the penguins' features are correlated *within* a species), and — the heart — that **classification
often survives the violation anyway** (Domingos & Pazzani 1997). Everything by hand on penguins; the
Gaussian density is still deferred to NB 3 (we stay in NB 1's counting/binning world for the
mechanism).

## What the learner does by hand (all numbers measured; seed 0; `datasets.load_penguins()`, 274 rows)

- **Two features:** `bill_length_mm`, `flipper_length_mm` (the module-00 pair).
- **The naive factorisation:** P(bill, flipper ∣ y) ≈ P(bill ∣ y)·P(flipper ∣ y) — multiply the two
  1-D binned likelihoods from NB 1.
- **Where it breaks — measured violation:** within-class correlation of the two features is
  **0.326 (Adélie) / 0.661 (Gentoo)** (the conditional quantity the assumption is about; the overall
  0.869 is mostly *between*-class and is the wrong number to quote). Real binned joint vs naive
  outer-product differ by up to **0.112** per cell (Gentoo, full 5×5 grid via `np.histogram2d`) — the
  correlation the product throws away.
- **Does classification survive? — the honest CV punchline:** GaussianNB (naive, diagonal per-class
  covariance) **0.9927** ties LDA (shared full covariance) **0.9927** and *beats* QDA (per-class full
  covariance) **0.9890** — 5-fold CV, seed 0, raw. The violation cost **no accuracy here**. Framed
  honestly: GaussianNB is **QDA with the off-diagonal covariances forced to zero**; the tie is an
  accuracy coincidence on **near-separable** 2-D data, *not* a model identity, and the assumption
  hurts the *probabilities* (calibration) more than the *decision* (shown on text in NB 5).
- A light **by-hand 2-feature prediction** extends NB 1's argmax to the product of marginals on a
  couple of example points (mechanism), without an accuracy claim from the crude binned model.

## Library / figures

**No `src/` additions.** Reuse `viz.use_course_style()`, `ml_course.colors` (`CLASS_CYCLE`, `COLORS`,
`CMAP_COUNT`, `CMAP_DIVERGING`), `datasets.load_penguins`/`penguins_xy`. sklearn for the punchline only:
`GaussianNB`, `QuadraticDiscriminantAnalysis`, `LinearDiscriminantAnalysis`, `cross_val_score`,
`StratifiedKFold` — **named as forward-references** (their Gaussian internals are built in NB 3–4; here
they are instruments to answer "does the violation hurt accuracy?"). Three in-notebook figures (charter
colours), all generated & eyeballed at plan time:
- **F1 scatter** `bill × flipper` by species — the two clouds, each visibly **tilted** (correlation).
- **F2 joint heatmaps** (one species): **real** binned joint P(bill,flipper∣Gentoo) (mass on a
  **diagonal** ridge) vs **naive** outer-product P(bill∣G)·P(flipper∣G) (rectangular spread), plus a
  **difference panel** (real − naive, `CMAP_DIVERGING`) so the correlation the product misses pops.
- **F3 CV-accuracy bars** GaussianNB / QDA / LDA (0.993 / 0.989 / 0.993), y-axis zoomed.
`pytest` stays 14.

## Cell-by-cell (~21 cells; one concept; "Read the figure" after every figure)

1. (md) **Header** — `# 02 — The "naive" assumption`; *notebook 2 of 5*; purpose; **Prerequisites:**
   NB 01 (Bayes from counts: prior/likelihood/posterior); module 00 NB 02 (the two features, feature
   space). **What you'll be able to do** (state conditional independence; factorise a joint likelihood;
   measure the violation via within-class correlation; explain why classification can survive it).
2. (code) imports + `np.random.seed(0)` + `viz.use_course_style()`; load `df`, `X, y = penguins_xy`;
   show the two features (`df.head()`, shapes).
3. (md) **Recap & the new problem** — NB 1 gave Bayes on **one** feature. Two features need the
   **joint** likelihood P(bill, flipper ∣ species). Re-establish **joint probability** ("both at once").
4. (md) **Why the joint is expensive** (intuition) — to estimate P(bill, flipper ∣ y) by counting we
   need a 2-D grid of bins; cells multiply, each holds few penguins; with d features the grid explodes
   (the curse, from KNN NB 5). We need a shortcut.
5. (code) show the 2-D binned joint counts for one species (a sparse-ish grid) to make the cost real.
6. (md) **Read the figure** — many cells, few penguins each; reliable estimates of every cell would
   need far more data. The shortcut must avoid the full grid.
7. (md) **The naive assumption** (intuition) — assume the features are **independent given the class**:
   P(x₁, x₂ ∣ y) = P(x₁ ∣ y)·P(x₂ ∣ y). Then we only need the **1-D marginals** from NB 1 (cheap).
   Define **conditional independence** plainly (independent *within* each class).
8. (code) build the **naive joint** = outer product of the two 1-D binned marginals; the **real joint**
   = 2-D binned counts; F2 (real / naive / difference heatmaps).
9. (md) **Read the figure** — real mass sits on a **diagonal** (long bills go with long flippers); the
   naive product spreads it into a **rectangle**; the difference panel shows exactly the correlation the
   product discards. The assumption is an **approximation**, not the truth.
10. (md) **Where it breaks** (intuition) — that diagonal is **correlation within a class**. Look at the
    raw cloud.
11. (code) F1 scatter + within-class correlation (0.326 Adélie, 0.661 Gentoo); print both, and the
    overall 0.869 for contrast.
12. (md) **Read the figure** — each species' cloud tilts: within a species, longer bills tend to go
    with longer flippers (corr 0.33 / 0.66). Independence assumes **no tilt** — so it is violated here,
    *moderately*. Honest note: the headline 0.869 is mostly the gap *between* species; the assumption
    is about the *within*-class number.
13. (md) **What the assumption buys** (intuition) — d features: the full joint needs exponentially many
    cells; the naive product needs only **d** 1-D marginals. Few parameters, fast, and it **scales to
    high dimensions** where the grid (and KNN's distances) fail — a forward-pointer to text (NB 5).
14. (md) **The real question** (intuition) — the assumption is **false** here. Does the *decision*
    break? Compare a model that ignores the correlation (naive) with ones that use it.
15. (code) `cross_val_score` (5-fold, seed 0) for **GaussianNB** (naive), **QDA** (per-class full cov),
    **LDA** (shared full cov); F3 bars. (Estimators named as forward-references — built in NB 3–4.)
16. (md) **Read the figure** — naive Bayes **0.993** ties LDA and **beats** QDA **0.989**: using the
    correlation did **not** help here. A wrong assumption, a sound decision (Domingos & Pazzani 1997).
17. (md) **Why — and the honest limits** — GaussianNB is **QDA with the off-diagonal covariances forced
    to zero**; the three tie because the two classes are **nearly separable** in 2-D (a property of
    *this* data, not a model identity — on correlated, overlapping data they diverge). And the
    assumption dents the **probabilities** (over-confidence) more than the **decision** — we measure
    that on text in NB 5.
18. (md) **Your turn** (2–3, tiered) — (a) compute the within-class correlation for a *different* pair
    and predict whether naive Bayes will suffer; (b) rebuild the naive vs real joint for **Adélie** and
    read the difference; (c) *harder* — explain how the argmax can stay correct even when the
    product-of-marginals gives the wrong *probability* (the seed of NB 5's calibration lesson).
19. (md) **What you built** — bullets: the joint likelihood; the naive (conditional-independence)
    factorisation; measuring the violation (within-class correlation); the measured punchline that the
    decision survived. Vocabulary added to the running glossary (joint likelihood, conditional
    independence, the naive assumption, what it buys / where it breaks).
20. (md) **Going further** (optional) — Domingos & Pazzani's argmax-robustness result; the calibration
    cost to come (NB 5); naive Bayes as a **generative** model (named, → ch 03 bridge).
21. (md) **References** — Domingos & Pazzani 1997 (DOI 10.1023/A:1007413511361); Hand & Yu 2001
    (DOI 10.1111/j.1751-5823.2001.tb00465.x); ISLR §4.4.4 (DOI 10.1007/978-1-0716-1418-1).
    `Previous: 01 — Bayes' rule, from counts` · `Next: 03 — The Gaussian likelihood`.

## Honest scoping (stated in the notebook)

- The naive assumption is **conditional** independence (within a class), only **moderately** violated
  here (within-class corr 0.33–0.66, not the between-class 0.869).
- The CV punchline uses sklearn estimators **named as forward-references** (Gaussian internals → NB 3–4);
  here they are instruments, not taught. GaussianNB = diagonal-covariance QDA, stated.
- The NB = LDA/QDA tie is a property of **near-separable** data, **not** a model identity; on correlated,
  overlapping data they diverge — said plainly.
- Classification surviving the violation does **not** mean the *probabilities* survive; calibration
  suffers — deferred to NB 5 (named here).
- Binning remains a teaching device (NB 3 replaces it with a Gaussian); the by-hand classifier is a
  *mechanism* demo, not an evaluation (the honest CV bars carry the accuracy claim).

## Verification

Build via `uv run python - < /tmp/build_nb2.py`. Numbers re-run & reconciled (within-class corr
0.326/0.661, overall 0.869; real−naive joint max 0.107; GaussianNB/QDA/LDA CV 0.9927/0.9890/0.9927).
Runs top-to-bottom (nbconvert to /tmp; **output-free**, `--clear-output --inplace` before commit);
`check_no_hardcoded_hex` passes; `gen_llms_txt` re-run; `pytest` green (14, no `src/` change); both
reviewers PASS (no BLOCK); Rémy validates visually; commit `feat(02_naive_bayes): notebook 02 — the
naive assumption`; merge `notebook → chapter`.
