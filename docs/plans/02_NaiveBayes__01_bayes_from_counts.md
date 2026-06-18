# Notebook plan — 02_NaiveBayes / 01_bayes_from_counts

> Status: **APPROVED** (2026-06-18, by Rémy — notebook plans are validated by Rémy alone; both
> reviewers return on the *built* notebook). Build via `uv run python - < /tmp/build_nb1.py` (stdin, to
> dodge the stray `/tmp/struct.py` shadow).

## Context

NB 1 of 5 — the chapter's foundation and the course's **first probabilistic classifier**. One concept:
**the posterior is proportional to prior × likelihood**, built entirely by **counting** on a single
feature. KNN voted by distance; here we ask, for a penguin's measurement, *how probable is each
species?* — and answer it with Bayes' rule, term by term. Everything is by hand (counts → fractions →
Bayes → argmax); no library estimator yet (`GaussianNB` arrives in NB 4). The penguins' `bill_length`,
discretized into three bins, even hands us a **live zero-frequency case** (no Adélie in the long-bill
bin) — the first crack we will later fix with smoothing (NB 4).

## What the learner does by hand (all numbers measured; seed 0; `datasets.load_penguins()`, 274 rows)

- **Bins** of `bill_length_mm` (range 32.1–59.6): `short [30,42)`, `medium [42,48)`, `long [48,62]`.
- **Contingency counts** (species × bin): Adélie **[135, 16, 0]**, Gentoo **[3, 67, 53]**.
- **Prior** P(species): Adélie **0.551**, Gentoo **0.449** (class frequencies).
- **Likelihood** P(bin∣species) (rows sum to 1): Adélie [0.894, 0.106, **0.000**], Gentoo [0.024, 0.545,
  0.431]. → the **zero-frequency case**: P(long∣Adélie)=0 (we never *saw* a long-billed Adélie).
- **Evidence** P(bin) (the normalizer): [0.504, 0.303, 0.193].
- **Posterior** P(species∣bin) = P(bin∣species)·P(species) / P(bin): short→Adélie **0.978**,
  medium→Gentoo **0.807** (a genuinely uncertain region — a probability, not a verdict), long→Gentoo
  **1.000** (exactly 1 *because* the Adélie likelihood is 0 there — the same zero-frequency
  over-confidence).
- **Predict by argmax:** short→Adélie, medium→Gentoo, long→Gentoo. "**Evidence cancels under argmax**"
  — P(bin) is the same denominator for both species in a bin, so comparing posteriors = comparing
  prior×likelihood; we needn't normalize to *decide* (a one-line consequence, not a fourth idea).

## Library / figures

**No `src/` additions.** Reuse `viz.use_course_style()`, `ml_course.colors` (`CLASS_CYCLE` for the two
species, `COLORS` roles), `viz.plot_class_balance` (the prior bars), `datasets.load_penguins`/
`penguins_xy`. The contingency table is a pandas `crosstab` (displayed); the likelihood and posterior
**grouped bar charts** and the prior bar are **one-off in-notebook figures** in charter colours. Code
uses keyword `axis=` on `.sum()` (pandas-4 clean). `pytest` stays 14.

## Cell-by-cell (~21 cells; one concept; "Read the figure" after every figure)

1. (md) **Header** — `# 01 — Bayes' rule, from counts`; *notebook 1 of 5 — Naive Bayes*; one-line
   purpose; **Prerequisites:** Module 01 (KNN — we classified by distance; now by probability);
   Module 00 NB 06 (class frequencies / baseline). **What you'll be able to do** (4 verbs: compute a
   prior, a likelihood, a posterior by counting; predict by argmax; explain the evidence; spot the
   zero-frequency trap). Warm one-line welcome.
2. (code) **Imports + setup** — numpy/pandas, `from ml_course import datasets, viz, colors`,
   `np.random.seed(0)`, `viz.use_course_style()`. Load `df = datasets.load_penguins()`; show
   `df.head()`, `df.shape`, `df["species"].value_counts()`.
3. (md) **Recap / footing** — re-establish, briefly but for real: **probability = a relative frequency**
   (count ÷ total); we have two species and one measurement; the goal is P(species ∣ measurement). The
   switch from KNN's distance vote to a probability.
4. (md) **The prior** (intuition) — *before* looking at any bill, how often is each species? That belief
   is the **prior** P(species).
5. (code) Compute the prior by counting (`value_counts(normalize=True)`); `viz.plot_class_balance` or a
   prior bar.
6. (md) **Read the figure** — Adélie 0.551, Gentoo 0.449; our belief with no measurement yet; a baseline
   to update.
7. (md) **One feature, discretized** (intuition) — to *count*, we group the continuous `bill_length`
   into three readable bins (short/medium/long). Why bin: counting needs categories (NB 3 will replace
   bins with a smooth curve — named, not now).
8. (code) `pd.cut` into the three bins; `pd.crosstab(species, bin)` → the **contingency table**; display.
9. (md) **Read the figure** — the counts: Adélie short-billed [135,16,0], Gentoo longer [3,67,53]; flag
   the **0** (no Adélie in the long bin) — we return to it.
10. (md) **The likelihood** (intuition) — *given* a species, how do its bills distribute across bins?
    Row-normalize the table: P(bin ∣ species). "If it's an Adélie, how likely a short bill?"
11. (code) Likelihood = contingency normalized per row (`.div(.sum(axis=1), axis=0)`); grouped bar.
12. (md) **Read the figure** — rows sum to 1; P(short∣Adélie)=0.894; **P(long∣Adélie)=0.000** — the
    **zero-frequency case**, named kindly: the model would call a long-billed Adélie *impossible*, only
    because we never observed one — an over-confidence we fix with smoothing in NB 4.
13. (md) **Bayes' rule** (intuition) — to go from P(bin∣species) to what we want, P(species∣bin), flip it
    with Bayes: posterior ∝ prior × likelihood, normalized by the **evidence** P(bin). Write the formula;
    name all four terms (prior, likelihood, posterior, evidence).
14. (code) By hand: `joint = likelihood * prior` (broadcast over species); `evidence = joint.sum(axis=0)`;
    `posterior = joint / evidence`; display the posterior table (columns sum to 1).
15. (md) **Read the figure** — short→Adélie 0.978, medium→**Gentoo 0.807** (genuinely uncertain — a
    probability, not a verdict), long→Gentoo **1.000** (exactly 1 because the Adélie likelihood was 0 —
    the same zero-frequency over-confidence, now in the posterior).
16. (code) **Predict by argmax** — per bin, the species with the largest posterior (`posterior.idxmax`);
    show the three predictions; one line showing prior×likelihood gives the *same* argmax as the full
    posterior (the evidence cancels).
17. (md) **Read** — the argmax decision; the **evidence-cancels** consequence in one line; we just built
    a classifier from counting. A small **vocabulary box** (prior / likelihood / posterior / evidence /
    argmax) — the running NB glossary starts here.
18. (md) **Your turn** (2–3, tiered) — (a) recompute a bin's posterior under a flat 50/50 prior and watch
    it shift; (b) re-bin with different edges (or 4 bins) and rebuild the contingency/posterior; (c)
    *harder* — the long-bin posterior is exactly 1.0: explain why that is over-confident and propose the
    fix (add 1 to every count — Laplace smoothing; we formalize it in NB 4). Scaffold only, not solved.
19. (md) **What you built** — concrete bullets: prior, likelihood, posterior, evidence, argmax; a working
    classifier by counting on one feature; you met the zero-frequency trap and named its fix.
20. (md) **Going further** (optional) — two forward-pointers: many features (NB 2 multiplies likelihoods —
    the "naive" step); continuous features without bins (NB 3's Gaussian density). Clearly optional.
21. (md) **References** — Bayes & Price (1763) DOI 10.1098/rstl.1763.0053; ISLR §4.4.4 DOI
    10.1007/978-1-0716-1418-1. `Previous: Module 01 — KNN` · `Next: 02 — the naive assumption`.

## Honest scoping (stated in the notebook)

- This builds the **rule** by counting on the full subset — it is the *mechanism*, not yet an honest
  held-out *evaluation* (that lands with the real estimator, NB 4–5). Said plainly, as chapter-01 NB 1
  did for the vote.
- Binning a continuous feature is a teaching device to enable counting; it throws away within-bin
  detail. NB 3 replaces it with a smooth Gaussian likelihood (named here, built there).
- The zero-frequency posterior (exactly 0 / exactly 1) is an **artifact of an unobserved combination**,
  not a true impossibility — the motivation for smoothing (NB 4). Shown, not hidden.

## Verification

Build via `uv run python - < /tmp/build_nb1.py` (stdin). Numbers re-run and reconciled
(priors 0.551/0.449; contingency [135,16,0]/[3,67,53]; likelihood incl. P(long∣Adélie)=0; posteriors
0.978/0.807/1.000; evidence 0.504/0.303/0.193). Runs top-to-bottom (nbconvert to /tmp; **output-free**,
`--clear-output --inplace` before commit); `check_no_hardcoded_hex` passes; `gen_llms_txt` re-run;
`pytest` green (14, no `src/` change); both reviewers PASS (no BLOCK) on the built notebook; Rémy
validates visually; commit `feat(02_naive_bayes): notebook 01 — Bayes' rule, from counts`; merge
`notebook → chapter`.
