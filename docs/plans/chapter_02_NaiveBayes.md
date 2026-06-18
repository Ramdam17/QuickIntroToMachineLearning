# Chapter plan — 02_NaiveBayes (Naive Bayes)

> Status: **APPROVED** (2026-06-18, by Rémy; reviewer-gated — ml-expert REVISE→1 BLOCK fixed (NB≠LDA;
> GaussianNB = diagonal-covariance QDA), re-checked PASS; pedagogy REVISE→no BLOCK, 4 MAJORs folded in).
> The course's **second method**, the first **probabilistic** one. Per the per-method arc: NB 1–3
> fundamentals (one concept each, by hand before the library), NB 4 the estimator & its parameters,
> NB 5 a demanding practical case. **Five notebooks, standard arc, no optional NB 6.** Drives the
> notebook loop in `docs/WORKFLOW.md`.

## Context

KNN classified by **distance**; Naive Bayes classifies by **probability**. It is the natural next
step: the first method that outputs *P(class | features)*, built entirely from counting and Bayes'
rule — so it lets us re-establish probability from scratch (prior, likelihood, posterior, the
evidence). It is also a beautiful honesty lesson: its defining "naive" assumption (features
independent **given the class**) is usually *false*, yet the classifier is often excellent — the gap
between *a wrong model* and *a useful decision* (Domingos & Pazzani 1997). Two contrasts land here:
**lazy vs eager** (KNN stores data; NB fits a handful of parameters, then discards the data) and
**distance vs probability** (NB has no metric, so it walks straight through the high-dimensional text
where KNN drowned in the curse — chapter 01, NB 5).

## Prerequisites (re-established briefly; **first-contacts flagged, not mislabelled as recaps**)

- **Module 00 (genuine recaps):** train/test split & leakage (NB 04); accuracy + baseline (NB 06);
  confusion matrix, precision/recall/F1 and the **PR curve** (NB 07–08) — reused for honest evaluation
  under imbalance; cross-validation (NB 10); `Pipeline` / fit-on-train-only (NB 11).
- **Chapter 01 (KNN):** the curse of dimensionality (NB 5) — text is very high-dimensional and NB
  *thrives* there (no distances); lazy vs eager learners.
- **Re-established from scratch (assumed from nobody):** probability as a relative frequency;
  conditional probability P(A∣B); joint vs marginal; **Bayes' rule** + its four terms (prior,
  likelihood, posterior, evidence); independence and **conditional** independence; argmax and why the
  evidence cancels.
- **FIRST CONTACTS (taught here for the first time — budget them, do not call them recaps):**
  (a) **continuous probability density** vs discrete mass + the **Gaussian pdf** (the learner has only
  *counted* so far; module-00 NB 08 showed score "humps" visually, never a formal density where *area*,
  not height, is probability and f(x) can exceed 1) — **NB 3**; (b) **bag-of-words / vectorization /
  the document-term matrix / sparsity** — a new *data type*, built by hand before `CountVectorizer`
  — **NB 5**; (c) **calibration**, the **reliability diagram**, the **Brier score** (NB 08 explicitly
  deferred these to this chapter) — bridged in **NB 4**, taught in full in **NB 5**; (d) **multi-class
  argmax** (the chapter, like the whole course so far, has only shown binary; one sentence that NB 1's
  argmax rule extends unchanged to >2 classes) — **NB 5**.

## Datasets (measured at plan time; seeds fixed. **Penguins numbers reproduced by the reviewer; text numbers are INDICATIVE and re-measured at build on the pinned sklearn — they are version-sensitive.**)

- **NB 1–4: Palmer penguins** — binary 2-feature subset (`load_penguins`/`penguins_xy`: Adélie 151 /
  Gentoo 123, `bill_length_mm`, `flipper_length_mm`). Continuous features → **Gaussian NB**, the
  fil-rouge vehicle. Measured facts the chapter rests on:
  - **The naive assumption is *conditional* (within-class) independence — only *moderately* violated
    here.** corr(bill, flipper): **overall 0.869** is mostly *between*-class structure (the wrong number
    to quote); the NB-relevant **within-class** value is **0.326 (Adélie) / 0.661 (Gentoo), pooled
    0.486**. NB 2 quotes the within-class number.
  - **What GaussianNB *is* (corrected per ml-expert BLOCK):** a per-class **diagonal**-covariance
    Gaussian classifier = **QDA with the off-diagonal covariances forced to zero** (QDA = per-class
    *full* covariance; LDA = a single *shared full* covariance — a different model on *two* axes). So
    the clean "what does the independence restriction cost?" contrast is **NB vs QDA** (same per-class
    structure, ± the off-diagonals), with **LDA as a third reference**.
  - **The restriction costs ~nothing on accuracy *here*** (5-fold CV, seed 0, 2 features, all **raw**):
    GaussianNB **0.9927**, QDA **0.9890**, LDA **0.9927**, LogReg **0.9927**. Three different covariance
    assumptions are **indistinguishable on this dataset because the two classes are nearly linearly
    separable in 2-D** — *not* because the models are the same. The honest takeaway: the independence
    restriction costs no accuracy **on this problem** (Domingos & Pazzani 1997, the argmax is robust),
    stated as a *measured* result, never as "NB = LDA".
  - **Penguins is too separable to expose NB's over-confidence**, so the calibration lesson is deferred
    to text. Held-out (30%, seed 0, positive=Gentoo, **both estimators raw** — GaussianNB is
    scale-invariant; single stated preprocessing, per the ml-expert fairness fix): GaussianNB Brier
    **0.0006** vs LogReg Brier **0.0023** — both excellent and well-calibrated; the separation hides any
    over-confidence pathology.
  - 3-class full penguins (4 numeric features) GaussianNB CV **0.968** — a multiclass option if needed.
- **NB 5: a 20-newsgroups subset (text)** — NB's authentic home. `sklearn.datasets.fetch_20newsgroups`
  (downloads ~14 MB **once**, then cached under `~/scikit_learn_data` — the same fetch-and-cache,
  visible-logging pattern as `datasets.load_penguins_full`; offline thereafter). **All numbers below are
  indicative and re-measured at build:**
  - 4 categories {comp.graphics, rec.sport.baseball, sci.med, talk.religion.misc}, headers/footers/
    quotes removed, train ≈2152 / test ≈1433: `CountVectorizer` + `MultinomialNB` → acc ≈ **0.89**,
    macro-F1 ≈ 0.88, **fit ≈ 2 ms** (NB's speed — a real selling point).
  - **Over-confidence, finally visible:** confusable binary {alt.atheism, talk.religion.misc} → MNB
    acc ≈ **0.70**, and **far more posteriors pile up at 0.99/0.01 than logistic regression's**, with a
    *worse* Brier despite that — NB ranks well, its probabilities are unreliable. (Exact counts drift by
    sklearn version; re-measured at build. The qualitative gap is robust.)
  - **alpha (Laplace/Lidstone smoothing):** a **flat plateau for small-but-nonzero α**, with a **cliff
    at the degenerate α→0** — strict α=0 collapses badly (a real *divide-by-zero in log* → non-finite
    `feature_log_prob_`, the zero-frequency catastrophe) versus a small α that recovers to the high-0.8s.
    The **collapse magnitude is itself category/version-sensitive** (≈0.41–0.68 across pairs) → not
    pinned; re-measured at build with `force_alpha=True` on the chosen category pair. **NB 4's own CV
    cell finds the optimum — the plan asserts no single "best α"** (version-sensitive; a finer grid may
    favour α≈0.1 over 1.0).
  - One-vs-rest imbalance for NB 5 (`sci.med` vs the rest): **test** split ≈ **396 vs 1037** (the
    evaluation-set imbalance; the train side is ≈594 vs 1558).
- All offline after first fetch; colours from `ml_course.colors`; "Read the figure" after every figure;
  a **"Your turn" (2–3 tiered exercises)** in every notebook; a running Naive-Bayes vocabulary box.

## Primordial concepts → notebooks 1–3 (one concept each; by hand before any library)

| NB | Title | The one concept | Done by hand | Key figure → "Read the figure" |
|----|-------|-----------------|--------------|-------------------------------|
| 1 | Bayes' rule, from counts | **Posterior ∝ prior × likelihood** on **one** feature: count → probability → Bayes; the **evidence** is the normalizer; predict = **argmax** posterior. "Evidence cancels under argmax" is a **one-line consequence**, not a fourth idea | discretize `bill_length` into 3 bins; species×bin contingency table; prior P(species), likelihood P(bin∣species), posterior P(species∣bin) by counting; predict by argmax. **Exploit the live zero-frequency case** the binning surfaces on penguins (a class with 0 count in a bin → posterior 0) to motivate smoothing (foreshadow NB 4) | contingency table + per-bin posterior bars; prior vs posterior after one feature |
| 2 | The "naive" assumption | **Conditional independence**: combine two features by *multiplying* per-feature likelihoods, P(x₁,x₂∣y) ≈ P(x₁∣y)·P(x₂∣y) — what it **buys** (few parameters, tractable, fast) and where it **breaks** (penguins' within-class corr 0.33–0.66). **Measured punchline, shown not asserted** (pin the CV scheme + the three numbers): NB (0.9927) classifies as well as the covariance-aware **QDA (0.9890)** / LDA (0.9927) **despite** the violation — the independence restriction costs no accuracy here (Domingos & Pazzani 1997) | estimate P(bill∣y), P(flipper∣y) separately, multiply; overlay the implied **independent (axis-aligned)** joint on the **real tilted** cloud; measure within-class corr; run NB vs **QDA** (primary contrast) vs LDA under one CV scheme | independent-vs-real density overlay (axis-aligned vs tilted); CV-accuracy bars NB / QDA / LDA |
| 3 | The Gaussian likelihood, computed safely | **Model P(feature∣class) with a density, and compute it without underflowing.** Re-scoped to *one* concept: **(a) first-contact with continuous density** — front-loaded as its own taught intuition ("until now every distribution was a *count*; here a smooth curve whose **area**, not height, is probability; f(x) can exceed 1") — then the **Gaussian** pdf (fit μ, σ per class per feature = Gaussian NB, upgrading NB 1's bins to a smooth bell); **(b) log-space** as the numeric coda (a product of many small likelihoods **underflows to 0.0** → sum of logs; argmax unchanged). **Multinomial/Bernoulli are *named, not built* here** — an explicit forward-pointer to NB 5 | fit per-class Gaussians on `bill_length`, overlay the bell on NB 1's histogram (axis relabelled *density*); assemble a 2-feature by-hand Gaussian NB; **demonstrate underflow** (a product → 0.0 in float) then redo with `np.log` | bell over histogram (mass→density); by-hand Gaussian-NB boundary; **underflow panel** — read it as *"the product is not wrong, it is **unrepresentable** (float hit 0.0); the log turns an impossible multiplication into a safe sum"* |

> **Split trigger (watched at NB-3 cell planning):** if NB 3 still reads as more than one concept once
> cell-by-cell, split **density-NB vs log-space-NB** (not family-menu vs log-space) — taking the chapter
> to 6. Default stays **5**.

## Notebook 4 — the estimators & their parameters

`sklearn.naive_bayes` for real. **Parity first:** by-hand Gaussian NB == `GaussianNB` on penguins (same
predictions, CV 0.9927). Then walk the knobs and their failure modes, each *shown*:
- **`var_smoothing`** (GaussianNB) — a variance floor; too little → a near-zero-variance feature becomes
  a spike one outlier dominates; too much → every class looks identically broad and the feature stops
  discriminating.
- **`alpha`** (Multinomial/Bernoulli) — Laplace/Lidstone smoothing and the **zero-frequency problem**
  (an unseen category/word → likelihood 0 that **annihilates the whole product**; the payoff of NB 3's
  log/underflow work and NB 1's live penguins zero-count). **NB 4's own CV cell finds α** on a small
  text snippet (foreshadowing NB 5); the plan asserts no fixed optimum, and shows the genuine α→0 cliff
  (a real log(0) divide-by-zero, a large collapse — magnitude re-measured at build) vs a small nonzero α.
- **`priors` / `class_prior` / `fit_prior`** — setting vs learning the prior; effect under imbalance.
- **Calibration — a *tight bridge*, not the full treatment (per pedagogy MAJOR-4):** one intuition +
  one figure + one read showing NB's posteriors are **over-confident** (the independence assumption
  double-counts correlated evidence) → hands off to NB 5, where the reliability diagram + Brier are
  load-bearing under imbalance. (`viz.plot_calibration_curve` lands in NB 5 where calibration is taught
  in full; NB 4 may preview it.)
- Choose a parameter honestly by `cross_val_score` / `GridSearchCV` on TRAIN; one sealed test eval.
- **Your turn:** reproduce the α behaviour and choose α by CV; nudge `var_smoothing` to the failure modes.

## Notebook 5 — the demanding practical case: **text classification**

Naive Bayes on a **20-newsgroups subset** — the data type where NB genuinely shines and is still a
strong, instant baseline. The full honest workflow, **with the new data type built by hand first**:
- **By-hand vectorization on-ramp (per pedagogy MAJOR-3 — built, not "just happening"):** 3–4 toy
  sentences → a hand-built vocabulary → a small **dense** count matrix the learner reads by eye → *then*
  reveal `CountVectorizer` does exactly this at scale, and why the real matrix is **sparse**. One "Read
  the figure" on the toy document-term matrix. This is what makes "multinomial likelihood = counts"
  (NB 1's idea, vectorized) concrete.
- look (class balance, document lengths) → `CountVectorizer` (fit on TRAIN only — NB 11 leakage rule) →
  `MultinomialNB` → one held-out evaluation. **Multi-class argmax** named (NB 1's rule extends to >2).
- **Honest evaluation under imbalance:** frame one topic as a rare positive (`sci.med` vs the rest,
  **test ≈ 396 vs 1037**) → report **precision/recall/F1 and the PR curve, not accuracy** (module-00
  NB 06–08); plus a confusable pair (atheism vs religion, acc ≈ 0.70) for genuinely hard, honest errors.
- **The calibration limit, taught in full here:** the over-confidence measured above (NB's posteriors
  pile at 0/1 far more than logistic regression's, worse Brier) → **reliability diagram + Brier**;
  *read* it as *"the ranking is still good — use the PR curve — but the number is not a probability."*
- **The Domingos–Pazzani loop closes:** in text the independence assumption is *wildly* violated (words
  co-occur), the feature space is huge — exactly where KNN died of the curse (ch 01) — yet NB is fast
  (≈2 ms fit) and strong. The honest contrast that explains *when to use NB*.
- **Bridge to chapter 03:** NB is **generative** (models P(x∣y), inverts via Bayes); logistic
  regression, next chapter, is **discriminative** (models P(y∣x) directly). Named (Ng & Jordan 2001) as
  the hand-off — **not** a sixth notebook.
- **Your turn:** swap in a different confusable newsgroup pair and do the honest error read; inspect the
  most "spammy" words per class via the fitted log-probabilities.

**Five notebooks (the standard arc); no optional NB 6.** Both reviewers agree on five. KNN earned a
sixth as *the* distance method; NB's generative/discriminative capstone belongs next to logistic
regression (ch 03), folded into NB 5's closing bridge. (Split only via the NB-3 trigger above.)

## Library additions (decided per-notebook, with tests; none forced now)

- **Likely (NB 5):** `datasets.load_newsgroups(categories=..., subset=...)` — a thin fetch-and-cache
  wrapper over `fetch_20newsgroups` with **visible logging** (mirrors `load_penguins_full`), returning
  texts / labels / target names.
- **Likely (NB 5, previewed NB 4):** `viz.plot_calibration_curve(y_true, proba)` — a reliability
  diagram in charter colours.
- NB 1's contingency bars, NB 2's independent-vs-real density overlay, NB 3's bells + underflow panel,
  NB 5's toy document-term matrix are **one-off in-notebook figures** (charter colours). Reuse
  `plot_decision_boundary` (Gaussian-NB boundary), `plot_confusion_matrix`, `plot_feature_histograms`,
  `plot_roc_curve`, `plot_score_threshold`. Each new helper ships with a `pytest` test; `pytest` grows
  from 14 only as helpers are added.

## Honest limits stated in the notebooks

- The "naive" conditional-independence assumption is **usually false**; we *measure* the violation
  (penguins within-class corr ~0.49; text far worse) and show **classification** survives — **without
  pretending the assumption holds**. The **probability estimates do not** survive: NB is **over-confident
  / poorly calibrated** (shown on text), so its posteriors are for *ranking*, not for trusting as
  calibrated probabilities.
- GaussianNB is **QDA with diagonal (independent) per-class covariance** — equal accuracy to QDA/LDA on
  penguins is a property of *this near-separable data*, not a model identity; on data with unequal,
  correlated per-class covariances they diverge. Stated explicitly.
- Gaussian NB assumes each feature is **Gaussian within each class** — false for skewed/multimodal
  features; named, with the fix (transform, or a different likelihood) pointed to, not taught.
- The **zero-frequency problem** is real (an unseen category zeroes the product; the literal α→0
  divide-by-zero) and is *why* smoothing exists; shown, not hidden.
- NB is **generative** and **eager** (fits μ/σ or counts, then discards the data — unlike lazy KNN);
  fast to train and predict, scales to high-dimensional sparse data (text) where distance methods fail.
  **When to use:** high-dimensional / sparse / text, a fast strong baseline, small data. **When not:**
  when calibrated probabilities matter, or when features are strongly dependent *and* the decision (not
  just the ranking) needs the joint structure (→ LDA/QDA, logistic regression, ch 03).
- moons/penguins are low-dimensional by design; the curse-beating claim is **measured on text**, not
  asserted from low-d.

## References (with DOI; all ten resolved by the ml-expert reviewer; re-verified at build)

- Bayes T, Price R (1763). *An essay towards solving a problem in the doctrine of chances.* Phil. Trans.
  R. Soc. 53:370–418. DOI: 10.1098/rstl.1763.0053 — the rule's origin.
- Domingos P, Pazzani M (1997). *On the optimality of the simple Bayesian classifier under zero-one
  loss.* Machine Learning 29:103–130. DOI: 10.1023/A:1007413511361 — the chapter's heart (argmax robust
  to the violated assumption).
- Hand DJ, Yu K (2001). *Idiot's Bayes — not so stupid after all?* Int. Stat. Rev. 69(3):385–398.
  DOI: 10.1111/j.1751-5823.2001.tb00465.x — why NB works despite being "naive".
- Zhang H (2004). *The optimality of naive Bayes.* FLAIRS-17 — conditions for optimality.
- Ng AY, Jordan MI (2001). *On discriminative vs. generative classifiers: a comparison of logistic
  regression and naive Bayes.* NeurIPS 14:841–848 — the bridge to chapter 03.
- Rennie JDM, Shih L, Teevan J, Karger DR (2003). *Tackling the poor assumptions of naive Bayes text
  classifiers.* ICML-2003 — text-NB pathologies (over-confidence, length, TF-IDF/Complement fixes).
- Manning CD, Raghavan P, Schütze H (2008). *Introduction to Information Retrieval*, ch. 13.
  DOI: 10.1017/CBO9780511809071 — NB 5 grounding.
- Niculescu-Mizil A, Caruana R (2005). *Predicting good probabilities with supervised learning.*
  ICML-2005. DOI: 10.1145/1102351.1102430 — NB is poorly calibrated (the calibration limit).
- ISLR (James et al., 2021) §4.4.4. DOI: 10.1007/978-1-0716-1418-1. ESL (Hastie et al., 2009) §6.6.3.
  DOI: 10.1007/978-0-387-84858-7.

## Verification (per notebook, at its commit)

Every number re-run and reconciled into prose at build (within-class corr 0.326/0.661/0.486;
**NB/QDA/LDA 0.9927/0.9890/0.9927 raw**; held-out Brier NB 0.0006 vs **raw** LogReg 0.0023; text acc,
α→0 collapse (log(0) divide-by-zero; magnitude unpinned), over-confidence gap — all re-measured on the
pinned sklearn, marked indicative until then). Runs top-to-bottom (nbconvert to /tmp; **output-free**,
`--clear-output --inplace` before commit); `check_no_hardcoded_hex` passes; `gen_llms_txt` re-run;
`pytest` green; `course_map.md` §02 aligned to final titles. Both reviewers pass (no BLOCK) on each
notebook; Rémy validates visually; commit per notebook; **chapter close via PR into `main`** (protected).

## Per-notebook status (update as the chapter is built)

| NB | Branch | Status |
|----|--------|--------|
| 1 | `notebook/02_NaiveBayes__01_bayes_from_counts` | **done** — built (21 cells), both reviewers PASS (4 MINORs folded), Rémy validated visually, merged to `chapter/02_NaiveBayes` |
| 2 | `notebook/02_NaiveBayes__02_naive_assumption` | **done** — built (21 cells), both reviewers PASS (shared MINOR "Read the table" folded), Rémy validated visually, merged to `chapter/02_NaiveBayes` |
| 3 | `notebook/02_NaiveBayes__03_gaussian_likelihood_logspace` | **done** — built (20 cells), both reviewers PASS (3 banned words + honesty/NaN MINORs folded), Rémy validated visually, merged to `chapter/02_NaiveBayes`. Stayed at 5 (split not pulled). |
| 4 | `notebook/02_NaiveBayes__04_estimators_and_parameters` | **done** — built (20 cells), both reviewers PASS (ml-expert REVISE→Brier-assertion reframed + 2 MINORs; pedagogy PASS); calibration named & deferred to NB 5; Rémy validated visually, merged to `chapter/02_NaiveBayes` |
| 5 | `notebook/02_NaiveBayes__05_text_classification` | planned |

## Reviewer notes

- **ML expert — REVISE (1 BLOCK, fixed in v2; re-check PASS):** BLOCK — "NB = LDA" was a model-identity
  error; GaussianNB is **QDA with off-diagonals zeroed** (verified at the parameter level: exact modulo
  ddof; QDA off-diagonals +0.326/+0.661 genuinely nonzero), the 0.9927 tie is an accuracy coincidence on
  near-separable 2-D data (LinearSVC 0.9964) → reframed; contrast is now NB-vs-QDA. MAJOR — unfair
  NB(raw)-vs-LogReg(scaled) comparison → both **raw** (LogReg CV 0.9927, Brier 0.0023). MAJOR — α-curve
  version-sensitive → all text numbers marked indicative/re-measured, NB 4 CV finds the optimum, α→0
  collapse = a real log(0) divide-by-zero (magnitude unpinned, ≈0.41–0.68 across pairs). MINOR —
  sci.med count now **test** (396 vs 1037). Praised: within-class framing, calibration deferral, the
  live penguins zero-frequency case (exploited in NB 1/4), DOIs all resolve.
- **Pedagogy — REVISE, no BLOCK (passes gate); 4 MAJORs folded in:** NB 3 re-scoped to one concept
  (Gaussian density + log-space), mass→density front-loaded as a *taught first-contact*,
  multinomial/Bernoulli demoted to "named, not built → NB 5", split-trigger noted. Density and
  bag-of-words relabelled **first contacts** in Prerequisites. NB 5 gains a **by-hand vectorization
  on-ramp**. NB 4 calibration reduced to a **tight bridge** (full treatment → NB 5). MINORs: named the
  two argument-figure readings; NB 2 punchline pinned as measured; "evidence cancels" = one-liner;
  **"Your turn" (2–3 tiered) added per notebook**; multi-class argmax sentence in NB 5. Praised: the
  fil-rouge, the honesty spine, declining NB 6.
