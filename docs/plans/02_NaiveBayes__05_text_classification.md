# Notebook plan — 02_NaiveBayes / 05_text_classification  (the demanding case — capstone)

> Status: **APPROVED** (2026-06-18, by Rémy — validated alone; reviewers gate the *built* notebook).
> Closes chapter 02 → PR `chapter/02_NaiveBayes` → `main`. Build via
> `uv run python - < /tmp/build_nb5.py` (stdin).

## Context

NB 5 of 5 — the chapter's **demanding case**, and Naive Bayes on its **home turf: text**. KNN drowned
in high dimensions (chapter 01, the curse); NB *thrives* there, because it never measures a distance —
it counts words. The full honest workflow on a real corpus: build the text→numbers representation **by
hand** first, fit `MultinomialNB`, then evaluate **honestly under imbalance** (precision/recall/F1/PR,
not accuracy) and finally confront the limit NB 4 named — **calibration**. Closes the **Domingos–Pazzani
loop** (independence absurdly violated by co-occurring words, yet NB is fast and strong) and **bridges
to chapter 03** (generative NB vs discriminative logistic regression). **Capstone → visualization-first;
~25 cells (≥20 is the floor).**

## Datasets & measured numbers (seed 0; 20-newsgroups, fetched-once-cached, `remove=(headers,footers,quotes)`)

- **4 categories** {comp.graphics, rec.sport.baseball, sci.med, talk.religion.misc}, train **2152** /
  test **1433**. `CountVectorizer(stop_words="english", min_df=2)` fit on TRAIN → **12 384-word**
  vocabulary; document-term matrix **sparse** (density **0.0043**). `MultinomialNB`: **fit ≈1.2 ms**,
  test **accuracy 0.887**, **macro-F1 0.882**; confusion matrix → talk.religion.misc the hardest.
- **Honest eval under imbalance — one-vs-rest `sci.med`**: train 594 vs 1558, **test 396 vs 1037**.
  Accuracy **0.930** but majority baseline **0.724**; honest read **precision 0.887 / recall 0.854 /
  F1 0.870**, PR **average-precision 0.935**.
- **Calibration:** MNB pushes **1205/1433** test probabilities past 0.99/0.01 (LogReg 534) — structural
  **over-confidence in shape**. On this *easy* task MNB Brier (**0.0559**) is *better* than LogReg
  (0.0797) → we do **not** claim "NB Brier worse"; the lesson is the **pile-up** (the number is not a
  calibrated probability) and that on a hard task it backfires: confusable pair {alt.atheism,
  talk.religion.misc} → MNB Brier **0.245 > LogReg 0.223** (a *Your turn*; the headline figure stays on
  the 4-cat one-vs-rest).
- **Most informative `sci.med` words:** patients, cancer, hiv, msg, candida, symptoms, disease, vitamin,
  cure (a *Your turn*).

## New library code (with tests; `pytest` 14 → 16)

- **`datasets.load_newsgroups(categories=None, subset="train", remove=("headers","footers","quotes"),
  random_state=0)`** → tidy `pandas.DataFrame[text, category]` (category = target-name string), **visible
  `logging`** on fetch (mirrors `load_penguins_full`). Test: fetch a 2-category subset, assert DataFrame
  with those columns and the requested categories (cache after first run, per the penguins-test precedent).
- **`viz.plot_calibration_curve(y_true, proba, *, n_bins=8, strategy="quantile", label=None, ax=None,
  color=None)`** → reliability diagram (mean predicted vs observed) + the calibrated diagonal when
  `ax is None` (same pattern as `viz.plot_roc_curve`); charter colours. Test: smoke → returns a figure.
  NumPy-style docstring (shapes, "when to use", example).

## Library / figures

Reuse `viz.use_course_style`, `viz.plot_confusion_matrix`, `ml_course.colors`. sklearn: `CountVectorizer`,
`MultinomialNB`, `LogisticRegression` (calibration foil + ch-03 bridge). The toy document-term matrix,
the PR curve, and the predicted-probability histograms are one-off in-notebook figures (charter colours).
Five figures (visualization-first): toy count matrix; 4-cat confusion matrix; PR curve (one-vs-rest);
predicted-probability histograms (MNB pile-up vs LogReg); reliability diagram (new helper).

## Cell-by-cell (~25 cells; capstone; "Read the figure" after every figure)

1. (md) **Header** — `# 05 — Text classification (the demanding case)`; *notebook 5 of 5*; purpose;
   **Prerequisites:** NB 01–04; module 00 NB 07–08 (confusion/precision/recall, PR); chapter 01 (curse).
   **What you'll be able to do** (text→counts; fit MultinomialNB; evaluate honestly under imbalance;
   judge whether to trust NB's probabilities).
2. (code) imports + seed + style; `datasets.load_newsgroups` for the 4-cat train & test; class balance +
   one example document.
3. (md) **Why text, and why Naive Bayes** — KNN drowned in high dimensions (ch 01); NB counts, never
   measures a distance → high-dimensional sparse text is its home. We leave penguins for a real corpus.
4. (md) **From text to numbers, by hand** — the **bag of words**: a vocabulary, then per document a
   vector of word **counts** (order discarded).
5. (code) toy: 4 sentences → hand-built sorted vocabulary → a small **dense** count matrix.
6. (md) **Read the figure** — the **document-term matrix**: row=document, column=word, entry=count. NB 1's
   counting, now per word.
7. (md) **At scale: `CountVectorizer`** — builds vocabulary + counts for thousands of docs; fit on
   **train only** (NB 11 leakage rule), then transform test.
8. (code) `CountVectorizer(stop_words="english", min_df=2)` fit on train; vocabulary size, sparse shape,
   density.
9. (md) **Read the output** — 12 384 words, 0.43 % non-zero: huge & **sparse**. Sparsity is why counting
   scales where distances (KNN) collapse.
10. (md) **MultinomialNB on text** — the **count likelihood** (NB 1's idea; the model NB 4 named) with
    `alpha`; multi-class **argmax** = NB 1's rule extended to four classes.
11. (code) fit `MultinomialNB` (print fit time), predict, accuracy 0.887, macro-F1; `plot_confusion_matrix`.
12. (md) **Read the figure** — 0.887, trained in ms; religion the hardest; fast & strong on 12 k features,
    where KNN's distances failed.
13. (md) **Honest evaluation under imbalance** — one topic as rare positive (`sci.med` vs rest); accuracy
    can flatter, so use module-00 metrics.
14. (code) one-vs-rest sci.med: accuracy 0.930 with **majority baseline 0.724** beside it;
    precision/recall/F1; **PR curve** (AP 0.935).
15. (md) **Read the figure** — 0.930 looks strong until the baseline (0.724); P/R/F1 + PR are the honest
    picture (module 00 NB 07–08).
16. (md) **Can we trust the probabilities? Calibration** — NB 4's warning; of the cases NB calls "90 %",
    are ~90 % really positive?
17. (code) predicted-probability **histograms** MNB vs LogReg (pile-up 1205 vs 534); reliability diagram
    via **`viz.plot_calibration_curve`**; print both **Brier** scores.
18. (md) **Read the figure** — MNB crushes probabilities to 0/1 (over-confident *in shape*); here its
    Brier is fine (0.056 < LogReg 0.080) because the task is easy — so the takeaway is **not** "NB Brier
    worse" but **the number is not a calibrated probability**: trust the *ranking*, recalibrate for the
    number. (Hard task → it backfires; *Your turn*.)
19. (md) **The naive assumption, absurdly violated — and it still works** — words co-occur, so
    conditional independence is wildly false; huge sparse space where KNN died; yet NB fast & strong
    (Domingos–Pazzani at scale). **When to use / when not.**
20. (md) **Bridge to chapter 03 — generative vs discriminative** — NB models P(words∣class) then Bayes;
    logistic regression models P(class∣words) directly (Ng & Jordan 2001).
21. (md) **Your turn** — (a) most "sci.med" words from the fitted log-probs; (b) the confusable pair →
    MNB Brier 0.245 > LogReg 0.223 (over-confidence finally costs); (c) *harder* — TfidfVectorizer /
    ComplementNB: does the over-confidence soften?
22. (md) **What you built** — text→counts; MultinomialNB on 12 k features in ms; honest eval under
    imbalance; calibration (number vs ranking); generative vs discriminative. Vocabulary.
23. (md) **What you built across the chapter** — counts→Bayes (1) → naive assumption (2) → Gaussian +
    log-space (3) → estimators & dials (4) → text (5). You can build, tune, and honestly judge a Naive
    Bayes classifier — and know when to reach for it.
24. (md) **Going further** — TF-IDF & `ComplementNB` (Rennie 2003); `CalibratedClassifierCV`; n-grams;
    multinomial-vs-Bernoulli.
25. (md) **References** — Manning/Raghavan/Schütze 2008 ch.13 (DOI 10.1017/CBO9780511809071); Rennie et al.
    2003 ICML; McCallum & Nigam 1998; Ng & Jordan 2001 NeurIPS; Niculescu-Mizil & Caruana 2005 (DOI
    10.1145/1102351.1102430); ISLR §4.4.4 (DOI 10.1007/978-1-0716-1418-1). `Previous: 04` ·
    `Next: Module 03 — Logistic Regression` (chapter 02 complete).

## Honest scoping (stated in the notebook)

- Independence **wildly** violated by co-occurring words — stated — and NB still works (decision
  survives; probabilities do not).
- **Calibration framed honestly:** robust signal = the **pile-up at 0/1**, *not* "Brier always worse"
  (here MNB's Brier is lower); the cost is shown on the hard confusable pair (Your turn). Trust the
  ranking; recalibrate for probabilities.
- `CountVectorizer` **fit on train only**; 20newsgroups uses its own train/test split, fetched-and-cached
  with visible logging (offline after first fetch); no synthetic data.
- Accuracy under imbalance reported **beside its baseline**, never alone.
- TF-IDF / ComplementNB / n-grams / CalibratedClassifierCV **named**, not taught.

## Verification

Build via `uv run python - < /tmp/build_nb5.py`. Numbers reconciled (vocab 12384, density 0.0043,
fit ≈ms, acc 0.887/macroF1 0.882; one-vs-rest 0.930 vs baseline 0.724, P/R/F1 0.887/0.854/0.870, AP
0.935; MNB Brier 0.056 vs LogReg 0.080, pile-up 1205 vs 534). Runs top-to-bottom (nbconvert to /tmp;
**output-free**, `--clear-output --inplace`); **new `src/` code** (`load_newsgroups`,
`plot_calibration_curve`) with tests → `pytest` **16**; `check_no_hardcoded_hex` passes; `gen_llms_txt`
re-run; `ruff`/`black` clean; both reviewers PASS (no BLOCK); Rémy validates; commit
`feat(02_naive_bayes): notebook 05 — text classification`; merge → chapter; **chapter 02 closes via PR
into `main`** (`gh pr create … --base main`, merge `--no-ff`).
