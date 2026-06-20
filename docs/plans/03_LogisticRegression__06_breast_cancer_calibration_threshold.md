# Notebook plan — 03_LogisticRegression / 06_breast_cancer_calibration_threshold

> Status: **APPROVED** (2026-06-20, by Rémy — validated alone; reviewers gate the *built* notebook).
> Build via `uv run python - < <scratchpad>/build_nb6.py` (stdin). **Final notebook; after it, CHAPTER 03
> closes via PR into `main`.**

## Context

NB **6 of 6** — the chapter's **demanding practical case and capstone**: logistic regression on
**breast-cancer diagnosis** (569 patients × 30 measurements), the full honest workflow where the
chapter's two payoffs *bite* — **calibrated probabilities** and **interpretable weights** — and where a
**threshold is a real medical decision**. The same dataset KNN met in chapter 01 NB 5 ("you felt the
curse with distances; now watch a linear model read it and report trustworthy probabilities"). It closes
the **generative-vs-discriminative loop** chapter 02 opened (LogReg vs GaussianNB calibration). Capstone
rule: **visualization-first**, ~20 cells a floor not a ceiling.

## Library addition (the chapter's only `src/` change)

- **`datasets.load_breast_cancer()`** — a thin **pandas-first** wrapper over
  `sklearn.datasets.load_breast_cancer`, returning a tidy `DataFrame` **(569, 31)**: the **30 named
  feature columns + a `target` column**, with INFO logging (mirrors `load_penguins_full`'s shape/logging;
  no download — sklearn-bundled). Docstring states sklearn's convention plainly (**target 0 = malignant,
  1 = benign**) so the notebook can flip to *malignant = positive* with eyes open.
- **Test** in `tests/test_datasets.py`: schema (shape (569, 31), the 30 expected feature names present +
  `target`, target values ⊆ {0, 1}, no NaN). **`pytest` 16 → 17.**

## Datasets & measured anchors (breast cancer; sklearn 1.9.0; re-measured at build)

**Positive class = MALIGNANT** (the costly miss). sklearn target is 0 = malignant / 1 = benign, so the
notebook sets `y = (target == 0)`. 569 patients, 30 features, **malignant 212 (37 %) / benign 357**. One
standardized `Pipeline` for both estimators; **70/30 split, `random_state=0`, stratified** (train 398 /
test 171, test malignant 64); CV = `StratifiedKFold(5, shuffle=True, random_state=0)`.

> **Several anchors are re-measured and differ from the chapter plan's preliminary figures** (flagged to
> Rémy). The qualitative stories are intact; the measured values are used.

- **CV accuracy (full data, std pipeline):** LogReg **0.979** vs GaussianNB **0.930** (chapter plan
  0.979/0.930 — matches).
- **Held-out test:** LogReg acc **0.953**; GaussianNB acc **0.895**.
- **Calibration (test, positive = malignant):** LogReg **Brier 0.033** vs GaussianNB **Brier 0.098**
  (≈ 3× better; chapter plan said 0.027/0.088 — re-measured 0.033/0.098). **Pile-up** of predicted
  probabilities past 0.99 / below 0.01: LogReg **119/171** vs GaussianNB **166/171** (GaussianNB
  over-confident — the ch 02 story). Honest nuance: near-separable data still leaves LogReg fairly
  confident (119/171) — *better*-calibrated, not perfectly.
- **Threshold (LogReg, test):** at the default **0.5**, malignant recall **0.938** (**4 of 64 cancers
  missed**, 4 false alarms); the recall plateaus at 0.953 (3 missed) over 0.4–0.2; to miss **only 1**
  (recall **0.984**) the threshold drops to **≈ 0.1**, at **14 false alarms** (precision ~0.82). (Chapter
  plan said 0.5→3-missed, 0.3→2-missed; re-measured to 0.5→4, 0.1→1 — the *cost-asymmetric* lesson is the
  same and stronger.)
- **L1 feature selection** (`l1_ratio=1`, `saga`, std train, high `max_iter`): nonzero **3 / 10 / 14** of
  30 at C = **0.02 / 0.2 / 1.0** (chapter plan 3/8/14 — re-measured middle 10).
- **Coefficient story** (std, L2, C=1): the largest |coef| push toward malignant — **radius error
  (+1.26), worst radius (+1.03), mean concave points (+0.94), worst symmetry (+0.94)** (compactness error
  −0.93) — bigger, more irregular tumours read malignant (clinically sensible).

## Library / figures

Reuse, all in `ml_course.viz`: **`plot_class_balance`**, **`plot_calibration_curve`** (overlay LogReg vs
GaussianNB via shared `ax` + `color` + `label`), **`plot_score_threshold`** (`scores`=P(malignant),
`threshold`, `class_names=("benign","malignant")`), **`plot_confusion_matrix`**, **`plot_roc_curve`** /
PR. One-off in-notebook (charter colours) for the coefficient-story bar and the L1-count / recall-vs-
threshold line. **Sklearn**: `LogisticRegression`, `GaussianNB`, `StandardScaler`, `Pipeline`,
`StratifiedKFold`, `cross_val_score`, `train_test_split`, `brier_score_loss`, `confusion_matrix`,
`recall_score`/`precision_score`. **`src/` grows by `load_breast_cancer` (+test) → `pytest` 17.**

- **Fig A — class balance** (`plot_class_balance`): malignant 37 % / benign 63 %.
- **Fig B — calibration / reliability** (`plot_calibration_curve` overlay): LogReg near the diagonal vs
  GaussianNB off it; Brier in the legend.
- **Fig C — score histogram + threshold** (`plot_score_threshold`) and a **recall/precision-vs-threshold**
  line beside it.
- **Fig D — confusion at 0.5 vs at 0.1** (`plot_confusion_matrix` ×2): 4 missed → 1 missed.
- **Fig E — coefficient story + L1 path**: top-|coef| bar (signed) and nonzero-count vs C.

## Cell-by-cell (~24 cells; visualization-first; "Read the figure" after every figure)

1. (md) **Header** — `# 06 — A demanding case: breast cancer`; *notebook 6 of 6*; warm welcome; this is
   where it all comes together. **Prerequisites:** NB 1–5 (the whole method); module 00 — confusion /
   precision / recall (NB 07), score → threshold & ROC/PR (NB 08), cross-validation (NB 10),
   Pipeline / fit-on-train (NB 11); chapter 01 NB 5 (KNN on this same dataset); chapter 02 NB 5
   (calibration, the reliability diagram, the Brier score, NB's over-confidence). **What you'll be able to
   do:** run an honest LogReg workflow end-to-end; read a reliability diagram and trust (or distrust) a
   probability; choose a decision threshold under asymmetric cost; read the coefficients and let L1 select
   features.
2. (code) **Imports + seed + style + data** — sklearn pieces + `ml_course`; `df = datasets.load_breast_cancer()`
   (the new wrapper); `X = df.drop(columns="target")`, **`y = (df["target"] == 0).astype(int)`** with a
   comment: *malignant = 1 = positive (the costly miss)*. Print shape and class counts.
3. (md) **Where we are & the stakes** — 569 patients, 30 measurements, **malignant 212 / benign 357**. The
   same dataset KNN met in chapter 01 (it felt the curse of dimensionality); a linear model will read it
   and, crucially, report **probabilities we can check**. Two decisions matter here that did not on
   penguins: **is the probability trustworthy?** and **where do we put the threshold**, when a missed
   cancer is far worse than a false alarm? Standardize in a `Pipeline`, fit on train only.
4. (code) **Fig A — class balance** (`plot_class_balance`); a quick `df.describe()` glance at a couple of
   features.
5. (md) **Read the figure (A)** — 37 % malignant: not balanced, so **accuracy alone can mislead** (a
   "benign-always" baseline scores 63 %). We will watch **recall on malignant** and **calibration**, not
   accuracy alone.
6. (md) **Intuition — the honest workflow** — split once; **cross-validate on train** to compare LogReg vs
   GaussianNB; pick the winner; read the **sealed test** once (module 00 NB 10). Both inside one
   standardized `Pipeline`, fit on train only (no leakage).
7. (code) **CV + fit** — `cross_val_score` (pinned CV) for `Pipeline(StandardScaler, LogReg)` vs
   `Pipeline(StandardScaler, GaussianNB)` → **0.979 vs 0.930**; then `train_test_split` (70/30, seed 0,
   stratified), fit LogReg on train, print held-out accuracy (**0.953**).
8. (md) **Read the result** — LogReg beats GaussianNB by ~5 CV points; held-out accuracy 0.953. But
   accuracy is not the whole story on a medical problem — next, do we trust the probabilities?
9. (md) **Intuition — calibration (the chapter-02 loop closes)** — a calibrated model's "0.9" means
   "right about 90 % of such times". Chapter 02 found naive Bayes **over-confident** (probabilities piled
   at 0 and 1). Discriminative logistic regression models P(y∣x) directly — let us compare them honestly,
   both standardized, on the same test set: reliability diagram + Brier score.
10. (code) **Fig B — reliability diagram** (`plot_calibration_curve` overlay LogReg vs GaussianNB on one
    `ax`); print **Brier 0.033 vs 0.098** and the **pile-up 119 vs 166 / 171**.
11. (md) **Read the figure (B)** — LogReg hugs the diagonal (a 0.9 really is ~0.9); GaussianNB swings off
    it and piles 166 of 171 predictions past 0.99/0.01 — **over-confident**, exactly as chapter 02 warned.
    Brier ≈ 3× better for LogReg. Honest nuance: near-separable data still leaves LogReg fairly confident
    (119/171) — **better-calibrated, not perfect**; recalibrate (`CalibratedClassifierCV`) if the number
    must be exact.
12. (md) **Intuition — the threshold is a policy, not a default** — predicting "malignant" when P ≥ 0.5 is
    one choice. A **missed cancer (false negative) is far costlier than a false alarm**, so we should be
    willing to flag more aggressively — lower the threshold — and accept more false alarms. The
    probability being calibrated is what makes this a *defensible* policy (module 00 NB 08, made real).
13. (code) **Fig C — score histogram + threshold** (`plot_score_threshold`, `scores = P(malignant)`,
    `class_names=("benign","malignant")`) beside a **recall & precision vs threshold** line.
14. (md) **Read the figure (C)** — the two classes' probability scores overlap in a middle band; the
    threshold cuts there. Sliding it left (lower) catches more malignant (recall up) at the cost of more
    benign flagged (precision down) — the trade the curve makes explicit.
15. (code) **Fig D — confusion at two thresholds** (`plot_confusion_matrix` ×2): **0.5** (recall 0.938,
    **4 of 64 missed**, 4 false alarms) vs **0.1** (recall 0.984, **1 missed**, 14 false alarms).
16. (md) **Read the figure (D)** — at the default 0.5 the model misses **4 cancers**; to miss only **1**,
    drop the threshold to ≈ 0.1 — now **14 benign patients are flagged** for follow-up. Which is right is
    a *clinical* choice about the cost ratio, and a calibrated probability is what lets you make it
    honestly rather than guess.
17. (md) **Intuition — reading the model** — logistic regression is **interpretable**: each standardized
    coefficient is a malignant-vs-benign log-odds contribution; and **L1** can prune the 30 features to a
    handful (automatic selection), a chance to see *which* measurements carry the signal.
18. (code) **Fig E — coefficient story + L1 path** — top-|coef| signed bar (radius error, worst radius,
    mean concave points, …); and nonzero-count vs C for L1 (`l1_ratio=1`, `saga`): **3 / 10 / 14** of 30
    at C = 0.02 / 0.2 / 1.0.
19. (md) **Read the figure (E)** — the strongest positive weights are size/irregularity measurements
    (larger, more concave tumours read malignant) — clinically sensible, and the kind of story KNN and
    naive Bayes could not tell. L1 shrinks the model to 14, then 3, features as the penalty strengthens —
    automatic selection.
20. (md) **Error analysis & honest limits** — the misses sit in the **overlap band** (P ≈ ½), where the
    model is appropriately unsure; accuracy hid them, **recall surfaced them**. Logistic regression draws
    a **linear** boundary: where the truth is curved it **underfits** — pointing to **decision trees**
    (chapter 04) and, far later, neural networks (the sigmoid neuron *is* a logistic unit; NB 4's gradient
    descent *is* how they train). **When to use:** a linear-ish boundary, interpretable weights,
    trustworthy probabilities, a strong baseline. **When not:** strongly non-linear structure.
21. (md) **Your turn** (3 tiered) — *easy*: from the recall/precision-vs-threshold curve, pick a threshold
    that misses at most 1 cancer and state the false-alarm cost; *medium*: read the top five
    malignant-driving coefficients and say what they measure; *harder*: wrap the model in
    `CalibratedClassifierCV` (isotonic or sigmoid) and check whether the Brier score improves.
22. (md) **What you built** — a full honest workflow on real diagnostic data: CV model choice, a sealed
    test, **calibration** (LogReg vs GaussianNB; the ch 02 loop closed), a **cost-asymmetric threshold**,
    and a **readable, L1-selectable** model. **Vocabulary box:** calibration · reliability diagram ·
    Brier score · decision threshold · false negative / recall · feature selection (L1) · cost asymmetry.
23. (md) **Chapter wrap — logistic regression, end to end** — from a score (NB 1) to a fitted, calibrated,
    interpretable classifier with a chosen threshold (NB 6). The first **discriminative** method, the first
    trained by **iterative optimization** — the engine the back half of the course is built on. Forward:
    **Module 04 — Decision Trees** (non-linear partitions where the linear boundary underfits).
24. (md) **References** — Cox 1958 (DOI 10.1111/j.2517-6161.1958.tb00292.x); Niculescu-Mizil & Caruana
    (2005), calibration, DOI 10.1145/1102351.1102430; Platt (1999), probabilistic outputs; ESL §4.4 (DOI
    10.1007/978-0-387-84858-7); ISLR §4.3 (DOI 10.1007/978-1-0716-1418-1); Street/Wolberg/Mangasarian
    (1993), the WDBC dataset, DOI 10.1117/12.148698. `Previous: 05 — The estimator & its parameters` ·
    `Next: Module 04 — Decision Trees`.

## Honest scoping (stated in the notebook)

- **Positive = malignant** stated up front; sklearn's 0=malignant convention flagged where `y` is built.
- **Numbers are re-measured** on the pinned split/CV (they differ slightly from the chapter plan's
  preliminary figures — the prose uses the measured values; nothing asserted that isn't shown).
- **Calibration is better, not perfect** — LogReg 119/171 still piled near the extremes on near-separable
  data; `CalibratedClassifierCV` named for when the number must be exact.
- **The threshold is a clinical policy** under an asymmetric cost, not a tuned hyperparameter — chosen for
  a stated recall target, with the false-alarm cost made explicit.
- **Linear underfits curved truth** — named, fix pointed to (trees ch 04), not taught here.
- No leakage: split → CV on train → one sealed test; standardization inside the `Pipeline`.

## Verification

Build via `uv run python - < <scratchpad>/build_nb6.py` (stdin); add `datasets.load_breast_cancer()` +
`tests/test_datasets.py` case (**pytest 16 → 17**). Re-measure at build: CV 0.979/0.930; test acc
0.953/0.895; Brier 0.033/0.098, pile-up 119/166; threshold 0.5 (4 missed) / 0.1 (1 missed, 14 FP); L1
3/10/14; top coefs. Runs top-to-bottom (nbconvert to scratchpad; tracked file **output-free**,
`--clear-output --inplace`); **banned-word scan over the JSON real text** = 0; `check_no_hardcoded_hex`
passes; `gen_llms_txt` re-run; `ruff`/`black` clean (incl. the new `src/`+test). Both reviewers PASS (no
BLOCK); Rémy validates visually; commit `feat(03_logistic_regression): notebook 06 — a demanding case:
breast cancer`; merge `notebook → chapter`; **then close CHAPTER 03 via PR into `main`** (`--no-ff`
preserved per-notebook history).
