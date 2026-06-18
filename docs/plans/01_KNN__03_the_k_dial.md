# Notebook plan — 01_KNN / 03_the_k_dial

> Status: **APPROVED** (2026-06-17, by Rémy). Notebook plans validated by Rémy alone; reviewers return
> on the built notebook. Drives `notebooks/01_KNN/03_the_k_dial.ipynb`.

## Context

NB 3 of the k-NN chapter, and the last of the three fundamentals notebooks. NB 1 gave the vote and
*k*; NB 2 fixed the yardstick (scale). This notebook turns **k itself into the object of study**: k is
the **bias–variance knob**. Small k (k=1) follows every point — it *memorizes*, jagged and noise-prone
(overfit). Large k over-smooths, ignoring real structure (underfit). The honest way to choose k is the
one module 00 already taught: **cross-validation on the training data** (NB 10), then a single, sealed
test evaluation. This notebook is where k-NN ties together NB 09 (over/under-fitting, the
generalization gap) and NB 10 (CV for model selection). One concept: **k is the bias–variance dial,
chosen by CV.** Prereqs: NB 1–2, plus 05 (decision boundary), 09 (over/under-fitting), 10 (CV).

## Design (measured — `make_moons(300, 0.30, 0)`, stratified 70/30 → 210/90, by-hand k-NN, original comparably-scaled features)

- **Train vs test error across k** (error = 1 − accuracy):

  | k | train err | test err |
  |---|---|---|
  | 1 | **0.000** | 0.067 |
  | 3 | 0.071 | 0.033 |
  | 5 | 0.071 | 0.044 |
  | 15 | 0.062 | 0.044 |
  | 25 | 0.081 | 0.078 |
  | 51 | 0.100 | 0.122 |
  | 151 | 0.224 | 0.322 |
  | 201 | 0.276 | 0.356 |

  k=1 fits training **perfectly** (err 0) yet errs 0.067 on test — the gap *is* overfitting (NB 09).
  Train error rises with k (less flexible); test error is a shallow U (best near k=3–15) then climbs
  (underfit).
- **5-fold CV on the TRAIN set** (by hand, `StratifiedKFold`, shuffle, seed 0): cv accuracy peaks at
  **k = 15** (0.919). Sequence: k=1 0.886, k=3 0.910, k=5 0.900, k=15 **0.919**, k=25 0.900, k=51
  0.890, k=151 0.743.
- **Honest evaluation:** the CV-chosen **k = 15** scores **0.956** on the sealed test set — vs **0.933**
  at k=1 (overfit) and **0.678** at k=151 (underfit). Note to state plainly: the *test-error* curve's
  own minimum is k=3, but we may **not** choose k by looking at the test set; CV picks 15 from training
  data alone, and we accept the one test number. (CV is an estimate — the small 3-vs-15 gap is exactly
  why a single split is not trusted.)
- **Decision boundaries at k = 1 / 15 / 151** (`viz.plot_decision_boundary` + the NB 2 `ByHandKNN`):
  k=1 jagged with single-point islands (memorizing noise), k=15 a smooth two-crescent split, k=151
  over-smooth. Contrast k-NN's **local** boundary with NB 05's single straight bisector (one global
  rule) — k-NN's inductive bias, with k setting how local.

## Library additions / figures

**None to `src/`.** Reuse `viz.plot_decision_boundary` (boundaries vs k, with the NB 2 `ByHandKNN`
wrapper, redefined in-notebook) and `viz.plot_train_test_curve` (train/test error vs k; and again for
CV accuracy vs k). **CV is done by hand** with `StratifiedKFold` (the splitting utility from NB 10) +
a short loop over folds — `KNeighborsClassifier` and `cross_val_score` proper stay in NB 4 (the
by-hand `ByHandKNN` is not a clone-able sklearn estimator, and NB 3 keeps the by-hand spirit). The
original (comparably-scaled) moons are used — standardization was NB 2's lesson and is noted, not
re-shown. `pytest` stays 14.

## Cell-by-cell (~19 cells; intuition → by-hand → "Read the figure")

1. (md) **Header** — `# 03 — The k dial`; *Module 01 · k-Nearest Neighbours — notebook 3 of 6*;
   purpose; `Prerequisites: 01–02, plus 05 (decision boundary), 09 (over/under-fitting), 10 (CV)`;
   objectives (see k as the bias–variance knob; recognize k=1 as memorizing/overfit and large k as
   underfit; diagnose with train/test error; **choose k by cross-validation on the training data**;
   evaluate the choice once on test); warm welcome closing the fundamentals trio.
2. (code) imports (numpy, matplotlib, `make_moons`, `train_test_split`, `StratifiedKFold`; `ml_course`
   viz + colors) + seed + `use_course_style()` + data + stratified 70/30 split + by-hand `knn_predict`
   and the `ByHandKNN` wrapper (from NB 2). Print sizes.
3. (md) **Recap & footing** — from NB 1: the vote and *k*; from NB 2: distance (we use the original,
   comparably-scaled moons — standardize when scales differ); from NB 09: over/under-fitting and the
   generalization gap; from NB 10: cross-validation. The question this notebook answers: **what k?**
4. (md) **k is a dial** — small k = flexible, local (follows each point); large k = smooth, global
   (averages many neighbours). Intuition before numbers; we will *see* it, then *measure* it.
5. (code) **boundaries at k = 1, 15, 151** — three panels via `plot_decision_boundary` + `ByHandKNN`.
6. (md) **Read the figure** — k=1 jagged, with little islands around single points (it memorizes every
   point, noise and all); k=15 a smooth split following the two crescents; k=151 over-smooth, washing
   out the real shape. Contrast NB 05's nearest-centroid: one straight bisector, a single global rule —
   k-NN's boundary is **local**, and k sets how local. This is k-NN's inductive bias.
7. (md) **Does k=1's perfect memory mean it is best?** — k=1 gets every *training* point right by
   construction (its nearest neighbour is itself). NB 09 warned: training accuracy alone is misleading.
   Let us measure train *and* test error as we turn the dial.
8. (code) sweep **odd** k (1…201) → train error and test error; `plot_train_test_curve`. (Odd k avoids
   2–2 ties in binary voting; the tie-break detail is NB 4.)
9. (md) **Read the figure** — train error climbs from 0 at k=1 (perfect memorization) as the model
   stiffens; test error is a shallow U — worst at the extremes, best in the middle. The **gap** at k=1
   (train 0.000 vs test 0.067) is the overfitting signature from NB 09; the high-k end is underfitting.
10. (md) **But that used the test set** — drawing the test curve is a teaching luxury. In a real
    project you never watch test error move; you choose k on the **training data** by cross-validation
    (NB 10), then look at test once. Let us do that honestly.
11. (code) **by-hand 5-fold CV on the train set** (`StratifiedKFold`, shuffle, seed) → cv accuracy per
    odd k; a CV-accuracy-vs-k curve; take the argmax → **k = 15**.
12. (md) **Read the figure** — CV accuracy (training data only) peaks at **k = 15**: our principled
    choice, made without the test set. (The test-error curve's own minimum was k=3; CV picked 15 — CV
    is an estimate, and that small difference is precisely why one split is not trusted. We commit to
    the CV choice and do not go back to peek at test.)
13. (code) **evaluate k = 15 once on the sealed test set** → 0.956; print alongside k=1 (0.933) and
    k=151 (0.678) for contrast.
14. (md) **Read the output** — the CV-chosen k=15 scores 0.956 on data it never saw, beating the
    overfit k=1 and crushing the underfit k=151. The honest workflow: **CV chooses, the test set
    confirms — once.**
15. (md) **The bias–variance trade-off** (name it) — small k = low bias, high variance (sways with
    noise); large k = high bias, low variance (over-smooth). k is the knob between the two. (Ties to NB
    09's bias–variance; the same trade-off every method has, here controlled by k.)
16. (md) **Your turn** — (a) k=1 has zero training error; why is that *not* evidence it is the best
    model? (b) why choose k by CV on the training set rather than by the test score? (c) "I tried every
    k and k=3 had the best *test* accuracy, so I'll use k=3." What is the methodological problem?
17. (md) **What you built** + vocabulary — the k dial, bias–variance trade-off, overfitting/
    underfitting (recall), cross-validation for model selection (recall), hyperparameter (k), and the
    rule: do not choose k on the test set.
18. (md) **References** — ISLR §2.2.3 (k-NN), §5.1 (cross-validation), §2.2.2 (bias–variance), DOI
    10.1007/978-1-0716-1418-1; ESL §2.5 & §7.10 (k-NN; CV / model selection), DOI
    10.1007/978-0-387-84858-7. `Previous: 02 — Distance & the scale trap` · `Next: 04 — The estimator &
    its parameters`.

## Honest limits / no pre-emption

- One concept — **k as the bias–variance dial, chosen by CV**. The new idea is "k is a knob"; the
  diagnosis (over/under-fit) and the chooser (CV) are *recalled* from NB 09/10 and applied, not taught
  fresh. NB 3 does **not** introduce `KNeighborsClassifier` or tune weights/metric (NB 4), and does
  **not** study metric geometry (NB 6).
- CV is done **by hand** (`StratifiedKFold` + loop with the by-hand `ByHandKNN`); `cross_val_score`
  with a real estimator belongs to NB 4. Stated.
- The train/test error curve uses the test set **to illustrate** the bias–variance pattern (as NB 09
  did); the notebook is explicit that **selection** uses CV on train only, never the test score — and
  flags that CV's pick (15) need not equal the test-curve minimum (3).
- Odd k is used to avoid 2–2 ties in binary voting; the deterministic tie-break is named as NB 4's.
- The CV-best k and the numbers are dataset- and seed-specific (stated); the *pattern* (overfit small
  k → underfit large k, CV finds the middle) is the lesson.

## Verification

Measured anchors (train/test error table; CV-best k=15 at cv 0.919; test acc 0.933 / 0.956 / 0.678 at
k=1/15/151) re-run in the notebook and reconciled into prose at build. Runs top-to-bottom (nbconvert
to /tmp; output-free); `check_no_hardcoded_hex` passes; `gen_llms_txt` re-run; `pytest` green (14, no
`src/` change); both reviewers pass (no BLOCK); Rémy validates visually; commit + merge `notebook →
chapter`.
