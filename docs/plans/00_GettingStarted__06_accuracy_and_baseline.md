# Notebook plan — 00_GettingStarted / 06_accuracy_and_baseline

> Status: **APPROVED** (2026-06-16, by Rémy). Notebook plans validated by Rémy alone; reviewers
> return on the built notebook. Drives `notebooks/00_GettingStarted/06_accuracy_and_baseline.ipynb`.

## Context

First of the evaluation trilogy (06 accuracy → 07 confusion/precision-recall → 08 scores/ROC). Name
and define **accuracy** (the "fraction right" of NB 04–05), measure it on the nearest-centroid
classifier, anchor it against the **majority baseline**, and expose its first limit — **class
imbalance** (the *accuracy paradox*). Fix **positive = Gentoo** for NB 07–08. Prereqs: 03, 04, 05.

## Framing guards

- Classifier reused via `sklearn.neighbors.NearestCentroid` (the NB 05 library twin) — focus stays
  on the metric.
- Imbalanced case is a labelled **what-if** built from the held-out test set (all Adélie + ~5%
  Gentoo) — no leakage, not the real evaluation.
- Confusion matrix / precision-recall motivated here, built NB 07; ROC/score NB 08.

## Verified numbers

Test: 38 Adélie / 31 Gentoo. Centroid accuracy **1.000**, majority baseline **0.551**. Imbalanced
what-if (40 = 38 Adélie + 2 Gentoo, 95% majority): Dummy **0.95** accuracy but **0/2** Gentoo found;
centroid **1.00**, **2/2** Gentoo found.

## Cell-by-cell (~20 cells)

1. Header (Prereqs 03, 04, 05; 4 objectives). 2. (code) imports; NB04 split; fit `NearestCentroid`.
3. Recap (NB05 → name the "fraction right"). 4. **Accuracy defined** (correct/total).
5. (code) test accuracy by hand `(pred==y_test).mean()` and `accuracy_score` — same (1.000).
6. Read — value + honest reminder (separable, NB05). 7. **Baseline** (most-frequent; must beat it).
8. (code) `DummyClassifier(most_frequent)` accuracy (0.551); bar classifier vs baseline (matplotlib, `model` vs `muted`).
9. Read figure — classifier well above baseline → learned something real.
10. **Where accuracy misleads** — equal weight per example → dominated by the majority.
11. (code) imbalanced what-if (all Adélie + ~5% Gentoo); `viz.plot_class_balance`.
12. (code) Dummy on it: 0.95 accuracy, 0/2 Gentoo found.
13. Read — **accuracy paradox**: 95% while finding zero of the rare class.
14. (code) centroid on the same set: 1.00, 2/2 Gentoo found.
15. Read — both "high accuracy", but only one finds the rare class; accuracy alone barely separates them.
16. **Positive class** — the class we care about (disease/fraud/rare species); fix **positive = Gentoo** for NB 07–08.
17. So what — accuracy can't show *which* errors; NB 07 opens the confusion matrix.
18. Vocabulary — accuracy, baseline, DummyClassifier, class imbalance, accuracy paradox, positive class.
19. Your turn — (a) accuracy from counts; (b) 99/1 split → majority accuracy & why misleading; (c) a real rare-positive problem.
20. What you built + References — Géron ch. 3; ISLR ch. 4. `Previous: 05` · `Next: 07`.

## Verification

`use_course_style()`; colours from `ml_course.colors`; seed; "Read" after every output; English;
banned words avoided; pandas-first. Runs top-to-bottom; outputs cleared; hex clean; pytest green;
gen_llms_txt re-run; both reviewers pass; Rémy validates; commit + merge.
