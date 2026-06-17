# Notebook plan — 00_GettingStarted / 07_confusion_matrix_precision_recall

> Status: **APPROVED** (2026-06-17, by Rémy). Notebook plans validated by Rémy alone; reviewers
> return on the built notebook. Drives `notebooks/00_GettingStarted/07_confusion_matrix_precision_recall.ipynb`.

## Context

Second of the evaluation trilogy. NB 06 showed accuracy cannot say *which* errors a model makes.
Open the **confusion matrix** (TP/FP/FN/TN, positive = Gentoo) and build **precision**, **recall**,
**F1** — by hand then `sklearn.metrics` — and close on **asymmetric error costs**. Prereqs: 06.

## Design decision

The 2-feature centroid was perfect on the clean test → diagonal matrix teaches nothing. So classify
on **bill length alone** (NB 03: bill overlaps more than flipper) to make real, honest errors. Framed
as a deliberate teaching device, not the best model.

## Verified numbers (NearestCentroid on bill_length_mm, test, positive = Gentoo)

cm `[[37,1],[2,29]]` → TN 37, FP 1, FN 2, TP 29. Accuracy 0.957. Precision 0.967 (29/30). Recall
0.935 (29/31). F1 0.951.

## Reuse (no new library code)

`datasets.penguins_xy()`; NB 04 split; `NearestCentroid`; `confusion_matrix / precision_score /
recall_score / f1_score` with `pos_label='Gentoo'`; `viz.plot_confusion_matrix` (rows true, cols pred).

## Cell-by-cell (~21 cells)

1 Header (Prereqs 06; 4 objectives). 2 (code) imports; split; fit `NearestCentroid` on bill only.
3 Recap (NB 06 → need to see *which* errors). 4 Why bill-only (teaching device; NB 03 overlap).
5 (code) accuracy 0.957. 6 Read — a few errors now. 7 Confusion matrix md (TP/FP/FN/TN, positive = Gentoo).
8 (code) `confusion_matrix` + `plot_confusion_matrix`; print TN/FP/FN/TP. 9 Read — 37/1/2/29, two kinds of error.
10 Precision md (TP/(TP+FP), punishes false alarms). 11 (code) by hand 29/30 + `precision_score`. 12 Read — 0.967.
13 Recall md (TP/(TP+FN), punishes misses). 14 (code) by hand 29/31 + `recall_score`. 15 Read — 0.935; the tension.
16 F1 md (harmonic mean). 17 (code) by hand + `f1_score`. 18 Read F1 + asymmetric costs (spam→precision, cancer→recall).
19 Your turn (compute P/R from a cm; recall-vs-precision scenarios; flip positive class). 20 What built + vocab.
21 References (Géron ch. 3; sklearn metrics). `Previous: 06` · `Next: 08`.

## Honest limits / no pre-emption

Bill-only stated as a teaching device; thresholds/scores/ROC motivated but deferred to NB 08; small
but real counts (1 FP, 2 FN) — the point is the *kinds* of error.

## Verification

`use_course_style()`; colours from `ml_course.colors`; seed; "Read" after cells 5/8/11/14/17; English;
banned words avoided; by-hand P/R/F1 cross-checked vs sklearn. Runs top-to-bottom; outputs cleared;
hex clean; pytest green; gen_llms_txt re-run; both reviewers pass; Rémy validates; commit + merge.
