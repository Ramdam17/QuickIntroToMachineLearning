# Notebook plan — 00_GettingStarted / 08_scores_thresholds_roc

> Status: **APPROVED** (2026-06-17, by Rémy). Notebook plans validated by Rémy alone; reviewers
> return on the built notebook. Drives `notebooks/00_GettingStarted/08_scores_thresholds_roc.ipynb`.

## Context

Last of the evaluation trilogy. NB 07 judged at one threshold; here we expose the dial: a classifier
emits a **score**, sliding the **threshold** trades precision↔recall, and **ROC/AUC** summarize
across all thresholds. Realizes the NB 05 score teaser. Prereqs: 05, 07.

## The score (pinned, verified)

`s(x) = d(x, μ_Adelie)² − d(x, μ_Gentoo)²` (positive = Gentoo); `s > 0` reproduces the nearest-centroid
label. Bill-only model (imperfect → real ROC): **AUC 0.989** (by hand = `roc_auc_score`). Operating
points: `s>−50` → P 0.79 / R 1.00; `s>0` → 0.97 / 0.94; `s>+50` → 1.00 / 0.61. Two-feature model:
**AUC 1.0** (contrast). Test: 31 Gentoo, 38 Adélie.

## Library additions (DONE, tested)

`viz.plot_roc_curve(y_true, scores, *, ax, label, color)` and
`viz.plot_score_threshold(scores, y_true, *, threshold, class_names, ax)` — charter-coloured, sklearn
imported lazily; tests added (pytest 12/12, ruff/black/hex clean). PR curve drawn inline.

## Cell-by-cell (~21 cells)

1 Header (Prereqs 05, 07; 4 objectives). 2 (code) imports; split; bill-only centroids; score fn; `y_true=(y_test=='Gentoo')`.
3 Recap (NB 07 = one threshold; find the dial). 4 From label to score (`s`, high=Gentoo; `s>0` = nearest centroid; NB 05 teaser).
5 (code) scores on test, a few sorted; confirm `s>0` matches NearestCentroid. 6 Read — the score ranks penguins; label = sign(s).
7 The threshold dial (raise → precision↑/recall↓; lower → recall↑/precision↓). 8 (code) `plot_score_threshold`; P/R at −50/0/+50.
9 Read fig+table — the two humps; trade-off numbers. 10 ROC md (sweep thresholds; TPR vs FPR).
11 (code) `plot_roc_curve(y_true, scores, label='bill only')`. 12 Read — AUC 0.989; AUC = P(random Gentoo scores above random Adélie); diagonal = chance.
13 PR curve md (precision vs recall; for rare positives). 14 (code) PR inline (`precision_recall_curve` + plt). 15 Read — precision holds until recall near 1.
16 Fixed vs swept (accuracy/P/R at one threshold; AUC across all = ranking quality). 17 (code) two-feature ROC overlaid (AUC 1.0).
18 Read — stronger ranker's curve dominates (1.0 vs 0.989). 19 Your turn (screening threshold; AUC 0.5 meaning; high AUC vs mediocre fixed-threshold accuracy).
20 What built + vocab (score, threshold, TPR/FPR, ROC, AUC, PR curve). 21 References (Fawcett 2006 DOI 10.1016/j.patrec.2005.10.010; Géron ch.3; sklearn). `Previous: 07` · `Next: 09`.

## Honest limits / no pre-emption

Bill-only keeps the ROC non-degenerate; two-feature contrast honest about AUC 1.0. Affine signed
score → by-hand AUC = `roc_auc_score` (rigor check); `NearestCentroid` has no `decision_function`, so
the by-hand score is the point. Probabilities/calibration NOT introduced (→ LogReg/NaiveBayes).

## Verification

New `viz` helpers tested; pytest green; ruff/black/hex clean. Runs top-to-bottom; outputs cleared;
gen_llms_txt re-run; by-hand AUC vs `roc_auc_score`; both reviewers pass; Rémy validates; commit + merge.
