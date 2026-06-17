# Notebook plan — 00_GettingStarted / 05_first_classifier_nearest_centroid

> Status: **APPROVED** (2026-06-16, by Rémy). Notebook plans validated by Rémy alone; reviewers
> return on the built notebook. Drives `notebooks/00_GettingStarted/05_first_classifier_nearest_centroid.ipynb`.

## Context

The centerpiece of module 00: the first real classifier, by hand, making `fit → predict → evaluate`
concrete. Fuses NB 02 (centroid = mean; Euclidean distance) and NB 04 (the stratified split). Prereqs:
02, 04.

## Framing guards

- This **is** "our first classifier". Boundary = perpendicular bisector of the two centroids (a line);
  state the inductive bias (equal isotropic spread) + a failure case + scale-sensitivity (→ NB 11).
- "Fraction right" informal; accuracy formalised NB 06. Compare to the majority baseline.
- Label-only here; the confidence **score** is a teaser for NB 08.

## Library addition (done, tested)

Extended `ml_course.viz.plot_decision_boundary` to be **pandas-first** and label-agnostic: accepts a
DataFrame `X` (column names → axis labels) or ndarray, and `y` int **or string** (colours follow
`sorted(unique(y))`); predictions mapped through the same code map. Backward compatible. New
`tests/test_viz.py` case (DataFrame + string labels → Figure with right axis labels). pytest 10/10,
ruff/black clean.

## By-hand classifier (in the notebook)

`NearestCentroidByHand` with `fit` (store class means) / `predict` (nearest centroid), checked against
`sklearn.neighbors.NearestCentroid` (identical predictions / score).

## Cell-by-cell (~21 cells)

1 Header (Prereqs 02, 04; 4 objectives). 2 Imports + same NB04 split. 3 Recap (02 + 04 → combine).
4 The idea (nearest centroid). 5 (code) centroids on train (`groupby`). 6 Read — centroids ARE the fit.
7 The predict rule. 8 (code) predict 2 test penguins by hand (distances + winner). 9 Read one example.
10 Wrap as estimator (fit/predict API). 11 (code) `NearestCentroidByHand`; fit on train.
12 (code) `plot_decision_boundary` + mark centroids. 13 Read — perpendicular bisector (a line).
14 The honest loop. 15 (code) predict test; fraction right vs baseline; sklearn agreement.
16 Read — 100% test vs 55% baseline; honest (sealed) but these species are unusually separable, so a
high held-out score here ≠ classification is always easy. 17 Assumes/breaks — isotropic-blob bias,
failure case, scale-sensitivity (→ NB 11). 18 Score teaser (→ NB 08). 19 Your turn (3, tiered).
20 What you built + vocabulary. 21 References (ESL §4.3 DOI 10.1007/978-0-387-84858-7; Tibshirani 2002
PNAS DOI 10.1073/pnas.082099299; sklearn NearestCentroid). `Previous: 04` · `Next: 06`.

## Verification

Numbers verified: centroids Adélie (39.0, 190.4) / Gentoo (47.4, 217.1); test 100% (by-hand =
sklearn, full agreement) vs baseline 55%. Runs top-to-bottom; outputs cleared; hex clean; pytest
10/10; gen_llms_txt re-run; both reviewers pass; Rémy validates visually; commit (NB05 + viz + test +
plan + STATE + llms.txt) + merge.
