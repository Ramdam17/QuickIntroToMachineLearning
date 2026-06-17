# Notebook plan — 00_GettingStarted / 02_features_and_feature_space

> Status: **APPROVED** (2026-06-16, by Rémy). Notebook plans validated by Rémy alone; reviewers
> return on the built notebook. Drives `notebooks/00_GettingStarted/02_features_and_feature_space.ipynb`.

## Context

Second notebook. NB 01 introduced features/label/example informally. NB 02 makes the **feature space**
precise and picks up the two geometric tools the rest of the module leans on: **the mean of a point
cloud** and the **Euclidean distance between two points** — exactly the primitives NB 05's
nearest-centroid classifier needs. One unifying concept: *data is a cloud of points in feature space,
and we can measure two things — where the middle is, and how far apart points are.* **No classifier
is built here** (that is NB 05). Prereqs: 01.

## What you'll be able to do

- Lay data out as a feature matrix `X` (n×d) and a label vector `y`, and name feature types.
- See each example as a **point** in feature space (and why we can only draw 2 features at a time).
- Compute the **mean (centroid)** of a set of points, by hand and per class.
- Compute the **Euclidean distance** between two points, by hand, and say what "closest" means.

## Reuse (no new library code)

`ml_course.datasets.load_penguins()` / `penguins_xy()`. pandas-first (`X.shape`, `X.mean()`,
`X.groupby(y).mean()`); numpy under the hood for distance. Plain matplotlib under
`viz.use_course_style()`, colours from `ml_course.colors`.

## Cell-by-cell (~20 cells)

1. Header (`# 02 — …`; purpose; `Prerequisites: 01`; 4 objectives; welcome).
2. Imports & setup (seed; `colors, datasets, viz`; `use_course_style()`; `FEATURES`, `SPECIES_ORDER`).
3. Recap / footing — features/label/example from NB 01; now feature space + two tools.
4. **X and y** — feature matrix (rows=examples, cols=features) + label vector.
5. (code) `penguins_xy()`; shapes; `X.head()`.
6. Read the output — `X` (274,2), `y` (274,); design matrix, n×d, target vector.
7. **Feature types** — numeric / categorical / ordinal; ours numeric → can go on axes.
8. **A point in feature space** — coordinates = feature values; 2 features → drawable; more → can't see all at once.
9. (code) scatter with one penguin singled out and its coordinates annotated.
10. Read the figure — row ↔ point; same scatter as NB 01, read as geometry.
11. **The mean of a point cloud** — average position per feature; the centroid.
12. (code) overall mean by hand (`X.mean()`); per-species means (`groupby`); plot cloud + class-mean points.
13. Read the figure — each mean in the middle; class means clearly apart. *Teaser: NB 05 turns "closest mean" into a classifier.*
14. **Distance between two points** — Euclidean / Pythagoras in 2-D; extends to more features.
15. (code) two penguins; distance by hand vs `np.linalg.norm`; draw the segment + length.
16. Read the figure — segment + number; honest caveat: distance mixes bill-mm and flipper-mm → **scale** matters (NB 11 resolves; not fixed here).
17. Vocabulary update — `X`, `y`, n_samples/n_features, feature types, feature space, mean/centroid, Euclidean distance.
18. Your turn — (a) shapes of `X`/`y`; (b) mean of three points by hand; (c) distance between two penguins, which of two is closer to a third. (No classifier.)
19. What you built — `X`/`y`, points, centroid, distance = the toolkit NB 05 assembles.
20. References — Géron ch. 2; ISLR §2.1. `Previous: 01` · `Next: 03`.

## Honest limits / no pre-emption

- Mean & distance are **tools only** — no classification yet (NB 05). Per-class means shown as a teaser.
- Euclidean distance is scale-dependent; flagged once (NB 11), not fixed.
- "Can't see >~3 features at once" motivates later care without naming the curse of dimensionality (01_KNN).

## Verification

`use_course_style()`; colours from `ml_course.colors`; seed; "Read the figure" after cells 9/12/15;
English; banned words avoided; pandas-first. At commit: runs top-to-bottom; outputs cleared;
`check_no_hardcoded_hex.py` passes; `pytest` green; `gen_llms_txt.py` re-run; both reviewers pass;
Rémy validates visually; commit + merge into `chapter/00_GettingStarted`.
