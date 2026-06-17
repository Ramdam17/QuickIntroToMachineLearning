# Notebook plan — 00_GettingStarted / 03_look_before_you_model

> Status: **APPROVED** (2026-06-16, by Rémy). Notebook plans validated by Rémy alone; reviewers
> return on the built notebook. Drives `notebooks/00_GettingStarted/03_look_before_you_model.ipynb`.

## Context

The EDA notebook the chapter-plan reviewers asked for. Before any model, **look at the data**: class
balance, per-feature distributions, feature ranges/scales. Pre-establishes the *imbalance* picture
(NB 06) and the *scale* picture (NB 11). One concept: *inspect a dataset before modelling it.*
Prereqs: 01, 02. No split, no model, no metric computed here.

## Library additions (built — done)

`ml_course.viz.plot_class_balance(y, ax=None)` and `plot_feature_histograms(df, features, by=None,
*, bins=20)` — colours from `ml_course.colors`, return a `Figure`; smoke tests added in
`tests/test_viz.py` (pytest 9/9, ruff/black clean).

## Cell-by-cell (~20 cells)

1. Header (`# 03 — Look before you model`; purpose; `Prerequisites: 01, 02`; 4 objectives; welcome).
2. (code) Imports & setup — seed; `colors, datasets, viz`; `use_course_style()`; `df`, `X`, `y`; `FEATURES`, `SPECIES_ORDER`.
3. Recap / footing — NB 01–02 gave X/y, points, mean, distance; before modelling, we look.
4. **Why look first** — imbalance / scales / outliers / mistakes; garbage in, garbage out; three questions.
5. **Class balance** — what it is, why it matters (lazy majority predictor; exploited NB 06).
6. (code) `y.value_counts()`; `viz.plot_class_balance(y)`.
7. Read the figure — 151 / 123, roughly balanced; what 95/5 would do to "accuracy" (→ NB 06).
8. **Distributions** — histogram shows shape/spread; split by class shows separation.
9. (code) `viz.plot_feature_histograms(df, FEATURES, by="species")`.
10. Read the figure — flipper separates better than bill; overlap zones; tie to the scatter.
11. **Ranges & scales** — `describe()` per feature.
12. (code) `df[FEATURES].describe()`.
13. Read the output — bill ≈ 32–60, flipper ≈ 172–231; wider flipper range; → NB 02 distance caveat, → NB 11.
14. **Re-read the scatter** — with all that in mind, predict where a simple rule struggles.
15. (code) the 2-D scatter coloured by species.
16. Read the figure — separation mostly along flipper; thin overlap band; what a model contends with. No model yet.
17. Vocabulary — EDA, distribution, histogram, class balance/imbalance, range, scale, outlier.
18. Your turn — (a) balanced? what would 95/5 do to majority-accuracy; (b) which feature separates better; (c) which feature has larger spread and why it matters for distance (recall NB 02).
19. What you built — the EDA habit + two reusable plots; balance, shapes, scales known before modelling.
20. References — Géron ch. 2; Tukey, *Exploratory Data Analysis* (1977). `Previous: 02` · `Next: 04`.

## Honest limits / no pre-emption

Describe only — no split/model/metric (NB 04–06). Imbalance & majority baseline previewed in words.
Scale named, deferred to NB 11. Outliers named, not removed (out of scope for 00).

## Verification

`use_course_style()`; colours from `ml_course.colors`; seed; "Read the figure" after cells 6/9/15;
English; banned words avoided; pandas-first. At commit: `pytest` green; runs top-to-bottom; outputs
cleared; `check_no_hardcoded_hex.py` passes; `gen_llms_txt.py` re-run; both reviewers pass; Rémy
validates visually; commit + merge into `chapter/00_GettingStarted`.
