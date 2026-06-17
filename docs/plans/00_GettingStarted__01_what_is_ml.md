# Notebook plan — 00_GettingStarted / 01_what_is_ml

> Status: **APPROVED** (2026-06-16, by Rémy). Notebook plans are validated by Rémy alone; reviewers
> return on the built notebook. Drives the build of `notebooks/00_GettingStarted/01_what_is_ml.ipynb`.

## Context

First notebook of the course. Make "machine learning" concrete and non-intimidating — learning a
*rule from examples* — and set up the through-line (penguins, the toolkit, the "read the figure"
habit). One concept: **what ML is** (supervised; classification vs regression). By hand: read the
data, name features vs label, read the feature-space scatter, predict by eye. Prereqs: none.

## Course-wide convention (recorded in CLAUDE.md + AGENTS.md)

**Pandas-first**: DataFrame/Series as the default data interface, numpy under the hood for numeric
kernels. `load_penguins()` returns a DataFrame; `viz` helpers gain DataFrame support just-in-time.

## What you'll be able to do

- Explain "learning a rule from examples" in your own words.
- Identify features and the label; one row = one example.
- Tell classification from regression.
- Read the course's feature-space scatter.

## Data / library (built — done)

`ml_course.datasets.load_penguins()` → DataFrame (274 rows: 151 Adélie, 123 Gentoo; columns
`bill_length_mm`, `flipper_length_mm`, `species`), offline from a vendored CSV; `penguins_xy()`
splits into `X`/`y`. Produced by `scripts/vendor_penguins.py`; `.gitignore` exception added; tests in
`tests/test_datasets.py` (green).

## Cell-by-cell (~18 cells, per docs/notebook_template.md)

1. (md) Header — `# 01 — What is machine learning?`; purpose; `Prerequisites: none`; objectives (4); warm welcome.
2. (code) Imports & setup — `np.random.seed(0)`; `from ml_course import viz, colors, datasets`; `viz.use_course_style()`; version echo (python/numpy/pandas/sklearn/matplotlib) — printed.
3. (md) Intuition: code the rule vs learn the rule.
4. (md) What ML is — supervised; model maps features → label. Re-established plainly.
5. (code) Meet the data — `load_penguins()`; `df.head()`; `df["species"].value_counts()`.
6. (md) Read the table — features vs label; row = example. Define feature/label/example.
7. (md) Classification vs regression — category → classification; number → regression; course = classification.
8. (code) The feature space — scatter bill vs flipper, coloured by species; axis labels from column names.
9. (md) Read the figure — axes, point, colours; mostly-separate clouds with some overlap.
10. (md) "Learning a rule" made tangible — where would you draw the line? Built for real in NB 05.
11. (code) Predict by eye — one unlabelled point (highlight colour) on the scatter.
12. (md) Read the figure — the point sits in one cloud → our guess = prediction.
13. (md) ML vocabulary box — feature, label, example, supervised, classification, regression, prediction.
14. (md) Your turn — (a) name features/label + count; (b) classify 3 tasks as classification/regression; (c) by eye, a separating rule + one penguin it misclassifies.
15. (md) What you built — celebrate the takeaways.
16. (md) Going further (optional) — supervised vs unsupervised vs reinforcement; this course is supervised.
17. (md) References — Géron ch.1; ISLR §2.1; penguins (Gorman 2014, DOI 10.1371/journal.pone.0090081; palmerpenguins 2020). `Previous: —` / `Next: 02`.

## Charter / rigor

`use_course_style()`; colours from `ml_course.colors`; seed + version echo; "Read the figure" after
cells 8 & 11; English; warm-but-rigorous; banned words avoided; pandas-first.

## Verification (at commit)

`uv run pytest` green; notebook runs top-to-bottom; outputs cleared; `check_no_hardcoded_hex.py`
passes; `gen_llms_txt.py` re-run; both reviewers pass (no BLOCK); Rémy validates visually; then
commit + merge into `chapter/00_GettingStarted`.
