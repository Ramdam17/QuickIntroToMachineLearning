# Notebook plan — 00_GettingStarted / 04_generalize_dont_memorize

> Status: **APPROVED** (2026-06-16, by Rémy). Notebook plans validated by Rémy alone; reviewers
> return on the built notebook. Drives `notebooks/00_GettingStarted/04_generalize_dont_memorize.ipynb`.

## Context

First **methodological** notebook. One concept: *we judge a model on data it has not seen, because
the goal is to generalize, not memorize.* Demonstrated with a deliberately silly **rote memorizer**
(100% on train, ~majority-baseline on test) — the visceral proof of the cardinal sin. Prereqs: 01,
02, 03. Decided with Rémy: use the memorizer demonstration.

## Framing guards (no pre-emption)

- The memorizer is a **strawman**, not a method and not "our first classifier" (that is NB 05's
  nearest centroid).
- "Fraction it gets right" stays **informal**; accuracy + its limits are formalised in NB 06. The
  memorizer's ~55% test score IS the majority baseline — named only in passing here.
- Scaling-before-split leakage named, deferred to NB 11.

## What you'll be able to do

- Explain the **cardinal sin** (scoring on training data) and why a perfect training score is no proof.
- Make a **stratified train/test split** and check it preserved class balance.
- State the **i.i.d. assumption** as a *choice*, and name **data leakage**.
- Describe the honest loop *fit → predict → evaluate on held-out data*.

## Reuse (no new library code)

`datasets.load_penguins()` / `penguins_xy()`; `sklearn.model_selection.train_test_split(...,
stratify=y, random_state=0)`. Train/test scatter uses palette roles `COLORS["train"]` /
`COLORS["test"]`. Memorizer = a dict lookup + majority fallback, a few lines in the notebook.

## Cell-by-cell (~20 cells)

1. Header (`# 04 — Generalize, don't memorize`; purpose; `Prerequisites: 01, 02, 03`; 4 objectives; welcome).
2. (code) Imports & setup — seed; `colors, datasets, viz`; `use_course_style()`; `train_test_split`; `df`, `X`, `y`, `FEATURES`, `SPECIES_ORDER`.
3. Recap / footing — NB 03 looked at the data; now set some aside.
4. **The cardinal sin** — exam answer-key analogy; scoring on training data measures memorization.
5. **Memorize vs generalize** — we want to do well on new penguins; estimate that only on unseen data.
6. **The train/test split** — hold out ~25% as a test set; train on the rest; stratify to keep class proportions.
7. (code) `train_test_split(..., stratify=y, random_state=0)`; print sizes + class proportions per split.
8. Read the output — sizes; proportions preserved; why stratify (rare class can vanish from a fold).
9. (code) Scatter: train vs test points (palette `train`/`test` roles).
10. Read the figure — test set sealed, untouched until the end; that discipline makes its score trustworthy.
11. **A rote memorizer** — a deliberately silly "model": stores training answers, recognizes only exact repeats, else majority. A cautionary tale, not a method.
12. (code) Build it (dict + majority fallback); fraction right on train vs test.
13. Read the output — ≈100% train vs ≈55% test; the gap is the lesson; on new data it's no better than guessing the common species. (accuracy formalised NB 06.)
14. **The i.i.d. assumption** — train/test from the same distribution; we *adopt* it; penguins span islands/years; structure can break it.
15. **Leakage** — the cardinal sin is the simplest leak; keep test sealed; scaling-before-split leak → NB 11.
16. **The honest loop** — fit → predict → evaluate on held-out; NB 05 runs it with the nearest centroid.
17. Vocabulary — train/test split, test (held-out) set, generalization, leakage, stratification, i.i.d., baseline (preview).
18. Your turn — (a) friend reports 100% train accuracy: what do you ask? (b) test fraction 10% vs 50% trade-offs; (c) a leakage scenario + why stratify a rare class.
19. What you built — the split, the cardinal sin felt, leakage/i.i.d. named, the honest loop ready for NB 05.
20. References — ISLR §2.2 & ch. 5; scikit-learn *Common pitfalls*. `Previous: 03` · `Next: 05`.

## Verification

`use_course_style()`; colours from `ml_course.colors` (train/test roles); seed; "Read the figure/
output" after cells 7/9/12; English; banned words avoided; pandas-first. At commit: runs
top-to-bottom; outputs cleared; `check_no_hardcoded_hex.py` passes; `pytest` green; `gen_llms_txt.py`
re-run; both reviewers pass; Rémy validates visually; commit + merge.
