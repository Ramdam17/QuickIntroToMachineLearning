# Notebook plan — 00_GettingStarted / 09_overfitting_generalization_gap

> Status: **APPROVED** (2026-06-17, by Rémy). Notebook plans validated by Rémy alone; reviewers
> return on the built notebook. Drives `notebooks/00_GettingStarted/09_overfitting_generalization_gap.ipynb`.

## Context

The conceptual pivot of module 00: over-/under-fitting, the **generalization gap** (named precisely,
≠ variance), **bias–variance** as the conceptual *why*, and the **learning curve**. Prereqs: 04, 06.

## Design (measured)

- **Dataset:** `make_moons(n_samples=300, noise=0.30, random_state=0)` — penguins are too separable
  to overfit (measured), so we switch honestly to a harder synthetic 2-D set (penguins return NB 11).
- **Complexity dial:** polynomial **degree**. Engine = `PolynomialFeatures(d)` → `StandardScaler` →
  `LogisticRegression(C=1e6)`, hidden behind a notebook helper `flexible_boundary(degree)` and framed
  as plumbing (linear classifier → ch. 03, scaling → NB 11); we study **only the degree**.
- **Verified complexity sweep** (error = 1−acc): train 0.148→0.019 (falls); test U — min at **degree 3**
  (0.089), rising to ~0.14 by degree 9; gap at deg 9 = 0.144 vs 0.019 (wide).
- **Learning curve:** computed **manually** (subsample train, evaluate on the held-out *test* set — no
  CV, so "cross-validation" is not pre-empted; NB 10 owns it). Trend: train error rises from ~0 as
  data grows, the train–test gap narrows.

## Library addition (DONE, tested)

`viz.plot_train_test_curve(x, train_scores, test_scores, *, xlabel, ylabel='error', ax=None)` — two
lines in the charter train/test colours; serves the complexity curve and the learning curve. Test
added (pytest 13/13, ruff/black/hex clean). `plot_decision_boundary` reused for the 3 boundary panels.

## Cell-by-cell (~19 cells)

1 Header (Prereqs 04, 06; 4 objectives). 2 Why a new dataset (penguins too easy → moons).
3 (code) `make_moons` + split + scatter. 4 Read — interleaving classes; need to bend; can dial how much.
5 Complexity dial md (degree; engine = plumbing, ch.03/NB11 forward-refs; study only the degree).
6 (code) `flexible_boundary(d)`; 3 panels (deg 1, 3, 9) via `plot_decision_boundary`. 7 Read — underfit / good / overfit.
8 (code) sweep deg 1–9; `plot_train_test_curve` (error vs degree). 9 Read — train↓, test U; the **generalization gap** (≠ variance); widens with overfitting; best deg 3.
10 Bias–variance md (too simple = high bias / both errors high; too complex = high variance / train low, test high; conceptual, decomposition is an average over datasets).
11 Learning curve md (fix complexity, vary data). 12 (code) manual learning curve at degree 6 (held-out test, no CV); `plot_train_test_curve` (error vs train size). 13 Read — gap wide with little data, narrows as data grows; train error climbs from ~0; more data cures variance, not bias.
14 Going further (optional) md — bias–variance decomposition in words; variance fan idea. 15 (code, optional) overlay degree-9 boundaries from several resamples. 16 Read — high-degree boundary jumps around (variance); low-degree barely moves.
17 Your turn (label under/overfit + pick degree; 100%/70% diagnosis + a fix; would more data help underfitting?). 18 What built + vocab. 19 References (ISLR §2.2; Domingos 2012 DOI 10.1145/2347736.2347755; Géron ch.4). `Previous: 08` · `Next: 10`.

## Honest limits / no pre-emption

Penguins→moons stated plainly; engine = plumbing (forward refs); generalization gap ≠ variance;
bias–variance conceptual + optional fan; **cross-validation not used** (manual held-out learning
curve) — NB 10 owns CV; here we pick degree by eye and flag that repeatedly tuning on test would leak.

## Verification

`plot_train_test_curve` tested; pytest green; ruff/black/hex clean. Runs top-to-bottom; outputs
cleared; gen_llms_txt re-run; numbers reconciled at build; both reviewers pass; Rémy validates; commit + merge.
