# Notebook plan — 00_GettingStarted / 10_cross_validation

> Status: **APPROVED** (2026-06-17, by Rémy). Notebook plans validated by Rémy alone; reviewers
> return on the built notebook. Drives `notebooks/00_GettingStarted/10_cross_validation.ipynb`.
> Structure: **a single notebook** (Rémy chose the single NB 10 over a 10a/10b split — content
> integrity over the ~20-cell target; budget ~26-28 cells, no compression).

## Context

The conceptual completion of the evaluation arc (06-09). Notebook 09 ended on an honest loose end:
*we chose the polynomial degree by reading the **test** error* — and warned that doing so repeatedly
makes the test set stop being a fair judge. This notebook closes that loop. It teaches how to choose a
hyperparameter **honestly**, never touching the test set: parameters vs hyperparameters, the
**validation set**, **stratified k-fold cross-validation** (built by hand), **model selection**, and
**tuning-on-test inflation**. Prereqs: **04** (the split), **09** (the picked-the-degree-on-test loose
end). It continues NB 09's setup verbatim: `make_moons(n_samples=300, noise=0.30, random_state=0)`,
`train_test_split(test_size=0.30, random_state=0, stratify=y)`, and the `flexible_boundary(degree)`
pipeline (`PolynomialFeatures(degree) → StandardScaler → LogisticRegression(C=1e6, max_iter=5000)`).

## Design (measured on the NB 09 setup — `uv run python`, not guessed)

- **5-fold stratified CV picks degree 3** — the same honest sweet spot NB 09 found by eye on the test
  curve, now chosen **without touching test**. CV accuracy (the by-hand folds, as shipped): deg1 0.843,
  deg2 0.829, deg3 **0.914** (peak), easing to ~0.86-0.88 by deg 6-9. (Small non-monotone wobble at
  deg 2, echoing NB 09's train kink — read the trend, not each step. A pre-build run with scikit-learn's
  own StratifiedKFold folds gave deg3 0.905 — same pick, a slightly different fold partition.)
- **By-hand k-fold mean == `cross_val_score` exactly** (0.904762 both; the rigor check, true because we
  feed sklearn the same folds we built).
- **A single validation split is unstable:** the val-picked degree across 6 seeds = **3, 3, 5, 6, 3, 9**
  — motivates k-fold by measurement, not assertion.
- **Final honest estimate:** CV-picked degree 3, refit on the full training set, scored **once** on the
  sealed test = **0.9111**.
- **Tuning-on-test inflation** (100 random splits): honest CV protocol mean test acc **0.9197** vs
  picking the degree by best test acc **0.9338** → **+0.014** optimistic bias. Modest but real; on any
  single split the two may coincide (seed 0: both pick degree 3), so the bias only shows as an **average
  over splits** — which is exactly why the experiment is repeated, and it grows with the number of
  settings tried.

## Library additions

**None.** Reuse `viz.plot_train_test_curve` for the validation curve and the CV train/val curve. The
**k-fold scheme diagram** and the **inflation bar** are one-off in-notebook figures pulling **all**
colours from `ml_course.colors` (validation fold = `COLORS["test"]`, train folds = `COLORS["train"]`;
honest vs cheat bars = `COLORS["model"]` / `COLORS["highlight"]`) so the hex guard passes. `pytest`
stays 13 (no `src/` change). A reusable `plot_kfold_scheme` is **deferred** — not bulk-added
speculatively; promote it only when a later chapter actually needs it.

## Cell-by-cell (~26-28 cells; intuition → implementation → "Read the figure"; 4 figures)

1. (md) **Header** — `# 10 — Validating honestly: cross-validation`; purpose (choose the degree without
   ever touching test); `Prerequisites: 04, 09`; 4 objectives (parameters vs hyperparameters; why
   tuning on test inflates; build stratified k-fold CV by hand and select the degree; quote one honest
   final estimate); warm welcome.
2. (code) Imports + `np.random.seed(0)` + `viz.use_course_style()`; re-establish `make_moons` + the
   stratified split + `flexible_boundary(degree)` so the notebook stands alone.
3. (md) **Recap & the loose end** — NB 04's train/test; NB 09 chose the degree by reading the *test*
   curve; why that quietly spoils the test set as an unbiased judge (a choice adapted to it is no longer
   tested by it).
4. (md) **Parameters vs hyperparameters** — weights are learned by `fit`; the degree is set by us
   *before* `fit`. `fit` will not choose the degree — we must, and honestly.
5. (md) **Intuition: a third slice, the validation set** — seal the test set; carve a validation set out
   of training; choose the degree on validation. Three slices, three jobs.
6. (code) Three-way split; per degree fit on the train part, score train vs validation; **figure 1** =
   `plot_train_test_curve(degrees, train_err, val_err, xlabel="polynomial degree")`; print the
   val-picked degree.
7. (md) **Read figure 1** — validation-error U picks a degree, chosen without touching test (a
   legitimate stand-in for the NB 09 test curve).
8. (md) **The catch** — one validation split wastes data and the pick is unstable (depends which points
   land in validation).
9. (code) **Wobble demo** — val-picked degree across ~6 seeds (measured 3,3,5,6,3,9); print each.
10. (md) Interpret — unstable + wasteful → use *all* the data for validating, each point once → k-fold.
11. (md) **Intuition: k-fold CV** — k folds; each takes a turn as validation while the rest train;
    average the k scores; every point validates exactly once; **stratified** preserves class proportions.
12. (code) **figure 2** = k-fold scheme diagram (k rows of coloured blocks; validation = `COLORS["test"]`,
    train = `COLORS["train"]`), in-notebook, charter colours only.
13. (md) **Read figure 2** — what the rotation shows; the stratification note.
14. (code) **Stratified k-fold BY HAND** — a function returning k `(train_idx, val_idx)` pairs that
    preserve each class's proportion; print fold sizes + per-fold class balance (the stratification check).
15. (md) **Read** — folds ~equal; class proportions preserved in each (brief; may merge into 16).
16. (code) **CV per degree with the by-hand folds** — per-fold train & val error; assemble a **pandas
    DataFrame** (degree × folds + mean) for display; identify the CV-picked degree (= 3).
17. (code) **figure 3** = `plot_train_test_curve(degrees, cv_train_err, cv_val_err, xlabel="polynomial degree")`;
    mark the chosen degree.
18. (md) **Read figure 3** — CV validation error U, minimum at degree 3 = the honest pick; it agrees
    with NB 09's eyeballed degree — but was chosen without touching test.
19. (code) **By-hand == sklearn** — `cross_val_score` fed the *same* folds reproduces the by-hand mean
    exactly (0.904762); `StratifiedKFold` is the library version of what we built; print the match.
20. (md) Interpret — the library just runs the loop we wrote (brief; may merge into 21).
21. (md) **The final honest estimate** — degree chosen on training via CV; refit on *all* training;
    score the sealed test **once**.
22. (code) Refit `flexible_boundary(3)` on full `X_train`; score `X_test` once (= 0.9111); print.
23. (md) **Intuition: why not pick the degree on test?** tuning-on-test / selection inflation — trying
    many settings on test and keeping the best fits the test's noise through the choice.
24. (code) **figure 4** = inflation experiment over ~100 random splits: honest (CV-pick → test once) vs
    cheat (best-of-degrees test acc); bar of the two means (0.9197 vs 0.9338), charter colours.
25. (md) **Read figure 4** — cheat reports higher than it generalizes (gap ≈ 0.014 here, grows with the
    number of settings tried); on a single split the two may coincide, so the bias is an average effect
    → this is why the test set is touched once, at the very end.
26. (md) **Your turn** — (a) 1000 samples, 5-fold: how many fits to evaluate one degree, on how many
    points each? (b) tiny data → argue for a larger k (toward leave-one-out) and name its cost; (c)
    "99% accuracy after trying 50 settings on the test set" — what is wrong, and what should they have
    done?
27. (md) **What you built** + vocabulary — hyperparameter vs parameter, validation set, (stratified)
    k-fold cross-validation, model selection, the touch-test-once rule, tuning-on-test / selection
    inflation.
28. (md) **Going further (optional)** — nested CV (select *and* estimate); LOOCV as k = n; CV estimates
    themselves have variance (folds are correlated). Clearly optional.
29. (md) **References** — ISLR §5.1 (cross-validation; validation-set approach, LOOCV, k-fold) DOI
    10.1007/978-1-0716-1418-1; Kohavi (1995), *A study of cross-validation and bootstrap for accuracy
    estimation and model selection*, IJCAI (stratified k-fold & model selection — exact cite/DOI
    verified at build); Géron ch. 2 (cross-validation). `Previous: 09` · `Next: 11`.

(Code+Read pairs 14/15 and 19/20 may merge at build to land nearer ~26 cells — no content dropped.)

## Honest limits / no pre-emption

- The degree and the linear engine remain **plumbing** (forward refs: logistic regression → ch. 03,
  scaling → NB 11); the lesson here is **CV**, not the classifier.
- `flexible_boundary` is a `Pipeline`, so the scaler is refit inside each fold's training data already —
  no scale leakage by construction. The **leakage deep-dive is NB 11's** job; here we note it is handled,
  not belabour it.
- CV estimates have variance too (correlated folds); the final single test number is one draw — flagged;
  **nested CV deferred to Going further** (it is not a prerequisite for NB 11).
- The inflation is shown as a **measured average over splits**, not asserted, and its magnitude (~0.014)
  is stated honestly as modest and dependent on how many settings were tried.

## Verification

Measured numbers above re-run inside the notebook and reconciled into the prose at build. Notebook runs
top-to-bottom (`jupyter nbconvert --execute` to /tmp, tracked file kept output-free);
`check_no_hardcoded_hex.py` passes (in-notebook diagrams use `ml_course.colors` only); `gen_llms_txt.py`
re-run; `pytest` green (still 13, no `src/` change); `course_map.md` 00 section marks NB 10 done; both
reviewers pass (no BLOCK); Rémy validates visually; commit (`feat(00_getting_started): notebook 10 —
validating honestly: cross-validation`) + merge `notebook → chapter`.
