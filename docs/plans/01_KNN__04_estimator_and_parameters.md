# Notebook plan — 01_KNN / 04_estimator_and_parameters

> Status: **APPROVED** (2026-06-17, by Rémy). Notebook plans validated by Rémy alone; reviewers return
> on the built notebook. Drives `notebooks/01_KNN/04_estimator_and_parameters.ipynb`.

## Context

NB 4 of the k-NN chapter — and the notebook where the **role changes**. NBs 1–3 built k-NN by hand,
one concept at a time. Now we meet the real estimator, **`sklearn.neighbors.KNeighborsClassifier`**,
confirm it is exactly the rule we built, and walk its **hyperparameters**: `n_neighbors` (the NB 3
dial), `weights` (uniform vs distance), and `metric` / Minkowski `p` (the NB 2 metric, now a tunable
knob). We show what each knob does, where it fails, and how to choose it — by cross-validation (NB 10),
which now plugs straight in because the estimator is sklearn-compatible. We also surface the **even-k
tie** and how sklearn resolves it. This is the "method & its parameters" notebook, so it integrates
several knobs rather than teaching one concept. Prereqs: NB 1–3, plus 05 (decision boundary).

## Design (measured — `make_moons(300, 0.30, 0)`, stratified 70/30 → 210/90, original comparably-scaled features)

- **The library is our rule, optimized.** by-hand k-NN vs `KNeighborsClassifier` at k=15: **identical
  predictions (0 differ)**, test acc **0.956**. And `cross_val_score(KNeighborsClassifier(15), X_train,
  y_train, cv=5-fold)` = **0.919** — the *exact* number NB 3's by-hand CV produced. cross_val_score
  (NB 10) now plugs in directly (NB 3 had to hand-roll CV because `ByHandKNN` was not clone-able).
- **`weights`: uniform vs distance** (test acc): k=15 0.956/0.967, k=51 0.878/0.922, k=101 0.733/0.889,
  **k=151 0.678/0.833**. Distance-weighting (weight ∝ 1/distance) **resists over-smoothing** — the gap
  widens as k grows, because near neighbours keep their say while a uniform vote drowns in the far
  crowd.
- **`metric` / Minkowski p** at k=15: Manhattan p=1 **0.967**, Euclidean p=2 0.956, Chebyshev 0.956. A
  small effect (one test point), echoing NB 2's "scale ≫ metric"; still worth trying as a hyperparameter
  (the deep metric geometry — L∞, Mahalanobis — is NB 6).
- **Choosing `weights` by CV** (the "how to choose" payoff): `cross_val_score` at k=15 gives uniform
  **0.919** vs distance **0.924** → pick distance; it confirms on the sealed test at **0.967** (up from
  uniform's 0.956). The same recipe tunes any knob: CV on train, confirm on test once.
- **Even-k tie** (k=2): with `weights="uniform"`, sklearn breaks a 1-1 tie by the **lowest class label
  (0)** — deterministic, but arbitrary: it ignores which neighbour is nearer (measured test point: 2-NN
  labels [1, 0] with the **class-1** point nearer, yet it predicts 0). Hence **odd k for two classes**.
  (With `weights="distance"` the nearer neighbour wins instead — a one-line contrast.)

## Library additions / figures

**None to `src/`.** This notebook uses `sklearn.neighbors.KNeighborsClassifier` (introduced here, per
the chapter plan) and `sklearn.model_selection.cross_val_score` / `StratifiedKFold` (NB 10). Reuse
`viz.plot_decision_boundary` for the `weights` boundary panels (KNeighborsClassifier has `predict`, so
no wrapper needed). The by-hand `knn_predict` is kept once, only for the parity check. `pytest` stays 14.

## Cell-by-cell (~21 cells; meet the estimator → walk each knob → "Read the figure/output")

1. (md) **Header** — `# 04 — The estimator & its parameters`; *Module 01 · k-NN — notebook 4 of 6*;
   purpose; `Prerequisites: 01–03, plus 05 (decision boundary)`; objectives (use
   `KNeighborsClassifier`; confirm it equals our by-hand rule; tune `n_neighbors`, `weights`,
   `metric`; choose a knob by CV; understand the even-k tie); a warm "graduating from by-hand to the
   library" welcome.
2. (code) imports (numpy, matplotlib, `make_moons`, `train_test_split`, `StratifiedKFold`,
   `cross_val_score`, `KNeighborsClassifier`; `ml_course` viz/colors) + seed + `use_course_style()` +
   data + split + the by-hand `knn_predict` (kept for the parity check). Print sizes.
3. (md) **From by-hand to the library** — we built k-NN three times by hand; meet
   `KNeighborsClassifier`, the same rule, optimized (it builds a KD-/ball-tree at fit to skip
   comparisons — NB 1's lazy-cost nuance), behind the `fit`/`predict` interface we wrapped in NB 2.
   Two honest checks before we trust it: does it predict like ours, and does `cross_val_score`
   reproduce our by-hand CV?
4. (code) **parity** — by-hand vs `KNeighborsClassifier` at k=15 (count differences, print accuracy);
   and `cross_val_score(KNeighborsClassifier(15), X_train, y_train, cv=skf)`.
5. (md) **Read the output** — 0 predictions differ (acc 0.956), and cross_val_score returns 0.919,
   exactly NB 3's by-hand CV number. The library is our rule made fast, and NB 10's `cross_val_score`
   now plugs straight in. Three knobs to explore: `n_neighbors`, `weights`, `metric`.
6. (md) **Knob 1 — `n_neighbors`** — this is the NB 3 dial: the bias–variance knob, chosen by CV. The
   estimator's default is k = 5. (Brief — NB 3 did the depth; here it takes its place among the knobs.)
7. (md) **Knob 2 — `weights`** — `uniform` (every neighbour counts equally) vs `distance` (a neighbour's
   vote is weighted by 1/distance, so nearer points matter more). Intuition: distance-weighting lets a
   large neighbourhood add context without letting far points overrule close ones.
8. (code) `weights` uniform vs distance, test accuracy across k = 15/51/101/151.
9. (md) **Read the output** — close at moderate k; as k grows, uniform underfits (the far crowd drowns
   the signal: 0.678 at k=151) while distance holds up (0.833) — it resists over-smoothing.
10. (code) **boundary** — uniform vs distance at a large k (built at **k = 151**, not the planned 51:
    it matches the table's most dramatic row, uniform 0.678 vs distance 0.833), two panels via
    `plot_decision_boundary`.
11. (md) **Read the figure** — uniform at k=51 is washed out; distance at k=51 stays responsive near
    the data, the boundary bending back toward the crescents. Same k, different weighting.
12. (md) **Knob 3 — `metric` (Minkowski `p`)** — `p=2` Euclidean (default), `p=1` Manhattan, and
    others (e.g. Chebyshev). The NB 2 metric, now a tunable parameter. (The geometry of metrics — L1/
    L2/L∞, Mahalanobis — is NB 6.)
13. (code) metric comparison at k=15: Manhattan (p=1), Euclidean (p=2), Chebyshev.
14. (md) **Read the output** — the metric nudges accuracy (Manhattan a touch better here, one test
    point) — small, as NB 2 found; still worth trying as a hyperparameter, and the best choice is
    data-dependent.
15. (code) **choosing a knob honestly** — `cross_val_score` on the train set for `weights` at k=15
    (uniform vs distance) → pick the winner; confirm once on the sealed test set.
16. (md) **Read the output** — CV (training data only) prefers `distance` (0.924 vs 0.919); the test
    set confirms it (0.967 vs uniform's 0.956). This is how you set any of these knobs — CV chooses,
    the test set confirms once (the NB 3 discipline, now with the library's `cross_val_score`).
17. (md) **The even-k tie** — with two classes and an even k, the vote can tie (k=2: one of each). How
    does the estimator decide?
18. (code) even-k tie demo — find k=2 queries whose two neighbours split 1-1; show the prediction (and
    that the nearer neighbour can be the *losing* class).
19. (md) **Read the output** — with `uniform` weights, sklearn breaks the tie by the **lowest class
    label**: deterministic, but arbitrary — it ignores which neighbour is nearer (the shown point's
    nearest is class 1, yet it predicts 0). That is why **odd k** is the default for two classes: no
    ties. (`weights="distance"` would instead break toward the nearer neighbour.)
20. (md) **Your turn** — (a) why does `weights="distance"` help more at large k than at small k? (b)
    with **three** classes, can an *odd* k still produce a tie? (c) on a new dataset, how would you
    decide between `p=1` and `p=2`?
21. (md) **What you built** + vocabulary — `KNeighborsClassifier`, `n_neighbors`, `weights`
    (uniform/distance), `metric` / Minkowski `p`, the deterministic tie-break, the `fit`/`predict`
    interface, tuning by `cross_val_score`. **References** — scikit-learn `KNeighborsClassifier` user
    guide / API; ISLR §2.2.3; Cover & Hart (1967), DOI 10.1109/TIT.1967.1053964. `Previous: 03 — The k
    dial` · `Next: 05 — Demanding case: breast cancer & the curse`. *(Split into two md cells —
    "What you built" + "References" — if it reads long.)*

## Honest limits / no pre-emption

- NB 4 **introduces `KNeighborsClassifier`** (the chapter plan places it here) and tunes its knobs. It
  does **not** do the deep metric geometry / Mahalanobis (NB 6) or the demanding 30-D case + curse (NB
  5). The metric is shown as a tunable dial with a small effect, not a geometry study.
- Three classes: odd k does **not** guarantee no ties (only binary does) — surfaced in the exercises so
  the "odd k" rule is not over-generalized.
- `weights` and `metric` effects are dataset-specific (isotropic moons); the *behaviours* (distance
  resists over-smoothing; metric is a fine dial; ties are broken deterministically) are the lesson, and
  the exact numbers are flagged as seed/dataset-specific.
- Original (comparably-scaled) moons are used; standardization inside a `Pipeline` is the NB 5 workflow,
  noted not re-shown.

## Verification

Measured anchors (by-hand == sklearn 0-diff/0.956; cross_val_score 0.919 == NB 3; weights
.956/.967…/.678/.833; metric p=1 0.967 / p=2 0.956 / Chebyshev 0.956; CV weights 0.919 vs 0.924 → test
0.967; even-k tie → lowest label 0) re-run in the notebook and reconciled into prose at build. Runs
top-to-bottom (nbconvert to /tmp; output-free; **`--clear-output --inplace` before commit**);
`check_no_hardcoded_hex` passes; `gen_llms_txt` re-run; `pytest` green (14, no `src/` change); both
reviewers pass (no BLOCK); Rémy validates visually; commit + merge `notebook → chapter`.
