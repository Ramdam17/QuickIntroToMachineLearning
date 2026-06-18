# Notebook plan — 01_KNN / 01_neighbourhood_vote

> Status: **APPROVED** (2026-06-17, by Rémy). Notebook plans validated by Rémy alone; reviewers return
> on the built notebook. Drives `notebooks/01_KNN/01_neighbourhood_vote.ipynb` (create the
> `notebooks/01_KNN/` folder at build).

## Context

The course's first method notebook. Introduce **k-NN**: classify a point by the **majority vote of its
k nearest neighbours**, built **by hand** on `make_moons`. Establish the rule, **k = neighbourhood
size**, and that k-NN is **lazy / instance-based** — fitting just stores the data; predicting carries
the cost. One concept: *the vote*. Prereqs: 02 (Euclidean distance), 04 (the train/test split).

## Design (measured — `make_moons(n_samples=300, noise=0.30, random_state=0)`, stratified 70/30 → 210 train / 90 test)

- **Running query (borderline, vivid):** `q = (0.60, 0.14)`, true class **0**. Its 5 nearest training
  neighbours have labels **[1, 0, 0, 1, 0]** at distances [0.07, 0.13, 0.14, 0.15, 0.15]. Votes: **k=1
  → 1** (the single nearest is a class-1 point — likely noise), **k=3 → 0**, **k=5 → 0** (the truth).
  A perfect "why more than one neighbour": k=1 overfits a noisy neighbour; the vote (k ≥ 3) recovers
  the true label. (Foreshadows the k-dial, NB 3, without pre-empting it.)
- **Lazy cost:** by-hand "fit" = store the array (trivial); "predict one point" = Euclidean distances
  to all n training points (O(n·d) per query) → time **grows with n** while fit stays ~free (measured;
  honest nuance: libraries build a KD-/ball-tree at fit to speed this, but the cost still lives at
  predict).

## Library additions

**None.** The "query + its k neighbours" picture is a one-off in-notebook figure (scatter + ringed
neighbours + segments) using `ml_course.colors`. `KNeighborsClassifier` is deferred to NB 4 (NB 1 is
by hand). `pytest` stays 14. (If later notebooks reuse a neighbours-highlight, promote it to `viz` then.)

## Cell-by-cell (~18 cells; intuition → by-hand → "Read the figure")

1. (md) **Header** — `# 01 — Predict by the neighbourhood vote`; *Module 01 · k-Nearest Neighbours —
   notebook 1 of 6*; purpose; `Prerequisites: 02, 04`; 4 objectives (state the k-NN rule; compute it by
   hand — distances → k nearest → vote; see k as the neighbourhood size; understand "lazy" and feel the
   predict cost); a warm "your first real method" welcome.
2. (code) Imports (numpy, matplotlib, `make_moons`, `train_test_split`; `ml_course` colors/viz) +
   `np.random.seed(0)` + `viz.use_course_style()` + `make_moons(300, 0.30, 0)` + stratified 70/30 split;
   print train/test sizes.
3. (md) **Recap & the new dataset** — module 00 gave us **Euclidean distance** (NB 02) and the sealed
   **train/test split** (NB 04). New here: a *method*. And a new dataset — `make_moons` (two interleaving
   noisy crescents): penguins separate so cleanly that every method nails them, so to see k-NN's
   character (and its dials in the notebooks ahead) we use a harder 2-D set (the honest switch from NB 09).
4. (code) scatter of the **training** moons, coloured by class.
5. (md) **Read the figure** — two interleaving classes with overlap in the middle; near the boundary a
   point's neighbours are a mix of both classes — exactly where "who are your neighbours?" becomes a
   real question.
6. (md) **The idea: k-nearest neighbours** — to classify a new point, find the k training points
   **closest** to it (distance, NB 02) and take the **majority vote** of their labels. k = how many
   neighbours vote. There is no equation to fit — the training data *is* the model.
7. (code) **by hand, one query** `q = (0.60, 0.14)`: Euclidean distances to all 210 training points →
   the 5 nearest; print their labels `[1, 0, 0, 1, 0]` and distances; the majority vote → class 0.
8. (code) **figure 1** — training scatter + the query (star) + its 5 nearest neighbours ringed + line
   segments to them; title shows the 3–2 vote.
9. (md) **Read the figure** — the star is the new point; the 5 ringed points are its nearest neighbours;
   their labels are [1, 0, 0, 1, 0] → 3 vs 2 → predict class 0. The whole rule, in one picture.
10. (md) **k is the neighbourhood size** — change k and you change who votes. The single nearest point
    (k=1) here is a class-1 point sitting close — but is it representative, or noise?
11. (code) the **same query at k = 1, 3, 5** → votes 1, 0, 0; print each with its neighbour labels.
12. (md) **Read the output** — k=1 trusts the one closest point (a stray class-1 neighbour) and says 1;
    widening to k=3 or 5 lets the surrounding class-0 points outvote that stray, recovering the true
    label (0). Too small a k chases noise; *which* k is right is notebook 3.
13. (md) **k-NN is a "lazy" learner** — most methods train hard then predict cheaply; k-NN is the
    opposite. Fitting just **stores** the training data; all the work happens at **predict** time, when
    it measures the new point against every stored point.
14. (code) **feel it** — "fit" = keep the array (instant); "predict one point" = distances to all n;
    time it at n = 300 / 3000 / 30000 → predict time grows with n while fit stays ~free. (Note:
    libraries build a spatial index at fit to speed this; the cost still lives at predict.)
15. (md) **Read the output** — fitting is essentially free (k-NN just remembers); predicting carries the
    cost, and it grows with the training-set size. Cheap to train, expensive to predict, whole dataset
    in memory — k-NN's signature, and the honest contrast to the "eager" methods in later chapters.
16. (md) **Your turn** — (a) a query's 3 nearest labels are [1, 0, 1]: what does k=3 predict? (b) why
    can k=1 be fooled near a noisy point? (c) if you triple the training set, what happens to predict
    time, and why?
17. (md) **What you built** + vocabulary — k-nearest neighbours, neighbour, the (majority) vote,
    k = neighbourhood size, lazy / instance-based learner, predict-time cost.
18. (md) **References** — Cover & Hart (1967), *Nearest neighbor pattern classification*, IEEE Trans.
    Inf. Theory 13(1):21–27, DOI 10.1109/TIT.1967.1053964; Fix & Hodges (1951); ISLR §2.2.3 (DOI
    10.1007/978-1-0716-1418-1). `Previous: —` (end of module 00) · `Next: 02 — Distance & the scale trap`.

## Honest limits / no pre-emption

- One concept only — *the vote*. The **decision boundary** (regions) and **choosing k** are NB 3; the
  **scale trap** is NB 2; the **estimator & parameters** are NB 4 — NB 1 stays on a single-query
  intuition, by hand, no `KNeighborsClassifier`.
- moons is synthetic, used because penguins is too separable (NB 09's honest switch) — stated.
- The lazy-cost timing is illustrative (machine-dependent); the notebook reports the *trend* (predict
  grows with n; fit ~free), not machine-specific milliseconds, and flags the KD-tree nuance.

## Verification

Measured anchors (the q=(0.60,0.14) neighbour labels/votes; the predict-vs-fit timing trend) re-run in
the notebook and reconciled into prose at build. Runs top-to-bottom (nbconvert to /tmp; output-free);
`check_no_hardcoded_hex` passes (figure uses `ml_course.colors`); `gen_llms_txt` re-run; `pytest` green
(14, no `src/` change); both reviewers pass (no BLOCK); Rémy validates visually; commit + merge
`notebook → chapter`.
