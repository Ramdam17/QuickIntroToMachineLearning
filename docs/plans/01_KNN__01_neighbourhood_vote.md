# Notebook plan — 01_KNN / 01_neighbourhood_vote

> Status: **APPROVED** (2026-06-17, by Rémy). Notebook plans validated by Rémy alone; reviewers return
> on the built notebook. Drives `notebooks/01_KNN/01_neighbourhood_vote.ipynb` (create the
> `notebooks/01_KNN/` folder at build).
>
> **Correction (2026-06-17, approved by Rémy).** The originally approved running query `q=(0.60, 0.14)`
> was re-measured at build and found to sit on the Bayes boundary (~50/50 class mix), so calling its
> vote "the true class 0" was not honest. Replaced with **`q=(-0.23, 0.75)`** (Option A): a point in the
> left region where **only the class-0 crescent lives** (the class-1 crescent does not reach `x≈-0.23`),
> so a large same-noise sample is ~85% class 0 there — class 0 is the right answer, yet the single
> nearest training point is a stray class-1 noise point. The narrative (k=1 fooled → the vote recovers
> the truth, class 0; vote sequence k1=1 / k3=0 / k5=0) is preserved **and now honest**.

## Context

The course's first method notebook. Introduce **k-NN**: classify a point by the **majority vote of its
k nearest neighbours**, built **by hand** on `make_moons`. Establish the rule, **k = neighbourhood
size**, and that k-NN is **lazy / instance-based** — fitting just stores the data; predicting carries
the cost. One concept: *the vote*. Prereqs: 02 (Euclidean distance), 04 (the train/test split).

## Design (measured — `make_moons(n_samples=300, noise=0.30, random_state=0)`, stratified 70/30 → 210 train / 90 test)

- **Running query (clearly class-0 territory, with a stray nearest neighbour):** `q = (-0.23, 0.75)`.
  At `x≈-0.23` only the class-0 crescent has points (the class-1 crescent spans `x∈[0,2]`), so a large
  same-noise sample is **~85% class 0** there (measured at K=501/1501/5001: 0.87/0.84/0.85) → **class 0
  is the right answer**. Its 5 nearest **training** neighbours have labels **[1, 0, 0, 0, 0]** at
  distances **[0.085, 0.122, 0.147, 0.156, 0.170]**. Votes: **k=1 → 1** (the single nearest is a stray
  class-1 noise point), **k=3 → 0**, **k=5 → 0** (4 vs 1). A clean "why more than one neighbour": k=1
  follows one noisy straggler; the vote (k ≥ 3) follows the class-0 majority that actually populates the
  region. (Foreshadows the k-dial, NB 3, without pre-empting it.)
- **Lazy cost:** by-hand "fit" = store the array (trivial); "predict one point" = Euclidean distances
  to all n training points (O(n·d) per query) → time **grows with n** while fit stays ~free (measured
  live at n = 300 / 3000 / 30000; honest nuance: libraries build a KD-/ball-tree at fit to speed this,
  but the cost still lives at predict).
- The "true class" is established **visually + by region**, not by a hardcoded statistic: the figure
  shows q sitting among class-0 points; the prose says this part of the plane is populated by the
  class-0 crescent, with class-1 points appearing only as noise-scattered strays.

## Library additions

**None.** The "query + its k neighbours" picture is a one-off in-notebook figure (scatter + ringed
neighbours + segments) using `ml_course.colors` (`CLASS_CYCLE`, `COLORS`). `KNeighborsClassifier` is
deferred to NB 4 (NB 1 is by hand). `pytest` stays 14. (If later notebooks reuse a neighbours-highlight,
promote it to `viz` then.)

## Cell-by-cell (~18 cells; intuition → by-hand → "Read the figure")

1. (md) **Header** — `# 01 — Predict by the neighbourhood vote`; *Module 01 · k-Nearest Neighbours —
   notebook 1 of 6*; purpose; `Prerequisites: 02, 04`; 4 objectives (state the k-NN rule; compute it by
   hand — distances → k nearest → vote; see k as the neighbourhood size; understand "lazy" and feel the
   predict cost); a warm "your first real method" welcome.
2. (code) Imports (`time`, numpy, matplotlib, `make_moons`, `train_test_split`; `ml_course` viz +
   `CLASS_CYCLE`/`COLORS`) + `np.random.seed(0)` + `viz.use_course_style()` + `make_moons(300, 0.30, 0)`
   + stratified 70/30 split (`random_state=0`); print train/test sizes + train class balance.
3. (md) **Recap & the new dataset** — module 00 gave us **Euclidean distance** (NB 02) and the sealed
   **train/test split** (NB 04). New here: a *method*. And a new dataset — `make_moons` (two interleaving
   noisy crescents): penguins separate so cleanly that every method nails them, so to see k-NN's
   character (and its dials in the notebooks ahead) we use a harder 2-D set (the honest switch from NB 09).
4. (code) scatter of the **training** moons, coloured by class (`CLASS_CYCLE[0/1]`).
5. (md) **Read the figure** — two interleaving classes with overlap in the middle; near the boundary a
   point's neighbours are a mix of both classes — exactly where "who are your neighbours?" becomes a
   real question.
6. (md) **The idea: k-nearest neighbours** — to classify a new point, find the k training points
   **closest** to it (distance, NB 02) and take the **majority vote** of their labels. k = how many
   neighbours vote. There is no equation to fit — the training data *is* the model.
7. (code) **by hand, one query** `q = (-0.23, 0.75)`: Euclidean distances to all 210 training points →
   the 5 nearest; print their labels `[1, 0, 0, 0, 0]` and distances; the vote counts (class 0 → 4,
   class 1 → 1) → class 0.
8. (code) **figure 1** — training scatter + the query (star) + its 5 nearest neighbours ringed + line
   segments to them; title shows the 4–1 vote → class 0.
9. (md) **Read the figure** — the star is the new point; it sits in the left part of the plane where
   only the class-0 crescent has points; the 5 ringed points are its nearest neighbours; their labels
   are [1, 0, 0, 0, 0] → 4 vs 1 → predict class 0. The whole rule, in one picture.
10. (md) **k is the neighbourhood size** — change k and you change who votes. The single nearest point
    (k=1) here is a class-1 point sitting close — but in a region populated by class 0, is that one
    point representative, or a noise-scattered stray?
11. (code) the **same query at k = 1, 3, 5** → votes 1, 0, 0; print each with its neighbour labels.
12. (md) **Read the output** — k=1 trusts the one closest point (a stray class-1 neighbour) and says 1;
    widening to k=3 or 5 lets the class-0 points that actually fill this region outvote that stray,
    recovering the right label (0). Too small a k chases noise; *which* k is right is notebook 3.
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
- The query's "true class 0" is honest: it is in a region populated by the class-0 crescent (~85% class
  0 in a large sample), shown via the figure and the region, not a hardcoded statistic.
- The lazy-cost timing is illustrative (machine-dependent); the notebook reports the *trend* (predict
  grows with n; fit ~free), not machine-specific milliseconds, and flags the KD-tree nuance.

## Verification

Measured anchors (q=(-0.23,0.75): 5-NN labels [1,0,0,0,0], votes k1=1/k3=0/k5=0; the ~85%-class-0
region; the predict-vs-fit timing trend) re-run in the notebook and reconciled into prose at build.
Runs top-to-bottom (nbconvert to /tmp; output-free); `check_no_hardcoded_hex` passes (figure uses
`ml_course.colors`); `gen_llms_txt` re-run; `pytest` green (14, no `src/` change); both reviewers pass
(no BLOCK); Rémy validates visually; commit + merge `notebook → chapter`.
