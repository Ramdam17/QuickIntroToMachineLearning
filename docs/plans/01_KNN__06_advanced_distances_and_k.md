# Notebook plan — 01_KNN / 06_advanced_distances_and_k

> Status: **APPROVED** (2026-06-18, by Rémy). Notebook plans validated by Rémy alone; reviewers return
> on the built notebook. Drives `notebooks/01_KNN/06_advanced_distances_and_k.ipynb`. Build via
> `uv run python - < /tmp/build_nb6.py` (stdin) to dodge the stray `/tmp/struct.py` shadow.

## Context

The chapter's optional **Advanced** notebook (a Rémy-approved exception to the 5-notebook ceiling),
written to answer the two questions Rémy raised: **(1) does the *definition* of distance matter for
classification — L1/L2/L∞, Mahalanobis, cosine?** and **(2) is there a principled way to choose k —
and what is that "silhouette" algorithm?** It deepens NB 2 (distance) and NB 3 (choosing k) without
introducing a new method. Prereqs: NB 1–5.

## Design (measured)

**Q1 — the metric is a choice (geometry first, then where it bites).**
- **Minkowski unit balls** (geometry, no fitting): the set {x : ‖x‖_p = 1} is a **diamond (p=1)**,
  **circle (p=2)**, **square (p=∞)**. The unit ball *is* the shape of "within distance r" — the metric
  defines the neighbourhood's shape.
- **Mahalanobis = covariance-aware distance.** Standardization (NB 2) fixes per-feature *scale* but not
  *correlation*; Mahalanobis whitens by the covariance, so its unit ball is an **ellipse aligned to the
  data**. Shown **geometrically** (Euclidean circle vs Mahalanobis ellipse on correlated data). **Honest
  scope note:** we do *not* stage an accuracy race — measured, Mahalanobis ≈ Euclidean ≈ standardized
  (~0.85) on correlated 2-class data, because a clean Mahalanobis win needs the *within-class*
  covariance (LDA territory, out of scope). We show what it *is*, not a contrived victory. **Cosine** is
  named (angle not magnitude; for direction/text data).
- **Metric × the curse** (Aggarwal 2001, ties to NB 5): smaller p resists distance concentration.
  Measured **near/far ratio** on pure N(0,1) noise (lower = better contrast):

  | dims | p=0.5 | p=1 | p=2 |
  |---|---|---|---|
  | 2 | 0.032 | 0.037 | 0.040 |
  | 20 | 0.383 | 0.427 | 0.447 |
  | 100 | 0.653 | 0.690 | 0.711 |
  | 500 | 0.832 | 0.852 | 0.864 |
  | 1000 | 0.874 | 0.890 | 0.899 |

  And in **accuracy** (breast_cancer, standardized, + noise dims; k=7): p=1 vs p=2 = **0.947/0.947** at
  0 noise, **0.906/0.860** at +200, **0.877/0.842** at +1000 → in high dimensions Manhattan (smaller p)
  degrades **slower** than Euclidean. The bigger lever is still reducing dimensionality (NB 5); the
  metric is a secondary, real high-d effect.

**Q2 — choosing k, rigorously.**
- **Nested CV.** Choosing k by CV *and* reporting that same CV score is optimistic (you keep the
  luckiest k — a winner's curse). Nest: inner CV tunes k, outer CV estimates honestly. Measured on
  breast_cancer: naive (best inner-CV) **0.967** vs nested **0.960** — small here, but in the expected
  direction; the principle is essential (NB 5's single held-out test is the simplest honest estimate).
- **"silhouette" — the honest clarification (Rémy's question).** The silhouette coefficient
  (Rousseeuw 1987) is real and *does* pick a "k" — but the number of **clusters** for an
  **unsupervised** method like k-means, by within- vs between-cluster distances. Measured on 3-blob
  data, k-means silhouette peaks at **n_clusters = 3** (the true count). It does **not** choose k-NN's
  k: k-NN is **supervised** (we have labels) → choose k by **cross-validation against accuracy**. Same
  word "k", different problems.

## Library additions / figures

**None to `src/`.** Uses `KNeighborsClassifier` (incl. `metric="mahalanobis"`/Minkowski `p`),
`sklearn.cluster.KMeans`, `silhouette_score`, `GridSearchCV`, `cross_val_score`, `StratifiedKFold`,
`pairwise_distances`, `make_blobs`, `load_breast_cancer`, matplotlib for the unit balls / ellipse.
Figures use `ml_course.colors`/`viz`. `pytest` stays 14.

## Cell-by-cell (~21 cells)

1. (md) **Header** — `# 06 — Advanced: distances & choosing k`; *Module 01 · k-NN — notebook 6 of 6
   (optional, advanced)*; purpose (answer two deeper questions); `Prerequisites: 01–05`; objectives
   (see how the metric reshapes the neighbourhood; meet Mahalanobis & cosine; see why the metric
   matters more in high-d; choose k by nested CV; tell silhouette apart from k-NN's k). Frame it as
   optional depth for the curious.
2. (code) imports + seed + `use_course_style()`.
3. (md) **Q1 — distance is a choice.** The Minkowski family (`p`): p=2 Euclidean, p=1 Manhattan, p→∞
   Chebyshev. We will *see* each as a neighbourhood shape, then meet two non-Minkowski distances.
4. (code) **unit balls** — plot {‖x‖_p = 1} for p = 1, 2, ∞ (one panel, three curves).
5. (md) **Read the figure** — diamond / circle / square. The unit ball is the set of points "within
   distance 1"; the metric is literally the shape of a neighbourhood, so it changes who is nearest.
6. (md) **Mahalanobis — covariance-aware.** Standardization equalized feature *scales* (NB 2) but left
   *correlations*; Mahalanobis whitens by the covariance Σ, so its unit ball {x : xᵀΣ⁻¹x = 1} is an
   ellipse aligned to the data.
7. (code) correlated 2-D cloud + Euclidean unit **circle** vs Mahalanobis unit **ellipse** (from the
   sample covariance).
8. (md) **Read the figure** — the circle treats every direction equally; the ellipse stretches along
   the cloud, so a step across the data's narrow direction counts as "farther". (Honest note: a clean
   accuracy win needs the within-class covariance — LDA territory, beyond this chapter; here we show
   the geometry. **Cosine distance** is named: it compares *direction*, ignoring magnitude — natural
   for text / counts.)
9. (md) **Metric × the curse** — in high dimensions the metric matters more (Aggarwal 2001): smaller p
   resists the distance concentration we felt in NB 5.
10. (code) (a) near/far ratio on pure noise for p = 0.5 / 1 / 2 across dims; (b) accuracy on
    breast_cancer + noise dims, p=1 vs p=2. Print both; plot the ratio curves.
11. (md) **Read the output** — smaller p concentrates a little slower, and at +1000 noise dims
    Manhattan (0.877) beats Euclidean (0.842). So in high-d prefer smaller p — but the dominant fix is
    still fewer dimensions (NB 5); the metric is a secondary lever.
12. (md) **Q2 — choosing k, rigorously.** We picked k by CV (NB 3). Two refinements follow.
13. (md) **Nested CV.** If you tune k on CV and then quote that CV number, it is optimistic — you kept
    the luckiest k. Nest it: an inner CV tunes k, an outer CV estimates performance on data the tuning
    never saw.
14. (code) naive (`GridSearchCV.best_score_`) vs nested (`cross_val_score` of the grid) on
    breast_cancer.
15. (md) **Read the output** — naive 0.967 vs nested 0.960: the tuned CV score is optimistic; nested CV
    is the honest estimate. (Small gap here; larger when grids are big or data small. The single
    held-out test in NB 5 is the simplest honest estimate of all.)
16. (md) **"silhouette" — a clarification.** The algorithm Rémy recalled is real (Rousseeuw 1987): it
    scores how tight-and-separated clusters are, and is used to pick the number of **clusters** for
    **unsupervised** methods like k-means — not k-NN's k.
17. (code) silhouette vs `n_clusters` for k-means on 3 well-separated blobs.
18. (md) **Read the output** — silhouette peaks at **3**, the true cluster count. That is a different
    question: k-means has no labels and asks "how many groups?"; k-NN has labels and asks "how many
    neighbours vote?" — chosen by CV against accuracy. Same letter k, different problems.
19. (md) **Your turn** — (a) why is reporting the tuned CV score optimistic, and what does nested CV
    fix? (b) on a 500-feature problem, which metric would you reach for first, and why? (c) you have
    labelled data and want k for k-NN — silhouette or cross-validation? why?
20. (md) **What you built** + vocabulary — Minkowski p & the unit ball; Mahalanobis (covariance-aware);
    cosine; metric × the curse; nested CV; silhouette (clusters) ≠ k-NN's k.
21. (md) **References** — C. Aggarwal, A. Hinneburg, D. Keim (2001), *On the surprising behavior of
    distance metrics in high dimensional space* (ICDT); K. Beyer et al. (1999); P. Mahalanobis (1936);
    P. Rousseeuw (1987), *Silhouettes*, J. Comp. Appl. Math. 20:53–65, DOI 10.1016/0377-0427(87)90125-7;
    ESL §13.3. `Previous: 05 — Demanding case` · `Next: Module 02 — Naive Bayes` (chapter 01 complete).

## Honest limits / no pre-emption

- This is the **optional advanced** capstone of the chapter; it deepens NB 2/NB 3, introduces no new
  method, and stays within k-NN.
- **Mahalanobis is shown geometrically, not as an accuracy win** — because a fair win requires the
  within-class covariance (LDA), which is out of scope; stated plainly rather than faked.
- The metric × curse effect is **real but secondary** to dimensionality reduction (NB 5) — stated, so
  the learner does not over-rate metric choice.
- Nested-CV gap is **small on this dataset** (0.967 vs 0.960); the notebook teaches the *principle*
  (the tuned score is biased up) and flags that the gap grows with bigger grids / smaller data.
- The silhouette section is a **conceptual clarification** (supervised CV vs unsupervised silhouette),
  not a clustering tutorial; k-means is used only to make the contrast concrete.

## Verification

Measured anchors (unit-ball geometry; Mahalanobis ellipse; near/far ratio p=0.5<1<2; metric×curse
accuracy p=1 0.877 vs p=2 0.842 at +1000; nested 0.960 vs naive 0.967; silhouette peak at 3) re-run in
the notebook and reconciled at build. Runs top-to-bottom (nbconvert to /tmp; output-free;
**`--clear-output --inplace` before commit**); `check_no_hardcoded_hex` passes; `gen_llms_txt` re-run;
`pytest` green (14); both reviewers pass (no BLOCK); Rémy validates visually; commit + merge `notebook
→ chapter`; **then chapter 01 closes via PR into `main`**.
