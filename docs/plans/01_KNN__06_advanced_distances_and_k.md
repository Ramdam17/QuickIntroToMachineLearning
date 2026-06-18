# Notebook plan — 01_KNN / 06_advanced_distances_and_k  (v2 — rebuilt from scratch)

> Status: **APPROVED** (2026-06-18, by Rémy). v1 was scrapped (too few visualizations); this v2 is
> visualization-first and freed from the ~20-cell ceiling. Build via `uv run python - <
> /tmp/build_nb6.py` (stdin) to dodge the stray `/tmp/struct.py` shadow.

## Context

The chapter's optional **Advanced** capstone, **rebuilt from scratch** after v1 was scrapped: v1 was
table-heavy and visually poor for a notebook whose whole subject — the *geometry of distance* — begs
to be **seen**. This version is **visualization-first** (nine figures), features **both** course
datasets (penguins for 2-D geometry, breast cancer for high-D), and is **freed from the ~20-cell
ceiling** (≈28–30 cells; that ceiling is a floor for fundamentals, not a cap on a serious capstone). It
answers Rémy's two questions: **(1) does the *definition* of distance matter?** and **(2) how do you
choose k rigorously?** Deepens NB 2 (distance) and NB 3 (choosing k); no new method. Prereqs: NB 1–5.

## Design (every figure below was generated and visually inspected at plan time)

**Part 1 — a distance is a choice, and it has a geometry that reaches the decision boundary.**
- **Unit balls** {‖x‖_p = 1}: diamond (p=1) / circle (p=2) / square (p=∞).
- **The metric reshapes the decision boundary.** k-NN(15) boundaries on **moons** (overlap reveals it)
  for p=1 / 2 / ∞: Manhattan gives axis-aligned, staircase segments; Euclidean is smooth; Chebyshev is
  blocky/diagonal — *confirmed visually distinct*. (moons, not penguins, because penguins separate so
  cleanly the three boundaries nearly coincide — itself the Part-3 point.)

**Part 2 — Mahalanobis: a distance that learns the data's shape (on the penguins).**
- Standardization (NB 2) equalizes *scale* but not *correlation*; penguins' two features correlate
  **0.869** even standardized. **Unit ball → ellipse:** Euclidean circle vs Mahalanobis ellipse on the
  standardized penguin cloud (tilts ~45° along the cloud). *Confirmed.*
- **The consequence — the boundary reshapes:** Euclidean vs Mahalanobis **k-NN decision boundary** on
  the penguins, side by side — Mahalanobis bends the boundary along the correlation. *Confirmed
  visually distinct.* Honest: both separate the penguins well; the metric changes the *shape* of the
  decision, shown as geometry — a fair accuracy contest needs the within-class covariance (LDA), out of
  scope. **Cosine** named (angle, not magnitude).

**Part 3 — when does the metric matter? Low vs high dimension — both datasets.**
- **The curse, made visceral:** histogram of pairwise-distance / mean-distance at **d = 2 / 50 / 1000**
  (pure noise). At d=2 distances span 0–4× the mean; at d=1000 they collapse onto 1 — "nearest" ≈
  "farthest". *Confirmed striking.* (Charter colours in the build.)
- **near/far ratio by Minkowski p** (0.5 / 1 / 2) vs dims: smaller p concentrates more slowly
  (p=0.5 < 1 < 2 at every dim) — the robust mechanism (Aggarwal 2001; a simpler cousin of their
  *relative contrast*, which favours fractional p<1).
- **Accuracy, both datasets:** **penguins** = a *wash* (Manhattan/Euclidean/Chebyshev all **0.993** at
  k=5). **breast cancer + noise** (k=7, **averaged over 8 noise draws**, mean±std): a two-line curve,
  p=1 vs p=2, vs #noise-dims {0,50,100,200,500,1000} = (0.947/0.947), (0.938/0.928), (0.928/0.909),
  (0.920/0.902), (0.885/0.858), (**0.839/0.803**). p=1 ≥ p=2 at every level and the **gap widens with
  dimension**. The single-draw spread (±0.02–0.04) is comparable to the effect — so we average and say
  so. **Lesson:** the metric is a wash in low dimensions (penguins) but matters — modestly, on
  average — in high ones (noisy breast cancer); reducing dimensions (NB 5) is the bigger lever.

**Part 4 — choosing k rigorously: nested cross-validation.**
- A **schematic** of the nested loops (outer folds for estimation; inside each, an inner CV picks k).
- **The winner's curse, visualized:** the 5 outer-fold scores [0.938, 0.988, 0.938, 0.975, 0.962] as a
  strip/box, with the **nested mean 0.960** and the **optimistic naive best-inner-CV 0.967** marked —
  the naive number sits above the honest mean and ignores the fold-to-fold spread. *Confirmed.*

## Library additions / figures

**None to `src/`.** Reuse `viz.plot_decision_boundary` (metric & Mahalanobis boundaries),
`viz.use_course_style`, `ml_course.colors`, `ml_course.datasets.penguins_xy`. sklearn:
`KNeighborsClassifier` (Minkowski p, `metric="mahalanobis"`/`"chebyshev"`), `GridSearchCV`,
`cross_val_score`, `StratifiedKFold`, `train_test_split`, `StandardScaler`, `make_pipeline`,
`pairwise_distances`, `make_moons`, `load_breast_cancer`. Unit balls, concentration histogram, ratio
curve, accuracy curve, nested-CV schematic & fold-strip are one-off in-notebook figures (charter
colours). `pytest` stays 14.

## Cell-by-cell (~28–30 cells; visualization-first; "Read the figure" after every figure)

1. (md) **Header** — `# 06 — Advanced: distances & choosing k`; *notebook 6 of 6 (optional, advanced)*;
   purpose; `Prerequisites: 01–05`; objectives; a note that this is a longer, optional deep dive.
2. (code) imports + seed + `use_course_style()` + load penguins & make_moons & breast_cancer up front;
   print their shapes.
3. (md) **Part 1 — a distance is a choice.** Minkowski p (formula); the unit ball.
4. (code) unit balls p=1/2/∞.
5. (md) Read the figure — diamond/circle/square; the metric is the shape of a neighbourhood.
6. (md) **From the ball to the boundary** — that shape propagates: k-NN's boundary inherits the
   metric's geometry. We use moons (overlap makes it visible).
7. (code) k-NN(15) boundary on moons, p=1/2/∞ (3 panels).
8. (md) Read the figure — Manhattan's axis-aligned steps, Euclidean's smooth curve, Chebyshev's blocky
   diagonal: the metric's signature, in the decision itself.
9. (md) **Part 2 — Mahalanobis: a distance that learns the data's shape.** Scale vs correlation; the
   penguins correlate 0.869 even standardized; Mahalanobis whitens by Σ.
10. (code) Mahalanobis ellipse vs Euclidean circle on standardized penguins.
11. (md) Read the figure — the ellipse tilts along the cloud; that tilt is the correlation
    standardization left behind.
12. (code) Euclidean vs Mahalanobis **k-NN boundary** on penguins (2 panels).
13. (md) Read the figure — Mahalanobis bends the boundary along the correlation. Honest note (geometry,
    not an accuracy race; LDA out of scope). **Cosine** named.
14. (md) **Part 3 — when does the metric matter? It depends on the dimension.** Set up the question on
    both datasets; recall the curse (NB 5) and Aggarwal's prediction.
15. (code) distance-concentration histogram, d = 2 / 50 / 1000 (charter colours).
16. (md) Read the figure — at d=1000 all distances collapse onto the mean: "nearest" stops carrying
    information.
17. (code) near/far ratio by p (0.5/1/2) vs dims.
18. (md) Read the figure — smaller p concentrates more slowly (Aggarwal 2001; relative contrast;
    fractional p).
19. (code) accuracy: penguins (all metrics ≈ 0.993) printed; breast_cancer + noise, p=1 vs p=2,
    averaged over draws → the two-line curve vs #noise-dims.
20. (md) Read the figure/output — penguins a wash (low-d); breast cancer the gap opens with dimension
    (p=1 ahead, modest, averaged, per-draw noisy). The contrast **is** the lesson; NB 5 is the bigger
    lever.
21. (md) **Part 4 — choosing k rigorously: nested cross-validation.** The winner's curse of reporting
    a tuned CV score.
22. (code) nested-CV **schematic** (drawn rectangles: outer folds; inner CV inside one).
23. (md) Read the figure — inner loop picks k, outer loop estimates on data the tuning never saw.
24. (code) naive (`GridSearchCV.best_score_`) vs nested (`cross_val_score` of the grid) + the 5
    outer-fold scores; figure = fold scores as a strip with naive (0.967) and nested-mean (0.960) lines.
25. (md) Read the figure — the naive point sits above the honest mean and hides the fold-to-fold
    spread; nested CV reports both. NB 5's single sealed test is the simplest honest estimate.
26. (md) **Your turn** — (a) why the tuned CV score is optimistic & what nested CV fixes; (b) what
    decides whether the metric matters (low-d vs high-d); (c) which p first on 500 features, and what
    helps more (NB 5); (d) Mahalanobis reshaped the boundary but barely moved accuracy on penguins — why?
27. (md) **What you built** + vocabulary (Minkowski p / unit ball; metric → boundary shape;
    Mahalanobis; cosine; distance concentration / metric × dimension; nested CV) + the **chapter close**.
28. (md) **References** — Aggarwal 2001; Beyer 1999; Mahalanobis 1936; ESL §13.3 (+ §7.10 for CV).
    `Previous: 05` · `Next: Module 02 — Naive Bayes` (chapter 01 complete).

## Honest limits / no pre-emption

- Optional advanced capstone; deepens NB 2/3; no new method; LDA explicitly out of scope.
- **Mahalanobis is shown as geometry + boundary shape, not a faked accuracy win** (within-class
  covariance / LDA out of scope) — stated.
- **Metric-accuracy effect is modest and per-draw noisy** — shown by averaging (mean ± std) and a
  curve; the robust mechanism is the near/far ratio. Metric is *secondary* to dimensionality reduction
  (NB 5) — stated.
- moons is used for the metric-boundary panels (separable penguins would hide the effect — itself the
  Part-3 lesson); penguins carry Mahalanobis + the low-d wash; breast cancer carries the high-d curve.
- No silhouette (it concerns k-means cluster count, not k-NN's k — dropped in v1 review as out-of-scope
  noise for a learner).

## Verification

All nine figures were generated and inspected at plan time. At build, every number is re-run and
reconciled (penguins corr 0.869; metrics 0.993; accuracy curve incl. 0.839/0.803 at +1000; near/far
p=0.5<1<2; nested 0.967 vs 0.960, folds [0.938,0.988,0.938,0.975,0.962]). Runs top-to-bottom
(nbconvert to /tmp; output-free; `--clear-output --inplace` before commit); `check_no_hardcoded_hex`
passes (charter colours in every figure); `gen_llms_txt` re-run; `pytest` green (14); both reviewers
pass (no BLOCK); Rémy validates visually; commit + merge `notebook → chapter`; **then chapter 01
closes via PR into `main`**.
