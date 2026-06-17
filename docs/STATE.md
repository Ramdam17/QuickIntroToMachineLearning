# STATE — where the course build is right now

> The compaction lifeline. Update this at **every** phase transition. On any session start / after a
> compaction, read this first, then `docs/WORKFLOW.md`, then the active plan. See the phase
> vocabulary in `docs/WORKFLOW.md`.

| Field | Value |
|---|---|
| Current chapter | `00_GettingStarted` (4 of 11 notebooks done) |
| Current notebook | `05_first_classifier_nearest_centroid` |
| Phase | `notebook-plan` |
| Active branch | `notebook/00_GettingStarted__05_first_classifier_nearest_centroid` |
| Active plan | `docs/plans/chapter_00_GettingStarted.md` (APPROVED) → notebook plan `docs/plans/00_GettingStarted__05_first_classifier_nearest_centroid.md` (pending, written on approval) |
| Next concrete action | In plan mode: draft NB 05 — nearest-centroid by hand (centroids = train class means via `groupby`; assign by nearest Euclidean distance), wrapped as a `NearestCentroidByHand` class with fit/predict (the estimator API), compared to `sklearn.neighbors.NearestCentroid`; decision boundary via `viz.plot_decision_boundary` (**extend the helper to accept a DataFrame X + string labels, pandas-first**, with a test); the honest fit→predict→evaluate loop on the NB 04 split (fraction right vs baseline, accuracy formalised NB 06); inductive bias (bisector = hyperplane; equal isotropic spread; a failure case) + scale-sensitivity (→ NB 11); signed-score teaser (→ NB 08). ExitPlanMode for approval; then build. |

## Notes / blockers

- NB 01 shipped: `ml_course.datasets.load_penguins()` + vendored `penguins.csv` + tests are in place
  and reusable by later notebooks. pandas-first convention recorded in CLAUDE.md/AGENTS.md.
- Through-line reminder: Palmer penguins (binary, 2 features), nearest-centroid first classifier (NB
  05), polynomial-degree complexity dial (NB 09–10).
- Confirmed: notebooks all-English; maths re-established not presupposed; git history = preserve
  per-notebook, mark chapters (`--no-ff` chapter → main).

## Progress log (most recent first)

- NB 04 (generalize, don't memorize — stratified split + rote-memorizer demo) built, reviewer-gated
  (both PASS; convergent honesty MINOR fixed), Rémy validated, merged.
- NB 03 (look before you model — EDA) built (+ `viz.plot_class_balance` / `plot_feature_histograms`
  + tests), reviewer-gated (both PASS; polish applied), Rémy validated, merged. Also fixed NB 01 c06's
  dangling "fuller dataset" forward-reference (flagged by pedagogy reviewer).
- NB 02 (features, labels, the feature space) built, reviewer-gated (both PASS; 2 minor polish
  applied), Rémy validated visually, committed and merged into `chapter/00_GettingStarted`.
- NB 01 (what is machine learning?) built, reviewer-gated (pedagogy PASS; ml-expert REVISE→fixed:
  corrected the figure reading, softened the line claim, added subset honesty), Rémy validated
  visually, committed and merged into `chapter/00_GettingStarted`.
- Chapter 00 plan approved and persisted; `course_map.md` 00 section rewritten to 11 notebooks.
- Chapter 00 plan reviewer-gated (both REVISE→incorporated).
- Course scaffold + build workflow set up on `main` (3 infra commits).
