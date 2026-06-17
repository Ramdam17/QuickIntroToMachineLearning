# STATE — where the course build is right now

> The compaction lifeline. Update this at **every** phase transition. On any session start / after a
> compaction, read this first, then `docs/WORKFLOW.md`, then the active plan. See the phase
> vocabulary in `docs/WORKFLOW.md`.

| Field | Value |
|---|---|
| Current chapter | `00_GettingStarted` (1 of 11 notebooks done) |
| Current notebook | `02_features_and_feature_space` |
| Phase | `notebook-plan` |
| Active branch | `notebook/00_GettingStarted__02_features_and_feature_space` |
| Active plan | `docs/plans/chapter_00_GettingStarted.md` (APPROVED) → notebook plan `docs/plans/00_GettingStarted__02_features_and_feature_space.md` (pending, written on approval) |
| Next concrete action | In plan mode: draft the cell-by-cell plan for NB 02 from the chapter plan's NB 02 row — `X` (n×d) / `y`; feature types; row = example; the feature space as points; **the mean of a point cloud and Euclidean distance between two points** (the primitives NB 05's nearest-centroid needs), without yet building any classifier. ExitPlanMode for Rémy's approval; then write + commit the notebook plan and build. Reuse `ml_course.datasets` (penguins_xy); pandas-first (mean via `groupby`), numpy for distances. |

## Notes / blockers

- NB 01 shipped: `ml_course.datasets.load_penguins()` + vendored `penguins.csv` + tests are in place
  and reusable by later notebooks. pandas-first convention recorded in CLAUDE.md/AGENTS.md.
- Through-line reminder: Palmer penguins (binary, 2 features), nearest-centroid first classifier (NB
  05), polynomial-degree complexity dial (NB 09–10).
- Confirmed: notebooks all-English; maths re-established not presupposed; git history = preserve
  per-notebook, mark chapters (`--no-ff` chapter → main).

## Progress log (most recent first)

- NB 01 (what is machine learning?) built, reviewer-gated (pedagogy PASS; ml-expert REVISE→fixed:
  corrected the figure reading, softened the line claim, added subset honesty), Rémy validated
  visually, committed and merged into `chapter/00_GettingStarted`.
- Chapter 00 plan approved and persisted; `course_map.md` 00 section rewritten to 11 notebooks.
- Chapter 00 plan reviewer-gated (both REVISE→incorporated).
- Course scaffold + build workflow set up on `main` (3 infra commits).
