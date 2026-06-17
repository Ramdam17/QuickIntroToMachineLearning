# STATE — where the course build is right now

> The compaction lifeline. Update this at **every** phase transition. On any session start / after a
> compaction, read this first, then `docs/WORKFLOW.md`, then the active plan. See the phase
> vocabulary in `docs/WORKFLOW.md`.

| Field | Value |
|---|---|
| Current chapter | `00_GettingStarted` (2 of 11 notebooks done) |
| Current notebook | `03_look_before_you_model` |
| Phase | `notebook-plan` |
| Active branch | `notebook/00_GettingStarted__03_look_before_you_model` |
| Active plan | `docs/plans/chapter_00_GettingStarted.md` (APPROVED) → notebook plan `docs/plans/00_GettingStarted__03_look_before_you_model.md` (pending, written on approval) |
| Next concrete action | In plan mode: draft the cell-by-cell plan for NB 03 (EDA) from the chapter plan's NB 03 row — class balance, per-feature distributions (histograms split by class), feature ranges/scales (`describe`), re-read the scatter; pre-establish imbalance (→ NB 06) and scale (→ NB 11). Plan the just-in-time `viz` helpers `plot_class_balance` + `plot_feature_histograms` (ax/Figure, colours from `ml_course.colors`) + smoke tests. ExitPlanMode for Rémy's approval; then write + commit the plan and build. |

## Notes / blockers

- NB 01 shipped: `ml_course.datasets.load_penguins()` + vendored `penguins.csv` + tests are in place
  and reusable by later notebooks. pandas-first convention recorded in CLAUDE.md/AGENTS.md.
- Through-line reminder: Palmer penguins (binary, 2 features), nearest-centroid first classifier (NB
  05), polynomial-degree complexity dial (NB 09–10).
- Confirmed: notebooks all-English; maths re-established not presupposed; git history = preserve
  per-notebook, mark chapters (`--no-ff` chapter → main).

## Progress log (most recent first)

- NB 02 (features, labels, the feature space) built, reviewer-gated (both PASS; 2 minor polish
  applied), Rémy validated visually, committed and merged into `chapter/00_GettingStarted`.
- NB 01 (what is machine learning?) built, reviewer-gated (pedagogy PASS; ml-expert REVISE→fixed:
  corrected the figure reading, softened the line claim, added subset honesty), Rémy validated
  visually, committed and merged into `chapter/00_GettingStarted`.
- Chapter 00 plan approved and persisted; `course_map.md` 00 section rewritten to 11 notebooks.
- Chapter 00 plan reviewer-gated (both REVISE→incorporated).
- Course scaffold + build workflow set up on `main` (3 infra commits).
