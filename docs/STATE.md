# STATE — where the course build is right now

> The compaction lifeline. Update this at **every** phase transition. On any session start / after a
> compaction, read this first, then `docs/WORKFLOW.md`, then the active plan. See the phase
> vocabulary in `docs/WORKFLOW.md`.

| Field | Value |
|---|---|
| Current chapter | `00_GettingStarted` |
| Current notebook | `01_what_is_ml` |
| Phase | `notebook-plan` |
| Active branch | `notebook/00_GettingStarted__01_what_is_ml` |
| Active plan | `docs/plans/chapter_00_GettingStarted.md` (APPROVED) → notebook plan `docs/plans/00_GettingStarted__01_what_is_ml.md` (pending, written on approval) |
| Next concrete action | In plan mode: draft the cell-by-cell plan for NB 01 (from the chapter plan's NB 01 row: what ML is / supervised / classification vs regression / meet penguins + toolkit + "read a figure" + version echo); ExitPlanMode for Rémy's approval; then write + commit the notebook plan and build the `.ipynb`. NB 01 needs `ml_course.datasets.load_penguins()` + vendored `penguins.csv` + `.gitignore` exception, built just-in-time. |

## Notes / blockers

- Chapter 00 plan APPROVED (11 notebooks). Through-line: Palmer penguins (binary, 2 features),
  nearest-centroid first classifier, polynomial-degree complexity dial (NB 09–10).
- First library need (NB 01): `ml_course.datasets.load_penguins()` + vendored `penguins.csv` +
  `.gitignore` exception. Build it just-in-time within the NB 01 work.
- Confirmed: notebooks all-English; maths re-established not presupposed; git history = preserve
  per-notebook, mark chapters (`--no-ff` chapter → main).

## Progress log (most recent first)

- Chapter 00 plan approved and persisted; `course_map.md` 00 section rewritten to 11 notebooks.
- Chapter 00 plan reviewer-gated (ml-expert + pedagogy, both REVISE→incorporated: EDA notebook added,
  mean/distance footing, affine signed-distance score, bias–variance precision, stratification).
- Opened chapter `00_GettingStarted` (branch created); chapter planning.
- Course scaffold + build workflow set up on `main` (3 infra commits).
