# STATE — where the course build is right now

> The compaction lifeline. Update this at **every** phase transition. On any session start / after a
> compaction, read this first, then `docs/WORKFLOW.md`, then the active plan. See the phase
> vocabulary in `docs/WORKFLOW.md`.

| Field | Value |
|---|---|
| Current chapter | `00_GettingStarted` |
| Current notebook | — (about to open 01) |
| Phase | `chapter-plan-approved` |
| Active branch | `chapter/00_GettingStarted` |
| Active plan | `docs/plans/chapter_00_GettingStarted.md` (APPROVED, 11 notebooks) |
| Next concrete action | Open notebook 01: `git switch -c notebook/00_GettingStarted__01_what_is_ml` off the chapter branch; set phase `notebook-plan`; enter plan mode; draft the cell-by-cell notebook plan from the chapter plan's NB 01 row; Rémy approves; then build. |

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
