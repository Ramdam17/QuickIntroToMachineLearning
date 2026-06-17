# STATE — where the course build is right now

> The compaction lifeline. Update this at **every** phase transition. On any session start / after a
> compaction, read this first, then `docs/WORKFLOW.md`, then the active plan. See the phase
> vocabulary in `docs/WORKFLOW.md`.

| Field | Value |
|---|---|
| Current chapter | `00_GettingStarted` (2 of 11 notebooks done) |
| Current notebook | — (02 merged; next is 03) |
| Phase | `chapter-plan-approved` (ready to open the next notebook) |
| Active branch | `chapter/00_GettingStarted` |
| Active plan | `docs/plans/chapter_00_GettingStarted.md` (APPROVED) |
| Next concrete action | Open notebook 03: `git switch -c notebook/00_GettingStarted__03_look_before_you_model` off the chapter branch; phase `notebook-plan`; enter plan mode; draft the cell-by-cell plan from the chapter plan's NB 03 row (EDA: per-feature distributions/histograms, class balance, feature ranges/scales, re-read the scatter for overlap — pre-establishing the imbalance picture NB 06 needs and the scale picture NB 11 needs). Likely a small new `viz` helper (`plot_feature_histograms` / `plot_class_balance`) built just-in-time. Rémy approves; then build. |

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
