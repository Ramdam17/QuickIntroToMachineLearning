# STATE — where the course build is right now

> The compaction lifeline. Update this at **every** phase transition. On any session start / after a
> compaction, read this first, then `docs/WORKFLOW.md`, then the active plan. See the phase
> vocabulary in `docs/WORKFLOW.md`.

| Field | Value |
|---|---|
| Current chapter | `00_GettingStarted` (3 of 11 notebooks done) |
| Current notebook | `04_generalize_dont_memorize` |
| Phase | `notebook-plan` |
| Active branch | `notebook/00_GettingStarted__04_generalize_dont_memorize` |
| Active plan | `docs/plans/chapter_00_GettingStarted.md` (APPROVED) → notebook plan `docs/plans/00_GettingStarted__04_generalize_dont_memorize.md` (pending, written on approval) |
| Next concrete action | In plan mode: draft NB 04 from the chapter plan's NB 04 row — **stratified** `train_test_split`; the cardinal sin (scoring on training data); i.i.d. as a *chosen* assumption (penguins span islands/years); leakage intro; preview fit→predict→evaluate. Open fork to settle with Rémy: demonstrate memorize≠generalize via a deliberately silly "rote memorizer" (100% train / chance test), or stay model-free (split mechanics + analogy, deferring fit/predict to NB 05). Keep "accuracy" informal here (fraction correct), formalised in NB 06. ExitPlanMode for approval; then build. |

## Notes / blockers

- NB 01 shipped: `ml_course.datasets.load_penguins()` + vendored `penguins.csv` + tests are in place
  and reusable by later notebooks. pandas-first convention recorded in CLAUDE.md/AGENTS.md.
- Through-line reminder: Palmer penguins (binary, 2 features), nearest-centroid first classifier (NB
  05), polynomial-degree complexity dial (NB 09–10).
- Confirmed: notebooks all-English; maths re-established not presupposed; git history = preserve
  per-notebook, mark chapters (`--no-ff` chapter → main).

## Progress log (most recent first)

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
