# STATE — where the course build is right now

> The compaction lifeline. Update this at **every** phase transition. On any session start / after a
> compaction, read this first, then `docs/WORKFLOW.md`, then the active plan. See the phase
> vocabulary in `docs/WORKFLOW.md`.

| Field | Value |
|---|---|
| Current chapter | `00_GettingStarted` |
| Current notebook | — |
| Phase | `chapter-plan` |
| Active branch | `chapter/00_GettingStarted` |
| Active plan | `docs/plans/chapter_00_GettingStarted.md` (pending — written on approval) |
| Next concrete action | In plan mode: resolve the chapter-00 forks (notebook count/shape; the "first classifier"; the dataset), draft the chapter plan, run the reviewer gate (both reviewers), ExitPlanMode for Rémy's approval, then write + commit the plan and set phase `chapter-plan-approved`. |

## Notes / blockers

- `00_GettingStarted` is the foundational module; the per-method arc (NB 1–3 / NB 4 params / NB 5
  case) does not map cleanly — it has its own shape (to be settled in the plan).
- Confirmed: notebooks all-English; audience = young developers, maths re-established not presupposed;
  git history = preserve per-notebook, mark chapters (`--no-ff` chapter → main).

## Progress log (most recent first)

- Opened chapter `00_GettingStarted` (branch created); entering chapter planning.
- Course scaffold + build workflow set up on `main` (3 infra commits).
