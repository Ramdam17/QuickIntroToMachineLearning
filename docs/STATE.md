# STATE — where the course build is right now

> The compaction lifeline. Update this at **every** phase transition. On any session start / after a
> compaction, read this first, then `docs/WORKFLOW.md`, then the active plan. See the phase
> vocabulary in `docs/WORKFLOW.md`.

| Field | Value |
|---|---|
| Current chapter | — (none started) |
| Current notebook | — |
| Phase | `idle` |
| Active branch | `main` |
| Active plan | — |
| Next concrete action | Open the first chapter: `00_GettingStarted`. Create branch `chapter/00_GettingStarted`, enter plan mode, draft `docs/plans/chapter_00_GettingStarted.md`. |

## Notes / blockers

- Scaffold committed on `main` (`chore(scaffold)`). Build workflow encoded; nothing in progress yet.
- Confirmed: notebooks all-English; audience = young developers, maths re-established not presupposed;
  git history = preserve per-notebook, mark chapters (`--no-ff` chapter → main).

## Progress log (most recent first)

- Course scaffold + build workflow set up; awaiting kickoff of chapter `00_GettingStarted`.
