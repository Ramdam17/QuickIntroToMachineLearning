# Build workflow — the chapter → notebook loop

The single procedure for building this course. It is designed to be **compaction-safe**: at any
moment, `docs/STATE.md` + the active plan under `docs/plans/` hold everything needed to resume. If
this document and your memory disagree, **these files win**.

## The three durable artifacts

| Artifact | Holds | Updated |
|---|---|---|
| `docs/STATE.md` | the live position: current chapter, notebook, phase, **next concrete action**, active branch, active plan | at **every** phase transition |
| `docs/plans/chapter_NN_Method.md` | the validated chapter plan (concepts → notebooks 1–3, notebook 4, notebook 5 example) | when the chapter plan is drafted / revised / approved |
| `docs/plans/NN_Method__MM_title.md` | the validated notebook content plan (cell-by-cell intent) | when the notebook plan is drafted / revised / approved |

## Phase vocabulary (the value in `STATE.md`)

```
idle                     between chapters; nothing in progress
chapter-plan             drafting/iterating the chapter plan (plan mode)
chapter-plan-review      reviewers passing over the chapter plan
chapter-plan-approved    chapter plan validated by Rémy; ready to plan notebook 1
notebook-plan            drafting/iterating a notebook plan (plan mode)
notebook-plan-approved   notebook plan validated by Rémy; ready to build
notebook-build           building the notebook
notebook-review          reviewers passing over the built notebook
notebook-revise          applying review modifications
notebook-visual-check    awaiting Rémy's visual validation of the executed notebook
notebook-commit          doc update + clear outputs + guards + commit + merge to chapter
chapter-merge            chapter complete; merging chapter → main
```

## Branch & history model

- Chapter branch off `main`: `chapter/NN_Method` (e.g. `chapter/01_KNN`).
- Notebook branch off the **chapter** branch: `notebook/NN_Method__MM_title`
  (e.g. `notebook/01_KNN__01_see_knn`).
- `notebook → chapter`: keep the notebook's commits (no squash).
- `chapter → main`: **`--no-ff` merge commit** — preserves per-notebook history; each chapter is a
  visible unit.
- Never force-push `main`. Never merge a `wip` commit to `main`.

## Chapter loop

1. **Open the chapter.** `git switch main && git switch -c chapter/NN_Method`. Set `STATE.md`:
   chapter = NN_Method, phase = `chapter-plan`, next action = "draft chapter plan in plan mode".
2. **Plan (plan mode).** Enter plan mode. Draft `docs/plans/chapter_NN_Method.md` deciding together:
   1. the **primordial concepts** and how they distribute across **notebooks 1–3** (one concept per
      notebook),
   2. the **content of notebook 4** (the real estimator, its hyperparameters, failure modes, tuning),
   3. the **example for notebook 5** (the demanding practical case).
   Cross-check `docs/course_map.md`; refine it if the plan changes the promised coverage.
3. **Review the plan.** Phase = `chapter-plan-review`. Dispatch BOTH `@ml-expert-reviewer` and
   `@pedagogy-reviewer` on `docs/plans/chapter_NN_Method.md`. Apply fixes until no BLOCK.
4. **Validate.** Present the plan; Rémy approves (ExitPlanMode). Commit the plan
   (`docs(plan): chapter NN_Method`). Phase = `chapter-plan-approved`.

## Notebook loop (repeat for each notebook in the chapter)

1. **Open the notebook.** `git switch chapter/NN_Method && git switch -c notebook/NN_Method__MM_title`.
   Set `STATE.md`: notebook = MM_title, phase = `notebook-plan`, active branch, next action.
2. **Plan (plan mode).** Enter plan mode. Draft `docs/plans/NN_Method__MM_title.md`: the cell-by-cell
   intent following `docs/notebook_template.md` (~20 cells; header, recap, body sections with
   intuition → implementation → "Read the figure", Your turn, What you built, References).
3. **Validate the notebook plan.** Rémy approves (ExitPlanMode). Phase = `notebook-plan-approved`.
4. **Build.** Phase = `notebook-build`. Create the `.ipynb` (use `NotebookEdit`). One concept; by
   hand before library; `viz.use_course_style()`; colours from `ml_course.colors`; seeds fixed;
   every figure followed by a "Read the figure" paragraph; everything in English.
5. **Review.** Phase = `notebook-review`. Dispatch BOTH `@ml-expert-reviewer` and
   `@pedagogy-reviewer` on the notebook. A notebook passes only when **both** return no BLOCK.
6. **Revise.** Phase = `notebook-revise`. Apply the fixes. Re-review if a BLOCK was raised.
7. **Visual check.** Phase = `notebook-visual-check`. Rémy runs and visually validates the executed
   notebook. Iterate until agreed. (Do not commit before this validation.)
8. **End-of-notebook doc update + commit** (phase = `notebook-commit`) — run the checklist below,
   commit on the notebook branch, then merge into the chapter branch keeping the commits:
   `git switch chapter/NN_Method && git merge notebook/NN_Method__MM_title` (fast-forwards, so the
   notebook's commits land on the chapter branch). Update `STATE.md` to the next notebook, or to
   `chapter-merge` if this was the last notebook.

### End-of-notebook checklist (before the commit)

- [ ] **Clear all outputs** from the notebook.
- [ ] `uv run python scripts/check_no_hardcoded_hex.py` passes.
- [ ] `uv run python scripts/gen_llms_txt.py` re-run (the **LLM doc / machine index**).
- [ ] `uv run pytest` green (if the notebook added/changed any `src/` code).
- [ ] `docs/course_map.md` reflects what was actually built (mark the notebook done / refine).
- [ ] `docs/common_errors.md` extended if the notebook surfaced a new intuition trap.
- [ ] `docs/STATE.md` updated.
- [ ] Both reviewers passed (no BLOCK) and Rémy validated visually.
- [ ] Commit on the notebook branch: `feat(NN_method): notebook MM — <title>`, then merge into the
      chapter branch (keep commits).

## Chapter close

- Phase = `chapter-merge`. Ensure every notebook of the chapter is merged into the chapter branch and
  `llms.txt` is current.
- `git switch main && git merge --no-ff chapter/NN_Method` with a message
  `feat(NN_method): complete chapter — <method>`.
- Set `STATE.md`: phase = `idle`, next action = "open chapter (NN+1)".

## Reviewer gate (both levels)

- **Plan level:** reviewers read the plan markdown — pedagogy checks progression/exhaustiveness/no
  concept gap; ML expert checks correctness, honest limits, that notebook 5's example actually
  mobilizes the method and admits an honest evaluation.
- **Notebook level:** reviewers read the `.ipynb` — same dimensions, against the built artifact.
- Dispatch both **in parallel**. A plan or notebook advances only when **neither** returns a BLOCK.

## Compaction-resume protocol

On any session start, after a compaction, or after `/clear`:

1. Read `docs/STATE.md` → note chapter, notebook, **phase**, **next concrete action**, active branch.
2. `git status` / `git branch --show-current` to confirm you are on the active branch.
3. Read the active plan named in `STATE.md`.
4. Resume **exactly** at the named next action. Do not restart an earlier phase.

## Plan mode

Both planning steps (chapter, notebook) use plan mode: enter plan mode, draft the plan into
`docs/plans/`, run the reviewer gate, then ExitPlanMode for Rémy's approval. The plan file — not the
plan-mode buffer — is the durable record, so a compaction during planning loses nothing already
written to `docs/plans/`.
