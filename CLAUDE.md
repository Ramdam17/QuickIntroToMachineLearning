# CLAUDE.md — Quick Intro to Machine Learning (course build rules)

> Project-level rules for **building** this course. Global preferences in `~/.claude/CLAUDE.md`
> also apply (uv, ruff/black, never silence console output, ask before unsolicited changes, cite
> literature). When this file and the global file disagree on course-specific matters, this file
> wins; on everything else the global file wins.
>
> **Scope split.** This file governs agents *building* the course. `AGENTS.md` governs an
> assistant *tutoring a learner* through the published course. When both could apply to you, decide
> by task: building/restructuring → here; helping a learner → `AGENTS.md`.

## READ FIRST — every session, especially after a compaction / `/clear` / new conversation

Before doing anything, read, in order:

1. **`docs/STATE.md`** — where we are right now (current chapter, notebook, phase) and the **next
   concrete action**. This is the compaction lifeline.
2. **`docs/WORKFLOW.md`** — the chapter → notebook build loop, the git branch model, the review
   gates, and the end-of-notebook / end-of-chapter checklists.
3. The **active plan** named in `STATE.md` (under `docs/plans/`), if a chapter or notebook is in
   progress.

A mid-flight compaction must never disrupt the work: `STATE.md` + the active plan hold everything
needed to resume. **If your memory of the plan disagrees with these files, the files win.**

## What this course is

A hands-on, **learn-by-doing** introduction to twelve classical ML methods, for **young
developers** — comfortable with code, with mathematics welcomed and **always re-established from
scratch, never presupposed**. Each idea is built by hand, observed, then generalized.

Twelve methods, one module each, plus a getting-started module. Per method: **3 to 5 notebooks**,
~20 cells each.

## The per-method arc (non-negotiable shape)

For each method, the notebook series follows this arc:

1. **Notebooks 1–3 — fundamentals.** Build the core intuition of the method one concept at a time
   (one concept per notebook). The learner *does the thing by hand* before any library call.
2. **Notebook 4 — the method and its parameters.** Explore the real estimator: its hyperparameters,
   what each one does, how it fails, how to tune it.
3. **Notebook 5 — a demanding practical case.** A realistic, non-trivial problem that mobilizes
   everything. This is where rigor bites: honest evaluation, error analysis, stated limits.

Three notebooks is the floor (a method may not need 5); five is the ceiling. Never fewer than 3.

## Non-negotiable rules

- **One concept = one notebook.** Short focused notebooks build to the demanding case. Never cram
  several new concepts into one notebook.
- **Build one notebook at a time. Commit per notebook.** NEVER bulk-generate many notebooks in one
  unreviewed pass — that is the classic failure mode.
- **Two-reviewer gate (mandatory before each commit).** Every notebook is validated by BOTH:
  - `@ml-expert-reviewer` — refuses oversimplification; checks ML correctness, honest assumptions
    and limits, sound evaluation, citations.
  - `@pedagogy-reviewer` — checks progression, prerequisites, exhaustiveness (no concept gap),
    voice and charter compliance, exercise quality.
  A notebook ships only when both pass. **Making something understandable is an effort of
  presentation and accompaniment — never a simplification.**
- **Voice = warm, empowering, celebratory — AND rigorous.** Celebrate what the learner just
  achieved; frame difficulty as growth, never a wall. Be kind, never condescending. **Ban
  "obviously / simply / trivially / just" / "il suffit de / évidemment / il n'y a qu'à".** Warmth
  never softens the science (no false praise). **No decorative emojis** — warmth lives in the words.
- **Never presuppose prior knowledge.** A thing being well known is no reason to skip re-laying it.
  Re-establish each concept the series depends on, briefly but genuinely.
- **Structure: intuition → implementation → interpretation.** Every figure is followed by a
  **"Read the figure"** paragraph that says, kindly and concretely, what we are looking at.
- **Graphic charter:** the palette lives only in `ml_course.colors`; **no hardcoded hex** in
  notebooks or modules (enforced by `scripts/check_no_hardcoded_hex.py`). Call
  `ml_course.viz.use_course_style()`; fixed dims/DPI/fonts from the charter.
- **Pandas-first.** Use pandas (DataFrame/Series) as the default data interface — loading, tabular
  display, selection, `value_counts`, `describe`, `groupby` for class means — keeping **numpy under
  the hood** where numpy is right (distances, `meshgrid`, linear algebra). Datasets return
  DataFrames; `viz` helpers accept DataFrames/Series and use column names for labels.
- **Notebooks are output-free in git** (clear outputs before committing) and didactic, per the
  `notebook-quality` standard.
- **Rigor:** fix and document random seeds; validate numbers against known/closed forms where they
  exist; flag heuristic vs. established choices; cite sources with DOI in docstrings and references.

## Build workflow (chapter → notebook loop)

The full procedure, git commands, gates, and checklists live in **`docs/WORKFLOW.md`**; the live
position lives in **`docs/STATE.md`**. The shape:

- **Each chapter = a git branch** `chapter/NN_Method` off `main`. **Each notebook = its own branch**
  `notebook/NN_Method__MM_title` off the chapter branch.
- **Chapter planning (plan mode).** Plan the chapter together: (1) the primordial concepts and how
  they distribute across notebooks 1–3, (2) the content of notebook 4 (the method & its parameters),
  (3) the example for notebook 5 (the demanding case). The plan is written to `docs/plans/`. Both
  reviewers review the plan. Rémy validates before any notebook is built.
- **Notebook planning (plan mode).** For each notebook, plan its content cell-by-cell together;
  Rémy validates the plan before building.
- **Build → review → revise → visual check → commit.** Build the notebook; both reviewers pass over
  it (`@ml-expert-reviewer` + `@pedagogy-reviewer`, no BLOCK); apply fixes; **Rémy validates the
  notebook visually**; then update docs (incl. the LLM index `llms.txt`), clear outputs, run guards,
  commit the notebook, and merge `notebook → chapter`.
- **Chapter done → merge `chapter → main`** with a `--no-ff` merge commit (preserve per-notebook
  history; each chapter is a visible unit).
- **Update `docs/STATE.md` at every phase transition**, and the docs at every notebook end.

Never skip a gate, never bulk-build. One notebook at a time, validated by both reviewers and by
Rémy, before it ships.

## Course content language

**Everything in the notebooks is in English** — learner-facing prose, code, identifiers,
docstrings, comments — as are all instruction files (this file, `AGENTS.md`, agent definitions).

## Numbering & folders

`notebooks/NN_MethodName/MM_snake_title.ipynb` — plain `.ipynb` (not Jupyter Book), per-module
numbering. Module order is a pedagogical progression (instance-based → probabilistic → linear →
trees → margins → ensembles → neural):

```
00_GettingStarted        what ML is, the train/test split, a first classifier end-to-end
01_KNN                   k-nearest neighbours
02_NaiveBayes            naive Bayes
03_LogisticRegression    logistic regression
04_DecisionTree          decision trees
05_SVM                   support vector machines
06_RandomForest          random forests (bagging)
07_AdaBoost              adaptive boosting
08_GradientBoosting      gradient boosting
09_XGBoost               XGBoost
10_LightGBM              LightGBM
11_MLP                   multi-layer perceptron
12_NeuralNetworks        neural networks
```

The full per-method plan lives in `docs/course_map.md`; the cell-by-cell charter in
`docs/notebook_template.md`. Regenerate the machine index after adding notebooks:
`uv run python scripts/gen_llms_txt.py`.

## Running things

```bash
uv sync --extra dev --extra boosting   # set up the environment
uv run pytest                          # run the tests
uv run ruff check . && uv run black --check .
uv run jupyter lab                     # open the notebooks
```
