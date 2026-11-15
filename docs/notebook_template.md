# Notebook charter (ML course)

This charter says what every notebook of the course looks like. A notebook teaches **one concept**.
It is a plain `.ipynb` file, committed **without outputs**, and it runs from top to bottom on a
fresh kernel, including when its exercises are left unfilled.

**Nothing is presupposed.** A notion being well known is no reason to skip it: the notebook
re-establishes what it leans on.

## Voice

The voice is warm, empowering and celebratory, and it is rigorous. Celebrate what the learner has
achieved and frame difficulty as growth. Do not write "obviously", "simply", "trivially" or "just".
Do not use decorative emojis. Warmth never softens the science: no false praise, no hidden limit.
Everything in a notebook is in English: prose, code, identifiers and docstrings.

"You built" and "you wrote" are reserved for code the learner wrote in a cell to complete. For
code the notebook supplied, say what happened: "you ran", "you read", "you watched".

## Size and pace

A notebook has **25 to 30 cells**. When a concept needs more room, it is split into two notebooks;
it is never packed into denser cells. A method takes at least three notebooks and has no upper
limit: it takes as many notebooks as it has concepts.

The numbers in this charter (25 to 30 cells, 25 lines per code cell) are practical rules of this
course. They are not established results, and they are revised when a module shows they are badly
tuned.

## Code cells

- **A code cell does one thing.** The markdown cell before it says what that thing is.
- **Computation and plotting sit in separate cells.** A cell that computes a result does not draw
  it, and a cell that draws does not compute anything new.
- **A code cell has at most 25 lines of code.** Blank lines and comment-only lines do not count, so
  that an explanation is never cut to fit. The one exception is a single definition (one function
  or one class) that cannot be split without breaking it; the notebook plan names each exception.
- **Every numpy, pandas or matplotlib shortcut is explained at its first use in the module**, in
  one sentence, even when an earlier module introduced it. Broadcasting, boolean masks, `axis=`,
  slicing with `:`, `groupby` and `meshgrid` are shortcuts in this sense.

## Ideas and mathematics

- **A worked numeric example comes first.** Every new idea is carried out once on a single point,
  with plain numbers the learner can check by hand or with a calculator, before any vectorised
  code. The vectorised version then reproduces that number.
- **Every mathematical notion is re-established immediately before its first use**: what it is,
  what it measures, and one small example. This holds for notation as well (a sum sign, an index,
  a vector).
- **Formulas are written in LaTeX** (`$...$` and `$$...$$`), never in backticks or unicode. Each
  symbol is named in words the first time it appears.

## Exercises

Every exercise is **a code cell to complete**, followed by an automatic check and by its solution,
collapsed. An exercise asks only for what the notebook has taught. It takes three cells, in this
order.

1. **The cell to complete** (code). Its first line is the comment `# Exercise N`. The part the
   learner writes is marked `...  # YOUR CODE`.

   ```python
   # Exercise 1
   score = ...  # YOUR CODE
   ```

2. **The check** (code). It calls `ml_course.check`, which compares the answer with the expected
   value and answers with a kind message. On a cell left unfilled it reports "Not filled in yet"
   and does not raise.

   ```python
   check.equal("Exercise 1", score, expected=-0.25)
   ```

3. **The solution** (markdown), collapsed under "Show the solution". Its Python block is the whole
   cell to complete, filled in, starting with the same `# Exercise N` line. A short explanation of
   the reasoning may follow the block.

   ````markdown
   <details>
   <summary>Show the solution</summary>

   ```python
   # Exercise 1
   score = w1 * x1 + w2 * x2 + b
   ```

   The two products are 1.0 and -1.5. Their sum is -0.5, and the bias brings it to -0.25.

   </details>
   ````

Two rules keep the notebook runnable. **No later cell uses a name defined in a cell to complete**:
when the lesson needs the value, it computes it under its own name. And **the solution, put in
place of the cell to complete, makes the check pass**; `scripts/check_exercises.py` verifies it.

Exercises sit in two places. In the body, each by-hand build contains at least one cell the
learner completes, right after the worked example it rests on. At the end, the "Your turn" section
holds two or three exercises, from easy to harder.

## Cross-references

A cross-reference names the module and the notebook by their numbers, in one fixed wording. It is
plain text, not a link. The subject of the notebook may follow in parentheses:

```markdown
module 03, notebook 4 (gradient descent)
```

The same wording is used for the `Previous:` and `Next:` lines at the end of a notebook.
`scripts/check_crossrefs.py` verifies that every module and notebook number cited exists.

## Cell order

1. **Header** (markdown): `# NN — Title`, a one-line purpose, the **prerequisites** as
   cross-references, and **what you will be able to do** (three to five objectives, each starting
   with an action verb). A warm one-line welcome is encouraged.
2. **Imports** (code): standard library, third party, then `ml_course`. The random seed is fixed
   and documented. Call `viz.use_course_style()`. Never hardcode a colour: use `ml_course.colors`.
3. **Footing** (markdown): re-establish, briefly but genuinely, the ideas this notebook stands on.
   "You already know X" is not a recap.
4. **Body sections**, one step of the concept per section. Each section follows this path:
   - *Intuition* (markdown): the picture, before any formula.
   - *Worked numeric example* (markdown, then code): the idea on a single point, with plain numbers.
   - *Implementation by hand* (code): the mechanism, one thing per cell, with at least one cell to
     complete in each by-hand build.
   - *The library* (code): the real estimator, once the mechanism is understood, checked against
     the by-hand result.
   - *Read the figure* (markdown): after every figure, say kindly and concretely what we are
     looking at.
5. **Your turn** (markdown and code): two or three exercises in the form above.
6. **What you built** (markdown): the accomplishment in concrete bullets, in the wording the Voice
   section allows, and what the learner can now do.
7. **Going further** (markdown, optional): the formal or deeper view, clearly marked optional and
   never a prerequisite of the next notebook.
8. **References** (markdown): the cited papers with their DOI, then the `Previous:` and `Next:`
   lines.

## The three notebook roles

- **Fundamentals.** One concept per notebook, built by hand and shown in pictures. Afterwards the
  learner could explain the idea without help. A method has as many of these as it has concepts.
- **The method and its parameters.** The real estimator. For each hyperparameter: what it does,
  what too much and too little look like, and how to choose it on held-out data or by
  cross-validation. Failure modes are shown, not only mentioned.
- **The demanding case.** A realistic problem and a full honest workflow: look at the data,
  preprocess, choose, evaluate on held-out data, analyse the errors and state the limits.

## Figures

Use the `ml_course.viz` helpers and `ml_course.colors`. Dimensions, resolution and fonts come from
`viz.use_course_style()`. Every figure is followed by a "Read the figure" paragraph.

## Before commit

- All outputs are cleared.
- The notebook runs from top to bottom with its exercises unfilled.
- `uv run python scripts/check_no_hardcoded_hex.py` passes.
- `uv run python scripts/check_exercises.py` passes: every solution makes its check pass.
- `uv run python scripts/check_crossrefs.py` passes: every module and notebook number cited
  exists.
- `uv run python scripts/check_cells.py` reports no code cell above the limit, apart from the
  exceptions named in the notebook plan.
- `uv run python scripts/gen_llms_txt.py` is re-run when a notebook was added or renamed.
- `@ml-expert-reviewer` and `@pedagogy-reviewer` both pass, with no BLOCK and no MAJOR issue left
  open.
