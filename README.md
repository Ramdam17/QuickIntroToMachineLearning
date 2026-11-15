# Quick Intro to Machine Learning — a Hands-On Course

A **learn-by-doing** journey through twelve classical machine-learning methods, for developers who
want to *understand* them rather than only call them. Every method is built by hand first,
observed, then generalized, before the real library estimator appears.

The course is built **one concept per notebook**: focused notebooks build progressively, so each
idea lands before the next arrives. You can follow it on your own, with nothing but this
repository, or with an AI assistant open alongside (see [`AGENTS.md`](AGENTS.md)).

## The twelve methods

k-Nearest Neighbours · Naive Bayes · Logistic Regression · Decision Trees · Support Vector Machines ·
Random Forests · AdaBoost · Gradient Boosting · XGBoost · LightGBM · Multi-Layer Perceptron ·
Neural Networks.

Ordered as a progression — instance-based → probabilistic → linear → trees → margins → ensembles →
neural — so each method leans on the intuition built by the ones before it. A getting-started
module comes first: what machine learning is, the train/test split, a first classifier from end to
end, and how to measure it honestly.

## How each method is taught

Each method gets at least three notebooks, and as many as it has concepts, along one arc:

1. **Fundamentals** — the core intuition, one concept per notebook, built by hand.
2. **The method and its parameters** — the real estimator, its hyperparameters, how it fails and
   how to tune it.
3. **A demanding practical case** — a realistic problem with honest evaluation, error analysis,
   and stated limits.

The full plan is in [`docs/course_map.md`](docs/course_map.md).

## What you need

You need to be comfortable writing Python: functions, loops, lists, dictionaries, classes. No
machine learning is assumed. The notebooks set out to introduce the libraries they use (numpy,
pandas, matplotlib, scikit-learn) and to re-establish the mathematics each method needs.

On your machine you need `git` and [`uv`](https://docs.astral.sh/uv/getting-started/installation/).
`uv` installs the right version of Python (3.12 or later) and every library the course uses, in a
folder `.venv` inside the repository. Nothing is installed system-wide.

## Quick start

Run these three commands from the root of the repository.

```bash
uv sync --extra dev --extra boosting --extra deep
uv run pytest
uv run jupyter lab
```

The first command installs everything. The `boosting` extra brings XGBoost and LightGBM (modules
09 and 10), the `deep` extra brings PyTorch (the last notebooks of module 12), and the `dev` extra
brings the tests and the linters. The second command runs the tests of the small course library,
which tells you that the installation works. The third opens JupyterLab in your browser.

## Working in Jupyter

If you have never used a notebook, this is all you need to start.

- **Open a notebook.** JupyterLab shows the files of the repository in its left panel. Open
  `notebooks/00_GettingStarted/01_what_is_ml.ipynb`.
- **Cells.** A notebook is a sequence of cells. Text cells explain; code cells hold Python. Click
  a code cell and press `Shift+Enter`: the cell runs, its result appears under it, and the cursor
  moves to the next cell.
- **Run every cell, in order.** The notebooks are stored without their outputs, so figures and
  numbers appear only when you run the cells. Each code cell uses what the earlier ones defined: a
  skipped cell shows up further down as a `NameError`.
- **The marker on the left.** `[ ]` means the cell has not run, `[*]` that it is running, and
  `[3]` that it was the third cell to run.
- **Start again from a clean state.** The *kernel* is the Python process behind the notebook; it
  keeps every variable you created. When something looks inconsistent, use the menu **Kernel →
  Restart Kernel and Run All Cells…**: the kernel forgets everything and the notebook runs again
  from the top.
- **Datasets.** The first time a notebook needs a dataset (penguins, newsgroups, Fashion-MNIST and
  others), it downloads it and keeps a local copy, so you need a network connection once.
- **Running time.** Most notebooks run in a few seconds. On the machine the course was checked on,
  the slowest ones take between one and three minutes.
- **Change things.** Editing a cell and running it again is the best way to test what you
  understood. `git restore notebooks/NN_Module/MM_title.ipynb` brings the original notebook back.

## Following the course on your own

Take the modules from `00_GettingStarted` to `12_NeuralNetworks`, and the notebooks of a module in
the order of their numbers. Each notebook names its prerequisites at the top. A reference such as
"module 03, notebook 4" (written "Chapter 03, NB 4" in some notebooks) points to the fourth
notebook of the folder `notebooks/03_LogisticRegression/`.

In each notebook:

1. Read the header: what the notebook teaches and what you will be able to do at the end.
2. Before you run a cell that draws a figure or prints a number, say what you expect. Then run it
   and compare. The **"Read the figure"** paragraph that follows each figure tells you what to
   look at.
3. Do the exercises of **"Your turn"** before you move on. This is where the understanding is
   built; reading alone will not do it.
4. Read **"What you built"** at the end and check that you could explain each line to someone
   else.

**Exercises.** An exercise is a code cell to complete, where `...  # YOUR CODE` marks what is
missing. The cell after it checks your answer and tells you whether it matches; it says "Not
filled in yet" as long as you have written nothing, and it never stops the notebook. Under the
check, a collapsed block **"Show the solution"** holds the filled cell. Try first, and open it to
compare. The notebooks are being converted to this form one module at a time: in a module that is
not converted yet, the exercises are questions in prose, without a check or a written solution.

**When you are stuck.** Run the notebook again from the top (Kernel → Restart Kernel and Run All
Cells…). Go back to the prerequisite named in the header. Look in
[`docs/common_errors.md`](docs/common_errors.md), which lists the confusions met most often and
the notebook that addresses each one. To find where a concept is taught, search
[`llms.txt`](llms.txt), which lists every notebook with what it teaches. Each notebook ends with
its references.

**With an assistant.** If you work with an AI assistant, point it at [`AGENTS.md`](AGENTS.md): it
asks the assistant to tutor you, with hints first, rather than hand out answers.

## What the repository holds

| Path | What it is |
|---|---|
| `notebooks/NN_ModuleName/MM_snake_title.ipynb` | The course: one folder per module, plain `.ipynb` files numbered within their module. |
| `src/ml_course/` | The small library the notebooks import: `colors` (the palette), `viz` (the matplotlib style and plotting helpers), `datasets` (the teaching datasets, downloaded once and cached), `check` (the checks that follow each exercise). |
| `tests/` | The tests of the library and of the scripts. |
| `scripts/run_notebooks.py` | Executes notebooks from top to bottom on a fresh kernel, on a copy, and reports the ones that fail. |
| `scripts/check_*.py` | The guards of the course: exercises and their solutions, cross-references, length of code cells, no hardcoded colour. |
| `scripts/gen_llms_txt.py` | Regenerates `llms.txt`. |
| `scripts/vendor_*.py` | Download a dataset ahead of time, for instance before working offline. |
| `docs/course_map.md` | What each module teaches, notebook by notebook. |
| `docs/notebook_template.md` | The charter every notebook follows. |
| `docs/common_errors.md` | The confusions learners meet most often. |
| `llms.txt` | A machine-readable index of every notebook: path, title, what it teaches. |
| `AGENTS.md` | Instructions for an AI assistant that tutors a learner. |

To check that every notebook of a module runs on your machine:

```bash
uv run python scripts/run_notebooks.py "00_GettingStarted/*.ipynb"
```

## Design

The course inherits the warm-pastel **graphic charter** centralized in `ml_course.colors` (a family
resemblance with the PPSP courses): no hardcoded colours, a single matplotlib style, every figure
followed by a "Read the figure" reading. Voice: warm, empowering, and rigorous — making things
understandable is an effort of presentation, never a simplification. All course content is in English.
