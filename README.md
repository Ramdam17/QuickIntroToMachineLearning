# Quick Intro to Machine Learning — a Hands-On Course

A **learn-by-doing** journey through twelve classical machine-learning methods, for developers who
want to *understand* them — not just call them. Every method is built by hand first, observed, then
generalized, before the real library estimator appears.

The course is built **one concept per notebook**: short, focused notebooks build progressively, so
each idea lands before the next arrives. It is meant to be worked through with an AI tutor alongside
(see `AGENTS.md`).

## The twelve methods

k-Nearest Neighbours · Naive Bayes · Logistic Regression · Decision Trees · Support Vector Machines ·
Random Forests · AdaBoost · Gradient Boosting · XGBoost · LightGBM · Multi-Layer Perceptron ·
Neural Networks.

Ordered as a progression — instance-based → probabilistic → linear → trees → margins → ensembles →
neural — so each method leans on the intuition built by the ones before it.

## How each method is taught

Per method, **3 to 5 notebooks** of ~20 cells following one arc:

1. **Fundamentals (notebooks 1–3)** — the core intuition, one concept at a time, by hand.
2. **The method & its parameters (notebook 4)** — the real estimator, its hyperparameters, how it
   fails and how to tune it.
3. **A demanding practical case (notebook 5)** — a realistic problem with honest evaluation, error
   analysis, and stated limits.

The full plan is in [`docs/course_map.md`](docs/course_map.md).

## Quick start

```bash
uv sync --extra dev --extra boosting
uv run pytest
uv run jupyter lab
```

## How the course is structured

One module folder per method, plain `.ipynb` (not Jupyter Book), per-module numbering
`NN_ModuleName/MM_snake_title.ipynb`. Each notebook declares its prerequisites and builds on the
ones before it. To find which notebook covers a concept, read [`llms.txt`](llms.txt) or ask your
assistant.

## Design

The course inherits the warm-pastel **graphic charter** centralized in `ml_course.colors` (a family
resemblance with the PPSP courses): no hardcoded colours, a single matplotlib style, every figure
followed by a "Read the figure" reading. Voice: warm, empowering, and rigorous — making things
understandable is an effort of presentation, never a simplification. All course content is in English.
