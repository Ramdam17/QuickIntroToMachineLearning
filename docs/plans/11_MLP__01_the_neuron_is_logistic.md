# NB plan — 11_MLP / 01_the_neuron_is_logistic — the artificial neuron == the logistic unit you already built

> Status: **APPROVED by Rémy (via ExitPlanMode, 2026-06-29).** No reviewer gate at the NB-plan stage —
> both reviewers return on the built notebook. Anchors measured live (scikit-learn 1.9.0, SEED=0).

## Context
Chapter 11 (MLP), **NB 1 of 5** — the first fundamental. **ONE concept:** the artificial neuron = weighted
sum + bias + activation, and **a single sigmoid neuron IS the logistic-regression unit of ch 03** (a renamed
friend). This is the chapter's load-bearing bridge: the whole optimizer (sigmoid, log-loss, gradient
descent) was already built by hand in ch 03 — here we only **re-frame** it as "a neuron" and meet the
**pluggable activation**. By hand before the library; ~20 cells, 3 figures. *Why a non-linearity matters at
all* is NB 2's question — NB 1 just establishes the neuron and the bridge. NB-plan stage = **Rémy validates
alone** (no reviewer gate; reviewers return on the built notebook).

## Live anchors (measured, scikit-learn 1.9.0, SEED=0 — `measure_ch11_nb1.py`)
- `make_moons(n_samples=400, noise=0.20, random_state=0)`, stratified 75/25 split:
  **LogisticRegression test acc 0.9300 == MLPClassifier(hidden_layer_sizes=()) test acc 0.9300** (exact tie).
- **By-hand sigmoid neuron with logistic's (w, b): max|σ(Xw+b) − logistic.predict_proba| = 0.00e+00** (the
  exact bridge — same math, to the bit).
- Coefficients differ a touch (MLP's default `alpha` + `lbfgs` vs logistic's `C=1`): the **form and the
  predictions** match; the by-hand parity is the exact anchor. (Re-measure at build.)
- Activations: σ(0)=0.5 (range 0→1), tanh(0)=0 (range −1→1), relu(z)=max(0, z). σ(z)=(tanh(z/2)+1)/2.

## Cell-by-cell (~20 cells, 3 figures) — intuition → implementation → "Read the figure"
1. **(md) Header** — `# 11.1 — The artificial neuron == the logistic unit you already built`; one-line
   purpose; **Prerequisites** (ch 03 NB 1 sigmoid · NB 3 log-loss · NB 4 gradient descent; ch 00 train/test
   + accuracy); **What you'll be able to do** (describe a neuron as weighted sum + bias + activation; explain
   why a single sigmoid neuron *is* logistic regression; name sigmoid/tanh/ReLU and what each does; show an
   empty-hidden MLP reproduces logistic regression); warm one-line welcome (the first method beyond trees).
2. **(code) Imports** — `numpy`, `matplotlib.pyplot`, `scipy.special.expit`; sklearn `make_moons`,
   `train_test_split`, `LogisticRegression`, `MLPClassifier`; `from ml_course import viz, colors`;
   `viz.use_course_style()`; `SEED = 0`; `np.random.seed(SEED)`.
3. **(md) Recap / footing** — genuinely re-establish ch 03: the linear score `z = w·x + b`; the sigmoid
   `σ(z)=1/(1+e⁻ᶻ)` squashes it to a probability in (0,1); we fit `(w, b)` by gradient descent on the
   log-loss. NB 1 leans on all of this — recapped, not presupposed.
4. **(md) Intuition — a neuron in three steps:** weighted sum `w·x` → add bias `b` → squash with an
   **activation** `φ`: `a = φ(w·x + b)`. The activation is a *pluggable* choice.
5. **(code) Fig 1 — the neuron diagram** (matplotlib schematic: inputs `x₁, x₂` → weighted edges `w₁, w₂`
   → `Σ + b` node → `φ` → output `a`; charter colours only).
6. **(md) Read the figure** — walk the diagram: the weighted sum, the bias shift, the activation → one
   neuron's single output.
7. **(md) Intuition — meet three activations:** sigmoid (smooth 0→1, ch 03's), tanh (−1→1, sigmoid's
   cousin), ReLU (`max(0, z)`, the modern default). (*Why* a non-linearity matters is NB 2.)
8. **(code) Fig 2 — the three activations** over `z ∈ [−5, 5]` (three curves, charter colours).
9. **(md) Read the figure** — sigmoid saturates to 0/1 (a probability); tanh is centred at 0; ReLU is off
   below 0 then linear. Each squashes the same linear score differently.
10. **(md) Intuition — the bridge:** logistic regression computes `σ(w·x + b)`. That is *exactly* a sigmoid
    neuron. The neuron is **not a new model — it is logistic regression, renamed.**
11. **(code) By hand** — fit `LogisticRegression` on moons; take its `(w, b)`; compute `σ(Xw + b)` with
    `expit`; compare to `predict_proba` → print `max abs diff` = **0.00e+00**.
12. **(md) Read the result** — the by-hand sigmoid neuron reproduces logistic's probabilities to the bit.
    Same math, same model.
13. **(md) Intuition — the empty-hidden MLP:** scikit-learn's `MLPClassifier(hidden_layer_sizes=())` has
    only the output sigmoid neuron → it should behave like logistic regression. Let's check.
14. **(code) Empty-hidden MLP vs logistic** — fit `MLPClassifier(hidden_layer_sizes=(), activation='logistic',
    solver='lbfgs', alpha=1e-4, max_iter=5000, random_state=0)` and `LogisticRegression` on the same split;
    print both test accuracies (**0.9300 == 0.9300**).
15. **(code) Fig 3 — decision boundaries side by side** (`viz.plot_decision_boundary` on two axes:
    logistic | empty-hidden MLP) — the same straight boundary.
16. **(md) Read the figure** — both draw the *same straight* boundary and score 0.9300: the same model,
    found by different optimizers (coefficients differ a hair from regularization; the predictions match).
17. **(md) Your turn** — (easy) set the empty-hidden MLP's `activation='tanh'` — does the boundary stay
    straight? why? (core) change the moons `noise` and check the two test accuracies still track each other;
    (reach) relate tanh to sigmoid — verify `σ(z) = (tanh(z/2)+1)/2` numerically.
18. **(md) What you built** — reframed logistic regression as a single neuron; met sigmoid/tanh/ReLU; proved
    (to the bit) a sigmoid neuron == logistic regression; watched the empty-hidden MLP reproduce it. You can
    now say exactly what a neuron computes.
19. **(md) Going further (optional)** — Rosenblatt's perceptron (1958) used a hard *step* activation; the
    sigmoid neuron is its smooth, differentiable cousin — which is why gradient descent (ch 03) works on it.
20. **(md) References** — Rosenblatt 1958 (DOI 10.1037/h0042519); ch 03 NB 1/3/4 (sigmoid, log-loss,
    gradient descent). Previous: ch 10 (LightGBM). Next: **11.2 — why one neuron is not enough (the hidden
    layer)**.

## `src/` & guards
- **No `src/` change** (notebook-local numpy + matplotlib; `make_moons`; `LogisticRegression`/
  `MLPClassifier`; `viz.plot_decision_boundary`; pytest 20). Colours only from `ml_course.colors`; `SEED=0`;
  "Read the figure" after every figure; **never silence output**; banned-word scan 0; hex clean; ruff/black
  clean; output-free.
- Build from `build_ch11_nb1.py` (source of truth; rebuild right before `git add` — kernel-drift guard).
- Exit guards: nbconvert exit 0 (3 figures), banned 0, hex clean, ruff clean, output-free; **two-reviewer
  gate** (no BLOCK) → **Rémy visual** → end-of-NB checklist (`gen_llms_txt`, `common_errors` +rows,
  `course_map` §11 note, pytest 20, STATE) → commit `feat(11_mlp): notebook 01 — the neuron is logistic` →
  `git merge --ff-only` into `chapter/11_MLP`.

## Verification (end-to-end)
1. nbconvert a scratchpad copy → exit 0, 3 figures, anchors reproduce (logistic 0.9300 == MLP(()) 0.9300;
   by-hand parity 0.00e+00). 2. hex + banned + ruff clean. 3. pytest 20. 4. **Rémy validated this NB plan
   (no reviewer gate at plan stage); both reviewers return on the built notebook.**
