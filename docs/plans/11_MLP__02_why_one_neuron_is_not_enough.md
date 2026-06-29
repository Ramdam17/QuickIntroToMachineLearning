# NB plan — 11_MLP / 02_why_one_neuron_is_not_enough — the hidden layer

> Status: **APPROVED by Rémy (via ExitPlanMode, 2026-06-29).** No reviewer gate at the NB-plan stage —
> both reviewers return on the built notebook. Anchors measured live (scikit-learn 1.9.0, SEED=0).

## Context
Chapter 11 (MLP), **NB 2 of 5** — the second fundamental, building on NB 1 (a neuron = `φ(w·x+b)`; a single
sigmoid neuron *is* logistic regression, a **straight** boundary). **ONE concept:** a single neuron can draw
only a straight boundary; a **hidden layer + a non-linearity** carves curved ones — and **why the
non-linearity is essential** (a stack of *linear* layers collapses to one linear map). By hand: the `2→H→1`
forward pass, the linear-collapse, and the classic **hand-set ReLU XOR net** (no training — finding the
weights is **NB 3**). The universal-approximation *intuition* is seated here, stated honestly as an
**existence** result. ~21 cells, 3 figures. NB-plan stage = **Rémy validates alone** (no reviewer gate).

## Live anchors (measured, scikit-learn 1.9.0, SEED=0 — `measure_ch11_nb2.py`)
- **XOR** (4 points): `LogisticRegression` train acc **0.50** (one line can't); `MLPClassifier((4,), relu,
  lbfgs)` **1.0**. **Hand-set ReLU XOR net** (Goodfellow §6.1: `W1=[[1,1],[1,1]]`, `b1=[0,−1]`, `w2=[1,−2]`,
  `b2=0`) → forward-pass output **`[0,1,1,0]` == XOR exactly**, with no training.
- **circles** (`make_circles(n_samples=400, noise=0.10, factor=0.40, random_state=0)`, stratified 75/25):
  `LogisticRegression` test **0.41** (≈ a coin flip — a straight line cuts the rings); **`MLP((8,),
  identity)` test 0.41 == logistic** (the linear collapse — the hidden layer bought nothing); **`MLP((8,),
  tanh)` 1.0, `relu` 1.0, `logistic` 0.99** (a non-linearity separates the rings).
- **Linear collapse, by hand:** `(x@W1)@W2` == `x@(W1@W2)` to **2.22e-16**; `W1@W2` is a single `(2,1)` map
  — two identity-activated layers *are* one linear neuron.

## Cell-by-cell (~21 cells, 3 figures) — intuition → implementation → "Read the figure"
1. **(md) Header** — `# 02 — Why one neuron is not enough: the hidden layer`; purpose; **Prerequisites**
   (11.1 the neuron == logistic; ch 00 accuracy); **What you'll be able to do** (see one neuron's
   straight-line limit; build a `2→H→1` forward pass by hand; explain why a non-linearity is essential; see a
   hidden layer solve XOR and circles); warm welcome.
2. **(code) Imports** — `numpy`, `matplotlib.pyplot`; sklearn `make_circles`, `train_test_split`,
   `LogisticRegression`, `MLPClassifier`; `from ml_course import colors, viz`; `viz.use_course_style()`;
   `SEED = 0`; `np.random.seed(SEED)`.
3. **(md) Recap** — NB 1: a neuron is `φ(w·x+b)`; a single sigmoid neuron is logistic regression, which
   draws a **straight** boundary. The question this notebook answers: what can one straight line *not* do?
4. **(md) Intuition — not linearly separable.** Two classic problems no straight line can split: **XOR**
   (1 if exactly one input is on) and **concentric circles** (an inner disc inside an outer ring).
5. **(code) Fig 1 — one straight line is not enough** (2 panels: `plot_decision_boundary` of logistic on
   XOR | on circles) + print XOR acc 0.50, circles acc 0.41.
6. **(md) Read Fig 1** — the line gets ~half right either way (XOR 0.50; circles 0.41, no better than
   guessing). One neuron = one straight cut; these problems are not linearly separable.
7. **(md) Intuition — stack neurons into a hidden layer.** Put `H` neurons side by side (each its own
   `w, b, φ`), feed all their outputs to a final neuron. The hidden layer remaps the inputs into new
   features; the output neuron draws its straight line *in that new space*.
8. **(code) Fig 2 — the `2 → H → 1` architecture** (matplotlib schematic: 2 inputs → H hidden nodes →
   1 output; charter colours).
9. **(md) Read Fig 2** — H hidden neurons in parallel, then one output neuron combining them. Two questions
   follow: does this actually help, and what makes it help?
10. **(md) Intuition — why the non-linearity is essential.** If `φ` is the identity (no squash), two layers
    compose into one: `W₂(W₁x) = (W₂W₁)x` — a single linear map. Stack as many identity layers as you like;
    the boundary stays a straight line.
11. **(code) By hand — the linear collapse** — compose `W1 @ W2` and show `(x@W1)@W2 == x@(W1@W2)` to
    `~2e-16`; then `MLP((8,), activation="identity")` on circles scores **0.41 == logistic** (the hidden
    layer vanished).
12. **(md) Read** — with identity activations the hidden layer buys **nothing** (0.41, same as one neuron);
    the two layers *are* one linear map (`2.22e-16`). The non-linearity is what prevents the collapse.
13. **(md) Intuition — a hidden layer *with* a non-linearity, built by hand.** The classic 2-hidden-unit
    **ReLU** network for XOR (Goodfellow §6.1). We **set** the weights (finding them automatically is the
    next notebook); here we check that this tiny network computes XOR.
14. **(code) By hand — the hand-set ReLU XOR net** — `W1=[[1,1],[1,1]]`, `b1=[0,−1]`, `w2=[1,−2]`;
    `H = relu(X@W1+b1)`, `out = H@w2+b2`; print the truth table → **`[0,1,1,0]` == XOR**.
15. **(md) Read** — two ReLU hidden units, set by hand, compute XOR exactly — what one neuron (0.50) cannot.
    The hidden layer + the non-linearity did it (we *chose* these weights; **NB 3** is how a network *finds*
    them).
16. **(code) Fig 3 — a non-linear hidden layer succeeds** (2 panels: `MLP((4,), relu)` on XOR |
    `MLP((8,), tanh)` on circles, both fit with `lbfgs` as an **opaque** box) + print accs (XOR 1.0,
    circles 1.0).
17. **(md) Read Fig 3** — the same architecture that collapsed under identity now carves a closed boundary
    around the inner ring (1.0) and the XOR regions (1.0). A non-linear hidden layer is the engine. (We
    fitted these as black boxes — *how* the weights are found is NB 3.)
18. **(md) How far does this go? (universal approximation, honestly).** A hidden layer with enough
    non-linear units can approximate **any** continuous boundary (Cybenko 1989; Hornik 1991). State it
    honestly: this is an **existence** result — enough units *can* represent it; it does **not** promise that
    training will find that network, or that it will generalize. (Next: NB 3 — how a network *learns* the
    weights, by backpropagation.)
19. **(md) Your turn** — (warm-up) set the circles MLP's `activation="identity"` — predict the test accuracy
    before running, then check it (why?); (core) shrink the hidden layer to **1 unit** — does a 1-unit
    hidden layer help on circles, and why not? (reach) a single neuron *can* do AND and OR — hand-set or fit
    one for OR, then argue in one sentence why XOR is different (not linearly separable).
20. **(md) What you built** — saw one neuron's straight-line limit (XOR 0.50, circles 0.41); built a `2→H→1`
    forward pass; proved the linear collapse (`2.22e-16`; identity-MLP == one neuron); hand-built a ReLU
    network that computes XOR; watched a non-linear hidden layer separate the rings (1.0). Next: how a
    network *finds* these weights.
21. **(md) References** — Minsky & Papert 1969 (*Perceptrons* — the XOR limit of a single unit); Cybenko
    1989 (DOI 10.1007/BF02551274) & Hornik 1991 (DOI 10.1016/0893-6080(91)90009-T) — universal approximation;
    Goodfellow, Bengio & Courville 2016 §6.1 (the hand-set ReLU XOR net). Previous: **11.1 — the neuron ==
    logistic**. Next: **11.3 — how a network learns (backpropagation)**.

## `src/` & guards
- **No `src/` change** (notebook-local numpy + matplotlib; `make_circles`; `LogisticRegression`/
  `MLPClassifier`; `viz.plot_decision_boundary`; pytest 20). Colours only from `ml_course.colors`; `SEED=0`;
  "Read the figure" after every figure; **never silence output**; banned-word scan 0; hex clean; ruff/black
  clean; output-free.
- Build from `build_ch11_nb2.py` (source of truth; rebuild right before `git add` — kernel-drift guard).
- Exit guards: nbconvert exit 0 (3 figures), banned 0, hex clean, ruff clean, output-free; **two-reviewer
  gate** (no BLOCK) → **Rémy visual** → end-of-NB checklist (`gen_llms_txt`, `common_errors` +rows,
  `course_map` §11 note, pytest 20, STATE) → commit `feat(11_mlp): notebook 02 — why one neuron is not
  enough` → `git merge --ff-only` into `chapter/11_MLP`.

## Verification (end-to-end)
1. nbconvert a scratchpad copy → exit 0, 3 figures, anchors reproduce (XOR logistic 0.50 / hand-set net
   `[0,1,1,0]`; circles logistic 0.41 == identity-MLP 0.41; tanh/relu MLP 1.0; collapse `2.22e-16`).
2. hex + banned + ruff clean. 3. pytest 20. 4. **Rémy validates this NB plan (no reviewer gate); both
   reviewers return on the built notebook.**
