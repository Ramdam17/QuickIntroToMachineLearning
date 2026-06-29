# Chapter plan — 11_MLP (the Multi-Layer Perceptron)

> Status: **APPROVED by Rémy (via ExitPlanMode + the two-reviewer gate, 2026-06-29).** Per-method arc over
> **5 notebooks**; per-NB plans drafted and Rémy-validated one at a time before each build. All API facts
> measured live (scikit-learn 1.9.0 / numpy 2.4.6); every anchor re-measured/pinned at NB-plan/build.

## Context

Chapters 00–10 are complete and merged to `main` (the boosting family closed via PR #10). Chapter 11 opens
the two neural chapters — the **first method beyond trees: a model that learns its own features.** This plan
was built by the project chapter-plan process: a concept tour (`@concept-cartographer`, which read ch 03 +
measured every load-bearing anchor live on scikit-learn 1.9.0) → draft → the **two-reviewer gate**
(`@ml-expert-reviewer` REVISE→PASS-after-fold + `@pedagogy-reviewer` REVISE→no-BLOCK). All folds applied.

## What this chapter is

The whole chapter is built so one bridge is load-bearing:

> **A single sigmoid neuron is exactly the logistic-regression unit of chapter 03.** Weighted sum + bias
> `z = w·x + b` (ch 03 NB 1–2), squashed by the sigmoid (ch 03 NB 1), trained by gradient descent on the
> log-loss (ch 03 NB 3–4). The neuron is **not a new model — it is a renamed friend.**

The MLP is then **one new idea stacked on that unit**: insert a **hidden layer with a non-linearity**, and
the model can carve boundaries a single neuron *provably* cannot. The learner builds a small network by
hand (the forward pass, then **backpropagation = the chain rule applied layer by layer**), watches it solve
a problem one neuron fails (concentric circles / XOR), then drives the real `MLPClassifier` and ships an
honest end-to-end capstone (handwritten digits, 8×8) with scaling, seeds, a loss-curve diagnostic, limits.

**The ch 11 / ch 12 line (the crux).** Chapter 11 owns **"the MLP as scikit-learn ships it"**: **one**
hidden-layer concept, parameter-space gradient descent, a finite CPU-trainable estimator. Chapter 12 owns
**"neural networks as a paradigm"**: depth as a *representation hierarchy*, dropout, depth-driven gradient
pathologies, normalization, the framework transition.

**Builds on:** ch 03 (the entire single-neuron training stack — sigmoid, log-loss, the gradient `(P−y)x`,
the descent loop, the learning-rate panel, mini-batch SGD *named once in passing*); ch 00 (split & leakage,
scaling + `Pipeline` + fit-on-train-only, over/underfitting + the generalization gap + the learning curve,
CV, confusion matrix / precision-recall / ROC-PR-AUC / threshold). One-sentence contrast: ch 08 NB 2 did
gradient descent in *function* space; the MLP does it in *parameter* space — same idea, different space.

## Live API ground truth (measured, scikit-learn 1.9.0 / numpy 2.4.6)

`MLPClassifier` / `MLPRegressor` ship in scikit-learn (**no new package**).

- **Defaults** (`MLPClassifier`): `hidden_layer_sizes=(100,)`, **`activation='relu'`**, **`solver='adam'`**,
  `alpha=1e-4` (L2), `learning_rate_init=1e-3`, `batch_size='auto'`, `max_iter=200`, `early_stopping=False`,
  `momentum=0.9`, `tol=1e-4`. `solver ∈ {sgd, adam, lbfgs}`. `verbose=False` (per-iteration loss when True).
- **The bridge (verified to machine precision):** `MLPClassifier(hidden_layer_sizes=())` reproduces logistic
  regression — form (single sigmoid output), coefficients (1.07/−2.61 vs 1.00/−2.30, low `alpha`+`lbfgs`),
  number (test **0.9167 == logistic 0.9167** on two-moons). A single neuron *is* logistic regression.
- **Diagnostics:** `loss_curve_` (per-iteration training loss; digits ~179 long, 2.51 → 0.006);
  `validation_scores_`/`best_validation_score_` under `early_stopping=True`; `n_iter_`, `coefs_`.
- **Non-linearity is the engine** (`make_circles(noise=0.1, factor=0.4)`, 5 seeds): `identity` **0.528
  (chance)**; **`tanh`/`relu` (default adam) → ~0.999–1.0**. Linear∘linear = `W₂(W₁x)=Wx`. **Solver caveat
  (load-bearing):** sigmoid reaches **~0.996 only with `lbfgs` (full-batch)** — with default **`adam` it
  stalls near chance (~0.41–0.48)** on the flat-gradient plateau. The by-hand build (NB 1–3) uses sigmoid +
  full-batch GD (works, train acc 1.0); the sigmoid+adam stall is reframed as the measured motivation for
  **ReLU as the default in NB 4**.
- **What a hidden layer buys:** XOR → logistic **0.50**, MLP(4) **1.0**; circles → linear **0.528**, MLP(16)
  **0.994**.
- **By-hand parity (NB 3):** a hand-written **2-4-1** net (numpy forward + backprop; gradient-checked rel_err
  **~2e-9**; full-batch GD) reaches **train acc 1.0 on circles**, matching `MLPClassifier((4,), 'logistic',
  'lbfgs')`.
- **Symmetry breaking (NB 3):** zeros init → units cannot differentiate. Full zeros → all gradients
  identically **zero** (frozen); W1=0/W2≠0 → **identical-but-nonzero** columns. Random init breaks it.
- **Capstone (NB 5), `load_digits` (8×8, 1797×64, 10-class):** MLP(64) **~0.974–0.980 test, ~0.2 s**,
  ~165–179 iters; **seed spread 0.974–0.980** across 5 init seeds (single split — *init* variance). **The
  fair tree foil matches or beats it:** HistGB **~0.982** ≥ MLP, RF ~0.970, logistic ~0.972 — "fully
  competitive, not superior," not an MLP win.
- **Tabular-limit nuance (measured):** clean homogeneous `breast_cancer` (scaled), MLP(32) CV **0.9754**
  edged RF **0.9649** / HistGB **0.9701**. So the honest limit is "heterogeneous mixed-type small/medium
  tabular → trees safer + no scaling," **not** "trees always beat MLPs" — **this is where the genuine MLP
  win lives** (digits is competitive, not a win).

(Every number re-measured and reconciled at NB-plan/build; seeds pinned.)

## Prerequisites (from earlier chapters)

The key finding of the tour: **ch 03 already built the entire single-neuron training stack by hand.** Ch 11
*inserts one hidden layer* and *generalizes the gradient via the chain rule* — never re-teaches optimization.

| Prerequisite | Where built | How ch 11 uses it |
|---|---|---|
| Sigmoid σ(z), plotted | ch 03 NB 1 | The activation φ for the bridge neuron |
| Linear score `z=w·x+b`; log-odds; boundary | ch 03 NB 1–2 | The neuron's pre-activation |
| Log-loss = cross-entropy = NLL; **convexity** | ch 03 NB 3 | The loss — with the honest caveat that a *multi-layer* net is **non-convex** |
| Gradient `=(P−y)x`; update `θ←θ−η∇L`; convergence; the learning-rate panel; loss-vs-iteration curve (NB 4 fig (c)) | ch 03 NB 4 | The optimizer, re-used wholesale; backprop = "this gradient, layer by layer." **Epoch / mini-batch / batch_size are genuinely new** (ch 03 named mini-batch once; "epoch" never) — introduced in NB 4 |
| L1/L2 weight penalties | ch 03 NB 5 | `alpha` (L2 / weight decay) |
| Softmax / multinomial (full section: ch 03 NB 5 cells 15–17) | ch 03 NB 5 | The K-class output head (recap the formula; *build the head step* in NB 4) |
| Split, leakage; accuracy+baseline; confusion/PR/ROC/threshold | ch 00 NB 4, 6–8 | Capstone hygiene + evaluation |
| Over/underfitting, generalization gap, learning curve | ch 00 NB 9 | Reading the loss curve |
| CV; hyperparameters vs parameters | ch 00 NB 10 | Honest tuning |
| Scaling, `Pipeline`, fit-on-train-only | ch 00 NB 11 (+ scale trap ch 01 NB 2) | "Scaling is mandatory for NNs" — enforced every fit |
| GD in *function* space | ch 08 NB 2 | One-sentence contrast (function- vs parameter-space) |

**Re-established briefly:** sigmoid, log-loss, the descent loop, the learning rate, the loss curve, scaling,
over/underfitting, softmax-the-formula. **Genuinely new:** the neuron-as-pluggable-φ framing; tanh & ReLU;
*why* non-linearity; the hidden layer + weight matrix; the forward pass; **backprop (chain rule)**;
zeros-init symmetry breaking; **epoch/mini-batch/batch_size**; **the K-class softmax output head**;
`MLPClassifier` knobs; Adam/momentum named; universal approximation; depth vs width.

## The per-method arc (5 notebooks)

By hand before the library. **Sigmoid throughout the by-hand build (NB 1–3)** (ch 03 continuity + clean
`H·(1−H)` derivative), trained by **full-batch GD / `lbfgs`** (sigmoid+adam stalls — the measured ReLU
motivation in NB 4). **ReLU revealed in NB 4** (no saturation). Parity pins `lbfgs` (à la ch 03's `C=np.inf`).

### NB 1 — The artificial neuron == the logistic unit you already built
- **One concept:** neuron = weighted sum + bias + **activation** `a=φ(w·x+b)`; a single **sigmoid** neuron
  *is* logistic regression (the ch 03 bridge); φ is *pluggable* — sigmoid (recap), **tanh**, **ReLU**.
- **By hand:** re-frame σ and `w·x+b` as "a neuron"; plot tanh/ReLU from scratch; one-unit forward pass;
  show `MLPClassifier(hidden_layer_sizes=())` reproduces the ch 03 boundary.
- **Anchor:** empty-hidden MLP test **0.917 == logistic 0.917** on two-moons.
- ~3 figures (neuron diagram; the three activations; empty-hidden MLP boundary == logistic).

### NB 2 — Why one neuron is not enough: the hidden layer
- **One concept:** one neuron → only a **straight** boundary; a **hidden layer + non-linearity** → curved
  ones; **why non-linearity is essential** (linear∘linear = `Wx`). **Seat the universal-approximation
  intuition here** ("enough hidden units can approximate any boundary — Cybenko/Hornik, an *existence*
  result, stated not proved"); NB 5 recalls it as the caveat.
- **By hand:** `2→H→1` forward pass with weight matrices; show the linear-collapse (`identity` no better than
  one neuron); **hand-set** the weights (or call a solver as an opaque box — *how* weights are found is
  **NB 3's** concept) to separate circles/XOR; plot the curved boundary.
- **Anchors:** XOR logistic **0.50** vs MLP(4) **1.0**; circles `identity` **0.528** vs **`tanh`/`relu`
  ~0.999–1.0**. **Sigmoid's curved demo pins `solver='lbfgs'`** (adam stalls — saturation preview → NB 4),
  said so in the cell.
- ~3 figures (straight boundary failing on circles; layered forward-pass schematic; curved boundary winning).

### NB 3 — How a network learns: backpropagation (the chain rule)
- **One concept:** backprop = the **chain rule layer by layer** — forward caches activations, backward
  carries the error back; **init must not be zeros** (symmetry breaking).
- **By hand:** forward + backward for a `2-H-1` net (`d_out`; `dH = d_out @ W2.T * H·(1−H)`; `dW1`, `dW2`);
  **gradient-check vs finite differences** (rel_err ~2e-9); run GD (the ch 03 loop, now multi-layer);
  **demonstrate the zeros-init collapse**. General multi-layer algorithm as a **picture**, not a grind.
- **Anchor:** by-hand 2-4-1 GD → **train acc 1.0 on circles**, matching `MLPClassifier((4,),'logistic',
  'lbfgs')`. **Zeros-init (pin the exact init at build, align prose):** full zeros → all gradients **zero**
  (frozen); W1=0/W2≠0 → identical-nonzero columns (units never diverge); random init breaks it.
- ~3 figures (forward-cache → backward-error flow; loss falling, by-hand vs sklearn overlaid; zeros vs random
  init — columns identical/frozen vs distinct).

### NB 4 — The estimator `MLPClassifier`/`MLPRegressor` & its parameters
- **Integrative**, each knob from the concept that owns it: `hidden_layer_sizes` (depth × width — capacity);
  `activation` (**why ReLU is the default** — no saturation; **the measured sigmoid+adam stall from NB 2 is
  the concrete motivation**; depth-driven vanishing-gradient deferred to ch 12); `solver` (sgd/adam/lbfgs —
  **Adam named** as "an adaptive-step upgrade of the GD you built"); `alpha` (L2); `learning_rate_init`;
  `early_stopping`+`validation_fraction`; `max_iter`; **`batch_size` — epoch / mini-batch / iteration
  introduced as vocabulary** (they set the loss-curve x-axis; their genuine home); **scaling mandatory**; the
  **loss curve** (`loss_curve_`; *reading* recapped from ch 03 NB 4 fig (c)) as the convergence diagnostic.
- **The K-class output head (its explicit home):** a short subsection — **"from one output unit to K: the
  softmax head"** — turning NB 1–3's single sigmoid output into K softmax units with K-class cross-entropy
  (formula recapped from ch 03 NB 5), one schematic + "Read the figure". Built here, not absorbed silently.
- **Honest spine:** unscaled-vs-scaled on a **small synthetic 2-feature mismatched-scale problem** (robust
  ~0.96 → ~0.99 — `load_digits` is too homogeneous to break reliably: a lone ×1e6 feature is ignored, adam
  absorbs a uniform rescale); a capacity sweep (width/depth) reading the boundary + loss curve; `alpha`
  up/down; defaults-vs-`GridSearchCV` → **one sealed test**. (`ConvergenceWarning` stays visible — never
  silenced; `verbose=True` where per-iteration loss aids the lesson.)
- ~4 figures (capacity vs boundary & loss curve; softmax-head schematic; unscaled-vs-scaled loss curves;
  `alpha` path / default-vs-tuned).

### NB 5 — A demanding case: handwritten digits, end to end (visualization-first capstone)
- A full honest workflow on a real **multi-class** problem: **`load_digits`** (8×8, 1797×64, 10-class,
  offline, CPU-fast). `Pipeline(StandardScaler, MLPClassifier)`; tune lightly; held-out evaluation; read the
  **loss curve**; confusion matrix + per-digit error analysis; **seed-variance** check; a **fair tree-foil**
  — RF/HistGB on **raw** features (trees need no scaling) vs the MLP in its `StandardScaler` pipeline, same
  split/metric; **the preprocessing difference *is* the point.**
- **Honest verdict (measured):** on this homogeneous-pixel, image-like data the MLP is **fully competitive**
  with strong tree ensembles and earns the **"learned features"** framing as a *conceptual* virtue (the ch 12
  bridge) — but **the foil matches/beats it here** (HistGB ~0.982 ≥ MLP; not an MLP win), trees need **no
  scaling** and stay interpretable, the loss surface is **non-convex** (a minimum, not *the* minimum), the
  model is **seed-sensitive**, and universal approximation is an *existence* result. **The genuine MLP
  accuracy win lives elsewhere** (the `breast_cancer` edge in Honest scoping). Bridges to ch 12.
- **Arc** (≥6 figures, **~28–30 cells**, visualization-first): look at the digits → baseline → scaled MLP +
  loss curve → held-out metrics (confusion, per-class) → seed-variance → fair tree foil + honest limit →
  error gallery (which digits confuse).

## Honest scoping (the ml-expert bar)
- The single neuron **== logistic regression** (verified); the one new idea = hidden layer + backprop; the
  optimizer is ch 03 re-used.
- **Non-convex** loss (the one thing that breaks from ch 03): GD finds *a* minimum, not *the* minimum (8
  inits → 8 minima, measured).
- **Universal approximation = existence** (Cybenko 1989 / Hornik 1991), not a training/generalization promise
  (seated in NB 2), never oversold.
- **Tabular limit precise:** heterogeneous mixed-type small/medium → trees safer + no scaling; **MLP can win
  on clean homogeneous** (`breast_cancer` 0.9754 > RF 0.9649 / HistGB 0.9701 — the genuine win). Digits is
  **competitive, not superior** (HistGB ~0.982 ≥ MLP); NB 5 measures the **fair** foil on the actual data.
- **Scaling mandatory** — but digits won't break reliably; NB 4 stages it on a **synthetic 2-feature**
  problem (robust ~0.96 → ~0.99).
- **Sigmoid solver caveat (folded BLOCK):** sigmoid+adam stalls near chance on circles; by-hand build is
  sigmoid+full-batch-GD/`lbfgs`, library sigmoid demos pin `lbfgs`, the stall reframed as the ReLU
  motivation (NB 4).
- **Seed-sensitive** (non-convex): report seed variance, never a single lucky run.
- **Not interpretable** (vs ch 03 weights / ch 04 tree).
- **ch 11 / ch 12 boundary:** ch 11 = one hidden layer + sigmoid/tanh/ReLU + why-non-linearity + forward+
  backprop by hand + GD/SGD/mini-batch/epoch/lr + init + the K-softmax head + `MLPClassifier` knobs + scaling
  + loss curve + seeds + UAT stated. ch 12 = depth-as-hierarchy + **dropout** (absent from `MLPClassifier`,
  deferred not promised) + depth-driven gradient pathologies + modern init (He/Xavier) + normalization +
  the framework move.

## Datasets
- **Intuition (NB 2–3):** `make_circles(noise=0.1, factor=0.4)` (strongest discriminator); **XOR** (minimal
  historical). **`make_moons` is a POOR accuracy discriminator** (linear ~0.85–0.92) — boundary-shape only.
- **Capstone (NB 5):** `load_digits` (multi-class, offline, CPU-fast, image-like → ch 12 bridge). Fair tree
  foil RF/HistGB on raw features.
- **Scaling demo (NB 4):** a synthetic 2-feature mismatched-scale problem (robust break), not digits.
- **Avoid:** `breast_cancer` as the capstone (reuse; and MLP wins there — kept as the measured "MLP can win"
  fact, not the capstone); a 2-D toy as a capstone.

## `src/` & guards
No `src/` change expected (reuse `viz`/`colors`/`datasets`; numpy by-hand nets; `MLPClassifier` + `Pipeline`
+ `StandardScaler` + baselines/CV; `load_digits`/`make_circles`/`make_moons`/`make_classification`; pytest
20). By hand before library; colours only from `ml_course.colors` (no hardcoded hex); seeds fixed; "Read the
figure" after every figure; **never silence output** (`ConvergenceWarning` visible; `verbose=True` where it
aids; no `verbose=False`-as-suppression); banned-word scan 0; hex clean; ruff/black clean; output-free;
`gen_llms_txt`; two-reviewer gate + Rémy visual before each commit; ff-merge `notebook→chapter`;
**chapter→main via PR (`--no-ff`)** at close.

## References (chapter-level; per-NB DOIs at build)
Rumelhart–Hinton–Williams 1986 (backprop; DOI 10.1038/323533a0); Cybenko 1989 (UAT; 10.1007/BF02551274);
Hornik 1991 (UAT; 10.1016/0893-6080(91)90009-T); Rosenblatt 1958 (perceptron; 10.1037/h0042519);
Minsky & Papert 1969 (XOR); Glorot & Bengio 2010 (init/symmetry); Nair & Hinton 2010 (ReLU); Kingma & Ba
2015 (Adam; arXiv:1412.6980); Grinsztajn et al. 2022 (tabular limit; arXiv:2207.08815); Shwartz-Ziv & Armon
2022 (tabular limit; 10.1016/j.inffus.2021.11.011); ESL §11 (10.1007/978-0-387-84858-7); Goodfellow et al.
2016 §6 (→ ch 12).

## Open decisions (resolved in this plan)
1. **5 notebooks** (not 4): neuron / hidden layer / backprop do not bundle.
2. **Activation ordering + sigmoid solver caveat:** sigmoid by-hand (NB 1–3, full-batch GD/`lbfgs`), ReLU in
   NB 4; sigmoid+adam stalls → library sigmoid demos pin `lbfgs`, the stall is the ReLU motivation (folded
   ml-expert BLOCK).
3. **Backprop depth by hand:** full forward+backward on a 2-H-1 net (gradient-checked); the general algorithm
   as a picture.
4. **Scaling demo:** synthetic 2-feature problem in NB 4 (digits too homogeneous — folded ml-expert MAJOR).
5. **Tabular limit framing:** heterogeneous → trees; MLP wins on clean homogeneous (`breast_cancer`); digits
   **competitive, not superior** (HistGB ≥ MLP); fair foil (MLP scaled, trees raw).
6. **ch 11 / ch 12 split:** saturation/vanishing-gradient *intuition* in ch 11 NB 4; depth-driven pathologies
   + dropout + normalization + modern init → ch 12.
7. **Capstone dataset:** `load_digits` (a spiral noted as an alternative, but digits is the stronger story).
8. **By-hand parity solver:** pin `lbfgs` for the NB-3 parity cell; capstone uses default `adam`.
9. **Placement (pedagogy MAJORs):** the **K-class softmax head** gets an explicit home in NB 4; **epoch /
   mini-batch / batch_size** introduced in NB 4; the loss-curve reading anchored on ch 03 NB 4 fig (c);
   the universal-approximation intuition seated in NB 2; per-NB "Your turn" tiers sketched at each NB-plan.

## Reviewer gate outcome
Both reviewers ran live checks on scikit-learn 1.9.0 / numpy 2.4.6; **no BLOCK remains after folding**.

- **`@ml-expert-reviewer`: REVISE → PASS-after-fold.** The science verified to machine precision (bridge
  `hidden_layer_sizes=()` == logistic 0.9167; backprop `dH=d_out@W2.T·H(1−H)` rel_err 2e-9; 2-4-1 parity;
  XOR/identity; non-convexity; UAT-as-existence; the `breast_cancer` win; all DOIs). **BLOCK folded:** the
  false "sigmoid/tanh/relu all ~0.99 on circles" anchor — sigmoid+`adam` actually stalls near chance, so the
  by-hand build is sigmoid+full-batch-GD/`lbfgs` and the library sigmoid demo pins `lbfgs`, the stall
  reframed as the measured ReLU motivation. **MAJORs folded:** the wild-scale demo moved to a robust
  synthetic 2-feature problem (digits won't break reliably); the digits capstone reframed "shines" → "fully
  competitive" with the foil that matches/beats it (HistGB ~0.982). **MINOR/NIT folded:** HistGB
  breast_cancer 0.9701 (not 0.9648); zeros-init full-zero-vs-W1=0 precision; seed-variance single-split
  caveat.
- **`@pedagogy-reviewer`: REVISE → no-BLOCK.** Bridge load-bearing & verified; 5-NB split justified; voice/
  charter clean (0 banned/hex/emoji); honesty woven in. **MAJORs folded:** the K-class softmax output head
  given an explicit home in NB 4; the prereq table's mini-batch/epoch framing reconciled (epoch absent in
  ch 03; introduced here) with a home in NB 4. **MINOR/NIT folded:** loss-curve reading anchored on ch 03
  NB 4 fig (c); NB 2 hand-sets weights (training is NB 3's concept); NB 5 fair-foil protocol; universal-
  approx intuition seated in NB 2; per-NB "Your turn" tiers at NB-plan.
