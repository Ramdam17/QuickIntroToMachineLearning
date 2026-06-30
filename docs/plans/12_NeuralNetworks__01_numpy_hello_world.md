# NB plan — 12_NeuralNetworks / 01_numpy_hello_world — a neural network from scratch in numpy

> Status: **APPROVED by Rémy (via ExitPlanMode, 2026-06-29).** NB-plan stage = Rémy validates alone (no
> reviewer gate; both reviewers return on the built notebook). Anchors measured live (numpy + sklearn
> `make_blobs`, SEED=0; `measure_ch12_nb1.py` / `_nb1b.py`). **NB 1 of 10** (chapter 12 — the PyTorch
> finale). **Pure numpy — NO torch** (the `deep` extra lands at NB 7). Chapter plan:
> `docs/plans/chapter_12_NeuralNetworks.md` (APPROVED).

## Context
Chapter 12 (the course finale), **NB 1 of 10 — the hello-world.** Assemble, in one clean place, the
**complete machinery** of a neural net — forward → loss → backward → the training loop (the optimizer) →
train/test → eval — **by hand in numpy**, on a small **2-D 3-class** toy we can see. This is the
**reference implementation**: NB 2 stacks it into a deep net; **NB 7 re-instantiates the SAME architecture
in PyTorch** (autograd computes the backward we write here). Per Rémy: we **SHOW** the machinery (the
learner reads it; challenges live in "Your turn"), and the NB may be thick. The **only genuinely-new math is
the softmax-cross-entropy gradient**; the rest is ch 11 (backprop = chain rule, the softmax head) + ch 00
(train/test, accuracy, scaling) — re-established, never presupposed.

## Recap vs new (the ch-11 boundary — pedagogy/ml-expert fold)
- **RECAP (black-boxed, NOT re-derived):** the chain-rule backward through a hidden layer (ch 11 NB 3);
  ReLU & softmax (ch 11 NB 1/4); the GD loop (ch 03 / ch 11); train/test + accuracy + scaling (ch 00).
- **NEW (the one derivation):** the **softmax-cross-entropy gradient** at the logits —
  `d_logits = (softmax(logits) − onehot(y)) / n` — the **same clean `(p − y)` form** as the binary
  sigmoid-CE gradient (a renamed friend). ch 11 only ever ran the softmax head *inside* `MLPClassifier`;
  here we derive its gradient by hand.

## Live anchors (measured; re-pinned at build)
- **Data:** `make_blobs(n_samples=300, centers=3, n_features=2, cluster_std=1.2, random_state=0)`; stratified
  75/25 → train (225, 2) / test (75, 2); **standardized (fit on train)**.
- **Net:** `2 → 16 (ReLU) → 3 (softmax)`; **small random init (scale 0.1** — "break symmetry", ch-11 recap;
  **He init deferred to NB 4**); biases 0.
- **Untrained net ≈ uniform → initial CE loss ≈ ln(3) ≈ 1.10** (1.09–1.12 across seeds) — the random-guess
  baseline (the lovely "before training, the net guesses" anchor; small init delivers it).
- **Gradient-check** (analytic backward vs central finite differences, eps 1e-6): **rel_err ~3e-8** (order
  1e-8) — the by-hand softmax-CE gradient is correct.
- **Training:** full-batch GD, **lr=0.5, epochs=400 → loss 1.10 → 0.31; train acc 0.87 / test acc 0.89**
  (chance 0.33); robust across seeds (test 0.89–0.91) and lr 0.3–1.0.
- **Decision regions:** 3 carved regions; the few errors sit in the **overlap where the blobs touch** (honest).

## Cell-by-cell (~24 cells, 4 figures) — intuition → implementation → "Read the figure"; **SHOW (never assign)**; ends with "Your turn"
1. **(md) Header** — `# 01 — A neural network from scratch in numpy`; the finale opens by assembling the
   whole machinery in one place — the **reference** we'll re-meet in PyTorch; **Prerequisites** (ch 11: the
   neuron, the hidden layer, backprop = chain rule, the softmax head *shown*; ch 00: train/test, accuracy,
   scaling); **What you'll do** (build a complete multi-class net by hand, train it, evaluate it).
2. **(md) The machinery we'll assemble** — name the pieces (forward · loss · backward/gradient · the training
   loop = optimizer · train/test · eval); most is **recap**, the **one new piece is the softmax-CE gradient**;
   ch 11 built a *binary* net by hand — here a clean, reusable, *multi-class* version.
3. **(code) Imports + data** — `make_blobs` 3-class 2-D std=1.2; stratified 75/25; **standardize on train**;
   print shapes + class balance; `use_course_style()`.
4. **(md) The problem** — 3 classes, 2 features (so we can **see** everything); clean-but-overlapping; the
   binary neuron → K classes needs the **softmax head**.
5. **(code) Fig 1 — look at the data** — 3-class 2-D scatter (train), charter `CLASS_CYCLE`.
6. **(md) Read Fig 1** — three blobs; overlap where they touch (honest errors will live there).
7. **(md) The architecture `2 → 16 → 3`** — a hidden layer (ReLU — carves structure, ch 11) + a **softmax
   output head** (K scores → K probabilities that sum to 1); show the softmax formula.
8. **(code) Fig 2 — the network schematic** (2 → 16 ReLU → 3 softmax; charter colours; matplotlib-drawn).
9. **(md) Read Fig 2** — the shapes `W1 (2×16)`/`b1`/`W2 (16×3)`/`b2`; the forward path.
10. **(md) The forward pass** — `z1=XW1+b1`; `h=ReLU(z1)`; `logits=hW2+b2`; `p=softmax(logits)`. Recap
    ReLU/softmax (ch 11).
11. **(code)** implement small-init params + forward + softmax; **show the untrained net ≈ uniform → initial
    loss ≈ ln(3) ≈ 1.10** (print).
12. **(md) The loss** — cross-entropy = −mean log p[correct class]; ≈ ln(K) when guessing uniformly (1.10 for
    K=3) — the number to drive down.
13. **(md) The backward pass — the ONE new derivation:** the softmax-CE gradient `d_logits=(p−onehot)/n` —
    the same clean `(p−y)` form as the binary case (a renamed friend). Then the rest is the **ch-11 chain
    rule (black-boxed)**: `dW2=hᵀd_logits`, `dh=d_logits·W2ᵀ`, `dz1=dh⊙(z1>0)` [ReLU mask], `dW1=Xᵀdz1`. Say
    it: "the binary backward is ch 11 NB 3 — we package it; the new piece is the softmax-CE gradient."
14. **(code)** implement backward; **gradient-check vs finite differences → rel_err ~3e-8** (print). "We
    trust the gradient because we checked it" (the ch-11 habit).
15. **(md) Read the gradient-check** — rel_err ~1e-8 → analytic == numerical; the math is right.
16. **(md) The training loop (the optimizer)** — full-batch GD: repeat {forward → loss → backward →
    `θ ← θ − η∇L`} for N epochs; define **epoch** (one full pass over the data); the optimizer = the
    ch-11/03 GD loop, now multi-layer + multi-class.
17. **(code)** train (lr=0.5, epochs=400); record the loss curve; **Fig 3 — the loss curve (1.10 → 0.31)**;
    print final loss.
18. **(md) Read Fig 3** — from the random-guess baseline ln(3) down to ~0.31; converged.
19. **(md) Evaluation (the ch-00 habit)** — accuracy on the **held-out test set**, never just train.
20. **(code) Fig 4 — the learned decision regions** on the 2-D toy (inline meshgrid → forward → argmax
    `contourf`; charter colours) + the test points; print **train 0.87 / test 0.89**.
21. **(md) Read Fig 4** — three carved regions; the few errors sit in the overlap (honest); train 0.87 /
    test 0.89 vs chance 0.33 — it clearly learned.
22. **(md) What you built — the reference** — the complete machinery (forward, loss, the softmax-CE
    gradient, the training loop, eval) for a multi-class net, by hand. This exact net is the **reference NB 7
    re-instantiates in PyTorch** and NB 2 stacks into a deep net. The only new math was the softmax-CE
    gradient; the rest is ch 11 + ch 00, packaged.
23. **(md) Your turn** — (warm-up) change `H` (hidden units) and re-train — does the boundary change?;
    (core) change the learning rate (0.05 vs 2.0) — what happens to the loss curve (too slow / unstable)?;
    (core) raise `cluster_std` to 2.0 — how does test accuracy degrade, and *where* do the errors go?;
    (reach) add a second hidden layer (`2→16→16→3`) — extend the forward + backward by one layer (a taste of
    NB 2's depth).
24. **(md) Where this goes next** — NB 2 (depth as a representation hierarchy — what stacking *many* layers
    does; NB 3 finds what breaks) + **References** (ch 11 NB 3 backprop; softmax/CE recap ch 03 NB 5 / ch 11
    NB 4; Goodfellow et al. 2016 §6; the chapter plan).

## `src/` & guards
- **No `src/` change** (notebook-local numpy net; `make_blobs` / `train_test_split`; `viz.use_course_style` /
  `colors`; the multi-class decision-region plot is **inline** — `viz.plot_decision_boundary` expects a
  `.predict` model, so either a tiny predict wrapper or an inline meshgrid contour). **pytest stays 20** (NB 1
  adds no `src/`). **torch NOT used** (the `deep` extra arrives at NB 7).
- Colours only from `ml_course.colors` (no hardcoded hex); `SEED=0`; **"Read the figure" after every figure**
  (incl. the schematic); **never silence output**; banned-word scan 0; ruff/black clean; output-free.
- Build from `build_ch12_nb1.py` (source of truth; rebuild right before `git add` — kernel-drift guard).
- Exit guards: nbconvert exit 0 (4 figures), anchors reproduce (initial loss ≈ 1.10; gradcheck ~3e-8; loss
  1.10 → 0.31; train 0.87 / test 0.89), banned 0, hex clean, ruff clean, output-free; **two-reviewer gate**
  (`@ml-expert-reviewer` + `@pedagogy-reviewer`, no BLOCK) → **Rémy visual** → end-of-NB checklist
  (`gen_llms_txt`, `common_errors` +rows, `course_map` §12 → NB 1 built, pytest 20, STATE) → commit
  `feat(12_neuralnetworks): notebook 01 — a neural network from scratch in numpy` → `git merge --ff-only`
  into `chapter/12_NeuralNetworks`.

## Verification (end-to-end)
1. nbconvert a scratchpad copy → exit 0, 4 figures, anchors reproduce (initial loss ≈ 1.10; gradient-check
   ~3e-8; loss 1.10 → 0.31; train 0.87 / test 0.89).
2. hex + banned + ruff clean. 3. pytest 20. 4. **Rémy validated this NB plan (no reviewer gate; both
   reviewers return on the built notebook).**
