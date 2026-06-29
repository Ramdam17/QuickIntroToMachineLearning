# NB plan — 11_MLP / 03_backpropagation — how a network learns

> Status: **APPROVED by Rémy (via ExitPlanMode, 2026-06-29).** No reviewer gate at the NB-plan stage —
> both reviewers return on the built notebook. All anchors measured live (scikit-learn 1.9.0, SEED=0,
> `measure_ch11_nb3.py` + `_nb3b.py`). The two measured refinements below were presented and **approved**.

## Context
Chapter 11 (MLP), **NB 3 of 5** — the third fundamental. NB 2 left one question open: a hidden layer +
non-linearity *can* separate circles/XOR, but we **hand-set** the weights. **ONE concept here: how a
network *finds* its weights — backpropagation = the chain rule applied layer by layer.** By hand on a
`2-4-1` sigmoid net: a forward pass that **caches** the hidden activation; a backward pass that carries the
output error back (`d_out → dH → dW1,dW2`); **gradient-check vs finite differences** (the proof the math is
right); then the **ch 03 GD loop, now multi-layer**, trained until it solves circles; finally **why init
must break symmetry**. The general L-layer algorithm is stated as a **picture**, not a multi-index grind.
~22 cells, 3 figures. NB-plan stage = **Rémy validates alone** (no reviewer gate).

## Measured refinements vs the chapter plan (APPROVED)
Two chapter-plan §NB 3 anchors did not survive live measurement; the measured versions were approved (both
make the NB *more* correct, and both lean on the chapter's own non-convexity / symmetry themes):

1. **Parity is init-dependent, not "1.0 == 1.0".** The chapter plan said the by-hand 2-4-1 GD net reaches
   train acc 1.0 "matching `MLPClassifier((4,),'logistic','lbfgs')`". Measured: the **by-hand net reaches
   train 1.0 / test 1.0 robustly (all 5 seeds)**, but **lbfgs lands init-dependent train 0.85–1.0 (mean
   0.94) / test 0.76–1.0 (mean 0.90)** — rs=0 → 0.98/0.95; more iterations don't move it. That gap **is**
   the non-convexity the chapter flags: different starts settle in different good minima. Framing: "the
   by-hand net solves it (1.0/1.0); the library reaches the same closed-ring boundary at comparable accuracy,
   the exact number init-dependent — a first sighting of non-convexity (NB 4 returns to it)."
   *(Footnote: sigmoid + `adam`/`sgd` STALL at chance here — 0.48/0.50 — so the by-hand build uses full-batch
   GD and the parity pins `lbfgs`; the saturation fix is ReLU in NB 4.)*

2. **The symmetry sub-claim was wrong.** The chapter plan said "W1=0/W2≠0 → identical-nonzero columns (units
   never diverge)". Measured: **W1=0/W2≠0 BREAKS symmetry** (W2's randomness propagates into dW1; col-spread
   0→3.37, train acc 1.0). The genuine frozen/stuck cases are: **full zeros** (all gradients *exactly* 0,
   loss frozen at ln2=0.6931, acc 0.50) and **fully-symmetric units** (identical W1 columns *and* identical
   W2 → identical gradients → col-spread stays 0→0, stuck at acc 0.68 = effective width 1). Demonstrated:
   those two vs random init (col-spread 0.47→3.66, acc 1.0).

## Live anchors (measured, scikit-learn 1.9.0, SEED=0)
- **Data:** the exact NB-2 circles — `make_circles(n_samples=400, noise=0.10, factor=0.40, random_state=0)`,
  stratified 75/25 → `X_tr (300,2)`, balanced 150/150, `mean(y_tr)=0.5000`.
- **Gradient check (H=4, 17 params, central differences ε=1e-6):** `rel_err = 6.0e-10` (the hand-derived
  backward pass matches the numerical gradient to ~10 digits).
- **By-hand 2-4-1 GD** (sigmoid hidden+output, no penalty, `lr=3.0`, `iters=8000`, init seed=SEED): **train
  1.0 / test 1.0** (robust across seeds 0–4); loss `~0.71 → ~0.004`.
- **Parity** `MLPClassifier((4,),'logistic','lbfgs',alpha=1e-4)`: rs=0 → **train 0.98 / test 0.95**; across
  rs 0–4 → train 0.85–1.0 (mean 0.94) / test 0.76–1.0 (mean 0.90), unchanged at `max_iter` 5000 vs 20000
  (non-convex; init-dependent). `adam` 0.48 / `sgd` 0.50 (stall) vs `lbfgs` 0.98 — why we pin `lbfgs`.
- **Symmetry breaking (lr=1.0, iters=15000):**
  - **full zeros** → `|grad0|=[0,0,0,0]`, loss `0.6931→0.6931` (=ln 2), train acc **0.50**, W1 col-spread 0→0.
  - **fully-symmetric** (identical W1 cols + identical W2) → loss `0.70→0.56`, train acc **0.68**, col-spread
    **0→0** (units never differentiate — effective width 1).
  - **random** → loss `0.70→0.007`, train acc **1.0**, col-spread **0.47→3.66** (units diverge → learns).

## Cell-by-cell (~22 cells, 3 figures) — intuition → implementation → "Read the figure"
1. **(md) Header** — `# 03 — How a network learns: backpropagation`; purpose; **Prerequisites** (11.1 the
   neuron == logistic; 11.2 hidden layer + why non-linearity; ch 03 NB 3–4: log-loss, the single-neuron
   gradient `(P−y)x`, the GD update `θ←θ−η∇L`); **What you'll be able to do** (run a forward pass that caches
   activations; derive & code the backward pass = the chain rule; gradient-check it to machine precision;
   train a 2-4-1 net by GD to solve circles; explain why init must break symmetry); warm welcome.
2. **(code) Imports + data** — `numpy`, `matplotlib.pyplot`; sklearn `make_circles`, `train_test_split`,
   `MLPClassifier`; `from ml_course import colors, viz`; `viz.use_course_style()`; `SEED=0`; build the exact
   NB-2 circles + stratified 75/25 split; print shape + balance.
3. **(md) Recap & the open question** — NB 2 hand-SET weights to solve XOR/circles; how does a network
   *find* them? Answer: gradient descent (ch 03) on the network's loss — and to get the gradient *through a
   hidden layer*, the chain rule. Recap ch 03: log-loss, the single-neuron gradient `(P−y)x`, the update.
4. **(md) Intuition — forward caches, backward carries the error back.** The net is a composition
   `x→(W1,b1)→σ→H→(W2,b2)→σ→P`. To nudge `W1` (buried inside) we need `∂L/∂W1`; the chain rule walks the
   dependency **backward**, layer by layer. Forward: compute and **cache** `H`. Backward: start at the
   output error, pull it back through `W2`, gate it by the hidden slope, reach `W1`.
5. **(code) The forward pass (with caching)** — `sigmoid`; `forward(params, X)` → `(Z1, H, Z2, P)`, `H`
   explicitly cached; build the 2-4-1 params; print shapes + `P` on a few rows.
6. **(md) The chain rule, written once (a picture, not a grind).** Three links: (i) output error
   `d_out = P − y` (ch 03 — the sigmoid+log-loss gradient, unchanged); (ii) pull back through `W2`: the
   hidden layer's share is `d_out · W2ᵀ`; (iii) gate by the hidden slope `σ'(Z1)=H·(1−H)`:
   `dH = (d_out · W2ᵀ) ⊙ H(1−H)`. Then each weight gradient is just **error × input**: `dW2 = Hᵀ d_out`,
   `dW1 = Xᵀ dH`. State the **mean-loss 1/n convention** (folded into `d_out`) — a flagged numerical choice.
7. **(code) The backward pass** — `backward(params, X, y)`: `d_out=(P−y)/n`, `dW2`, `db2`,
   `dH=(d_out@W2.T)*H*(1−H)`, `dW1`, `db1`; return the four gradients. (The heart of the notebook.)
8. **(md) Intuition — is our gradient correct? Gradient checking.** Before trusting it: nudge each weight
   by ±ε, measure the actual loss change, compare to the analytic gradient (central finite differences).
   Agreement to ~1e-9 means the chain-rule code is right — this is how every DL library is verified.
9. **(code) Gradient check** — central differences over all 17 params; print `rel_err = 6.0e-10`.
10. **(md) Read** — 6e-10: the hand-derived backward pass matches the numerical gradient to ~10 digits. The
    chain rule is implemented correctly. (Celebrate — quietly rigorous.)
11. **(code) Fig 1 — the backprop picture (schematic)** — matplotlib 2-4-1 schematic: forward arrows
    left→right (`x → H` cached `→ P`) in one charter colour; backward arrows right→left
    (`d_out → dH → dW1,dW2`) in another; annotate the chain-rule gates. No data — pure diagram.
12. **(md) Read Fig 1** — forward fills activations left→right (cache `H`); backward sends the error
    right→left, pulled through `W2` and gated by `H(1−H)`. The same two passes for any depth.
13. **(md) Intuition — now descend.** With a correct gradient, training is the ch 03 loop unchanged:
    `θ←θ−η∇L`, repeated. The only new thing is that `∇L` now comes from backprop.
14. **(code) Train by full-batch GD + Fig 2** — GD (`lr=3.0`, `iters=8000`) from the SEED init; track loss;
    print **train/test 1.0**. Fit `MLPClassifier((4,),'logistic','lbfgs')` for parity (rs=0 → 0.98/0.95).
    Fig 2: (a) the loss curve descending `0.71→0.004`; (b) the by-hand decision boundary on circles (closed
    ring) with the lbfgs boundary overlaid (dashed) — same shape.
15. **(md) Read Fig 2** — the loss falls to ~0.004; the by-hand net reaches **train/test 1.0**, carving the
    closed ring NB 2's single neuron could not. `MLPClassifier` (lbfgs) reaches the **same shape** at
    comparable accuracy (0.98/0.95 here; 0.85–1.0 across inits — the loss is **non-convex**, different starts
    find different good minima; NB 4 returns to this). The library automates exactly the forward/backward/
    descent we built. *(Footnote: full-batch GD / lbfgs here because sigmoid + mini-batch adam/sgd stalls on
    this problem — the saturation issue ReLU fixes in NB 4.)*
16. **(md) Intuition — why init must not be symmetric.** If two hidden units start identical, they compute
    the same thing, get the same gradient, and update identically — forever. They never differentiate; the
    width is wasted. The cure: random init breaks the symmetry.
17. **(code) Symmetry experiment + Fig 3** — train 3 inits (`lr=1.0`, `iters=15000`): **full zeros**,
    **fully-symmetric** (identical units), **random**. Print: zeros `|grad|=0` / loss flat 0.6931 / acc
    0.50; symmetric col-spread 0→0 / acc 0.68; random col-spread 0.47→3.66 / acc 1.0. Fig 3: (a) the 4
    hidden-unit incoming-weight vectors after training (symmetric → 1 point; random → 4 distinct); (b) the
    three loss curves (zeros flat at ln 2; symmetric stuck ~0.56; random → 0).
18. **(md) Read Fig 3** — full zeros: every gradient is exactly 0 (loss stuck at `ln 2 = 0.6931`, acc 0.50 —
    the net never moves). Identical units stay identical (the 4 vectors sit on one point; acc stuck 0.68 — an
    effective width of 1). Only random init lets the units diverge (4 distinct vectors) and the net learns
    (1.0). Random init is not a detail — it is what makes the hidden layer real.
19. **(md) The general algorithm (a picture).** For any number of layers, the same two passes: forward
    caches each layer's activation; backward sends the error back through each layer — pull through the
    weights (`Wᵀ`), gate by the activation's slope `φ'(z)`. That recursion **is** backpropagation
    (Rumelhart–Hinton–Williams 1986). State `δ^(l)=(W^(l+1)ᵀ δ^(l+1)) ⊙ φ'(z^(l))` once, read in words — no
    multi-index grind.
20. **(md) Your turn** — (warm-up) set `lr` to 0.1 then to 30 — predict, then watch the loss curve (too
    slow / unstable); (core) re-run the symmetric init but make **only W2 identical while W1 is random** —
    does symmetry break? (you measured the mechanism — which weights must differ?); (reach) add a second
    hidden layer (2-4-4-1): extend the forward cache and the backward chain by one link, and gradient-check
    it.
21. **(md) What you built** — derived & coded the backward pass (the chain rule); gradient-checked it to
    6e-10; trained a 2-4-1 net by GD to **train/test 1.0** on circles (the ring NB 2's neuron couldn't); saw
    why init must break symmetry (zeros frozen, identical units stuck, random learns). Next: NB 4 — the real
    `MLPClassifier`, its parameters, **ReLU**, and the **K-class softmax head**.
22. **(md) References** — Rumelhart, Hinton & Williams 1986 (backprop; DOI 10.1038/323533a0); LeCun, Bottou,
    Orr & Müller 1998, *Efficient BackProp* (gradient checking / practical backprop; DOI
    10.1007/3-540-49430-8_2); Glorot & Bengio 2010 (init / symmetry breaking; PMLR 9:249–256); Goodfellow,
    Bengio & Courville 2016 §6.5 (backprop). Previous: **11.2 — the hidden layer**. Next: **11.4 —
    MLPClassifier & its parameters**.

## `src/` & guards
- **No `src/` change** (notebook-local numpy by-hand net; `make_circles`; `MLPClassifier`;
  `viz.plot_decision_boundary`; pytest 20). Colours only from `ml_course.colors`; `SEED=0`; "Read the
  figure" after every figure; **never silence output** (no `verbose=False`-as-suppression; any
  `ConvergenceWarning` stays visible); banned-word scan 0; hex clean; ruff/black clean; output-free.
- Build from `build_ch11_nb3.py` (source of truth; rebuild right before `git add` — kernel-drift guard).
- Exit guards: nbconvert exit 0 (3 figures), banned 0, hex clean, ruff clean, output-free; **two-reviewer
  gate** (`@ml-expert-reviewer` + `@pedagogy-reviewer`, no BLOCK) → **Rémy visual** → end-of-NB checklist
  (`gen_llms_txt`, `common_errors` +rows, `course_map` §11 note, pytest 20, STATE) → commit
  `feat(11_mlp): notebook 03 — backpropagation` → `git merge --ff-only` into `chapter/11_MLP`.

## Verification (end-to-end)
1. nbconvert a scratchpad copy → exit 0, 3 figures, anchors reproduce (grad-check 6e-10; by-hand train/test
   1.0; parity lbfgs 0.98/0.95; full-zeros loss ln2/acc 0.50; symmetric col-spread 0/acc 0.68; random
   col-spread 3.66/acc 1.0).
2. hex + banned + ruff clean. 3. pytest 20. 4. **Rémy validates this NB plan (no reviewer gate); both
   reviewers return on the built notebook.**
