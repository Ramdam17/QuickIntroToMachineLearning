# Chapter plan — 12_NeuralNetworks (the course finale, 13th & final module)

> Status: **APPROVED by Rémy (via ExitPlanMode + the two-reviewer gate, 2026-06-29).** Framework = **PyTorch**
> (Rémy chose B *with eyes open*; the concept-cartographer had recommended A / stay-in-sklearn). **~10
> thicker notebooks** — Rémy deliberately **derogates the 5-NB ceiling for this finale** (NBs were kept
> light on purpose; the last chapter may be dense). Concept tour by `@concept-cartographer`; **two-reviewer
> gate run (`@ml-expert-reviewer` + `@pedagogy-reviewer`, both REVISE→no-BLOCK; all folds applied below).**
> Every sklearn/numpy anchor measured live (sklearn 1.9.0 / numpy 2.4.6); **torch anchors are build-time.**
> Per-NB plans drafted and Rémy-validated one at a time before each build.

## Context
Chapters 00–11 are complete and merged to `main` (ch 11 MLP via PR #11). Chapter 12 **closes the course**
(12/13 modules done): the move from **"the MLP scikit-learn ships" (ch 11)** to **"neural networks as a
paradigm."** It completes the per-method arc one last time and ends the whole 13-module journey.

Two decisions shape it, both Rémy's:
1. **Framework = B — introduce PyTorch.** The whole course is sklearn `.fit()`; ch 12 introduces a real
   deep-learning framework as the genuine modern tool. **The cartographer recommended A** (stay in
   sklearn + numpy) on dependency-weight / torch-reproducibility / charter-continuity grounds, having
   *verified every new concept is demonstrable in pure numpy*. Surfaced to Rémy, who chose **B with eyes
   open** for a faithful finale. Carried mitigations: see "The framework decision".
2. **~10 notebooks, thicker than usual.** The fundamentals stay one-concept-per-notebook; we **thicken, we
   do not pad**. Two parallel **"hello-world" notebooks** (a NN in numpy, then the same NN in PyTorch) are
   the spine of the framework move.

## The framework decision (the crux)
- **Plain PyTorch, SHOWN — never hand-coded by the learner.** Rémy's rule: *"on montre, comme on a toujours
  fait"* — the canonical torch hello-world (model, optimizer, the standard training loop, eval) is
  **demonstrated for the learner to read**, not framed as a code-it-yourself task. Challenges live in the
  **"Your turn"** exercises (and, for torch, must *modify* the shown loop — swap optimizer, add a layer,
  change init — never author it from blank). **No skorch / Lightning — do not complicate.**
- **Ethos intact:** ALL fundamentals (NB 1–6) are built **by hand in numpy**, before any framework. PyTorch
  appears only in NB 7–9. Both reviewers confirmed this is the *same* move as every prior chapter's
  estimator notebook ("build the mechanism by hand, then *show* the real tool's API"), not a break from it.
- **Value B delivers that sklearn can't:** real `nn.Dropout` / `nn.BatchNorm` / `nn.init` / optimizers /
  arbitrary architectures, and the actual modern workflow as the course's send-off. (`MLPClassifier` has no
  dropout and no batch-norm — verified.)
- **Reproducibility (the cartographer's + ml-expert's worry, addressed):** pinned seeds + `torch.manual_seed`
  + `torch.use_deterministic_algorithms(True)` + **CPU only** + a documented single thread; **pin the torch
  version** in the `deep` extra and **verify the determinism contract on the actual dev box** at the NB-7
  build before relying on any torch *number* as an anchor (CPU torch can wobble in the last decimal across
  BLAS/thread/version where sklearn+numpy is bit-stable). The one-time dataset fetch is **logged, never
  silenced**.
- **Dependency:** a new optional **`deep`** extra (CPU `torch`), mirroring the existing `boosting` extra;
  `uv sync --extra deep`. **torch install + a trivial deterministic example is the FIRST build step of NB 7**
  (flagged risk; macOS/py3.12 has CPU wheels — verify early).

## The ch 11 / ch 12 line (what is genuinely new vs the floor)
**Recap-only (ch 11 / earlier owns it):** neuron = φ(w·x+b); hidden layer + non-linearity; backprop = chain
rule; GD/SGD/mini-batch/epoch/learning-rate; sigmoid/tanh/ReLU (ReLU default; the *shallow* sigmoid+adam
stall); L2/`alpha` + early stopping; the K-class softmax head (shown via schematic + formula recap, **never
hand-coded**); scaling + `Pipeline` + fit-on-train-only; over/underfitting + the learning curve + CV +
confusion matrix; non-convexity + seed-variance; UAT = existence; "no universal best" + the tabular limit.

**Genuinely-new (ch 12 owns it):** **c01** depth as a *representation hierarchy*; **c02** vanishing/exploding
gradients (the pivot — what breaks when you stack many layers); **c03** modern variance-preserving init
(He / Xavier); **c04** normalization (batch / layer norm); **c05** dropout; **c06** deep-net diagnostics
(under/overfitting *and* optimization-failure — a flat loss curve from vanishing gradients ≠ convergence);
**c07** momentum / Adam, the *why-deep* (a deepened recap, not a new build); **c08** the framework move
(autograd + the canonical training loop, **shown** in PyTorch); **c09/c10/c11** CNN / RNN / transformer
(**named horizons**, not built — the finale's gesture). **One more genuinely-new derivation lives in NB 1:**
the **softmax-cross-entropy gradient** (ch 11 only ever ran the softmax head *inside* `MLPClassifier`).

The load-bearing edge: **c02 is the pivot.** It is just ch 11's backprop (the chain rule) followed across
*many* layers — "what happens when you do it ten times in a row?" — and c03 (init), c04 (norm), c06 (the
flat-curve diagnostic) all exist *because of* c02. That is how the chapter is genuinely new without
re-teaching ch 11.

## Live ground truth (measured live by both the cartographer AND the ml-expert reviewer; sklearn 1.9.0 / numpy 2.4.6; **torch anchors = build-time**)
- **Vanishing/exploding (the pivot):** a 10-layer stack's backward gradient RMS — **sigmoid + small init
  collapses 2.5e-1 → ~1e-16** (≈16 orders of magnitude); **ReLU + unit-Gaussian init explodes → ~5e6**.
- **As a training outcome:** a **5-layer sigmoid** `MLPClassifier` scores **0.500 (chance, all 5 seeds)** on
  circles, while **5-layer tanh / ReLU score ~1.000**. Depth makes the shallow sigmoid stall (ch 11 NB 4)
  *unrecoverable* — and sklearn's own estimator demonstrates it end to end via `activation`/`hidden_layer_sizes`.
- **He init fixes it:** holds ReLU's gradient RMS **~constant (0.71–0.80) across all 10 layers** (live).
  **Xavier holds tanh** in a healthy band — but **sigmoid is the awkward case** (not zero-centered; Xavier
  *slows* its decay but does not fully preserve it — Glorot recommends a larger gain for sigmoid). This
  itself motivates why tanh/ReLU displaced sigmoid as the default. (So: **He for ReLU, Xavier for tanh**;
  sigmoid the cautionary case — NOT "Xavier rescues sigmoid".)
- **Dropout:** inverted dropout preserves the expected activation (mean ~0 → ~0) while injecting variance
  (1.0 → 2.0) — an implicit ensemble.
- **Depth helps modestly & honestly** (MNIST-10k): MLP `(50,)` **0.941** → `(256,128,64)` **0.949** — the
  honest version, not hype. (NB: this is *more units arranged with depth*, NOT an equal-unit-budget claim —
  an equal-budget `(448,)` vs `(256,128,64)` check at NB-2 plan may be a near-wash, itself the honest lesson.)
- **Capstone foil — illustrative MNIST-10k proxy only** (live 5-fold CV, seed 0): MLP **0.944 ± 0.003** ≈ RF
  **0.946 ± 0.006** ≈ HistGB **0.949 ± 0.004** — all ~0.94–0.95, HistGB modestly ahead **but within ~1.5
  combined std (a near-tie, NOT a clean ordering)**. *(This corrects the chapter-plan-era "< HistGB ~0.958"
  — the exact tie-vs-ordering mistake ch 11 NB 5 already folded; a single split crowns a tree spuriously.)*
- **Capstone verdict rests on the ACTUAL data — Fashion-MNIST** (live 5-fold CV, 10k subset): MLP **0.859 ±
  0.010** ≈ RF **0.853 ± 0.007** < **HistGB 0.874 ± 0.006** — a tree **genuinely wins by ~1.4 pp (not within
  noise)**. The "competitive, not superior — and a tree even wins; the pixels have structure a dense net
  discards → CNN" punchline is **sharper and cleaner** on the real capstone data than on the MNIST proxy.
- **Tabular-humility aside (`breast_cancer`, live CV):** MLP **0.975** > RF **0.967** / HistGB **0.970** —
  the genuine "an MLP *can* win on clean homogeneous tabular" fact (re-used from ch 11, not the capstone).
- **Datasets verified offline / CPU:** `fetch_openml('mnist_784')` and `('Fashion-MNIST')` (70k×784,
  10-class), ~12 s first fetch then cached; a 10k-train / 5k-test subset trains in ~1–2 s.

(Every number re-measured and reconciled at NB-plan/build; seeds pinned. The torch anchors — autograd
gradient-match vs the NB-1 numpy net, dropout/BN effects, the capstone — are measured once torch is
installed at the NB-7 build, on the actual box, with the determinism contract verified.)

## Prerequisites (from earlier chapters)
The tour's key finding: ch 11 + ch 00 already built almost everything ch 12 *uses*; ch 12 adds what happens
**with depth**.

| Prerequisite | Where built | How ch 12 uses it |
|---|---|---|
| Neuron, hidden layer, forward pass, the weight matrix | ch 11 NB 1–2 | The layer we now **stack** (c01) |
| **Backprop = chain rule, layer by layer** (gradient-checked by hand, rel_err ~6e-10) | ch 11 NB 3 | **Restated/black-boxed** as a reusable function in NB 1 (NOT re-derived); followed across *many* layers → **vanishing/exploding** (c02); autograd later *computes* it (c08) |
| Softmax head + K-class cross-entropy (shown via schematic + formula, never hand-coded) | ch 11 NB 4 (← ch 03 NB 5) | NB 1 **derives the softmax-cross-entropy gradient by hand** for the first time (its one new derivation) |
| GD / SGD / mini-batch / epoch / batch_size / learning rate | ch 03 NB 4, ch 11 NB 4 | The training loop the torch hello-world **shows** (recap, not re-taught) |
| sigmoid / tanh / ReLU; the shallow sigmoid+adam stall | ch 11 NB 1, NB 4 | The *depth*-driven version of saturation (c02); why ReLU + He |
| Weight init must break symmetry (random, not zeros) | ch 11 NB 3 | The *existence* of the init problem → **variance-preserving** init (c03) |
| L2 / `alpha`; early stopping | ch 03 NB 5, ch 00 NB 9, ch 11 NB 4 | The regularizers **dropout** is contrasted against (c05) |
| Over/underfitting, generalization gap, the learning curve, CV | ch 00 NB 9–10 | Deep-net diagnostics incl. **optimization failure** (c06) |
| Scaling, `Pipeline`, fit-on-train-only, confusion matrix, baselines, seed-variance | ch 00 NB 11, ch 11 NB 5 | Capstone hygiene + the tree foil (applied, not re-taught) |
| Non-convexity; "no universal best"; the tabular limit; UAT = existence | ch 11 NB 3/5, ch 10 NB 5, ch 11 NB 2 | The honest capstone verdict + the CNN bridge |

**Re-established briefly (we *show*, never presuppose):** the forward pass, the loss, the backward pass, the
training loop, train/test, eval, the optimizer (consolidated cleanly in NB 1). **Genuinely new:** depth as a
hierarchy; vanishing/exploding gradients; He/Xavier; normalization; dropout; deep-net + optimization-failure
diagnostics; the softmax-CE gradient (NB 1); autograd + the framework move (PyTorch, shown);
CNN/RNN/transformer named.

## The per-method arc (10 notebooks)
By hand before the library. **NB 1–6 = pure numpy** (the ethos). **NB 7–9 = PyTorch, shown** (canonical, not
hand-coded). **NB 10 = the finale.** "Read the figure" after **every** figure (schematics included); every
NB ends with "Your turn" (NB 10's is reflective — see below).

### NB 1 — A neural network from scratch in numpy (the hello-world / our reference)
- **Concept:** assemble the complete machinery in one clean place — forward → loss → backward → the short GD
  **training loop** → train/test → **eval** → the optimizer. We **show** it end to end; this is the
  **reference implementation** the PyTorch hello-world (NB 7) re-instantiates.
- **Pinned (pedagogy MAJOR-1 + ml-expert MINOR):** the reference net is **multi-class (softmax + cross-entropy)**
  on a **small 2-D multi-class toy** (`make_blobs`, 3 classes) — so it is the honest reference for everything
  downstream, and the 2-D prediction figure is drawable. The **binary chain-rule backward is ch-11-NB-3
  recap** — *restate it as a black-boxed reusable function ("the derivation lives in 11.3; here we package
  it"), do NOT re-derive it.* NB 1's **one genuinely-new derivation is the softmax-cross-entropy gradient**;
  the rest is packaging + the reference role. NB 7 re-instantiates **this same architecture** in torch (the
  toy is the "same net"; NB 7 may *additionally* scale it once shown).
- Figs: the net schematic; the loss curve; the learned 3-class predictions on the 2-D toy.

### NB 2 — Depth is a representation hierarchy (c01)
- **Concept:** *why* stack layers — successive layers compose features; depth (not just width) is the bet.
- By hand: stack the NB-1 forward pass into an `L`-layer net; show what successive hidden layers represent on
  a 2-D problem; show — **honestly, measured** — that depth helps (MNIST-10k `(50,)` 0.941 → `(256,128,64)`
  0.949). **Drop any "equal unit-budget" claim** (the anchor compares 50 vs 448 units): land on "depth
  remaps the feature space; a modest, honest gain." *Optional NB-plan add:* an equal-budget `(448,)` vs
  `(256,128,64)` check — likely a near-wash, which is itself the honest lesson.
- *Honesty:* the *vivid* edge→part→object hierarchy is a CNN phenomenon (c09); NB 2 shows the honest version
  and defers the vivid picture to the horizon.
- Figs: layer-by-layer activation remap; depth-vs-accuracy (honest, modest).

### NB 3 — Vanishing & exploding gradients (c02 — the pivot)
- **Concept:** backprop through many layers multiplies many factors → the gradient signal collapses or blows
  up. The reason naive deep nets don't train.
- By hand: run ch 11's backward pass through a 10-layer stack; **measure the per-layer gradient RMS** (sigmoid
  + small init → ~1e-16; ReLU + unit-Gaussian → ~5e6); then the *training* proof — a 5-layer sigmoid net at
  **0.500** vs tanh/ReLU at **~1.000**.
- Figs: gradient-RMS-by-layer (vanish vs explode); the flat sigmoid loss curve vs the descending ReLU one;
  the chain-of-factors schematic.

### NB 4 — Initialization: He & Xavier (c03)
- **Concept:** choose the initial weight variance so signal/gradient magnitude is ~preserved across depth —
  the concrete fix for c02 (and the real content behind ch 11's "break symmetry"). **He for ReLU, Xavier for
  tanh** (the symmetric/linear regime). **Sigmoid is the awkward, non-zero-centered case** — Xavier slows but
  does not fully preserve it (Glorot's larger-gain note); say so in one clause, and let it *motivate* why
  tanh/ReLU are the defaults (ties back to c02). **Do not claim "Xavier rescues sigmoid".**
- By hand: derive the variance-preserving idea; implement He and Xavier; re-run NB 3's measurement → He
  flattens ReLU's gradient RMS to ~0.71–0.80 across all 10 layers.
- Figs: gradient-RMS before/after He & Xavier; activation distributions (drift vs preserved).

### NB 5 — Dropout (c05)
- **Concept:** randomly zero a fraction of activations each step (inverted dropout rescales) → an implicit
  ensemble; **how it differs from L2 / early-stopping** (stochastic co-adaptation breaking, not a weight
  penalty or a stop rule).
- By hand (numpy): inverted dropout preserving the expected activation (mean ~0, variance 1→2); show it reduce
  overfitting on a small net; contrast explicitly with `alpha` (L2). (`MLPClassifier` has no dropout — it
  *must* be by hand here; the real `nn.Dropout` layer arrives in NB 8.)
- Figs: the dropout mask / expected-activation preservation; train-vs-val with vs without dropout.

### NB 6 — Normalization: batch & layer norm (c04)
- **Concept:** re-center/re-scale activations per layer to keep them in a healthy range → stabler, faster
  training. The same "control the variance with depth" lever as init, applied *during* training.
- **By hand — a complete step (pedagogy MAJOR-2 + ml-expert MINOR):** implement a **batch-norm forward layer**
  in numpy (batch mean/var, normalize, the **learnable γ/β**), show it on the drifting-activation stack, AND
  demonstrate a **training-time effect** (a deeper net that stalled now trains / the loss descends faster) —
  so the by-hand mechanism is *complete* and earns the standalone NB. Then **name, in one sentence, what the
  real `nn.BatchNorm` (NB 8) adds beyond the forward transform: the running statistics and the train/eval-mode
  difference** — so the learner carries an honest, non-static model into NB 8 (no oversold from-scratch BN
  *trainer*; the train/eval+running-stats nuance is explicitly NB 8's).
- Figs: activation distributions drifting vs normalized; the train-time loss effect; the BN-layer schematic.

### NB 7 — Hello-world in PyTorch (c08 — the framework move, SHOWN)
- **Concept:** the real modern tool — define a model, **autograd builds the backward pass for you**, the
  canonical training loop, optimizer, train/eval mode. The **same net as NB 1**, now in torch.
- **Shown, not assigned** (Rémy): the standard `nn.Module` / `nn.Sequential`, the canonical loop
  (`zero_grad → forward → loss → backward → step`) demonstrated for the learner to read. **Plain torch, minimal.**
- **The by-hand↔shown bridge — an explicit beat, not a one-liner (pedagogy MAJOR-3):** the planned
  numpy-vs-torch side-by-side gets a "Read the figure" that maps the correspondence **component by component**
  — *your* `forward` ↔ `nn.Module.forward`; *your* hand-derived `backward` (NB 1/NB 3) ↔ `loss.backward()`
  (autograd); *your* `θ ← θ − η∇L` ↔ `optimizer.step()`; *your* loss ↔ the criterion — and **shows the one
  number**: the autograd gradient agrees with the NB-1 numpy gradient (a build anchor). Message: "the
  framework is *exactly what you built*, validated — not a black box that replaces understanding."
- **Overlap-watch (pedagogy MINOR):** epoch / mini-batch / loss-curve-as-diagnostic are **ch 11 NB 4 recap**
  (one sentence); the genuinely-new beats are **autograd** (the backward you no longer write), the **`nn.Module`
  idiom**, and **`.train()` / `.eval()` mode** (new — it matters the moment dropout/BN appear in NB 8).
- *Build-time first step:* install the `deep` extra + verify a trivial CPU-deterministic torch example **on
  the actual box**; pin the torch version.
- Figs: numpy-vs-torch side by side (same result + the matching gradient); the torch loss curve.

### NB 8 — The model and its parameters in PyTorch (the estimator notebook)
- **Integrative**, each knob from its owning concept: depth/width (capacity); **`activation` as the
  depth-pathology knob** (flip to sigmoid → a deep net stalls — c02, now a tunable failure); optimizers
  (SGD / momentum / Adam — **c07**, *why they matter more with depth*); learning rate; **real `nn.Dropout`
  (c05) and `nn.BatchNorm` (c04, incl. the running-stats + `.train()/.eval()` nuance NB 6 named)** as one-line
  layers; init via `torch.nn.init` (He/Xavier — c03); epochs / batch size; **train-vs-validation loss curves**
  to separate underfitting / overfitting / **optimization failure** (the flat-curve case — c06); honest tuning
  (a small grid) → one sealed test.
- *Overlap-watch:* `alpha`/early-stopping/Adam were named in ch 11 — recap in one tight paragraph; the new
  owners here are real dropout/BN layers, the activation-as-pathology knob, and the optimization-failure
  reading.
- Figs: deep-vs-shallow capacity & loss curves; the activation-knob stall; dropout/BN effect; init effect.

### NB 9 — Capstone: Fashion-MNIST, end to end (visualization-first; ≥6 figures, ~28–30 cells)
- A full honest workflow on **Fashion-MNIST** (70k×784 → 10k-train / 5k-test subset; offline, CPU): look at
  the data → baselines → a **deep, He-initialized, dropout-regularized** torch net (scaled pipeline) → loss
  curve & convergence → held-out **confusion matrix + error gallery** → **seed-variance** → a **fair
  cross-method foil on CV** (the torch net vs RF / HistGB on **raw** pixels, same protocol) → the **honest
  verdict.**
- **The workflow beats are applied, NOT re-taught (pedagogy MINOR):** baseline → loss curve → confusion →
  error gallery → seed-variance → CV foil are the learner's reflex by the finale; NB 9's genuinely-new content
  is **the torch deep net in the pipeline, the He/dropout stack as the thing under test, and the
  spatial-structure verdict** (the first-layer-weight gallery → the CNN bridge).
- **Honest verdict (measured; re-pinned at build):** on Fashion-MNIST the deep net is **competitive, and a
  tree even wins** — live 5-fold CV MLP **0.859** ≈ RF **0.853** < **HistGB 0.874** (a real ~1.4 pp gap);
  trees need no scaling; and — the finale's punchline — **the pixels have spatial structure a dense net
  throws away: this is a CNN's job (c09).** A short **tabular-humility aside** re-states ch 11's measured
  "an MLP *can* win on clean homogeneous tabular" (`breast_cancer` 0.975 > 0.967/0.970) so "no universal
  best" lands **both ways**.
- Figs: image gallery; loss curve; confusion matrix; error gallery; seed-variance strip; cross-method CV
  bars; (optional) a **first-layer-weight gallery** hinting at edge-like features → the CNN bridge.
- *Fashion-MNIST chosen over MNIST* (Rémy ✅): harder, **sharper measured verdict** (a tree genuinely wins),
  same offline/CPU profile.

### NB 10 — Where ML goes next, and the whole course (the finale, separate — Rémy ✅)
- **CNN / RNN / transformer named, not built** (c09/c10/c11), *motivated by NB 9's verdict*: a dense net
  flattens the image and discards spatial structure → **CNNs** (weight sharing + locality); **RNNs /
  transformers** extend "learn representations" to sequences and to the attention models behind today's LLMs.
- **The whole-course synthesis:** the spine — instance-based (KNN) → probabilistic (Naive Bayes) → linear
  (Logistic Regression) → partitions (Decision Tree) → margins (SVM) → ensembles (RF → AdaBoost → GBM →
  XGBoost → LightGBM) → learned representations (MLP → Neural Networks) — and the earned through-line:
  **there is no universal best model; the right tool depends on the data, the constraints, and what you can
  honestly evaluate.**
- **The ONE intentional exception (pedagogy MINOR):** NB 10 has **no by-hand mechanism and no executable
  "Your turn"** — it closes with a **reflective "Where to go next"** (directional prompts — "name which of
  the 13 methods you'd reach for on a dataset from your own world, and why"; "which problems in your field
  have the spatial/sequential structure that calls for a CNN/transformer" — plus the framework / next-course
  pointer, **not** code) and a **whole-course "What you built"** celebration. Stated as deliberate, so it
  reads as an earned finale, not a thinned NB. Mostly prose + 1–2 schematic figures (each with "Read the
  figure").

## Honest scoping (the ml-expert bar)
- **Depth helps modestly & honestly** — no hype; the true feature hierarchy is a CNN phenomenon (deferred to
  c09). NB 2 shows the honest version and **drops the equal-budget claim** its anchor doesn't establish.
- **Vanishing/exploding is a *mechanism* demonstration** on small/toy data with pinned seeds — not a claim
  about every architecture. Anchors (≈1e-16 / ~5e6; 0.500 vs ~1.000; He → flat RMS) verified live.
- **Init honesty:** He for ReLU, Xavier for tanh; **sigmoid the awkward case** (not "rescued") — which
  motivates the defaults.
- **Normalization** built by hand as a *complete forward layer* (γ/β + a training-time effect) in NB 6; the
  running-stats + train/eval nuance named there and realized as `nn.BatchNorm` in NB 8 — never an oversold
  from-scratch BN trainer.
- **Capstone verdict: competitive, NOT superior — and a tree even wins** on Fashion-MNIST (measured on the
  *same* data); the finale lands on "the data wants a CNN", not an MLP victory lap. **The MNIST proxy number
  is illustrative only** (a near-tie ~0.94–0.95), not a crowned ordering — the ch-11-NB-5 tie lesson, carried.
- **torch reproducibility** pinned (seeds + `use_deterministic_algorithms` + CPU + documented threads +
  pinned version, verified on-box); the one-time fetch **logged, never silenced**.
- **c07 (Adam/momentum) is a deepened recap**, not a new build — kept short inside NB 8 (avoid ch-11-NB-4
  overlap).
- **The training loop is shown, never assigned** — challenges in "Your turn" *modify* the shown loop.

## Datasets
- **Fundamentals (NB 1–6):** small synthetic — `make_blobs` (3-class, the NB-1 reference) / `make_circles` /
  `make_moons` / `make_classification` + the deep-stack toys; the MNIST-10k subset for the honest depth gain
  (NB 2).
- **Capstone (NB 9):** **Fashion-MNIST** subset (offline, CPU) + the tree foil (RF/HistGB on raw pixels). A
  tabular-humility aside re-uses ch 11's `breast_cancer` fact (measured, not the capstone).
- **Avoid:** `load_digits` (ch 11's capstone); a 2-D toy as the capstone.

## `src/` & guards — **FIRST chapter with real `src/` changes in a while (pytest count rises)**
- **`pyproject.toml`:** add an optional **`deep = ["torch>=2.2"]`** extra (CPU; mirrors `boosting`; **pin a
  concrete version at build** for determinism); document `uv sync --extra deep`. `torch` install + a trivial
  deterministic check **on the actual box** is the first NB-7 build step.
- **`datasets.py`:** add **`load_fashion_mnist()`** (and likely `load_mnist()`) wrapping `fetch_openml`, the
  existing **fetch-and-cache pattern** (INFO-logged, never silenced; cache under `src/ml_course/data/`,
  git-ignored as `*.npz`/`*.csv`), pandas-first or arrays + a subset helper; add **`scripts/vendor_fashion_mnist.py`**
  (mirror `vendor_penguins.py`). New **tests in `tests/test_datasets.py`** → **pytest rises from 20** (state
  the new total at build). Possibly a tiny torch-determinism helper (inline or a small util — decided at NB-7
  plan; keep minimal).
- **`viz.py`:** reuse `plot_confusion_matrix`; an image-gallery helper only if reused across NBs (else inline,
  as ch 11 did).
- **Guards (unchanged):** colours only from `ml_course.colors` (no hardcoded hex); seeds fixed; "Read the
  figure" after **every** figure (schematics included); **never silence output** (incl. the torch one-time
  fetch + any warnings — pinned determinism, not silencing); banned-word scan 0; ruff/black clean;
  output-free; `gen_llms_txt`; **two-reviewer gate + Rémy visual per NB**; ff-merge `notebook→chapter`;
  **chapter→main via PR (`--no-ff`)** at close (PR #12 — the course finale).

## References (chapter-level; per-NB DOIs at build)
Dropout: Srivastava et al. 2014 (JMLR 15:1929–1958). BatchNorm: Ioffe & Szegedy 2015 (arXiv:1502.03167);
LayerNorm: Ba et al. 2016 (arXiv:1607.06450). Xavier: Glorot & Bengio 2010 (PMLR 9:249–256). He: He et al.
2015 (DOI 10.1109/ICCV.2015.123; arXiv:1502.01852). Vanishing gradients: Bengio et al. 1994 (DOI
10.1109/72.279181); Hochreiter & Schmidhuber 1997 (LSTM; DOI 10.1162/neco.1997.9.8.1735). Deep Learning
text: Goodfellow, Bengio & Courville 2016 (ch 6–8). Adam: Kingma & Ba 2015 (arXiv:1412.6980 — recap).
Horizons: LeCun et al. 1998 (CNN/MNIST; DOI 10.1109/5.726791); Vaswani et al. 2017 (transformers;
arXiv:1706.03762). PyTorch: Paszke et al. 2019 (PyTorch; NeurIPS). (All DOIs verified by the ml-expert.)

## Open decisions (resolved in this plan)
1. **Framework = B (PyTorch)**, plain torch, **shown not hand-coded** (no skorch/Lightning). [Rémy, eyes open.]
2. **~10 notebooks** — derogate the 5-NB ceiling for the finale; thicken, don't pad. [Rémy.]
3. **Two parallel hello-worlds:** NB 1 numpy (the reference), NB 7 torch (the same net). [Rémy.]
4. **Capstone = Fashion-MNIST** subset (harder than MNIST → a tree genuinely wins → sharper "wants-a-CNN"). [Rémy ✅.]
5. **NB 10 (horizons + synthesis) separate**, the course's send-off — the **one intentional exception** to the
   by-hand + executable-"Your turn" pattern (reflective close). [Rémy ✅ / pedagogy.]
6. **NB 1 reference net = multi-class softmax + cross-entropy on a 2-D `make_blobs` toy**; the binary backward
   is **ch-11 recap (black-boxed, not re-derived)**; the **softmax-CE gradient is NB 1's one new derivation**;
   NB 7 re-instantiates the same architecture. [pedagogy MAJOR-1 + ml-expert MINOR.]
7. **NB 6 normalization thickened** to a complete by-hand BN forward layer (γ/β + a training-time effect);
   running-stats + train/eval nuance named there, realized as `nn.BatchNorm` in NB 8. [pedagogy MAJOR-2 + ml-expert MINOR.]
8. **NB 7 bridge is an explicit component-by-component "Read the figure" + the gradient-match number** (not a
   one-liner); overlap-watch names `.train()/.eval()` as the new beat. [pedagogy MAJOR-3 + MINOR.]
9. **Init framing:** He for ReLU, Xavier for tanh; **sigmoid the awkward non-zero-centered case** (not
   "rescued"). [ml-expert MINOR.]
10. **NB 2 drops the equal-unit-budget claim** (anchor is 50 vs 448 units); lands on "depth remaps + modest
    gain"; optional equal-budget check at NB-2 plan. [ml-expert NIT.]
11. **Capstone numbers:** the MNIST proxy is **illustrative only** (near-tie); the verdict rests on the
    **measured Fashion-MNIST** foil (a tree wins ~1.4 pp). [ml-expert MAJOR — the only near-blocking item, folded.]
12. **`src/` changes:** the `deep` extra + a Fashion-MNIST/MNIST loader + a vendor script + tests — **pytest
    count rises from 20** (first real `src/` change since the early chapters); validated by the tests.
13. **torch determinism pinned + verified on-box; version pinned**; the one-time fetch **logged, never
    silenced**; **CPU only**. [ml-expert NIT/registered concern.]
14. **"Your turn" tiers deferred to each NB-plan** (as ch 11); torch challenges **modify** the shown loop,
    never author it; NB 10's are reflective. [pedagogy NIT.]

## Reviewer gate outcome
Both reviewers ran live checks (sklearn 1.9.0 / numpy 2.4.6); **no BLOCK remains.**

- **`@ml-expert-reviewer`: REVISE → no-BLOCK.** Live-verified **every** sklearn/numpy anchor, several to 3–4
  decimals: 5-layer sigmoid **0.500** (all 5 seeds) vs tanh/ReLU **1.000**; 10-layer vanish **2.49e-1 →
  1.3e-16** / explode **5.0e6**; He → **0.71–0.80** flat across 10 layers; depth ladder **0.9406 → 0.9490**;
  inverted dropout mean **0** / variance **1.0 → 2.0**; `breast_cancer` **0.975 > 0.967/0.970**; Fashion-MNIST
  fetches offline. **MAJOR folded:** the stale capstone-foil "< HistGB ~0.958" (live CV: MLP 0.944 ≈ RF 0.946
  ≈ HistGB 0.949 — a near-tie, the exact ch-11-NB-5 mistake) → relabeled the MNIST line *illustrative proxy*
  and **rested the verdict on the measured Fashion-MNIST foil** (MLP 0.859 ≈ RF 0.853 < HistGB 0.874 — a tree
  genuinely wins, a *sharper* punchline). **MINORs folded:** Xavier-on-sigmoid overstated → "sigmoid the
  awkward case" (NB 4); NB-1 backprop boundary → black-box the ch-11 backward, softmax-CE gradient is the new
  bit; NB-6 positive framing (complete by-hand BN forward + name what NB 8 adds). **NITs:** NB-2 equal-budget
  claim dropped; torch determinism verified on-box + version pinned. **Framework B: no ML objection** to the
  plan as written (the real `nn.Dropout`/`nn.BatchNorm` gain is genuine; mitigations are the right ones).
- **`@pedagogy-reviewer`: REVISE → no-BLOCK.** Progression sound; the prerequisite table praised as the most
  careful in the course; the ch-11/ch-12 line drawn with unusual precision; no un-deferred forward
  references; honest scoping exemplary; the finale framing the right note to end on. **MAJORs folded:** NB-1
  class-count/dataset pinned (multi-class softmax-CE on a 2-D toy; the softmax-CE gradient the new
  derivation); NB-6 thickened to a complete by-hand BN layer; NB-7 bridge upgraded to an explicit
  component-by-component mapping + the gradient-match number. **MINORs folded:** NB-7 overlap-watch
  (`.train()/.eval()` named new); NB-9 beats applied-not-re-taught; NB-10 the intentional exception
  (reflective "Your turn"). **NITs:** "Read the figure" holds for schematics; per-NB "Your turn" tiers
  deferred (torch = modify-not-author).

## Verification (end-to-end, at build)
1. `uv sync --extra deep` installs `torch` (CPU, pinned version); a trivial example runs **deterministically**
   (pinned seeds + `use_deterministic_algorithms`) **on the dev box**.
2. Anchors reproduce: vanishing 10-layer RMS (~1e-16 sigmoid / ~5e6 ReLU); 5-layer sigmoid 0.500 vs tanh/ReLU
   ~1.000; He → flat RMS ~0.71–0.80; depth gain `(50,)` 0.941 → `(256,128,64)` 0.949; the **autograd gradient
   matches the NB-1 numpy net**; the Fashion-MNIST capstone foil (re-measured: MLP 0.859 ≈ RF 0.853 < HistGB
   0.874).
3. Guards: hex clean, banned 0, ruff/black clean, output-free, `gen_llms_txt`; **pytest green at the new
   count** (new dataset tests). 4. Per NB: nbconvert exit 0 with the planned figures; two-reviewer gate (no
   BLOCK) → Rémy visual. 5. **Chapter plan validated by Rémy via ExitPlanMode (this gate cleared with no BLOCK).**
