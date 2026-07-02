# NB plan — 12_NeuralNetworks / 04_initialization — He & Xavier (the fix)

> Status: **APPROVED by Rémy (via ExitPlanMode, 2026-06-30, no edits).** NB-plan stage = Rémy validates alone
> (no reviewer gate; both reviewers return on the built notebook). Anchors measured live (pure numpy + sklearn
> utilities, SEED=0; `measure_ch12_nb4.py`). **NB 4 of 10** (chapter 12 — the PyTorch finale). **Pure numpy —
> NO torch** (the `deep` extra lands at NB 7; `torch.nn.init` realizes He/Xavier in NB 8). Chapter plan:
> `docs/plans/chapter_12_NeuralNetworks.md` (APPROVED, §NB 4).

## Context
Chapter 12, **NB 4 of 10 — the cure for NB 3's pathology.** One concept: **initialization.** NB 3 diagnosed
that the per-layer factor (`W · activation'`) drifts from 1 → the gradient vanishes or explodes across depth.
We cannot change `activation'` much, but we **can choose the variance of `W`**. Choose it so each layer
preserves the signal's variance and the per-layer factor stays ≈ 1 — that is **He** (`std = sqrt(2/n)`, for
ReLU) and **Xavier/Glorot** (`std = 1/sqrt(n)`, for tanh). Built **by hand in pure numpy** (reuse NB-3's
10-layer stack): derive the variance-preserving rule, implement He/Xavier, re-run NB-3's per-layer RMS → the
gradient goes **flat**, and a deep net that **stalled** with naive init now **trains to 100%**. The honest
caveat: **sigmoid is the awkward non-zero-centered case** — Xavier *improves* it but does **not** preserve it
(still vanishing), which is *why* tanh/ReLU displaced sigmoid as the default. This is the real content behind
ch 11's hand-wave "init must break symmetry."

## Recap vs new (the boundary)
- **RECAP (black-boxed, NOT re-derived):** NB 3's vanishing/exploding mechanism (the per-layer factor, the
  product across depth); the `L`-layer net + by-hand backward (NB 1–3); ch 11 NB 3's "weights must break
  symmetry (random, not zeros)" — NB 4 is the *quantitative* version of that hand-wave; `make_circles` +
  `StandardScaler` (ch 00).
- **NEW (the one concept):** **variance-preserving initialization** — the rule `Var(Wx) = n·Var(W)·Var(x)` →
  `Var(W) = 1/n` (Xavier, ~linear/tanh) and `Var(W) = 2/n` (He, ReLU halves the variance); the measured fix
  (flat gradient + the net trains); the **sigmoid-is-awkward** honesty.
- **No forward references:** the **training loop / optimizer** is NB-1/3's (reused, not re-taught); the real
  `torch.nn.init` is **NB 8**; **normalization** (a *during-training* variance lever) is **NB 6** (named, not
  taught). No He/Xavier *torch* API here — pure numpy formulas.

## Live anchors (measured; re-pinned at build)
By-hand `L`-layer net (NB-3's machinery + a **fan-in-scaled init**: `he`=√(2/n_in), `xavier`=√(1/n_in),
`glorot`=√(2/(n_in+n_out))), SEED=0, on `make_circles(400, noise=0.1, factor=0.4)` standardized; 10 hidden
layers × width 16 for the diagnostics, 8 hidden layers for the training payoff.

- **Per-layer gradient RMS, input → output (the fix):**
  - **ReLU:** small `7e-8 → 1e-7` (alive but tiny) · large `2e3 → 5e3` (explode) · **he `4.6e-2 → 2.2e-1`
    (flat & healthy — min 0.03 / max 0.22 across all 11 layers)**.
  - **tanh:** small `8e-8` (tiny) · **xavier `6.1e-3 → 2.3e-2` (flat & healthy)**.
  - **sigmoid (the awkward case):** small `8.4e-14 → 2.9e-2` (vanish, NB 3) · **xavier `3.4e-9 → 7.8e-2`** —
    *improved ~5 orders but STILL vanishing* (input/output ratio **2.3e7**, vs tanh+xavier's **3.8**). Same
    Xavier init, opposite outcome: tanh preserved, sigmoid still dies.
- **Forward activation std across depth (variance preservation):** **ReLU+he 0.69 → 0.53** (preserved across
  10 layers) vs ReLU+small `0.07 → 0.00` (collapses to zero) vs ReLU+large (explodes). tanh+xavier `0.53 →
  0.24` (healthy).
- **Training payoff (by-hand GD, 8 hidden layers, lr 0.3, 300 ep, 3 seeds — robust):** **ReLU+small 0.500 /
  ReLU+large 0.500 (both stall at chance) → ReLU+he 1.000**; **tanh+small 0.500 → tanh+xavier 1.000.** The
  deep net of NB 3, cured by init alone.
- **The derivation (exact):** one linear layer `y = Wx`, `W ~ N(0, σ²)`, `n` inputs → `Var(y) = n·σ²·Var(x)`;
  preserve `Var(y) = Var(x)` ⇒ `σ² = 1/n` (Xavier). ReLU zeros ~half its inputs ⇒ variance halves ⇒ `σ² = 2/n`
  (He). (Glorot averages fan-in/fan-out: `σ² = 2/(n_in+n_out)`.)

## Cell-by-cell (~23 cells, 3 figures) — intuition → implementation → "Read the figure"; **SHOW**; ends with "Your turn"
1. **(md) Header** `# 04 — Initialization: He and Xavier`. **Prerequisites** (NB 3: the vanishing/exploding
   pathology — the per-layer factor; NB 1–2: the `L`-layer net by hand; ch 11 NB 3: "weights must break
   symmetry"). **What you'll do:** turn that hand-wave into a precise rule, implement He & Xavier, and watch a
   dead deep net come back to life.
2. **(md) From diagnosis to cure.** NB 3: the gradient vanishes/explodes because the per-layer factor
   (`W · activation'`) drifts from 1 across depth. We cannot reshape `activation'`, but the **scale of `W` is
   ours to choose**. What scale keeps each layer from shrinking or growing the signal?
3. **(md) Derive the rule — variance in, variance out.** Take one layer `y = Wx` with `n` inputs and weights
   drawn `~ N(0, σ²)`. Then `Var(y) = n · σ² · Var(x)`. For the signal to neither grow nor shrink we need
   `n · σ² = 1`, i.e. **`σ = 1/√n`** — this is **Xavier/Glorot** init. For **ReLU**, which zeros about half its
   inputs, the variance is halved, so we compensate with a factor of 2: **`σ = √(2/n)`** — this is **He** init.
   (Glorot's symmetric form averages the layer's inputs and outputs, `σ² = 2/(n_in + n_out)`.)
4. **(code) Imports + `use_course_style()`; implement the inits.** Reuse NB-3's net (forward/backward,
   sigmoid/tanh/ReLU); add a `fan_in`-scaled `init_params(kind)` with `he` / `xavier` / `glorot` / the naive
   `small` / `large`. Build the 10-layer × width-16 stack on standardized circles. Show the code.
5. **(md) The test: re-run NB 3's measurement.** We measure the per-layer gradient RMS exactly as in NB 3,
   now comparing the naive inits against He (for ReLU) and Xavier (for tanh).
6. **(code)** measure per-layer gradient RMS: ReLU small/large/he, tanh xavier; print (he flat 0.05–0.22; large
   ~1e3; small ~1e-7; tanh+xavier flat 6e-3–2e-2).
7. **(code) Fig 1 — gradient RMS per layer, before vs after** (two panels, log-y): *naive init* (sigmoid+small
   vanish, ReLU+large explode — NB 3's pathology) vs *variance-preserving init* (ReLU+he flat, tanh+xavier
   flat, **sigmoid+xavier still vanishing** — the caveat in the same frame). Charter colours.
8. **(md) Read Fig 1** — He flattens the ReLU gradient across all ten layers (~0.05–0.22; no vanish, no
   explode); Xavier does the same for tanh. The naive inits collapse or blow up. The cure is the **right
   scale** — nothing more. (And note the one curve that stays bent — sigmoid — which we return to.)
9. **(md) Why it works** — He's `√(2/n)` is exactly the scale that makes the per-layer factor ≈ 1 (it cancels
   ReLU's variance-halving); Xavier's `1/√n` does it for tanh's ~linear regime near 0. The factor that
   compounded into `1e-13` or `1e3` in NB 3 now compounds into ≈ 1.
10. **(code) Fig 2 — forward activation spread across depth** (std per hidden layer, or histograms at layers
    1/5/10): ReLU+he preserved (~0.69 → 0.53) vs ReLU+small (collapses to 0) vs ReLU+large (explodes). Charter
    colours.
11. **(md) Read Fig 2** — He keeps the activation spread roughly constant down the stack, so every layer sees a
    healthy signal — forward *and* backward. Small init starves the deep layers (spread → 0); large init
    saturates them. Preserving the variance is the whole game.
12. **(md) The awkward case: sigmoid.** tanh and ReLU are (near) zero-centered; **sigmoid is not** — its
    outputs live in (0, 1) with mean ~0.5, so a non-zero mean is injected at every layer and the
    variance-preserving rule no longer holds cleanly.
13. **(code)** same Xavier init, tanh vs sigmoid: print the input→output gradient ratio (tanh ~3.8 ≈ flat;
    **sigmoid ~2e7 — still vanishing**, input gradient 3.4e-9 vs a healthy ~1e-2).
14. **(md) Read** — **the same init that flattens tanh leaves sigmoid still dying.** Xavier *improves* sigmoid
    (from `1e-13` to `1e-9`) but cannot *preserve* it; Glorot's paper notes sigmoid needs a larger gain. This
    is the deeper reason — beyond NB 3's saturation — that **tanh and ReLU displaced sigmoid as the default.**
    (We are not "rescuing sigmoid"; we are showing why we avoid it.)
15. **(md) The payoff that matters: does the deep net now train?** A flat gradient is the means; the end is a
    network that learns. We train an **eight-hidden-layer** net by hand (the optimizer of NB 1/3), comparing
    naive init against He/Xavier.
16. **(code)** train 8-hidden-layer nets (by-hand GD, 3 seeds): ReLU small/large (0.500) vs ReLU+he (1.000);
    tanh small (0.500) vs tanh+xavier (1.000). Print.
17. **(code) Fig 3 — the training payoff** (bars): ReLU small / large / he and tanh small / xavier — chance
    (~0.5) for naive init, 1.000 for the matched variance-preserving init. Charter colours.
18. **(md) Read Fig 3** — an eight-layer network that was hopeless with naive weights (chance, whether the
    weights were too small *or* too large) trains to **100%** once initialized to preserve variance. NB 3's
    diagnosis, cured — by choosing the initial scale, nothing else.
19. **(md) What you built** — turned ch 11's "break symmetry" into a precise rule (`Var(Wx) = n·Var(W)·Var(x)`),
    derived and implemented **He** (ReLU) and **Xavier** (tanh), watched the gradient go flat and a dead deep
    net train, and saw **why sigmoid stays the awkward case**.
20. **(md) Where this goes next** — initialization fixes the variance *at the start*; **normalization**
    (notebook 6) keeps it healthy *during* training, even as the weights move. First, **notebook 5 — dropout**:
    a different lever — regularization, not signal flow.
21. **(md) Your turn** — *(warm-up)* apply He to the 10-layer ReLU stack and confirm the gradient RMS is flat
    (vs NB 3's vanish); *(core)* try **He on a tanh** net and **Xavier on a ReLU** net (mismatched) — does the
    mismatch hurt? (the factor of 2 matters); *(reach)* initialize a deep net with `√(2/n)` but set all biases
    to a large constant — does the signal still flow? (init is about more than the weights' std).
22. **(md) References** — Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training deep
    feedforward neural networks. *PMLR* 9:249–256 (Xavier/Glorot init). He, K., Zhang, X., Ren, S., & Sun, J.
    (2015). Delving deep into rectifiers: surpassing human-level performance on ImageNet classification. *ICCV*
    (He init), DOI 10.1109/ICCV.2015.123 (arXiv:1502.01852). Goodfellow, Bengio & Courville (2016), *Deep
    Learning*, ch 8. *Previous:* **12.3 — vanishing & exploding gradients.** *Next:* **12.5 — dropout.**

## `src/` & guards
- **No `src/` change** (notebook-local numpy `L`-layer net — NB-3's + a fan-in-scaled init; sklearn
  **utilities** `make_circles` / `StandardScaler` only; `viz.use_course_style` / `colors`; the before/after
  gradient figure, the activation-spread figure, and the training-payoff bars are **inline** matplotlib).
  **pytest stays 20.** **torch NOT used** (the `deep` extra arrives at NB 7; `torch.nn.init` at NB 8).
- **Decisions baked in (Rémy validated):** (1) **everything by-hand numpy** — the diagnostics *and* the
  training payoff (no `MLPClassifier` needed; init is naturally a by-hand topic). (2) **Sigmoid is the awkward
  case, NOT "rescued"** — measured: same Xavier, tanh flat (ratio 3.8) / sigmoid still vanishing (ratio 2e7);
  this *motivates* the tanh/ReLU default. (3) Report the **measured** flat values (gradient RMS ~0.05–0.22,
  forward std ~0.53–0.69), not the chapter-plan's "0.71–0.80" (a nearby figure for a different metric/width).
- Colours only from `ml_course.colors` (no hardcoded hex); `SEED=0`; **"Read the figure" after every figure**;
  **never silence output**; banned-word scan 0; ruff/black clean; output-free.
- Build from `build_ch12_nb4.py` (source of truth in the **ephemeral** scratchpad; rebuild right before
  `git add` — kernel-drift guard).
- Exit guards: nbconvert exit 0 (3 figures), anchors reproduce (ReLU+he gradient flat 0.05–0.22 & forward std
  0.53–0.69; tanh+xavier flat; sigmoid+xavier still vanishing ratio ~2e7; training ReLU small/large 0.500 →
  he 1.000, tanh xavier 1.000), banned 0, hex clean, ruff clean, output-free; **two-reviewer gate**
  (`@ml-expert-reviewer` + `@pedagogy-reviewer`, no BLOCK) → **Rémy visual** → end-of-NB checklist
  (`gen_llms_txt`, `common_errors` +rows, `course_map` §12 → NB 4 built, pytest 20, STATE) → commit
  `feat(12_neuralnetworks): notebook 04 — initialization (He and Xavier)` → `git merge --ff-only` into
  `chapter/12_NeuralNetworks`.

## Verification (end-to-end)
1. nbconvert a scratchpad copy → exit 0, 3 figures, anchors reproduce (He flattens the ReLU gradient/forward
   spread; sigmoid+xavier still vanishes; the 8-layer net trains 0.500 → 1.000 with He/Xavier).
2. hex + banned + ruff clean. 3. pytest 20 (no `src/` change). 4. **Rémy validated this NB plan** (no reviewer
   gate; both reviewers return on the built notebook).
