# NB plan — 12_NeuralNetworks / 06_normalization

> Status: **APPROVED by Rémy (via ExitPlanMode, 2026-06-30, no edits).** NB-plan stage = **Rémy validates
> alone** (no reviewer gate; both reviewers return on the built notebook). Anchors measured live (pure numpy +
> sklearn utilities, SEED=0, eps=1e-5; `measure_ch12_nb6.py`). **NB 6 of 10** (chapter 12 — the PyTorch
> finale). **Pure numpy — NO torch** (the real `nn.BatchNorm` arrives at NB 8; torch lands at NB 7). **The
> LAST by-hand numpy NB.** Chapter plan: `docs/plans/chapter_12_NeuralNetworks.md` (APPROVED, §NB 6).

## Context
Chapter 12, **NB 6 of 10 — the last by-hand numpy notebook.** One concept: **normalization** — re-center and
re-scale a layer's activations to a healthy range *during* training. Init (NB 4) fixes the activation variance
**at the start**; as the weights move, the per-layer distributions drift again (re-die / re-saturate /
re-explode — the NB 3 pathology, back). Normalization fixes it **at every step**: the same "control the
variance with depth" lever as init, now applied throughout training. The worked example is **batch
normalization** (Ioffe & Szegedy 2015), built **by hand as a complete layer — forward AND backward**
(gradient-checked), trained end-to-end via two learnable parameters γ/β. Its sibling **layer normalization**
(Ba et al. 2016) is **named and contrasted** (across features, not across the batch — what transformers use),
not a second by-hand build. `MLPClassifier` has no batch-norm — it *must* be by-hand here. Then we **name in
one sentence** what the real `nn.BatchNorm` adds beyond the forward transform — **running statistics + the
train/eval-mode split** — explicitly NB 8's, so no oversold from-scratch BN *trainer*.

## Recap vs new (the boundary)
- **RECAP (black-boxed, NOT re-derived):** vanishing/exploding from drifting activations (NB 3); init sets the
  variance *at initialization* (He/Xavier, NB 4); the `L`-layer net + by-hand forward/backward + the
  softmax-CE output (NB 1–5); ReLU/tanh/sigmoid (ch 11); mini-batch SGD (NB 2).
- **NEW (the one concept):** **normalization** — the batch-norm forward transform (per-feature batch mean/var
  → normalize → learnable γ/β), its backward (the BN Jacobian, gradient-checked), the **training-time effect**
  (a stalled deep net trains; robustness to init scale and learning rate), the **batch-coupling catch** → **layer
  norm** as the per-sample sibling, and what the real `nn.BatchNorm` adds (running stats + train/eval).
- **No forward references:** the real `nn.BatchNorm` (one line) + the running-stats/train-eval *machinery* is
  **NB 8**; PyTorch itself is **NB 7**; CNN/RNN/transformer (where LN lives) are **NB 10** horizons. No torch here.

## Live anchors (measured; re-pinned at build) — `measure_ch12_nb6.py`, SEED=0, eps=1e-5
By-hand `L`-layer net (NB-1–5 machinery) + a batch-norm layer placed **between the linear map and the
activation** (`z = a·W + b → BN → ReLU`). γ init 1, β init 0; population variance (ddof=0).

- **(A) Forward drift — 10-layer ReLU stack, width 32, per-layer activation std:**
  - small init, **no BN**: `0.079 → 0.001 → 0.000` (dies); large init, no BN: `0.790 → 121.3 → 128542` (explodes
    ~12 orders); he init, no BN: `0.790 → 0.474 → 0.490` (band 0.35–0.79, the NB-4 result).
  - **all three WITH BN: flat `0.575 → 0.613`, band `[0.555..0.615]`** across all ten layers — the per-layer
    distribution is pinned regardless of init scale. (BN pins the *pre-activation* to var 1; post-ReLU std ≈ 0.6,
    the point being it no longer drifts with depth.)
- **(B) BN backward gradient-check** (small net, central finite differences): `dW` rel-err **1.3e-10**, `dγ`
  **8.9e-09 / 2.5e-10**, `dβ` **2.8e-09** — correct to ~**1e-9**. Bonus: the pre-BN bias `b` has gradient
  **≈ 0** (2.7e-19) — BN re-centers, so **β is the effective bias** (why `nn.Linear(bias=False)` precedes
  `nn.BatchNorm`).
- **(C) Training-time effect — deep ReLU net (6 hidden × width 16) on circles, mini-batch SGD:**
  - **small init, lr 0.3: no BN stalls at acc 0.500 / loss 0.6934 (ln2, chance) → BN acc 1.000 / loss 0.0033.**
    The headline: a deep net stuck at chance trains perfectly once BN is inserted.
  - **learning-rate robustness:** he init, **lr 1.0**: no BN stalls **0.500** (the big lr breaks even He) → BN
    **1.000**. small lr 1.0: no BN 0.500 → BN 1.000.
  - tanh small lr 0.3: no BN 0.500 → BN 1.000; sigmoid small lr 0.3: no BN 0.500 → BN 1.000.
  - **Honest scope:** at a *good* init + *moderate* lr (he, lr 0.3), the plain net already trains (1.000, loss
    1e-4) and is **a touch faster than BN** (loss@ep20 0.0022 vs 0.0114). So BN's win on this small net is
    **robustness** (it trains from a bad init and tolerates a learning rate that stalls even He), **not raw
    speed** on an already-healthy net. The decisive *speed* wins are large vision nets (Ioffe & Szegedy 2015).
- **(D) BN vs LN** (same 6×4 input): BN per-feature (column) mean ≈ 0 / std ≈ 1; LN per-sample (row) mean ≈ 0 /
  std ≈ 1. BN normalizes each feature across the batch; LN normalizes each sample across its features.
- **(E) Running-stats motivation:** a feature ~ N(3, 2²); batch-size-32 stats over 6 draws wander (mean
  **2.49–3.25**, var **3.58–4.77**). Each batch sees different stats; at eval you may have one sample → the
  library keeps a **running average** and uses it at eval (the train/eval split — NB 8).
- **Honesty (the BN "why" is debated):** Ioffe & Szegedy framed BN as reducing "internal covariate shift";
  Santurkar et al. 2018 showed the mechanism is better explained by a **smoother loss landscape**. We report
  the **robust empirical effect** (pins the distributions; trains the stalled net; tolerates lr/init) and flag
  that the precise reason is still debated — we do not assert covariate shift as settled.

## Cell-by-cell (~22 cells, 3 figures) — intuition → implementation → "Read the figure"; **SHOW**; ends with "Your turn"
1. **(md) Header** `# 06 — Normalization: keeping activations healthy during training`. **Prerequisites**
   (NB 3: vanishing/exploding from drifting activations; NB 4: init sets the variance *at the start*; NB 1–5:
   the `L`-layer net, forward/backward, ReLU). **What you'll do:** build a batch-norm layer by hand (forward +
   backward), watch it pin every layer's activations to a healthy band regardless of init, train a deep net
   that had stalled, meet layer norm, and name what the real `nn.BatchNorm` adds. **This closes the by-hand
   numpy arc.**
2. **(md) The problem normalization solves.** Recap NB 4: init sets the activation variance *at
   initialization* — but as the weights move during training, the per-layer distributions **drift again**
   (re-die, re-saturate, re-explode — the NB 3 pathology returns). Idea: instead of only choosing the starting
   variance, **re-normalize the activations at every step**. Same "control the variance with depth" lever as
   init, applied throughout training.
3. **(md) The mechanism: batch normalization.** For each feature (each unit), over the current **mini-batch**:
   subtract the batch mean, divide by the batch std → mean 0, variance 1 (`ẑ = (z − μ)/√(σ²+ε)`). Then —
   because forcing every layer to exactly mean-0/var-1 is too rigid — give the layer back control with two
   **learnable** parameters per feature: a scale **γ** and a shift **β**: `out = γ·ẑ + β`. With γ=1, β=0 it's
   plain normalization; training adapts them (γ/β can even *undo* it if that helps). (Ioffe & Szegedy 2015.)
4. **(code)** implement `bn_forward(z, gamma, beta, eps)` (batch mean/var → ẑ → γẑ+β; return a cache); on a
   drifted input (a column with mean 5, std 0.1 and another with mean −3, std 8) print per-feature mean/std
   **before → after** (≈ 0 / ≈ 1 with γ=1, β=0). **SHOW** the few lines.
5. **(code) Fig 1 — the BN-layer schematic** (a batch `z` of n rows × d features → compute μ, σ² **per column /
   across the batch** → normalize ẑ → scale-shift `γ·ẑ+β` → out; the "across the batch" arrow drawn down the
   columns). Charter colours. [Placed early so the mechanism is visual before the experiments.]
6. **(md) Read Fig 1** — the four steps; "across the batch" = per-feature over the n rows of the mini-batch;
   γ/β are **learnable** (the layer keeps the freedom to rescale, or undo, the normalization).
7. **(md) Watch it fix the drift.** Reuse NB 3/4's deep stack (10 layers, ReLU). NB 4 showed: small init →
   activations die; large → explode; He → preserved *at the start only*. Insert BN after each linear and
   re-measure the per-layer activation std.
8. **(code)** the deep stack with optional BN per hidden layer; print the L1/L5/L10 std table — small no-BN
   `0.079→0.000`, large no-BN `0.790→128542`, he no-BN `0.790→0.490`; **all three BN flat ~0.58→0.61**
   (band 0.555–0.615).
9. **(code) Fig 2 — activation std across depth: drift vs BN** (2 panels): **left** no-BN small/large/he on a
   **log y** (die toward 0 / explode to 1e5 / He flat-ish) — the drift; **right** the same three **with BN**,
   linear y, collapsing onto **one flat ~0.6 band**. Charter colours (class cycle for the inits; model/sky for BN).
10. **(md) Read Fig 2** — without BN the activation scale is at the mercy of the init (vanishes, or explodes
    10⁵×, or only-He survives — and only at the start); **with BN every init gives the same flat healthy band
    across all ten layers.** BN pins the *pre-activation* to var 1 each layer; after ReLU the std settles at
    ~0.6, but — the point — it no longer drifts with depth.
11. **(md) Wiring BN into training — the backward pass.** BN sits between the linear map and the activation
    (`z = a·W+b → BN → ReLU`). Because BN re-centers, the pre-BN bias `b` becomes **redundant** (β is the
    effective bias). To train γ/β we need the gradient through BN — state the BN-backward formula in one line
    (`dz = (1/n)·istd·(n·dẑ − Σdẑ − ẑ·Σ(dẑ·ẑ))`, with `dγ = Σ(dout·ẑ)`, `dβ = Σdout`), and trust it only after
    a gradient-check.
12. **(code)** `bn_backward(dout, cache)` (returns dz, dγ, dβ) + the **gradient-check** vs central finite
    differences on a small net: `dW` ~**1e-10**, `dγ` ~**1e-9**, `dβ` ~**1e-9** — and `db` ≈ **0** (BN makes
    the pre-BN bias redundant). The rigor anchor. **SHOW** the backward.
13. **(md) Read the gradient-check** — analytic == numeric to ~**1e-9**, so the by-hand BN backward is correct;
    the net trains end-to-end with γ/β as ordinary learnable parameters. (And `db ≈ 0` under BN — which is why
    `nn.Linear(bias=False)` is paired with `nn.BatchNorm`.)
14. **(md) The training-time effect.** A deep ReLU net (6 layers) on circles with a *poor* small init stalls at
    chance — the NB-3/4 pathology. Insert BN and train both; also test a learning rate large enough to stall
    even a good (He) init.
15. **(code)** train none vs BN and print the table: **small init, lr 0.3 — none 0.500 / loss 0.693 → BN 1.000
    / loss 0.003**; **he init, lr 1.0 — none 0.500 → BN 1.000** (lr robustness); tanh/sigmoid small — none 0.500
    → BN 1.000; **he init, lr 0.3 — none already 1.000 and a touch faster than BN** (the honest caveat).
16. **(code) Fig 3 — training loss, none vs BN** (deep ReLU, small init): the no-BN curve **flat at ln2 ≈ 0.693**
    (stalled at chance) in `error` (coral), the BN curve **descending to ~0** in `model` (sky); a dotted `zero`
    guide at ln2. (Optional second pair: he lr 1.0 none-flat vs BN-descends — the lr-robustness case — or leave
    it to the table.) [Figures numbered by order of appearance: Fig 1 = schematic, Fig 2 = drift, Fig 3 = loss.]
17. **(md) Read Fig 3 + honest scope.** Without BN the loss never leaves ln2 (the NB-3 stall — stuck at chance);
    with BN it descends to ~0 and the net reaches 100%. **Honest scope:** at a *good* init and moderate lr the
    plain net already trains — and here is even a touch *faster* than BN; BN's win on this small net is
    **robustness** (it trains from a bad init, and tolerates a learning rate that stalls even He), **not raw
    speed**. On large vision nets the *speed-up* is decisive (Ioffe & Szegedy 2015). **And the precise reason
    BN helps is still debated** — the original "internal covariate shift" story was challenged by Santurkar et
    al. (2018), who attribute it to a smoother loss landscape; what's robust is the empirical effect we measured.
18. **(md) Batch norm's catch, and layer norm.** BN **couples the samples in a batch** (each activation is
    normalized using its batch-mates). Fine for big i.i.d. image batches; it breaks when the batch is tiny or
    the input is a sequence handled one step at a time. **Layer normalization** (Ba et al. 2016) normalizes
    across the **features of each sample** instead — batch-independent. It's the normalization in RNNs and
    **transformers** (NB 10's horizon).
19. **(code)** the BN-vs-LN contrast on one input matrix: BN normalizes **columns** (per feature, across the
    batch) → column mean ≈ 0 / std ≈ 1; LN normalizes **rows** (per sample, across features) → row mean ≈ 0 /
    std ≈ 1. Print both. (Plus the running-stats wander: a feature ~ N(3, 2²), 6 batches → mean 2.49–3.25 / var
    3.58–4.77, feeding cell 20.)
20. **(md) What the real `nn.BatchNorm` adds (NB 8).** Our by-hand layer is the complete forward+backward
    transform. The library layer adds one thing we left out: at **train** time it uses the batch's stats (as we
    did), but it also keeps a **running average** of mean/var over training and uses *that* at **eval** time —
    because at eval you may have a single sample, and (cell 19) batch stats wander. That **train/eval-mode
    split** is why `.train()` / `.eval()` matter the moment BN (or dropout) appears — realized as `nn.BatchNorm`
    in NB 8. We do **not** build the running-stats trainer here (it's NB 8's).
21. **(md) What you built** — a complete batch-norm layer by hand: forward (per-feature batch mean/var →
    normalize → learnable γ/β) and backward (the BN Jacobian, **gradient-checked to ~1e-9**); watched it pin
    every layer's activations to a flat healthy band regardless of init; trained a deep net that had stalled at
    chance; met layer norm; and named what `nn.BatchNorm` adds. **This closes the by-hand numpy arc (NB 1–6):**
    you built a deep net, saw its gradients vanish/explode (NB 3), and fixed it three complementary ways — good
    **init** (NB 4), **dropout** (NB 5), and **normalization** (here).
22. **(md) Where this goes next** — **NB 7: the same net, now in PyTorch** — autograd builds the backward you've
    been writing by hand. The real `nn.BatchNorm` / `nn.Dropout` / `torch.nn.init` arrive in **NB 8**.
23. **(md) Your turn** — *(warm-up)* sweep the init scale (0.05, 0.5, 5.0) and confirm BN trains from all three
    while plain training stalls; *(core)* implement **layer norm** as a one-line function (`(z − z.mean(1)) /
    √(z.var(1)+ε)`) and check it gives row-wise mean 0 / var 1; *(reach)* set γ = the batch std and β = the
    batch mean and confirm BN becomes the **identity** — the learnable parameters really can recover the
    un-normalized layer.
24. **(md) References** — Ioffe, S., & Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network
    Training by Reducing Internal Covariate Shift. *ICML* (arXiv:1502.03167). Ba, J. L., Kiros, J. R., & Hinton,
    G. E. (2016). Layer Normalization. arXiv:1607.06450. Santurkar, S., Tsipras, D., Ilyas, A., & Madry, A.
    (2018). How Does Batch Normalization Help Optimization? *NeurIPS* (arXiv:1805.11604). Goodfellow, Bengio &
    Courville (2016), *Deep Learning*, ch 8. *Previous:* **12.5 — dropout.** *Next:* **12.7 — hello-world in PyTorch.**

## `src/` & guards
- **No `src/` change** (notebook-local numpy `L`-layer net + the by-hand BN forward/backward; sklearn
  **utilities** `make_circles` / `StandardScaler`; `viz.use_course_style` / `colors`; the schematic, the
  drift figure and the loss figure are **inline** matplotlib). **pytest stays 20.** **torch NOT used** (the
  real `nn.BatchNorm` is NB 8; torch + the `deep` extra + Fashion-MNIST + new tests land at NB 7 → pytest
  rises then).
- **Decisions baked in (Rémy validated):** (1) **batch-norm built by hand as a complete forward+backward
  layer** (gradient-checked ~1e-9), trained end-to-end via γ/β — the strong core. (2) **Layer norm named and
  contrasted, not a second by-hand build** (one code cell + the transformer tie) — keeps one-concept focus.
  (3) **Honest scope:** BN's measured win on this small net is **robustness** (trains a stalled net; tolerates
  a too-large lr), **not raw speed** at a good init (where plain He is even a touch faster); decisive speed
  wins are large vision nets. (4) **The "why BN helps" is flagged as debated** (covariate shift vs Santurkar's
  smoother-landscape) — reported as empirical effect, not settled mechanism. (5) **What `nn.BatchNorm` adds
  (running stats + train/eval) is named in one paragraph, explicitly deferred to NB 8** — no from-scratch BN
  *trainer* here.
- Colours only from `ml_course.colors` (no hardcoded hex); `SEED=0`, `eps=1e-5`; **"Read the figure" after
  every figure** (schematic included); **never silence output**; banned-word scan 0; ruff/black clean;
  output-free.
- Build from `build_ch12_nb6.py` (source of truth in the **ephemeral** scratchpad; rebuild right before
  `git add` — kernel-drift guard).
- Exit guards: nbconvert exit 0 (3 figures), anchors reproduce (drift table small→0 / large→1e5 / BN flat ~0.6;
  gradient-check ~1e-9; training none 0.500 → BN 1.000 small-init; BN-vs-LN; running-stats wander), banned 0,
  hex clean, ruff clean, output-free; **two-reviewer gate** (`@ml-expert-reviewer` + `@pedagogy-reviewer`, no
  BLOCK) → **Rémy visual** → end-of-NB checklist (`gen_llms_txt`, `common_errors` +rows, `course_map` §12 → NB 6
  built, pytest 20, STATE) → commit `feat(12_neuralnetworks): notebook 06 — normalization` → `git merge
  --ff-only` into `chapter/12_NeuralNetworks`.

## Verification (end-to-end)
1. nbconvert a scratchpad copy → exit 0, 3 figures, anchors reproduce (forward drift small→0 / large→1e5 / all
   BN flat ~0.6; BN backward gradient-check ~1e-9 + db≈0; training small-init none 0.500 → BN 1.000, he lr 1.0
   none 0.500 → BN 1.000; BN cols vs LN rows mean0/std1; batch-stats wander).
2. hex + banned + ruff clean. 3. pytest 20 (no `src/` change). 4. **Rémy validated this NB plan** (no reviewer
   gate; both reviewers return on the built notebook).
