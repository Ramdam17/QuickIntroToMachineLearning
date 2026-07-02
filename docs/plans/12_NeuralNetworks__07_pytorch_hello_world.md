# NB plan — 12_NeuralNetworks / 07_pytorch_hello_world

> Status: **APPROVED by Rémy (via ExitPlanMode, 2026-06-30, no edits).** NB-plan stage = **Rémy validates
> alone** (no reviewer gate; both reviewers return on the built notebook). **NB 7 of 10** (chapter 12 — the
> PyTorch finale). **First torch notebook + first `src/` change of the chapter.** torch anchors measured
> **on-box** (torch 2.12.1, CPU, determinism verified). Chapter plan: `docs/plans/chapter_12_NeuralNetworks.md`
> (APPROVED, §NB 7) + open-decisions 1/3/8.

## Context
Chapter 12, **NB 7 of 10 — the framework move.** For six notebooks we wrote forward, backward, and the
training loop **by hand in numpy** — to know exactly what happens. NB 7 hands **the same network as NB 1**
(2 → 16 → 3, ReLU hidden, softmax + cross-entropy, on `make_blobs`) to **PyTorch**, the real modern tool. The
one decisive change: **you stop writing the backward pass** — you describe the forward computation and
**autograd** computes every gradient. Everything is **SHOWN, not assigned** (Rémy's rule: *"on montre, comme
on a toujours fait"*) — the learner reads the canonical `nn.Module` + training loop and recognizes every piece
as something they already built; the "Your turn" challenges *modify* the shown loop, never author it from
blank. Plain torch — **no skorch / Lightning.** The emotional beat: the framework is not a black box that
replaces understanding — it is **exactly what you built, automated and validated** (the autograd gradient
equals your by-hand gradient to machine precision).

## Recap vs new (the boundary)
- **RECAP (one sentence each, NOT re-taught):** epoch / mini-batch / learning rate / the loss curve as a
  diagnostic (ch 11 NB 4); the forward → loss → backward → GD-loop → train/test → eval machinery (NB 1); the
  softmax + cross-entropy gradient `(p − y)/n` (NB 1); ReLU (ch 11); the `2→16→3` net (NB 1, our reference).
- **NEW (the framework move):** **autograd** (the backward pass you no longer write — `loss.backward()`); the
  **`nn.Module` / `nn.Sequential`** idiom (parameters + `forward` bundled); the **canonical training loop**
  (`zero_grad → forward → loss → backward → step`); **`.train()` / `.eval()` mode** (introduced now, matters
  the moment dropout/BN appear in NB 8); `nn.CrossEntropyLoss` = softmax + CE folded into one stable step.
- **No forward references:** real `nn.Dropout` / `nn.BatchNorm` / `torch.nn.init` / optimizers-as-a-menu are
  **NB 8**; the Fashion-MNIST capstone is **NB 9**; CNN/RNN/transformer are **NB 10** horizons.

## Live anchors (measured ON-BOX; re-pinned at build) — `measure_ch12_nb7.py`, `verify_torch_determinism.py`
torch **2.12.1**, CPU (CUDA absent). Determinism contract: `torch.manual_seed(0)` + `np.random.seed(0)` +
`torch.use_deterministic_algorithms(True)` + `torch.set_num_threads(1)` → **bit-identical params across two
separate process runs** (param-hash `5c633ac9…` reproduced; within-process re-seed identical). So torch numbers
are usable as bit-stable anchors here.

- **THE build anchor — autograd gradient == NB-1 numpy gradient** (weights synced, float64, same batch):
  untrained loss numpy **1.1011530975** == torch **1.1011530975** (≈ ln 3; |Δ| 3e-12); per-parameter
  **max|Δgrad|: dW1 6.9e-18, db1 1.7e-17, dW2 6.9e-18, db2 2.4e-17** — equal to **machine precision**. (The
  match is done in **float64** to isolate the *mathematics* from float precision; training below uses float32,
  torch's default.)
- **Training parity — the same net in torch (float32, canonical loop, NB-1's init synced, SGD 400 ep):**
  - **lr 0.5: final loss 0.308, train 0.867 / test 0.893** — *identical* to NB 1's by-hand reference
    (train 0.867 / test 0.893); lr 0.1 → 0.876/0.907; lr 1.0 → 0.876/0.893. The loss falls **ln 3 → ~0.31**,
    matching NB 1. (Same architecture + same init + same data + same optimizer → the same result.)
- **`nn.Linear` shape note (for the honest bridge):** torch stores the weight as `(out, in)` = our `Wᵀ`, and
  folds in the bias; otherwise it is the `X·W + b` we wrote. `nn.CrossEntropyLoss` expects **raw logits** and
  applies log-softmax internally (no softmax layer in the model).

## Cell-by-cell (~21 cells, 2 figures) — intuition → SHOW → "Read the figure"; ends with "Your turn"
1. **(md) Header** `# 07 — Hello-world in PyTorch`. **Prerequisites** (NB 1 the reference `2→16→3` net; NB 3
   backprop = the chain rule; ch 11 GD / epoch / mini-batch / learning rate). **What you'll do:** meet
   `nn.Module`, autograd, and the canonical loop; **see torch's gradient equal your by-hand gradient to machine
   precision**; train the same net to the same accuracy; and meet the `.train()`/`.eval()` switch. **You will
   read this code, not write it from blank — the point is to recognize every line as something you built.**
2. **(md) Why a framework now.** Six notebooks by hand were the point — to know exactly what happens. Real work
   uses a framework for one decisive reason: **you stop writing the backward pass.** You describe the forward
   computation; **autograd** records it and computes every gradient. PyTorch is the tool; the ideas are the ones
   you already built. (Plain torch — no skorch/Lightning.)
3. **(md) The three things torch gives you.** (a) **tensors + autograd** — like numpy arrays that *remember*
   the operations done to them, so `loss.backward()` fills in every gradient; (b) **`nn.Module` /
   `nn.Sequential`** — parameters + a `forward`, bundled; (c) **optimizers** (`optim.SGD`) — the `θ ← θ − η·∇L`
   step you wrote by hand. Plus the reproducibility contract (seeds + `use_deterministic_algorithms(True)` +
   CPU + one thread → bit-identical runs, verified on this machine).
4. **(code)** `import torch`; set the determinism contract (`torch.manual_seed(0)`, `np.random.seed(0)`,
   `torch.use_deterministic_algorithms(True)`, `torch.set_num_threads(1)`); print `torch.__version__` and
   `torch.cuda.is_available()` (False → CPU). Load NB-1's data (`make_blobs` 3-class, split, `StandardScaler`
   on train). **SHOW.**
5. **(md) The same net, in torch.** NB 1's net was `2→16→3`: linear → ReLU → linear → softmax+CE. In torch:
   `nn.Sequential(nn.Linear(2,16), nn.ReLU(), nn.Linear(16,3))`, with the softmax folded into the loss
   (**`nn.CrossEntropyLoss` = softmax + cross-entropy** in one numerically-stable step). Note: `nn.Linear`
   stores its weight as `(out, in)` — the transpose of our `W` — and includes the bias; otherwise it is the
   `X·W + b` you wrote.
6. **(code)** define the torch net (`nn.Sequential`) + `criterion = nn.CrossEntropyLoss()`; print the module and
   its parameter shapes (`Linear` weights `(16,2)`, `(3,16)`) and the total parameter count. **SHOW.**
7. **(md) Autograd: the backward pass you no longer write.** In NB 1 you derived `(p − y)/n` and pushed it back
   through ReLU and the weight matrices by hand. Torch records every operation on a graph; `loss.backward()`
   walks it and deposits `∂loss/∂θ` in each parameter's `.grad`. To prove it is the **same** computation, we
   copy NB-1's exact weights into the torch net and compare the two gradients on the same batch.
8. **(code)** define NB-1's by-hand numpy net (init/forward/backward — the reference); copy its weights into the
   torch net (`weight = Wᵀ`); run torch forward + `cross_entropy` + `backward`; print the untrained loss (numpy
   **1.10115** == torch **1.10115** ≈ ln 3) and the per-parameter **max|Δgrad|** (dW1 6.9e-18 … all ~**1e-17**,
   float64). **SHOW both backwards side by side.** The build anchor.
9. **(code) Fig 1 — numpy ↔ torch, side by side** (2 panels): **left**, the component-by-component
   correspondence (your `forward` ↔ `nn.Module.forward`; your hand-derived `backward` ↔ `loss.backward()`
   /autograd; your `θ ← θ − η∇L` ↔ `optimizer.step()`; your loss ↔ the criterion) as a labelled schematic;
   **right**, a scatter of **numpy gradient vs torch gradient** for every parameter — all points on the `y = x`
   line (identical), with the **max|Δ| ≈ 1e-17** annotated on the panel. Charter colours.
10. **(md) Read Fig 1** — each piece of the framework is a piece you already built; and torch's autograd
    gradient equals your hand-derived gradient to ~`1e-17` (machine precision). The framework is **exactly what
    you built, automated** — not a black box that replaces understanding.
11. **(md) The canonical training loop.** Every torch loop is the same five lines, each one you have written by
    hand: `optimizer.zero_grad()` (clear last step's grads) · `out = model(X)` (forward) ·
    `loss = criterion(out, y)` · `loss.backward()` (autograd fills `.grad`) · `optimizer.step()` (the update).
    This shape is the heartbeat of all deep learning.
12. **(code)** the canonical loop, **SHOWN**: instantiate the net (NB-1's init synced for a fair parity),
    `optim.SGD(lr=0.5)`, 400 full-batch epochs recording the loss; print final loss + train/test accuracy
    (**loss ~0.31, train 0.867 / test 0.893 — the NB-1 numbers**). Uses `loss.item()` (no grad-scalar warning).
13. **(code) Fig 2 — numpy vs torch loss curve** (overlaid): NB-1's by-hand numpy loss (float64) and the torch
    loss (float32), both falling **ln 3 → ~0.31** — two curves, one line. Charter `train` / `model` colours.
14. **(md) Read Fig 2** — the torch net's loss falls from `ln 3` (knowing nothing) to ~`0.31` and reaches train
    0.867 / test 0.893 — **the same numbers as NB 1's by-hand net, because it is the same net** (same data,
    same starting weights, same optimizer). The framework changed *how we express* the computation, not *what*
    it computes.
15. **(md) `.train()` and `.eval()` — the mode switch.** One genuinely new idea: a module has two modes,
    `model.train()` and `model.eval()`. For this plain net they behave identically — but the moment you add
    **dropout** or **batch norm** (NB 8) they differ (dropout active only in train; batch norm uses batch stats
    in train, the **running stats** from NB 6 in eval). Flipping the switch tells the network "we are learning"
    vs "we are predicting." We meet it now so it is familiar when it starts to matter.
16. **(code)** toggle `net.train()` / `net.eval()` and print `net.training` (True/False); note predictions are
    unchanged here (no dropout/BN yet) — a one-line confirmation. **SHOW.**
17. **(md) What carried over, what's new.** Recap unchanged from ch 11 / NB 1–6: epoch, mini-batch, learning
    rate, the loss curve as a diagnostic. New in torch: **autograd** (stop writing the backward), the
    **`nn.Module`** idiom, and **`.train()`/`.eval()`** modes. The framework move is a change of *tooling*, not
    of *ideas*.
18. **(md) What you built** — the same `2→16→3` net as NB 1, now in PyTorch: `nn.Sequential` +
    `nn.CrossEntropyLoss`; the canonical loop (`zero_grad → forward → loss → backward → step`); the proof that
    autograd's gradient equals your by-hand gradient to machine precision; the same train/test accuracy; and the
    `.train()`/`.eval()` switch. You can now read any PyTorch training script and recognize every line.
19. **(md) Where this goes next** — **NB 8**: the model and its parameters in torch — real `nn.Dropout`,
    `nn.BatchNorm`, `torch.nn.init` (He/Xavier), and optimizers (SGD / momentum / Adam) as one-liners on top of
    this loop, and where `.train()`/`.eval()` starts to matter.
20. **(md) Your turn** — *(warm-up)* swap `optim.SGD` for `optim.Adam` in the shown loop and compare the loss
    curve; *(core)* add a second hidden layer (`nn.Linear(16,16), nn.ReLU()`) to the `nn.Sequential` and retrain
    — does it help on this easy 2-D problem?; *(reach)* re-initialize the net with `torch.nn.init.kaiming_normal_`
    (He, NB 4) and re-run — then re-run the gradient-match with torch's default init and confirm the gradients
    still agree (autograd is exact regardless of init). **(Every task modifies the shown loop — you never write
    the backward pass.)**
21. **(md) References** — Paszke, A., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning
    Library. *Advances in Neural Information Processing Systems* (NeurIPS). + pytorch.org documentation.
    *Previous:* **12.6 — normalization.** *Next:* **12.8 — the model and its parameters in PyTorch.**

## `src/` & guards
- **`src/` change (first of ch 12):** the **`deep = ["torch>=2.2"]`** extra is added to `pyproject.toml` (torch
  **2.12.1** resolved, CPU, pinned in `uv.lock`; **already committed** `build(deps)` 08fbcf2 after verifying
  on-box determinism). NB 7 adds a **`tests/test_torch_determinism.py`** (torch imports + CPU-only; a seeded
  computation is reproducible) → **pytest rises from 20** (state the exact new total at build).
- **Loader-scope decision (recommendation — Rémy to confirm via ExitPlanMode):** the **Fashion-MNIST loader +
  `scripts/vendor_fashion_mnist.py` + loader tests land at NB 9** (just-in-time, where the capstone uses them),
  **not** here. NB 7 uses only NB-1's synthetic `make_blobs`, so adding an unused loader now would break the
  just-in-time / one-concept discipline. (This narrows my earlier STATE note that bundled the loader into NB 7.)
- **Determinism / never-silence:** the determinism contract is set in the notebook (seeds +
  `use_deterministic_algorithms(True)` + `set_num_threads(1)`), verified on-box; **no output silenced** (torch
  prints, warnings visible; `loss.item()` used to avoid the requires-grad-scalar `UserWarning`). No dataset
  fetch here (make_blobs is synthetic; the Fashion-MNIST fetch is NB 9, logged-not-silenced).
- **Shown, not assigned:** the model + loop are demonstrated to read; the three "Your turn" tiers **modify** the
  shown loop (swap optimizer / add a layer / change init), never author it from blank (chapter open-decision 14).
- Colours only from `ml_course.colors` (no hardcoded hex); `SEED=0`; **"Read the figure" after every figure**;
  banned-word scan 0; ruff/black clean; output-free.
- Build from `build_ch12_nb7.py` (source of truth in the **ephemeral** scratchpad; rebuild right before
  `git add` — kernel-drift guard).
- Exit guards: nbconvert exit 0 (2 figures) with the torch determinism set; anchors reproduce (autograd gradient
  == numpy gradient ~1e-17; untrained loss ln 3; training loss ln3→~0.31, train 0.867 / test 0.893); banned 0,
  hex clean, ruff clean, output-free; **pytest at the new count** (torch-determinism test); **two-reviewer gate**
  (`@ml-expert-reviewer` + `@pedagogy-reviewer`, no BLOCK) → **Rémy visual** → end-of-NB checklist
  (`gen_llms_txt`, `common_errors` +rows, `course_map` §12 → NB 7 built, STATE) → commit
  `feat(12_neuralnetworks): notebook 07 — pytorch hello-world` → `git merge --ff-only` into `chapter/12_NeuralNetworks`.

## Verification (end-to-end)
1. nbconvert a scratchpad copy (with the determinism contract) → exit 0, 2 figures, anchors reproduce (gradient
   match ~1e-17; untrained loss ln 3; loss ln3→~0.31; train 0.867 / test 0.893; `.train()/.eval()` toggles).
2. hex + banned + ruff clean; output-free. 3. **pytest at the new count** (the torch-determinism test added;
   state the number). 4. **Rémy validates this NB plan** (no reviewer gate; both reviewers return on the built
   notebook).
