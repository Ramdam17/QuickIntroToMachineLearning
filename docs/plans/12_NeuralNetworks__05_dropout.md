# NB plan — 12_NeuralNetworks / 05_dropout

> Status: **APPROVED by Rémy (via ExitPlanMode, 2026-06-30, no edits).** NB-plan stage = Rémy validates alone
> (no reviewer gate; both reviewers return on the built notebook). Anchors measured live (pure numpy + sklearn
> utilities, SEED=0; `measure_ch12_nb5.py` / `_nb5b.py` / `_nb5c.py`). **NB 5 of 10** (chapter 12 — the PyTorch
> finale). **Pure numpy — NO torch** (the real `nn.Dropout` arrives at NB 8). Chapter plan:
> `docs/plans/chapter_12_NeuralNetworks.md` (APPROVED, §NB 5).

## Context
Chapter 12, **NB 5 of 10.** One concept: **dropout** — a regularizer specific to neural networks. Each
training step, randomly **switch off** a fraction `p` of hidden units; **inverted dropout** rescales the
survivors by `1/(1−p)` so the expected activation is unchanged. The effect is an **implicit ensemble**: every
step trains a different thinned sub-network, weights shared, so units cannot **co-adapt** (lean on specific
partners that might vanish). Built **by hand in pure numpy** (`MLPClassifier` has **no dropout** — it *must*
be by-hand here; the real `nn.Dropout` is NB 8). The honest finding, measured: on a small by-hand net dropout
is a **gentle but real** regularizer (a couple of held-out points, comparable to L2); its **decisive** wins are
large high-dimensional nets (Srivastava et al. 2014, vision). It differs from **L2** (a weight penalty) and
**early stopping** (a stop rule): dropout injects noise to break co-adaptation.

## Recap vs new (the boundary)
- **RECAP (black-boxed, NOT re-derived):** over/underfitting + the train/val gap (ch 00 NB 9–10); **L2 /
  `alpha`** and **early stopping** (ch 03 NB 5, ch 00 NB 9, ch 11 NB 4) — the regularizers dropout is contrasted
  *against*; the `L`-layer net + by-hand forward/backward (NB 1–4); He init (NB 4); ReLU (ch 11).
- **NEW (the one concept):** **dropout** — inverted dropout (mean preserved, variance injected), the
  implicit-ensemble / co-adaptation-breaking intuition, wiring it into the forward+backward, and how it differs
  from L2 / early stopping.
- **No forward references:** the real `nn.Dropout` (one line) is **NB 8**; normalization (NB 6) named, not
  taught. No torch here.

## Live anchors (measured; re-pinned at build)
By-hand `L`-layer net (NB-1–4's machinery + inverted dropout: a per-step mask on the hidden activations, and
the **same mask gating the backward gradient**; train mode applies it, eval mode does not). He init. SEED=0.

- **The mechanism (inverted dropout on `N(0,1)` activations, 100k samples):** mean **preserved** (≈0 → ≈0),
  variance **injected**, `E[mask] = 1.0`:
  - `p=0.2` → var `1.00 → 1.25`; `p=0.5` → var `1.00 → 1.99` (**the headline ~2×**); `p=0.8` → var `1.00 → 5.07`.
- **Overfitting reduction (50 features / 10 informative, 200 train / 1000 val, net `[50,256,256,2]`, He, ReLU,
  mini-batch SGD lr 0.1 batch 32, 1500 ep):**
  - none: train **1.000** / val **0.752** / gap **0.248**
  - **dropout p=0.3: val 0.780 / gap 0.220** (the sweet spot here — beats L2)
  - dropout p=0.5: val 0.771 / gap 0.229
  - L2 `alpha=1e-2`: val 0.776 / gap 0.224
  - **Honest:** a real but **modest** gain (~+2–3 val points), **comparable to L2**; the decisive wins are large
    vision nets. The 2-D moons toy (noise 0.35, 80 train) gives an even smaller effect (val 0.805 → 0.815) — so
    the demo uses the higher-dimensional problem where co-adaptation actually bites.
- **Train-vs-val over epochs (none vs dropout p=0.5):** both train → 1.000 fast; val **plateaus at ~0.75
  (none) vs ~0.77 (dropout)** — dropout holds the held-out score a couple of points higher, a visibly smaller
  train–val gap. (No dramatic peak-then-collapse on this data — stated honestly.)

## Cell-by-cell (~20 cells, 2 figures) — intuition → implementation → "Read the figure"; **SHOW**; ends with "Your turn"
1. **(md) Header** `# 05 — Dropout`. **Prerequisites** (ch 00 NB 9–10: over/underfitting + the train/val gap;
   ch 03 / ch 11 NB 4: L2 / `alpha` and early stopping; NB 1–4: the `L`-layer net, He init). **What you'll do:**
   build dropout by hand, see it preserve the signal while injecting noise, watch it shrink the train–val gap,
   and place it against the regularizers you already know.
2. **(md) Another lever on overfitting.** Recap: a network with spare capacity memorizes the training set
   (train ≫ val). You already have two cures — **L2** (penalize large weights) and **early stopping** (stop
   before it memorizes). **Dropout** is a third, specific to nets: during training, randomly switch units off.
3. **(md) The mechanism: inverted dropout.** Each step, keep each hidden unit with probability `1−p`, zero it
   with probability `p`. To leave the signal's *expected* size unchanged, rescale the survivors by `1/(1−p)`
   ("**inverted**" dropout). At test time we use the whole network with **no** dropout and no rescale — the
   training-time rescale already compensated, so train and test see the same expected activations.
4. **(code)** implement `inverted_dropout(a, p, rng)`; on `N(0,1)` activations print mean preserved & variance
   injected for `p=0.2/0.5/0.8` (`1.00→1.25 / 1.99 / 5.07`; `E[mask]=1.0`).
5. **(code) Fig 1 — the mask & the preservation** (2 panels): a single activation vector → mask → result (some
   units zeroed, survivors scaled ×2 at `p=0.5`); a small bar showing mean (≈0→≈0) and variance (1.0→~2.0).
   Charter colours.
6. **(md) Read Fig 1** — half the units switch off; the survivors are scaled up so the average is unchanged,
   but the spread doubles. The network sees a **different random sub-network every step**.
7. **(md) Why it helps: an implicit ensemble.** Each step trains a different thinned sub-network (there are
   `2^H` of them), all sharing one set of weights. Using the full network at test time (with the rescale)
   approximates **averaging that ensemble**. And because any partner unit may vanish, units cannot
   **co-adapt** — they must each carry useful, robust signal. That is the regularization.
8. **(md) Wiring it into the net.** Dropout multiplies the hidden activations by the mask in the **forward**
   pass (train mode only); the **backward** pass gates the gradient by the *same* mask (a dropped unit passes
   no gradient). Eval mode: no mask. We reuse the `L`-layer net (He init, NB 4).
9. **(code)** the net + dropout forward/backward (show); a quick gradient sanity note or a small print that
   train-mode applies the mask and eval-mode does not.
10. **(md) The test.** A **wide** net on a **small, higher-dimensional** problem (50 features, 200 training
    rows) — plenty of capacity to memorize. We train none / dropout / L2 and read the held-out gap. (Higher
    dimensions because co-adaptation — and so dropout — bites hardest where there are many features to lean on;
    on a 2-D toy the effect is tiny.)
11. **(code)** train none / dropout 0.3 / dropout 0.5 / L2 1e-2; print the train / val / gap table (none val
    0.752 gap 0.248; dropout 0.3 val 0.780 gap 0.220; dropout 0.5 0.771; L2 0.776).
12. **(code) Fig 2 — train-vs-val over epochs, none vs dropout** (`p=0.5`): both train curves → 1.0; the val
    curve sits a couple of points higher with dropout (a smaller gap). Charter `train`/`test` colours.
13. **(md) Read Fig 2** — without dropout the net memorizes (train 1.0, val ~0.75); with dropout the held-out
    score holds a couple of points higher and the gap narrows. A real, **modest** improvement on this net.
14. **(md) Honest scope.** Dropout is a **gentle** regularizer on a small net like this — here it buys ~2–3
    held-out points, about the same as L2. Its **decisive** wins are large, high-dimensional networks (Srivastava
    et al. 2014, on vision), where there is far more co-adaptation to break. We show the honest, modest version.
15. **(md) Dropout vs L2 vs early stopping** — three different levers: **L2** shrinks weights (penalty in the
    loss); **early stopping** halts before overfitting (a stop rule); **dropout** injects noise to break
    co-adaptation (an implicit ensemble). They target the same disease differently and **can be combined**.
16. **(md) What you built** — inverted dropout by hand (mean preserved, variance injected `1→2`), wired into
    the forward + backward (mask the activations *and* the gradient), watched it shrink the train–val gap, and
    placed it against L2 / early stopping.
17. **(md) Where this goes next** — **NB 6 — normalization**: another way to keep activations healthy *during*
    training. The real one-line **`nn.Dropout`** layer arrives in **NB 8** (PyTorch).
18. **(md) Your turn** — *(warm-up)* sweep `p` from 0.1 to 0.7 and find the value with the best val accuracy;
    *(core)* combine dropout **and** L2 — does the pair beat either alone here?; *(reach)* measure the variance
    of the net's first hidden layer with vs without dropout at `p=0.5` — does it match the `1→2` you saw on
    `N(0,1)`?
19. **(md) References** — Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014).
    Dropout: a simple way to prevent neural networks from overfitting. *JMLR* 15:1929–1958. Hinton, G., et al.
    (2012). Improving neural networks by preventing co-adaptation of feature detectors. arXiv:1207.0580.
    Goodfellow, Bengio & Courville (2016), *Deep Learning*, ch 7. *Previous:* **12.4 — initialization.**
    *Next:* **12.6 — normalization.**

## `src/` & guards
- **No `src/` change** (notebook-local numpy `L`-layer net + inverted dropout; sklearn **utilities**
  `make_classification` / `train_test_split` / `StandardScaler`; `viz.use_course_style` / `colors`; the
  mask/preservation figure and the train-vs-val curves are **inline** matplotlib). **pytest stays 20.** **torch
  NOT used** (the real `nn.Dropout` is NB 8).
- **Decisions baked in (Rémy validated):** (1) **all by-hand numpy** (`MLPClassifier` has no dropout — it must
  be by-hand). (2) **Honest scope** — dropout is a *gentle* regularizer here (~+2–3 val points, comparable to
  L2); decisive wins are large vision nets; the demo uses a **higher-dimensional** problem (50 features) where
  co-adaptation bites, since the 2-D toy effect is negligible — stated, not hidden. (3) The **mechanism**
  (mean-preserved / variance `1→2` / implicit ensemble) is the strong core; the overfitting reduction is real
  but modest.
- Colours only from `ml_course.colors` (no hardcoded hex); `SEED=0`; **"Read the figure" after every figure**;
  **never silence output**; banned-word scan 0; ruff/black clean; output-free.
- Build from `build_ch12_nb5.py` (source of truth in the **ephemeral** scratchpad; rebuild right before
  `git add` — kernel-drift guard).
- Exit guards: nbconvert exit 0 (2 figures), anchors reproduce (variance `1→2` at p=0.5; none val 0.752 / gap
  0.248 → dropout 0.3 val 0.780 / gap 0.220; train-vs-val curves), banned 0, hex clean, ruff clean,
  output-free; **two-reviewer gate** (`@ml-expert-reviewer` + `@pedagogy-reviewer`, no BLOCK) → **Rémy visual** →
  end-of-NB checklist (`gen_llms_txt`, `common_errors` +rows, `course_map` §12 → NB 5 built, pytest 20, STATE) →
  commit `feat(12_neuralnetworks): notebook 05 — dropout` → `git merge --ff-only` into `chapter/12_NeuralNetworks`.

## Verification (end-to-end)
1. nbconvert a scratchpad copy → exit 0, 2 figures, anchors reproduce (inverted-dropout variance `1→2` at p=0.5
   & mean preserved; dropout shrinks the train–val gap 0.248 → 0.220; the train-vs-val curves).
2. hex + banned + ruff clean. 3. pytest 20 (no `src/` change). 4. **Rémy validated this NB plan** (no reviewer
   gate; both reviewers return on the built notebook).
