# NB plan — 12_NeuralNetworks / 03_vanishing_exploding_gradients — the pivot

> Status: **APPROVED by Rémy (via ExitPlanMode, 2026-06-30, no edits).** NB-plan stage = Rémy validates alone
> (no reviewer gate; both reviewers return on the built notebook). Anchors measured live (pure numpy + sklearn
> utilities, SEED=0; `measure_ch12_nb3.py` / `_nb3b.py` / `_nb3c.py` / `_nb3d.py`). **NB 3 of 10** (chapter 12
> — the PyTorch finale). **Pure numpy — NO torch** (the `deep` extra lands at NB 7). Chapter plan:
> `docs/plans/chapter_12_NeuralNetworks.md` (APPROVED, §NB 3 — **the pivot**).

## Context
Chapter 12, **NB 3 of 10 — THE PIVOT.** One concept: **why naive deep nets don't train.** Backprop carries the
gradient backward layer by layer, each step multiplying by one factor (`W · activation'`). For an early
layer, the gradient is a **product of one factor per layer above it** — multiply many numbers and the signal
either **collapses to ~0 (vanishing)** or **blows up (exploding)**. Built **by hand in pure numpy** (reuse the
NB-1/NB-2 `L`-layer net): run *one* forward+backward through a **10-layer** stack and **measure the per-layer
gradient RMS**, then watch a real **5-layer sigmoid** net stall at chance while tanh/ReLU train. This is the
hinge of the chapter — **c03 (init), c04 (normalization), c06 (the flat-curve diagnostic) all exist because of
this.** The fix (initialization) is **NB 4** — NB 3 only *diagnoses*.

## Recap vs new (the boundary)
- **RECAP (black-boxed, NOT re-derived):** the `L`-layer forward+backward (NB 1/NB 2); backprop = the chain
  rule layer by layer (ch 11 NB 3); **sigmoid saturates → σ' ≤ 0.25** (ch 11 NB 4 — the *shallow* sigmoid+adam
  stall); the `ln 2` "knowing-nothing" loss baseline (NB 1); small init = scale 0.1 (NB 1/2).
- **NEW (the one concept):** the gradient as a **product of per-layer factors** across depth → **vanishing**
  (factor < 1, e.g. sigmoid's ≤ 0.25) and **exploding** (factor > 1, e.g. large weights); that this is *why*
  naive deep nets don't train (a deep sigmoid stalls *unrecoverably*, even with the optimizer that rescued it
  shallow).
- **No forward references:** the **fix** (variance-preserving init He/Xavier) is **NB 4** — NB 3 names that a
  "balanced scale" exists and points forward, but does **not** teach or plot it (no `sqrt(2/n)` in the code).
  Normalization (NB 6) named as a second lever, not taught.

## Live anchors (measured; re-pinned at build)
By-hand `L`-layer net (NB-1/2 machinery + a **sigmoid hidden option** and an **init-scale knob**: `small`=std
0.1, `large`=std 1.0), SEED=0, on `make_circles(400, noise=0.1, factor=0.4)` standardized; 10 hidden layers ×
width 16 (11 weight matrices). The *training* proof uses **`MLPClassifier`** (sklearn's own estimator,
end-to-end — its internal init keeps NB 3 free of the NB-4 init topic).

- **Backward gradient RMS per layer (the headline):**
  - **sigmoid + small → VANISH:** output-layer `2.9e-2` → input-layer **`8.4e-14`** (clean monotone fall over
    ~12 orders of magnitude; the early layers get ~zero gradient).
  - **ReLU + large(std 1) → EXPLODE:** **`~1e3–5e3`** at every layer (vs a healthy ~`1e-2`). *(Magnitude grows
    with width/depth: width 32 → ~`1e4`; the order, not the exact value, is the lesson.)*
- **Forward activation RMS per layer (the visceral companion):** sigmoid+small **flat ~0.5** (the signal
  collapses to a constant — saturated, carrying no information); ReLU+large **`0.8 → 7800`** (the activations
  literally blow up across depth).
- **Training proof — `MLPClassifier` (16,)×5 on circles, default adam, 5 seeds:** **sigmoid (logistic)
  `0.500` (chance, all seeds)**, loss curve **flat at `ln 2 ≈ 0.6932`** (0.72 → 0.693, n_iter 31); **tanh
  `1.000`** (loss → 0.006), **ReLU `1.000`** (loss → 0.002). The vanishing gradient, in a real run.
- **"Unrecoverable" honest check — `lbfgs` (the solver that rescued the *shallow* sigmoid in ch 11):** sigmoid
  depth **1 → `1.000`**, depth **5 → `0.478` (chance)**; tanh/ReLU depth 5 → `1.000`. So depth (not the
  optimizer) is the cause — the shallow stall was recoverable, the deep one is not. *(Honest nuance: with adam,
  sigmoid stalls at every depth — ch 11 NB 4; the clean "depth makes it unrecoverable" contrast needs lbfgs,
  where shallow genuinely trained.)*
- **Exercise answers (verified):** sigmoid ×20 layers → input gradient **`4.4e-24`** (deeper = worse);
  **ReLU + small** → gradient **small-but-flat ~`1e-7`** (does NOT explode — explosion needs large weights;
  the scale is the lever → NB 4).

## Cell-by-cell (~21 cells, 3 figures) — intuition → implementation → "Read the figure"; **SHOW**; ends with "Your turn"
1. **(md) Header** `# 03 — Vanishing and exploding gradients`. **Prerequisites** (NB 1–2: the `L`-layer net,
   forward+backward by hand; ch 11 NB 3: backprop = the chain rule; ch 11 NB 4: sigmoid saturates → tiny
   gradients). **What you'll do:** see *why* naive deep nets won't train — measure the gradient collapse/blow-up
   by hand, then watch it stall a real 5-layer net.
2. **(md) The pivot.** NB 2: two or three layers trained fine — but deep learning's bet is *many* layers. Stack
   ten the naive way and the net stops learning, its loss flat from the first epoch. *Why?* This notebook
   answers it, and it is the hinge the rest of the chapter turns on (initialization, normalization all exist
   because of this).
3. **(md) The mechanism: the chain rule across many layers.** Recap (ch 11 NB 3): backprop carries the gradient
   backward, each layer multiplying it by a factor (`W · activation'`). For an **early** layer the gradient is a
   **product of one factor per layer above it**. Multiply many numbers: each `< 1` → product collapses toward 0
   (**vanish**); each `> 1` → product blows up (**explode**). The deeper the net, the more factors — the more
   extreme the outcome.
4. **(code) Fig 1 — the chain-of-factors schematic** (matplotlib-drawn): a backward signal entering at the
   output, passing through `L` layers; one path multiplied by ~0.3 each step (shrinks → vanish), one by ~2 each
   step (grows → explode). Charter colours.
5. **(md) Read Fig 1** — the same arithmetic that lets depth *compose* representations (NB 2) also compounds the
   gradient: a chain of small factors annihilates it, a chain of large ones detonates it.
6. **(md) Let's measure the real thing.** Reuse the NB-1/2 net (forward + backward by hand), now with a
   **sigmoid** option and a knob for the **initial weight scale**. Build a deep **10-hidden-layer** stack, run
   *one* forward+backward, and read the gradient size at each layer. Two telling setups: **sigmoid + small
   weights**, and **ReLU + large (std-1) weights**.
7. **(code)** generalize the net (add sigmoid hidden + `scale` param — shown), build the 10-layer stack on
   standardized circles, compute **per-layer gradient RMS** and **forward activation RMS** for both setups;
   print (sigmoid gradient `2.9e-2 → 8.4e-14`; ReLU+large `~1e3–5e3`; forward sigmoid flat `~0.5`, ReLU+large
   `0.8 → 7800`). Never silence.
8. **(code) Fig 2 — two panels, log-y:** forward activation RMS per layer (left) and backward gradient RMS per
   layer (right), **sigmoid+small (vanish)** vs **ReLU+large (explode)**. Charter colours.
9. **(md) Read Fig 2** — vanish: the sigmoid gradient falls ~12 orders to `~1e-13` at the input layer, and its
   forward signal flatlines at `0.5` (saturated — every unit pinned mid-range, carrying nothing); explode:
   ReLU+large's activations climb `0.8 → ~8000` and its gradient sits at `~10³`. Either way the early layers are
   unusable — frozen or blown up.
10. **(md) Why.** **Sigmoid:** `σ'(z) ≤ 0.25` always (recap ch 11 NB 4 — the flat tails), so each layer shrinks
    the backward signal by at least ~4× — ten layers ≤ `0.25¹⁰ ≈ 1e-6`, and the small weights push it far
    lower. **ReLU + large weights:** `ReLU'` is 0 or 1 (no shrink), but std-1 weights scale the signal *up*
    every layer → the product explodes. The **factor per layer** is the whole story. *(There is a scale that
    keeps each factor ≈ 1 and the signal steady across depth — that is **initialization**, NB 4.)*
11. **(md) The consequence.** If the early layers receive ~0 gradient, they never move from their random start —
    the deep net is stuck. Let's confirm it in a real training run.
12. **(code)** train **5-hidden-layer** nets with **`MLPClassifier`** (sklearn's own estimator, end-to-end,
    default adam) on circles — sigmoid vs tanh vs ReLU; read `loss_curve_` + accuracy. Print sigmoid loss
    `0.72 → 0.693` (= `ln 2`) acc **0.500**; ReLU `→ 0.002` acc **1.000**; tanh `→ 0.006` acc **1.000**.
13. **(code) Fig 3 — the loss curves:** 5-layer sigmoid **flat at `ln 2 ≈ 0.69`** vs tanh/ReLU descending to
    ~0. Charter colours; annotate the `ln 2` baseline (NB 1's "knowing nothing").
14. **(md) Read Fig 3** — the sigmoid net's loss never leaves `ln 2` (NB 1's random-guess baseline): its early
    layers get no gradient, so it never learns. tanh and ReLU descend to ~0. This is the vanishing gradient,
    live, in a real 5-layer net.
15. **(md) "Is it just a bad optimizer?"** The honest check. In ch 11 a *shallow* sigmoid net was fine (1 layer
    + `lbfgs` → 1.0). Is the deep stall merely adam's fault? Test the solver that rescued the shallow sigmoid.
16. **(code)** `lbfgs`: sigmoid depth **1 → 1.000**, depth **5 → 0.478** (chance); tanh/ReLU depth 5 → **1.000**.
    Print.
17. **(md) Read** — **depth, not the optimizer, is the cause:** the shallow sigmoid was recoverable (swap to
    lbfgs); the deep one is not — no optimizer rescues a gradient that has already vanished. And the
    **activation matters**: sigmoid saturates worst, so it breaks first; tanh/ReLU survive depth 5 here — but
    stack enough layers and they break too. The general cure is next.
18. **(md) What you built** — diagnosed the **vanishing/exploding gradient by hand** (per-layer RMS through a
    10-layer stack), understood the **chain-of-factors** mechanism, and watched it **stall a real 5-layer
    sigmoid net** (loss pinned at `ln 2`). This is the pivot of the chapter: everything after it is a cure.
19. **(md) Where this goes next** — **NB 4: initialization** (He & Xavier) — choose the weight scale so each
    per-layer factor stays ≈ 1 and the signal survives across depth: the direct fix for what we just diagnosed.
    (Normalization, NB 6, is a second lever.)
20. **(md) Your turn** — *(warm-up)* deepen the sigmoid stack from 10 to **20** layers and re-measure the
    input-layer gradient — how much smaller? (it falls to `~1e-24`); *(core)* try **ReLU with small weights
    (0.1)** instead of large — does it still explode? (no — explosion needs large weights; the gradient goes
    small-but-flat — the scale is the lever, a taste of NB 4); *(reach)* read `loss_curve_` for the sigmoid net
    and confirm it sits at `ln 2` from the very first epoch.
21. **(md) Where this goes next / References** — Hochreiter (1991, *Untersuchungen zu dynamischen neuronalen
    Netzen*, diploma thesis — the original diagnosis); Bengio, Simard & Frasconi (1994), *Learning long-term
    dependencies with gradient descent is difficult*, IEEE TNN 5(2):157–166, DOI 10.1109/72.279181; Glorot &
    Bengio (2010), *Understanding the difficulty of training deep feedforward neural networks*, PMLR 9:249–256
    (the analysis behind the init fix → NB 4); Goodfellow, Bengio & Courville (2016), *Deep Learning*, ch 8.
    *Previous:* **12.2 — depth is a representation hierarchy.** *Next:* **12.4 — initialization: He & Xavier.**

## `src/` & guards
- **No `src/` change** (notebook-local numpy `L`-layer net — NB-2's, + a sigmoid option & init-scale knob;
  sklearn **utilities** `make_circles` / `StandardScaler` + **`MLPClassifier`** for the training proof;
  `viz.use_course_style` / `colors`; the schematic, the 2-panel RMS figure, and the loss curves are **inline**
  matplotlib). **pytest stays 20.** **torch NOT used** (the `deep` extra arrives at NB 7).
- **Decisions baked in (Rémy validated):** (1) the **mechanism** (per-layer gradient/forward RMS) is **by-hand
  numpy**; the **training proof** is **`MLPClassifier`** (sklearn's internal init keeps the NB-4 init topic out
  of NB 3 — no forward reference). (2) **No `he`/`xavier` in NB-3 code** — only `small` (0.1) and `large` (1.0)
  init; the "balanced scale" is *named* as NB 4's job, not shown. (3) Report the **measured** explode magnitude
  (`~1e3–5e3` gradient / `~8000` forward at width 16), not the chapter-plan's order-of-magnitude "~5e6" — note
  it grows with width/depth.
- Colours only from `ml_course.colors` (no hardcoded hex); `SEED=0`; **"Read the figure" after every figure**;
  **never silence output**; banned-word scan 0; ruff/black clean; output-free.
- Build from `build_ch12_nb3.py` (source of truth in the **ephemeral** scratchpad; rebuild right before
  `git add` — kernel-drift guard).
- Exit guards: nbconvert exit 0 (3 figures), anchors reproduce (sigmoid gradient `2.9e-2 → ~8e-14`; ReLU+large
  `~1e3+`; sigmoid 5-layer loss flat at `ln 2`, acc 0.500 vs tanh/ReLU 1.000; lbfgs depth 1→1.0 / 5→chance),
  banned 0, hex clean, ruff clean, output-free; **two-reviewer gate** (`@ml-expert-reviewer` +
  `@pedagogy-reviewer`, no BLOCK) → **Rémy visual** → end-of-NB checklist (`gen_llms_txt`, `common_errors`
  +rows, `course_map` §12 → NB 3 built, pytest 20, STATE) → commit `feat(12_neuralnetworks): notebook 03 —
  vanishing and exploding gradients` → `git merge --ff-only` into `chapter/12_NeuralNetworks`.

## Verification (end-to-end)
1. nbconvert a scratchpad copy → exit 0, 3 figures, anchors reproduce (gradient vanish `~8e-14` / explode
   `~1e3`; sigmoid 5-layer loss flat at `ln 2`, acc 0.500 vs tanh/ReLU 1.000; lbfgs depth 1→1.0 / 5→chance).
2. hex + banned + ruff clean. 3. pytest 20 (no `src/` change). 4. **Rémy validated this NB plan** (no reviewer
   gate; both reviewers return on the built notebook).
