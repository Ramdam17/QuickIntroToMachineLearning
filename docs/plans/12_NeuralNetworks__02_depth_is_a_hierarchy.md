# NB plan — 12_NeuralNetworks / 02_depth_is_a_hierarchy — depth is a representation hierarchy

> Status: **APPROVED by Rémy (via ExitPlanMode, 2026-06-29, no edits).** NB-plan stage = Rémy validates alone
> (no reviewer gate; both reviewers return on the built notebook). Anchors measured live (pure numpy + sklearn
> utilities, SEED=0; `measure_ch12_nb2_remap.py` / `_depth.py` / `_equalbudget.py` / `_init.py`). **NB 2 of
> 10** (chapter 12 — the PyTorch finale). **Pure numpy — NO torch** (the `deep` extra lands at NB 7). Chapter
> plan: `docs/plans/chapter_12_NeuralNetworks.md` (APPROVED, §NB 2).

## Context
Chapter 12, **NB 2 of 10.** One concept: **why stack layers.** A hidden layer *remaps* the input into a new
representation; stacking layers *composes* remaps; **depth — not just width — is the bet.** Built **by hand
in pure numpy** by generalizing NB-1's net to `L` layers. The lesson is delivered **honestly, measured**: we
*see* depth untangle a 2-D problem into linear separability, then *measure* that on flat data depth's gain is
**modest** (width does most of the lifting; at equal budget depth is ≈ a near-wash) — and we name *why* the
**vivid** feature hierarchy (edges→parts→objects) needs spatial **structure**, i.e. a CNN (NB 10's horizon).
It closes by foreshadowing **NB 3**: deep nets are hard to *train* (vanishing/exploding gradients).

## Recap vs new (the boundary)
- **RECAP (black-boxed, NOT re-derived):** NB-1's forward / backward / GD machinery, now **generalized to an
  `L`-layer loop** (the chain rule lives in 11.3 / NB 1 — we *package* it, re-checking the gradient once);
  flat small init (scale 0.1 — *same as NB 1*; "breaks symmetry, the principled scale is **NB 4**"); ReLU /
  tanh / softmax / sigmoid (ch 11 / NB 1); train/test + accuracy + scaling (ch 00); **linear separability /
  what a straight line can split** (ch 03).
- **NEW (the one concept):** **depth as a representation hierarchy** — successive layers *compose* remaps;
  what depth buys (measured, honest) vs width; *why* the vivid hierarchy needs structure (CNN, named).
- **No forward references:** **flat-0.1 init (NB-1's) trains every net here** — measured — so He/Xavier stay
  in **NB 4**; vanishing/exploding stays in **NB 3** (only *foreshadowed*); CNNs are *named*, met in NB 10.

## Live anchors (measured; re-pinned at build)
All nets **by hand, pure numpy, flat scale-0.1 init (NB-1's), SEED=0.** sklearn used only for **utilities**
(`make_moons`, `fetch_openml`, `train_test_split`, `StandardScaler`) and a **linear-separability probe**
(`LogisticRegression` — a *measurement* tool on the representation, never the model under study).

- **The remap (moons `[2,2,2,1]`, tanh hidden + sigmoid out, lr 0.5, 6000 epochs):** all-width-2 so every
  layer's activation is 2-D and plottable. **Linear-sep probe rises with depth: input 0.877 → hidden-1 ~0.917
  → hidden-2 ~0.973** (the moons untangle into linear separability); net acc **~0.96** (0.958–0.963 across 5
  seeds; last-hidden sep 0.965–0.973). The output neuron then just draws the line.
- **Width is another route (honest aside):** a single *wide* layer `[2,8,1]` also untangles moons (**~0.965**
  in one layer) — for this toy, depth is **not required**.
- **Depth gain on MNIST (10k train / 5k test, ReLU+softmax, /255, mini-batch SGD lr 0.2 batch 128, 60
  epochs):** at a **fixed unit budget**: 50u **0.9432** → 448u **wide 0.9538** → 448u **deep (256,128,64)
  0.9558** (all train ~1.0). **Width jump (50→448): +1.1 pp.** **Equal-budget depth (wide→deep): +0.2 pp — a
  near-wash** (robust: deep−wide **+0.002…+0.004** over seeds 0–2 at 60 epochs, both at train 1.0).
- **sklearn cross-check** (`MLPClassifier`, adam, `StandardScaler`): `(50,)` **0.940** → `(448,)` **0.9486** →
  `(256,128,64)` **0.9490** — *same shape*; equal-budget **+0.0004**. Confirms the by-hand story is not an
  artifact; **the NB stays pure numpy** (this number is corroboration, mentioned in one clause, not imported).
- **Gradient-check the L-layer net:** analytic backward vs central finite differences → **rel_err ~1e-8**
  (re-confirm at build) — the generalization to `L` layers is correct.
- **Foreshadow NB 3 (honest, seed-dependent):** the *deeper* all-width-2 net `[2,2,2,2,1]` **sometimes stalls**
  (seed0 0.887 vs others ~0.96) — a preview that deep nets are hard to *train* (vanishing gradients, NB 3).

## Cell-by-cell (~24 cells, 3 figures) — intuition → implementation → "Read the figure"; **SHOW**; ends with "Your turn"
1. **(md) Header** `# 02 — Depth is a representation hierarchy`. **Prerequisites** (NB 1: the net we built by
   hand — forward / backward / GD / train-test; ch 00: scaling, train/test; ch 03: linear separability — what
   a straight line can split). **What you'll do:** stack NB-1's net into many layers, *see* depth remap a
   problem into linear separability, *measure* depth's honest gain, learn why the vivid hierarchy needs a CNN.
2. **(md) Recap & the question** — NB 1 built **one** hidden layer (`2→16→3`) and already bends boundaries. So
   why go *deeper*? Name the bet: **depth** — more *layers*, not just more units (*width*).
3. **(md) The idea: a layer remaps; depth composes remaps** — a hidden layer transforms the input into a new
   representation (ReLU/tanh carve & fold space, ch 11 / NB 1). Stack layers → each remaps the *previous*
   representation → **composition**. The hope: successive remaps make the data progressively easier —
   eventually **linearly separable**. "Deep" = many layers.
4. **(code) Imports + `use_course_style()`; the `L`-layer net** — generalize NB-1's forward to a list of
   `(W, b)` + a loop (ReLU hidden, softmax/sigmoid out); **flat-0.1 init (same as NB 1)** — comment: "small,
   breaks symmetry; the principled scale is **NB 4**". Show the code.
5. **(md) It's NB-1's machinery, in a loop** — the `L`-layer forward + backward are exactly NB-1's pieces
   iterated over layers (the chain rule lives in NB 1 / 11.3; we *package* it). We'll trust it the same way:
   by checking the gradient.
6. **(code) Gradient-check the `L`-layer net** — print **rel_err ~1e-8** on a small instance. "Checked, so we
   trust it" (the ch-11 habit). Brief.
7. **(md) Let's *see* a remap** — moons (two interleaving crescents — a straight line can't separate them).
   Build an **all-width-2** net `2→2→2→1` so *every* layer's activation is 2-D and we can plot it.
8. **(code) Train the moons untangler** (by hand, flat-0.1, lr 0.5, 6000 ep); print loss ↓ and net acc ~0.96;
   print the **linear-separability probe** at input / hidden-1 / hidden-2 (**0.877 → ~0.917 → ~0.973**).
9. **(code) Fig 1 — the layer-by-layer remap** (3 panels: input / after hidden-1 / after hidden-2; points
   colored by class — the crescents untangle). Charter `CLASS_CYCLE`.
10. **(md) Read Fig 1** — input: crescents interleave (no line works, 0.877). Each layer **warps** the space.
    By hidden-2 the two classes sit on opposite sides of a line (~0.973) — the output neuron only has to draw
    it. **Depth = composing remaps until the problem is linearly separable.**
11. **(code) Fig 2 — linear separability rises with depth** (the probe accuracy at input → h1 → h2 as a short
    bar/step plot; charter colours).
12. **(md) Read Fig 2** — separability climbs layer by layer (0.877 → 0.917 → 0.973): a **quantified** view of
    the untangling. *Honest note:* the probe is a **descriptive** read on the representation (train-set), not a
    generalization claim — the honest accuracy story is next, with a held-out test set.
13. **(md) Honest aside: width is another route** — a single *wide* layer `2→8→1` also untangles moons
    (~0.965). For this toy, depth is **not required**. So does depth actually pay on a *real* problem?
14. **(md) The honest test on MNIST** — does depth help? Compare at a **fixed unit budget** so we separate
    "more units" from "more layers": 50 units; 448 units **wide**; the *same* 448 units arranged **deep**
    (256,128,64). (Recap: MNIST = 28×28 grayscale digits, flattened to 784; we use a 10k/5k subset.)
15. **(code) Train the three nets on MNIST 10k/5k** (by hand, flat-0.1, ReLU+softmax, /255, 60 ep); print test
    accs (**50u ~0.943 / 448u-wide ~0.954 / deep ~0.956**); never silence the run.
16. **(code) Fig 3 — depth vs accuracy** (3 bars: 50u / 448u wide / 448u deep; charter colours) — honest,
    modest.
17. **(md) Read Fig 3** — 50→448 units = the bigger move (~+1 pp); rearranging the *same* 448 units wide→deep
    adds only ~+0.2 pp (a **near-wash**, robust across seeds). **Capacity (units) helped; depth *per se* barely
    moved the needle on flat pixels.** (scikit-learn's own MLP shows the same shape — 0.94→0.95.)
18. **(md) Why isn't depth a blow-out here?** — flat pixels carry **no structure** for a hierarchy to exploit;
    a dense net treats pixel 1 and pixel 784 as unrelated and **discards the grid**. The **vivid** hierarchy
    (edges → parts → objects) needs **spatial structure** → that's a **convolutional network (CNN)** — named
    here, met as a horizon in **NB 10**. This is the honest version of "depth is a representation hierarchy."
19. **(md) A crack we'll fix next** — stacking *many* width-2 layers got **fragile** (a deeper net sometimes
    stalls near chance — try it in "Your turn"). Deep nets are genuinely hard to **train** — **NB 3**
    (vanishing & exploding gradients) explains exactly why, and **NB 4** (initialization) gives the first fix.
20. **(md) Your turn** — *(warm-up)* add a third width-2 layer (`2→2→2→2→1`) to the moons net — does it untangle
    *better*, or sometimes **stall**?; *(core)* change the MNIST width/depth (e.g. `(128,64,32)` vs `(224,)`)
    and re-measure the gain — does depth ever *clearly* win?; *(reach)* run the linear-sep probe **per layer**
    on the MNIST net — does separability rise with depth there too?
21. **(md) What you built** — generalized the net to `L` layers (still by hand); *saw* depth remap moons into
    linear separability; *measured* depth's honest, modest gain (and that **width did most of the lifting**);
    learned why the vivid hierarchy needs structure (CNN). The reference net (NB 1) is now a *deep* net.
22. **(md) Where this goes next** — **NB 3**: why deep nets are hard to train (vanishing & exploding gradients
    — the pivot of the chapter).
23. **(md) References** — Goodfellow, Bengio & Courville 2016 (*Deep Learning*, ch 6, MIT Press); Cybenko 1989
    (DOI 10.1007/BF02551274) / Hornik 1991 (DOI 10.1016/0893-6080(91)90009-T) — UAT existence (recap, ch 11 NB
    2); Olah 2014, "Neural Networks, Manifolds, and Topology" (colah.github.io — the untangling intuition);
    LeCun, Bottou, Bengio & Haffner 1998 (DOI 10.1109/5.726791 — CNN, the horizon). The chapter plan.
24. **(md) Previous / Next** — *Previous:* 01 — A neural network from scratch in numpy. *Next:* 03 — Vanishing
    & exploding gradients.

## `src/` & guards
- **No `src/` change** (notebook-local numpy `L`-layer net; sklearn **utilities** `make_moons` /
  `fetch_openml` / `train_test_split` / `StandardScaler` + `LogisticRegression` **as a linear-sep probe**;
  `viz.use_course_style` / `colors`; the moons-remap scatters + the MNIST bars are **inline** matplotlib). **pytest
  stays 20.** **torch NOT used** (the `deep` extra arrives at NB 7).
- **Decision baked in (Rémy validated):** the MNIST depth-gain is **by-hand numpy** (in-ethos, fast ~15 s
  total, reuses NB-1's net), **not** `MLPClassifier`; sklearn's matching number is cited in one clause as
  corroboration, not imported. **Init = flat 0.1 (NB-1's)** for *both* nets — measured to train everything,
  so **He stays in NB 4**.
- Colours only from `ml_course.colors` (no hardcoded hex); `SEED=0`; **"Read the figure" after every figure**;
  **never silence output**; banned-word scan 0; ruff/black clean; output-free.
- Build from `build_ch12_nb2.py` (source of truth in the **ephemeral** scratchpad; rebuild right before
  `git add` — kernel-drift guard).
- Exit guards: nbconvert exit 0 (3 figures), anchors reproduce (gradcheck ~1e-8; moons linsep 0.877→~0.92→~0.97
  & acc ~0.96; MNIST 50u ~0.943 / wide ~0.954 / deep ~0.956 with equal-budget near-wash), banned 0, hex clean,
  ruff clean, output-free; **two-reviewer gate** (`@ml-expert-reviewer` + `@pedagogy-reviewer`, no BLOCK) →
  **Rémy visual** → end-of-NB checklist (`gen_llms_txt`, `common_errors` +rows, `course_map` §12 → NB 2 built,
  pytest 20, STATE) → commit `feat(12_neuralnetworks): notebook 02 — depth is a representation hierarchy` →
  `git merge --ff-only` into `chapter/12_NeuralNetworks`.

## Verification (end-to-end)
1. nbconvert a scratchpad copy → exit 0, 3 figures, anchors reproduce (gradient-check ~1e-8; moons untangler
   acc ~0.96 + rising linsep 0.877→~0.97; MNIST 50u ~0.943 / wide ~0.954 / deep ~0.956; equal-budget near-wash).
2. hex + banned + ruff clean. 3. pytest 20 (no `src/` change). 4. **Rémy validated this NB plan** (no reviewer
   gate; both reviewers return on the built notebook).
