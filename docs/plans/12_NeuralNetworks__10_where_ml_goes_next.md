# NB plan — 12_NeuralNetworks / 10_where_ml_goes_next

> Status: **APPROVED by Rémy via ExitPlanMode (no edits) — 2026-07-02.** NB-plan stage = **Rémy validates
> alone** (no reviewer gate; both reviewers return on the built notebook). **NB 10 of 10 — THE COURSE FINALE.**
> The **one intentional exception** to the by-hand + executable-"Your turn" pattern (chapter plan §NB 10,
> Rémy ✅): a **reflective close** — no new mechanism, no measured anchors, no executable exercise. Mostly prose
> + **2 schematic figures** (each with a "Read the figure"). Chapter plan: `docs/plans/chapter_12_NeuralNetworks.md`
> (APPROVED, §NB 10). **No `src/` change** (pytest stays 26).

## Context
The last notebook of the last chapter — the send-off for the whole course. It does two things: **look back**
(synthesize the thirteen modules into one picture and name the earned through-lines) and **look forward** (where
machine learning goes next — CNNs, RNNs, transformers — *named, motivated by NB 9's verdict, not built*). It is
deliberately unlike every other notebook: no mechanism built by hand, no executable exercise. That is stated up
front so it reads as an **earned finale**, not a thinned notebook. The voice is warm and celebratory — and still
honest (no false praise, no overclaiming the horizons).

## Why this notebook breaks the pattern (stated, deliberate)
Every prior notebook built an idea by hand, then met the real tool, then set an executable challenge. A synthesis
cannot be "built by hand," and a reflective send-off should send the learner *out* of the notebook — into their
own problems and the next course — not into one more code cell. So NB 10's "Your turn" is a set of **reflective,
directional prompts** (which method for which data; what to read next), not code. The chapter plan approved this
as the single exception.

## What it must get right (the bars)
- **Faithful synthesis** — the twelve methods, in the order built, each with one honest line (what it represents,
  its one real strength). No method mischaracterized; no method dropped. GettingStarted is the *foundations*
  (nearest-centroid, train/test), not one of the twelve.
- **Honest horizons** — CNN / RNN / transformer are **named and motivated**, never built or benchmarked here; no
  claim about them is presented as something this course measured. The one measured bridge is NB 9's flatten
  verdict.
- **The two earned through-lines**, both from measured course results: **(1) no universal best model** (the
  digits tie, `breast_cancer` MLP-competitive, Fashion-MNIST tree-wins — all measured in-course), and **(2)
  honest evaluation is the transferable skill** (baselines, held-out data, CV, report-with-spread, the noise
  ruler, error analysis).
- **Voice** — warm, celebratory, rigorous; frames the learner's growth; **banned words** ("obviously / simply /
  trivially / just / clearly" + FR) absent; **no hardcoded hex** (charter only); no emojis.

## The 2 schematic figures (drawn, not data — deterministic, charter colours)
- **Fig 1 — the course map.** The twelve methods laid out along the pedagogical spine, grouped into the seven
  families (instance-based → probabilistic → linear → partitions → margins → ensembles → learned
  representations), each family in a charter class-colour. A single visual the learner can hold the whole course
  in. (matplotlib boxes + labels; `CLASS_CYCLE` / `COLORS`; no hex.)
- **Fig 2 — flatten vs. convolution.** The NB-9 bridge made visual: a 28×28 grid flattened into a 784-row that a
  dense layer wires all-to-all (locality lost), beside the same grid with a small filter sliding across it
  (locality + weight-sharing kept). Contrasts the two inductive biases in one picture — why images want a CNN.
  (schematic; charter colours.)

## Cell-by-cell (~17 cells, 2 schematic figures) — reflective; every figure has a "Read the figure"
1. **(md) Header** `# 10 — Where machine learning goes next, and the whole course`. This is the finale.
   **Prerequisites:** all thirteen modules — from the getting-started foundations to this chapter. **What this
   notebook is:** a look back (one picture of everything you built) and a look forward (where ML goes next),
   deliberately without new code to run.
2. **(md) A note on this notebook — and why it is different.** Say it plainly: every other notebook built a
   mechanism by hand and set a coding challenge; this one does neither, on purpose. A synthesis is not built by
   hand, and a send-off should point you *outward*. You have earned a look back.
3. **(md) Part I — what you built.** The arc in one paragraph: you started with the simplest possible classifier
   (nearest-centroid, module 0), and by the end you were building neural networks and driving PyTorch. At every
   step you built the idea by hand *first*, then met the real tool — so you own the machinery, not the imports.
4. **(code) Fig 1 — the course map.** The twelve methods along the spine, grouped by family, charter colours.
   (Setup folded here: `import` + `viz.use_course_style()`.)
5. **(md) Read Fig 1.** Walk the spine family by family, one honest line each:
   **instance-based — k-NN** (predict from the closest stored examples; no training, distance is everything);
   **probabilistic — Naive Bayes** (class probabilities from per-feature likelihoods under a bold independence
   assumption); **linear — Logistic Regression** (a weighted sum squashed to a calibrated probability; the
   weights are readable evidence); **partitions — Decision Tree** (axis-aligned if/then rules; interpretable,
   the building block of what follows); **margins — SVM** (the widest-margin boundary; kernels bend it);
   **ensembles — Random Forest** (bagging: many decorrelated trees averaged, variance down) **→ AdaBoost →
   Gradient Boosting → XGBoost → LightGBM** (boosting: add weak learners that fix the current mistakes —
   residual by residual, then regularized and made fast); **learned representations — MLP → Neural Networks**
   (hidden layers *learn* the features instead of you hand-crafting them; depth, and the discipline to train it).
6. **(md) The one idea under all of it.** Every method is a different answer to the same question — *how do you
   turn examples into a decision boundary?* Distances, probabilities, a line, rules, a margin, a committee, a
   learned representation. Naming them as variations on one question is the point of the map.
7. **(md) The through-line you earned #1 — there is no universal best model.** Not an opinion — you *measured*
   it, and honestly: the digits capstone was a three-way statistical **tie**; on `breast_cancer` the MLP was at
   least the **equal** of the trees (a within-noise gap — competitive, not a decisive win, exactly as NB 9 was
   careful to state); on Fashion-MNIST a boosted tree **won** by a real margin. Same three tools, three different
   outcomes. The right model depends on the data's shape, your constraints, and what you can honestly evaluate —
   and reporting each result *with its spread* is what let you tell a real gap from a tie. (This must stay
   consistent with NB 9's corrected framing — no "MLP wins on breast_cancer" overclaim.)
8. **(md) The through-line you earned #2 — honest evaluation is the real skill.** The reflexes that outlast any
   single method: a train/test split, baselines before celebration, cross-validation, reporting a number *with*
   its spread, a noise ruler for what counts as a real difference, and reading a confusion matrix / error gallery
   / loss curve. These transfer to every model you will ever meet, including the ones not in this course.
9. **(md) Part II — where machine learning goes next.** The bridge from NB 9: our dense network flattened the
   image and left its spatial structure on the table. That gap points straight at the architectures that own the
   modern frontier. Three, named and motivated — not built.
10. **(md) Convolutional networks (CNNs) — for grids.** The direct fix for NB 9's flattening: instead of wiring
    every pixel to every neuron, slide a small **filter** across the image so that **locality** and
    **weight-sharing** are built in. That is the right inductive bias for images (and audio spectrograms, and
    anything on a grid). Named as the next tool, motivated by the verdict you measured. (LeCun et al. 1998.)
11. **(code) Fig 2 — flatten vs. convolution.** The two inductive biases side by side: the 28×28 grid flattened
    to a row and wired all-to-all (dense — locality lost) vs. a small filter sliding over the grid (conv —
    locality + sharing kept). Charter colours.
12. **(md) Read Fig 2.** The dense network on the left cannot even *see* that two pixels are neighbours — it
    relearns that from data, if at all. The convolution on the right is built to exploit it, with far fewer
    weights. That single change is most of why vision runs on CNNs.
13. **(md) Sequences — RNNs and transformers.** Text, audio and time series are *ordered*. **RNNs** carry a
    running state along the sequence; **transformers** replace that with **attention** — each position looks
    directly at every other and weighs what matters. Attention is the engine behind today's large language
    models. Both extend the same idea you built — *learn the representation* — from vectors and grids to
    sequences. Named, not built. (Hochreiter & Schmidhuber 1997; Vaswani et al. 2017.)
14. **(md) The frontier is built on what you built.** A reassuring, honest closer to Part II: the largest modern
    models are still layers, still trained by backpropagation and gradient descent, still learning
    representations — the very things you built by hand in this chapter. The scale and the architectures are new;
    the foundations are the ones you now own. Pointers for going deeper (the Deep Learning text; the frameworks;
    reading original papers) — **not** code.
15. **(md) Where to go next — your turn (reflective).** Explicitly not a coding exercise — questions to carry
    out of the course: *(1)* pick a dataset from **your own world** and name which of the twelve methods you
    would reach for first, and why; *(2)* find a problem in your field with **spatial or sequential** structure —
    which horizon (CNN / transformer) does it call for?; *(3)* choose one method you want to understand more
    deeply and read its **original paper** (the references across the course are your map). The habit to keep:
    always start with a baseline and an honest evaluation.
16. **(md) What you built — the whole-course celebration.** The earned close: thirteen modules, each idea built
    by hand before the library, from nearest-centroid to neural networks and PyTorch — and, above all, the
    judgement to choose a model for a problem and evaluate it honestly. That judgement, not any one algorithm, is
    what you take with you. A warm, genuine send-off (no false praise — you did the work).
17. **(md) References.** Horizons: LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998), *Proc. IEEE*
    86(11):2278–2324 (DOI 10.1109/5.726791) — CNNs; Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term
    Memory. *Neural Computation* 9(8):1735–1780 (DOI 10.1162/neco.1997.9.8.1735) — RNNs/LSTM; Vaswani, A., et
    al. (2017). Attention Is All You Need. *NeurIPS*. arXiv:1706.03762 — transformers. Foundations: Goodfellow,
    I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. *Previous:* **12.9 — capstone:
    Fashion-MNIST.** *This is the end of the course.*

## `src/` & guards
- **No `src/` change** — two inline schematic figures only; `viz.use_course_style()`; colours from
  `ml_course.colors` (no hardcoded hex). **pytest stays 26.**
- **Decisions baked in (validated by Rémy):** (1) **the one intentional exception** — no by-hand mechanism, no
  executable "Your turn" (reflective prompts instead); stated in-notebook as deliberate. (2) **2 schematic
  figures** (course map; flatten-vs-conv), each with a "Read the figure". (3) **Horizons named, not built** —
  CNN/RNN/transformer motivated by NB 9's measured verdict, never benchmarked here. (4) **The two through-lines
  rest on measured course results** (no-universal-best; honest evaluation). (5) **Faithful synthesis** — twelve
  methods, GettingStarted as foundations.
- Determinism: the figures are schematic (no randomness/data); if any jitter is used for layout, seed it.
  **"Read the figure" after both figures**; banned-word scan 0; ruff/black clean; output-free.
- Build from `build_ch12_nb10.py` (source of truth in the **ephemeral** scratchpad; rebuild right before
  `git add` — kernel-drift guard).

## Verification (end-to-end, at build)
1. nbconvert a scratchpad copy → **exit 0, 2 figures**, both render cleanly (course map legible; flatten-vs-conv
   reads at a glance). 2. hex clean; **banned-word scan 0**; ruff clean; output-free. 3. **pytest 26** (no `src/`
   change). 4. **Two-reviewer gate** (`@ml-expert-reviewer` — synthesis faithful, horizons not overclaimed, the
   two through-lines correctly attributed to measured results; `@pedagogy-reviewer` — the exception is well-framed
   and earned, the reflective prompts land, voice/charter clean; no BLOCK) → **Rémy visual** → end-of-NB checklist
   (`gen_llms_txt`; `course_map` §12 → **COMPLETE (10/10)**; `common_errors` if a trap surfaced; STATE) → commit
   `feat(12_neuralnetworks): notebook 10 — where ML goes next & synthesis` → `git merge --ff-only` into
   `chapter/12`. 5. **Then the chapter closes: PR #12 (`chapter/12_NeuralNetworks → main`, `--no-ff`) — the course
   finale — on Rémy's explicit go.**
