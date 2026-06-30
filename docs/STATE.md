# STATE — where the course build is right now

> The compaction lifeline. Update this at **every** phase transition. On any session start / after a
> compaction, read this first, then `docs/WORKFLOW.md`, then the active plan. See the phase
> vocabulary in `docs/WORKFLOW.md`.

| Field | Value |
|---|---|
| Current chapter | **`12_NeuralNetworks` — in build (the course finale; 13th & final module). NB 1–5 shipped (5/10).** Earlier chapters merged to `main`: ch 11 PR #11 (`0ce9d93`), ch 10 PR #10 (`6609afb`), ch 09 PR #9 (`fe295aa`), ch 08 PR #8 (`4775fe2`), ch 07 PR #7 (`b256580`), ch 06 PR #6 (`9f18507`), ch 05 PR #5 (`b5c00f7`). **12/13 modules complete on `main`; this is the last — 10 NBs planned (PyTorch finale).** |
| Current notebook | **`06_normalization`** (NB 6 of 10) — **OPENED** (phase `notebook-plan`). The **LAST by-hand numpy NB**. 5/10 shipped. |
| Phase | **`notebook-plan`** — measuring anchors live (a complete by-hand batch-norm forward layer in numpy), then drafting the cell-by-cell plan → ExitPlanMode (Rémy alone, no reviewer gate). |
| Active branch | **`notebook/12_NeuralNetworks__06_normalization`** (off `chapter/12_NeuralNetworks` @ `db388cd`). |
| Active plan | Planning NB 6 (normalization — batch & layer norm, c04). Chapter: `docs/plans/chapter_12_NeuralNetworks.md` (APPROVED, §NB 6). Prior NBs 1–5 DONE. |
| Next concrete action | **Measure anchors live, then draft NB 6 cell-by-cell → ExitPlanMode (Rémy alone, no reviewer gate).** A **complete by-hand batch-norm forward layer** in numpy (batch mean/var → normalize → learnable **γ/β**, **with the backward through it** so the net trains end-to-end); show it on the **drifting-activation stack** (NB 3/4) AND demonstrate a **training-time effect** (a deeper net that stalled / trained slowly now trains / loss descends faster); then **name in one sentence** what the real `nn.BatchNorm` adds beyond the forward transform — **running statistics + the train/eval-mode difference** (explicitly NB 8's; no oversold from-scratch BN *trainer*). **Layer norm** named as the per-sample sibling (BN normalizes across the batch, LN across features; LN is what transformers use — ties to NB 10 horizons) — contrasted in a sentence, not a second by-hand build. Figs: activation distributions drifting vs normalized; the train-time loss effect; the BN-layer schematic. **Still pure numpy** (torch arrives NB 7 — the `deep` extra + Fashion-MNIST loader + tests land there → pytest rises from 20). On approval: write `docs/plans/12_NeuralNetworks__06_normalization.md` + commit (`docs(plan): NB 6 — normalization (approved)`), then build from `build_ch12_nb6.py` (**ephemeral** scratchpad — [[scratchpad-build-scripts-ephemeral]]). |

## Notes / blockers

- **Resolved (NB-4 cycle): ruff debt in ch 08 NB 1–3 fixed** (Rémy: "oui on s'occupe des erreurs"). The 8
  pre-existing errors (`B007` unused loop var; `E501` long lines; `B905` zip-without-strict) are cleared via
  behaviour-preserving edits **in the build scripts** (`for m`→`for _`; wrapped print / `np.array` / `ax.plot`
  / `AdaBoostClassifier(...)` lines; `zip(..., strict=False)`), notebooks rebuilt. `ruff check .` → **All
  checks passed!** (green across the repo); all three re-execute (nbconvert exit 0); the diff is the 8 lines
  only. Committed on `chapter/08_GradientBoosting`.
- **Resolved (lint debt):** Rémy chose option B — fix the notebooks. NB 01–09 made ruff-clean
  (explicit `zip(strict=False)`, unused `pandas` imports removed, long lines wrapped; all seven
  re-executed end-to-end), committed as `f84eec6`. `ruff check .` is now fully green across the repo.
- NB 01 shipped: `ml_course.datasets.load_penguins()` + vendored `penguins.csv` + tests are in place
  and reusable by later notebooks. pandas-first convention recorded in CLAUDE.md/AGENTS.md.
- Through-line reminder: Palmer penguins (binary, 2 features), nearest-centroid first classifier (NB
  05), polynomial-degree complexity dial (NB 09–10).
- Confirmed: notebooks all-English; maths re-established not presupposed; git history = preserve
  per-notebook, mark chapters (`--no-ff` chapter → main).
- **Build gotcha (editor kernel-drift):** opening a notebook in the VS Code Jupyter editor rewrites its
  kernelspec `display_name` to `ml-course (3.12.12.final.0)` — cosmetic (the kernel resolves by
  `name: "python3"`), but it drifted NB 3's committed metadata once (normalized in `c493293`). **Habit:
  rebuild each notebook from its build script right before `git add`** (after Rémy's visual validation)
  so the canonical `ml-course (3.12.12)` ships; or `git restore` the file if only the display_name drifted.

## Progress log (most recent first)

- **NB 6 (normalization: batch & layer norm — c04) OPENED.** Branch
  `notebook/12_NeuralNetworks__06_normalization` off `chapter/12_NeuralNetworks` (@ `db388cd`). Phase
  `notebook-plan`: measuring anchors live, then drafting the cell-by-cell plan. **One concept (chapter plan
  §NB 6):** **normalization** — re-center/re-scale a layer's activations to a healthy range *during* training
  (the same "control the variance with depth" lever as init [NB 4], now applied at every step). **By hand — a
  complete batch-norm forward layer** in numpy (batch mean/var → normalize → the **learnable γ/β**, with the
  **backward through it** so the net trains end-to-end); show it on the **drifting-activation stack** (NB 3/4)
  AND demonstrate a **training-time effect** (a deeper net that stalled / trained slowly now trains / loss
  descends faster). Then **name in one sentence** what the real `nn.BatchNorm` adds beyond the forward
  transform: **running statistics + the train/eval-mode difference** (explicitly NB 8's — no oversold
  from-scratch BN *trainer*). **Layer norm** named as the per-sample sibling (BN across the batch, LN across
  features; LN is what transformers use — ties to NB 10 horizons) — contrasted in a sentence, not a second
  by-hand build. **The LAST by-hand numpy NB** (NB 7 = the torch hello-world; the `deep` extra + Fashion-MNIST
  loader + tests land there → pytest rises from 20). Figs: activation distributions drifting vs normalized; the
  train-time loss effect; the BN-layer schematic. **Still pure numpy — no torch.** NB-plan = **Rémy validates
  alone** (no reviewer gate; both reviewers return on the built notebook). Refs: Ioffe & Szegedy 2015
  (BatchNorm; arXiv:1502.03167); Ba et al. 2016 (LayerNorm; arXiv:1607.06450); Goodfellow et al. 2016 ch 8.
- **NB 5 (dropout — c05) OPENED.** Branch `notebook/12_NeuralNetworks__05_dropout` off
  `chapter/12_NeuralNetworks` (@ `cc1546a`). Phase `notebook-plan`: measuring anchors live, then drafting the
  cell-by-cell plan. **One concept (chapter plan §NB 5):** **dropout** — randomly zero a fraction `p` of
  activations each training step (**inverted dropout** rescales the survivors by `1/(1−p)` so the expected
  activation is preserved) → an **implicit ensemble** that breaks co-adaptation; and **how it differs from L2 /
  early-stopping** (stochastic co-adaptation-breaking, not a weight penalty or a stop rule). By hand (numpy):
  inverted dropout preserving the expected activation (mean ~0, variance 1→2); show it **reduce overfitting**
  on a small net; contrast explicitly with `alpha` (L2). **`MLPClassifier` has NO dropout — it must be by-hand
  here; the real `nn.Dropout` layer arrives in NB 8.** Figs: the dropout mask / expected-activation
  preservation; train-vs-val with vs without dropout. **Still pure numpy — no torch** (the `deep` extra + torch
  land at NB 7). NB-plan = **Rémy validates alone** (no reviewer gate; both reviewers return on the built
  notebook). **Anchors measured live** (`measure_ch12_nb5.py` / `_nb5b.py` / `_nb5c.py`): mechanism on N(0,1)
  100k — inverted dropout preserves the mean (≈0→≈0), injects variance, E[mask]=1.0 (**p=0.5 var 1.00→1.99**;
  p=0.2→1.25; p=0.8→5.07); overfitting (50 feat/10 inf, 200 train/1000 val, `[50,256,256,2]`, He, ReLU,
  mini-batch lr0.1 batch32 1500ep) none train 1.0/val **0.752**/gap 0.248 → **dropout 0.3 val 0.780/gap 0.220**
  (beats L2 1e-2 val 0.776) / dropout 0.5 0.771; train-vs-val curves none plateau ~0.75 vs dropout ~0.77
  (modest, real; no peak-then-collapse on this data). **Honest scope: dropout gentle on a small net (~+2–3 val
  pts, ≈ L2); decisive wins are large vision nets (Srivastava 2014) — demo uses high-dim (50 feat) since the
  2-D toy effect is negligible (0.805→0.815).** **Decisions: all by-hand (MLPClassifier has no dropout);
  mechanism is the strong core, overfitting reduction real-but-modest, scope stated not hidden.** **Plan
  APPROVED by Rémy via ExitPlanMode (no edits) & persisted** (`docs/plans/12_NeuralNetworks__05_dropout.md`).
  **BUILT (19 cells: 5 code / 14 md, 2 figures; pure numpy, all by-hand) — awaiting Rémy's visual validation.**
  Reused the `L`-layer net (He init) + inverted dropout (mask the hidden activations in forward [train only];
  gate the backward gradient by the same mask; eval = no dropout). **Anchors reproduced (nbconvert exit 0, 2
  figures):** mechanism on N(0,1) p=0.5 var **1.0→1.99** mean preserved E[mask]≈1.0 (p=0.2→1.25, p=0.8→5.07);
  overfitting (50 feat/10 inf, 200 train/1000 val, `[50,256,256,2]`) none val **0.752** gap 0.248 → dropout 0.3
  val **0.780** gap 0.220 / dropout 0.5 0.771 / L2 1e-2 0.776; train-vs-val curves none ~0.75 vs dropout ~0.77.
  **Both reviewers no-BLOCK** — `@ml-expert-reviewer` **PASS** (the by-hand dropout **backward gradient-checked
  to 8.1e-08** — correct; inverted-dropout mean/variance match closed forms `1/(1−p)`; zero train/val overlap;
  no torch leak; 2 MINOR folded); `@pedagogy-reviewer` **PASS** (one-concept arc, no forward refs, 2/2
  Read-the-figure match pixels, honest scope empowering; 2 MINOR + 1 NIT, the load-bearing ones folded).
  **Folds (3, applied, markdown):** cell 14 — added the **single-split-noise caveat** on the dropout-vs-L2
  ordering (the ch-11-NB-5 lesson: both beat *none*, but which wins by a hair is within split noise — ml-expert
  MINOR-1, the most valuable); cell 7 — "averaging" → named it a **geometric mean** (Srivastava §7) + removed
  the "exactly"/"approximates" adjacency; cell 1 — "almost absurdly simple" → "almost suspiciously simple"
  (banned-family-adjacent). Guards: ruff *All checks passed* (1 E501 fixed), hex clean, **banned 0**,
  output-free; black skips `.ipynb`. **No `src/` change** (notebook-local numpy net + inverted dropout;
  `make_classification`/`train_test_split`/`StandardScaler`; `viz`/`colors`; inline matplotlib). Next: Rémy
  visual → end-of-NB checklist (rebuild [kernel-drift], `gen_llms_txt`, `common_errors` +rows, `course_map` §12
  → NB 5 built, pytest 20) → commit + ff-merge into `chapter/12`.
  **Rémy validated visually (2026-06-30). End-of-NB checklist done:** rebuilt from `build_ch12_nb5.py`
  (kernel-drift guard), `llms.txt` **92**, `course_map` §12 → **NB 1–5 built**, `common_errors` **+2 rows**
  (dropout-is-inverted-rescale-not-deletion / implicit-ensemble; dropout-gentle-here-decisive-at-scale /
  ordering-within-split-noise), **pytest 20** (no `src/` change). **No `src/` change** (notebook-local numpy net
  + inverted dropout; `make_classification`/`train_test_split`/`StandardScaler`; `viz`/`colors`; inline
  matplotlib). **COMMITTED & ff-merged into `chapter/12`.** Next: open & plan NB 6 (normalization — the LAST
  by-hand numpy NB) on Rémy's go.
- **NB 4 (initialization: He & Xavier — c03, the fix) OPENED.** Branch
  `notebook/12_NeuralNetworks__04_initialization` off `chapter/12_NeuralNetworks` (@ `71aa5e8`). Phase
  `notebook-plan`: measuring anchors live, then drafting the cell-by-cell plan. **One concept (chapter plan
  §NB 4):** the **fix** for NB 3's pathology — choose the initial weight **variance** so the per-layer factor
  stays ≈1 and signal/gradient magnitude is ~preserved across depth (the real content behind ch 11's "break
  symmetry"). **He `sqrt(2/n)` for ReLU**, **Xavier `1/sqrt(n)` for tanh** (the symmetric/linear regime);
  **sigmoid the awkward non-zero-centered case** — Xavier *slows* but does not fully preserve it (Glorot's
  larger-gain note), which itself motivates why tanh/ReLU displaced sigmoid as the default. **Do NOT claim
  "Xavier rescues sigmoid".** By hand (pure numpy, reuse NB-3's 10-layer stack): derive the variance-preserving
  idea, implement He & Xavier, re-run NB-3's per-layer RMS → He flattens ReLU's gradient/activation RMS across
  all 10 layers; plus the *training* payoff (a deep ReLU net that stalled with naive init now trains with He).
  Figs: gradient-RMS before/after He & Xavier; activation distributions (drift vs preserved). **Still pure
  numpy — no torch** (the `deep` extra + torch land at NB 7; `torch.nn.init` realizes He/Xavier in NB 8).
  NB-plan = **Rémy validates alone** (no reviewer gate; both reviewers return on the built notebook).
  **Anchors measured live** (`measure_ch12_nb4.py`): derivation `Var(Wx)=n·Var(W)·Var(x)` → Xavier `σ=1/√n` /
  He `σ=√(2/n)` (ReLU halves variance); per-layer gradient RMS **ReLU+he flat 4.6e-2→2.2e-1** (min 0.03/max
  0.22) / tanh+xavier flat 6.1e-3→2.3e-2 / **sigmoid+xavier still vanishing 3.4e-9→7.8e-2 ratio 2.3e7** (vs
  tanh+xavier ratio 3.8 — same Xavier, opposite outcome → sigmoid the awkward non-zero-centered case, NOT
  "rescued"); forward activation std **ReLU+he 0.69→0.53 preserved** vs ReLU+small 0.07→0.00 (dies); **training
  8-hidden-layer (by-hand GD, 3 seeds): ReLU small/large 0.500 → he 1.000; tanh small 0.500 → xavier 1.000.**
  **Decisions: all by-hand numpy (diagnostics + training payoff, no MLPClassifier); sigmoid awkward not rescued;
  report measured flat values (grad 0.05–0.22, fwd std 0.53–0.69), not the chapter-plan's "0.71–0.80".** **Plan
  APPROVED by Rémy via ExitPlanMode (no edits) & persisted** (`docs/plans/12_NeuralNetworks__04_initialization.md`).
  **BUILT (22 cells: 7 code / 15 md, 3 figures; pure numpy, all by-hand) — awaiting Rémy's visual validation.**
  Reused NB-3's `L`-layer net + a fan-in-scaled init (he/xavier/glorot/small/large). **Anchors reproduced
  (nbconvert exit 0, 3 figures):** derivation `Var(Wx)=n·σ²·Var(x)` → Xavier `1/√n` / He `√(2/n)`; gradient RMS
  ReLU+he flat **0.03–0.22** (0.22 = softmax output layer; hidden band tighter), tanh+xavier flat 5.7e-3–2.3e-2,
  **sigmoid+xavier still vanishing ratio 2.3e7** (vs tanh 3.8; input 8.4e-14→3.4e-9); forward std ReLU+he
  **0.69→0.53** preserved vs small 0.07→0.00; training 8-layer ReLU small/large **0.500** → he **1.000**, tanh
  small 0.500 → xavier 1.000. **Both reviewers no-BLOCK** — `@ml-expert-reviewer` **PASS** (re-ran every anchor
  bit-for-bit; confirmed the 0.22 max is the softmax layer & hidden layers sit in a tight 0.029–0.084 band;
  verified seed-robustness 0/1/2; sigmoid mechanism measured [Xavier sigmoid means ~0.5/layer vs tanh ~1e-4];
  no torch leak; 2 MINOR folded); `@pedagogy-reviewer` **PASS** (cure framing exact, derivation from scratch,
  3/3 Read-the-figure match pixels, sigmoid honesty empowering; 1 MINOR + 1 NIT folded). **Folds (4, applied):**
  (ml-expert) cell 3 — ReLU "halves the **second moment** `E[x²]`" not variance (rigor, matches the sigmoid
  argument); cell 21 exo 2 — reframed toward the gradient (mismatch often still trains on this easy net; gap
  grows with depth); (pedagogy) cell 8 — "0.05–0.22" → **0.03–0.22** (match the printout) + the 0.22-is-softmax
  note; cell 10 — **print the forward-std anchors** (0.69→0.53 / 0.07→0.00) for checkability. Guards: ruff *All
  checks passed* (1 E501 fixed), hex clean, **banned 0**, output-free; black skips `.ipynb`. **No `src/` change**
  (notebook-local numpy net; `make_circles`/`StandardScaler`; `viz`/`colors`; inline matplotlib). Next: Rémy
  visual → end-of-NB checklist (rebuild [kernel-drift], `gen_llms_txt`, `common_errors` +rows, `course_map` §12
  → NB 4 built, pytest 20) → commit + ff-merge into `chapter/12`.
  **Rémy validated visually (2026-06-30). End-of-NB checklist done:** rebuilt from `build_ch12_nb4.py`
  (kernel-drift guard), `llms.txt` **91**, `course_map` §12 → **NB 1–4 built**, `common_errors` **+2 rows**
  (init-scale-is-precise-not-arbitrary / He-Xavier-formulas + the 8-layer chance→100% payoff;
  init-is-matched-to-the-activation / sigmoid-awkward-not-rescued), **pytest 20** (no `src/` change). **No
  `src/` change** (notebook-local numpy net; `make_circles`/`StandardScaler`; `viz`/`colors`; inline
  matplotlib). **COMMITTED & ff-merged into `chapter/12`.** Next: open & plan NB 5 (dropout) on Rémy's go.
- **NB 3 (vanishing & exploding gradients — c02, the pivot) OPENED.** Branch
  `notebook/12_NeuralNetworks__03_vanishing_exploding_gradients` off `chapter/12_NeuralNetworks` (@ `199d900`).
  Phase `notebook-plan`: measuring anchors live, then drafting the cell-by-cell plan. **One concept (chapter
  plan §NB 3 — THE PIVOT):** backprop through many layers multiplies one factor per layer → the gradient signal
  **collapses (vanish)** or **blows up (explode)** — the reason naive deep nets don't train. By hand (pure
  numpy, reuse NB-2's `L`-layer net): run the backward pass through a **10-layer** stack and **measure the
  per-layer gradient RMS** (**sigmoid + small init → ~1e-16**; **ReLU + unit-Gaussian init → ~5e6**); then the
  *training* proof — a **5-layer sigmoid** net stalls at **0.500 (chance)** vs **tanh/ReLU ~1.000** on circles.
  Figs: gradient-RMS-by-layer (vanish vs explode); the flat sigmoid loss curve vs the descending ReLU one; the
  chain-of-factors schematic. **This is why c03 (init), c04 (norm), c06 (the flat-curve diagnostic) all exist.**
  **Still pure numpy — no torch** (the `deep` extra + torch land at NB 7). NB-plan = **Rémy validates alone**
  (no reviewer gate; both reviewers return on the built notebook). **Anchors measured live** (`measure_ch12_nb3.py`
  / `_nb3b.py` / `_nb3c.py` / `_nb3d.py`): mechanism by-hand (10-layer × width-16 stack on circles) — **sigmoid +
  small gradient 2.9e-2 → 8.4e-14 (vanish, ~12 orders)**, **ReLU + large(std1) ~1e3–5e3 (explode)**; forward
  activations sigmoid flat ~0.5 (saturated) / ReLU+large 0.8 → 7800; training proof `MLPClassifier` (16,)×5 adam
  — **sigmoid 0.500 loss flat at ln2 (0.6932)**, tanh/ReLU 1.000; **lbfgs depth 1 → 1.0, depth 5 → 0.478
  (chance)** = unrecoverable (depth, not optimizer; the clean contrast needs lbfgs since sigmoid+adam stalls at
  every depth — ch 11 NB 4). Exercise: sigmoid ×20 → 4.4e-24; ReLU+small → small-but-flat ~1e-7 (no explode →
  scale is the lever, NB 4). **Decisions: mechanism by-hand numpy / training proof MLPClassifier (no He
  forward-ref); only small+large init in code; report measured explode (~1e3, not the chapter-plan's order "5e6").**
  **Plan APPROVED by Rémy via ExitPlanMode (no edits) & persisted**
  (`docs/plans/12_NeuralNetworks__03_vanishing_exploding_gradients.md`). **BUILT (21 cells: 6 code / 15 md,
  3 figures) — awaiting Rémy's visual validation.** Reused NB-2's `L`-layer net + a sigmoid option + an
  init-scale knob (`small` 0.1 / `large` 1.0 — **no He/Xavier**); mechanism by-hand, training proof via
  `MLPClassifier`. **Anchors reproduced (nbconvert exit 0, 3 figures):** sigmoid+small gradient **8.38e-14
  (input) → 2.94e-2 (output)** (vanish); ReLU+large **~2e3** gradient, forward **0.8 → 7783** (explode);
  `MLPClassifier` (16,)×5 adam — sigmoid **0.500** loss flat at **ln2 (0.6932)**, tanh/ReLU **1.000**; lbfgs
  sigmoid depth **1 → 1.0 / 5 → 0.507** (unrecoverable). **Both reviewers no-BLOCK** — `@ml-expert-reviewer`
  **PASS** (re-verified every anchor live, σ'≤0.25 to machine precision, 0.25¹⁰≈9.5e-7, by-hand backward ==
  NB-1/2 chain rule, lbfgs choice *principled* not cherry-picked [adam stalls sigmoid at every depth], no
  He/Xavier leak, DOIs resolve; 1 MINOR = schematic fixed-factors, explicitly "no change required");
  `@pedagogy-reviewer` **PASS** (clean pivot framing, deferrals textbook-clean, 3/3 Read-the-figure match the
  pixels, exercises bridge to NB 4; 2 MINOR folded). **Folds (markdown only):** cell 14 — tied `ln 2` to the
  two-class circles (`ln K` with K=2; NB 1 showed `ln 3`); cell 5 — softened "annihilates/detonates" register.
  Guards: ruff *All checks passed*, hex clean, **banned 0**, output-free; black skips `.ipynb`. **No `src/`
  change** (notebook-local numpy net; `make_circles`/`StandardScaler`/`MLPClassifier`; `viz`/`colors`; inline
  matplotlib). **Rémy validated visually (2026-06-30). End-of-NB checklist done:** rebuilt from
  `build_ch12_nb3.py` (kernel-drift guard), `llms.txt` **90**, `course_map` §12 → **NB 1–3 built**,
  `common_errors` **+2 rows** (deeper-isn't-free / vanishing-exploding-mechanism + flat-loss-from-epoch-1;
  depth-not-the-optimizer / fixes-are-structural), **pytest 20** (no `src/` change). **COMMITTED & ff-merged
  into `chapter/12`.** Next: open & plan NB 4 (initialization — He & Xavier, the fix) on Rémy's go.
- **NB 2 (depth is a representation hierarchy — c01) OPENED.** Branch
  `notebook/12_NeuralNetworks__02_depth_is_a_hierarchy` off `chapter/12_NeuralNetworks` (@ `9fffa02`). Phase
  `notebook-plan`: measuring anchors live, then drafting the cell-by-cell plan. **One concept (chapter plan
  §NB 2):** *why* stack layers — successive layers **compose / remap** features; depth (not just width) is the
  bet. By hand (pure numpy): stack the NB-1 forward pass into an `L`-layer net; show what successive hidden
  layers represent on a 2-D problem (the **layer-by-layer activation remap**); show — **honestly, measured** —
  that depth helps modestly (MNIST-10k `(50,)` ~0.941 → `(256,128,64)` ~0.949). **Drop the equal-unit-budget
  claim** (the anchor compares 50 vs 448 units) → land on "depth remaps the feature space; a modest, honest
  gain"; *optional* equal-budget `(448,)` vs `(256,128,64)` near-wash check (itself the honest lesson).
  *Honesty:* the vivid edge→part→object hierarchy is a **CNN** phenomenon (c09, NB 10 horizon) — NB 2 shows
  the honest version and defers the vivid picture. Figs: layer-by-layer activation remap; depth-vs-accuracy
  (honest, modest). **Still pure numpy — no torch** (the `deep` extra + torch land at NB 7). NB-plan = **Rémy
  validates alone** (no reviewer gate; both reviewers return on the built notebook). **Anchors measured live**
  (`measure_ch12_nb2_remap.py` / `_depth.py` / `_equalbudget.py` / `_init.py`): moons `[2,2,2,1]` linsep
  **0.877→0.917→0.973**, net acc ~0.96 (5 seeds 0.958–0.963); **MNIST 10k/5k 50u 0.9432 / 448u-wide 0.9538 /
  deep(256,128,64) 0.9558** (width +1.1pp; **equal-budget depth +0.2pp — near-wash**, robust +0.002–0.004/seed
  at 60 ep, both train 1.0); sklearn cross-check 0.940/0.9486/0.9490; **flat-0.1 init (NB-1's) trains every net
  → no He forward-ref**; `[2,8,1]` width alone untangles moons (~0.965); `[2,2,2,2,1]` deeper net sometimes
  stalls (seed0 0.887 — NB-3 foreshadow). **Plan APPROVED by Rémy via ExitPlanMode (no edits) & persisted**
  (`docs/plans/12_NeuralNetworks__02_depth_is_a_hierarchy.md`). **BUILT (22 cells: 7 code / 15 md, 3 figures;
  pure numpy) — awaiting Rémy's visual validation.** Unified machinery (NB-1's softmax+CE generalized to an
  `L`-layer loop; hidden act a choice tanh/ReLU; flat-0.1 init — **no He forward-ref**). **Anchors reproduced
  (nbconvert exit 0, 3 figures):** gradient-check **1.45e-7**; moons `[2,2,2,2]` (tanh, lr0.5, 8000 ep) loss
  ln2(0.693)→0.089, acc **0.965**; one wide layer `[2,8,2]` **0.965**; MNIST 10k/5k (ReLU, flat-0.1, 60 ep)
  **50u 0.9432 / 448u-wide 0.9538 / deep(256,128,64) 0.9558** (width +1.06pp; equal-budget depth +0.20pp,
  near-wash). **Both reviewers no-BLOCK** — `@pedagogy-reviewer` **PASS** (one-concept arc, no forward refs,
  3 Read-the-figure match the pixels, exemplary equal-budget control) + 2 NIT folded (named the mini-batch loop
  in cell 5; fixed "NB-1's single sigmoid" → "a sigmoid of ch 11" in cell 7); `@ml-expert-reviewer` **REVISE→
  no-BLOCK** — verified every anchor live, **MAJOR folded:** the linsep staircase used the **default
  regularized** `LogisticRegression` probe (understated h1 ~5pp → false "rises step by step"); switched to a
  near-unregularized probe (`C=1e6`) → honest **input 0.877 → h1 0.970 → h2 0.970** ("the **first** layer does
  almost all the untangling; the second consolidates" — strengthens the anti-hype thesis, ties to the wide-layer
  + MNIST near-wash) + 2 MINOR (gradient-check message softened to "finite-difference precision"; float-order
  "about" confirmed sufficient). Removed an extra "clearly" (cell 21, ml-expert-adjacent). Guards: ruff *All
  checks passed*, hex clean, **banned 0**, output-free; black skips `.ipynb` (no `black[jupyter]`, as NB 1).
  **Rémy validated visually (2026-06-30). End-of-NB checklist done:** rebuilt from `build_ch12_nb2.py`
  (kernel-drift guard), `llms.txt` **89**, `course_map` §12 → **NB 1–2 built**, `common_errors` **+2 rows**
  (deep-net-remaps-into-linear-separability-not-a-wiggly-boundary; deeper-isn't-always-better / equal-budget
  near-wash / vivid-hierarchy-needs-a-CNN), **pytest 20** (no `src/` change). **No `src/` change** (notebook-local
  numpy `L`-layer net; sklearn utilities `make_moons`/`fetch_openml`/`train_test_split`/`StandardScaler` +
  `LogisticRegression` as a linsep probe; `viz`/`colors`; inline matplotlib). **COMMITTED & ff-merged into
  `chapter/12`.** Next: open & plan NB 3 (vanishing & exploding gradients — the pivot) on Rémy's go.
- **NB 1 (a neural network from scratch in numpy — the hello-world / reference) OPENED.** Branch
  `notebook/12_NeuralNetworks__01_numpy_hello_world` off `chapter/12_NeuralNetworks` (@ `d009480`). Phase
  `notebook-plan`: measuring anchors live (a by-hand multi-class **softmax + cross-entropy** net on
  `make_blobs` 3-class — gradient-check the softmax-CE gradient vs finite differences; train via GD), then
  drafting the cell-by-cell plan. **NB-1 pins (approved chapter plan):** the reference net is multi-class
  softmax+CE on a 2-D `make_blobs` 3-class toy; the binary chain-rule backward is **ch-11-NB-3 recap
  (black-boxed, NOT re-derived)**; the **softmax-CE gradient is the one genuinely-new derivation**; it is the
  **same architecture NB 7 re-instantiates in torch**; **show** (never assign) the full machinery — forward /
  loss / backward / short GD loop / train-test / eval / optimizer. **Pure numpy — no torch** (the `deep`
  extra + torch land at NB 7). NB-plan = **Rémy validates alone** (no reviewer gate; both reviewers return on
  the built notebook). **Plan APPROVED & persisted.** **BUILT (25 cells: 7 code / 18 md, 4 figures; pure
  numpy) & ff-merged into `chapter/12` — Rémy validated visually.** Anchors reproduced (nbconvert exit 0):
  untrained loss **1.1012 ≈ ln3**; softmax-CE **gradient-check 5.28e-8**; loss **1.10→0.31**; **train 0.867
  / test 0.893** (chance 0.333). **Both reviewers PASS, no BLOCK** — `@ml-expert` re-verified every anchor
  live and derived the softmax-CE gradient from first principles (rel-err **2.4e-10**, exact); `@pedagogy`
  found no concept gap and a clean recap/new boundary. **Folds (markdown only):** added a *Going further*
  derivation of `(p−y)` (the softmax Jacobian and the `1/p_c` cancel); re-anchored the generalization
  verdict on the **gap-to-chance** (not the noisy 75-pt train/test ordering — the ch-11-NB-5 lesson);
  bridged the schematic (5 units drawn → `W₁` is (2×16)). Guards: ruff / hex / banned 0 / output-free;
  **pytest 20** (no `src/` change); `llms.txt` **88**; `course_map` §12 → NB 1 built; `common_errors`
  **+2 rows** (softmax-CE-gradient-is-(p−y); untrained-loss-is-lnK). Next: open & plan NB 2 (depth) on
  Rémy's go.
- **CHAPTER 12 (NeuralNetworks) OPENED — the course finale (13th & final module).** Chapter 11 (MLP)
  closed via PR #11 (merge `0ce9d93`; 5 NBs, per-notebook history preserved). Branch
  `chapter/12_NeuralNetworks` created off `main` (@ `0ce9d93`). Phase `chapter-plan`: concept tour
  (`@concept-cartographer`, neural-networks scope) → draft the 5-NB chapter plan per `course_map.md`
  §12 and the per-method arc → **two-reviewer gate** (`@ml-expert-reviewer` + `@pedagogy-reviewer`, no
  BLOCK) → Rémy approves via ExitPlanMode. The deferred ch-11-close housekeeping (`course_map` §11 →
  "merged via PR #11") is folded into this opening commit (on the chapter branch, not protected
  `main`). **The key planning decision is the ch 11 / ch 12 scope boundary:** ch 11 already built the
  hidden layer, backprop by hand, ReLU, the K-class softmax head, mandatory scaling, and the loss
  curve — so ch 12 must teach the genuinely-new deep-learning material (depth as a representation
  *hierarchy*; dropout; depth-driven gradient pathologies — vanishing/exploding; modern init /
  normalization; the deep-learning-framework move) without repeating ch 11 or overloading. **`src/` WILL change** (the `deep` extra + a Fashion-MNIST loader + tests → pytest rises
  from 20 at the NB-7/NB-9 builds). **Concept tour DONE** (full map c01–c11 + dependency graph + live
  anchors + verified offline datasets + refs); the cartographer **recommended A (stay sklearn)**, but
  **Rémy chose B (PyTorch) with eyes open** for a faithful finale; structural direction = **~10 thicker
  NBs**, two hello-worlds (numpy NB 1 / torch NB 7), torch **shown not hand-coded** (skorch dropped).
  **Chapter plan drafted → two-reviewer gate (both REVISE→no-BLOCK; the ml-expert live-verified every
  sklearn anchor and measured the capstone-foil fix — Fashion-MNIST MLP 0.859 ≈ RF 0.853 < HistGB 0.874,
  a tree genuinely wins) → APPROVED by Rémy via ExitPlanMode (2026-06-29) & persisted**
  (`docs/plans/chapter_12_NeuralNetworks.md`); `course_map` §12 refined to the 10-NB decomposition.
  Next: open & plan NB 1 (numpy hello-world) on Rémy's go.
- **NB 5 (the demanding case — handwritten digits, the visualization-first capstone) OPENED.** Branch
  `notebook/11_MLP__05_digits_capstone` off `chapter/11_MLP` (@ `9b1394d`). Phase `notebook-plan`: measuring
  anchors live, then drafting the capstone. **The LAST NB of ch 11.** Scope (chapter plan §NB 5): a full,
  honest, multi-class workflow on **`load_digits`** (8×8, 1797×64, 10-class, offline, CPU-fast).
  `Pipeline(StandardScaler, MLPClassifier)`; tune lightly; held-out eval; read the **loss curve**; **confusion
  matrix + per-digit error analysis**; **seed-variance** check (single split → init variance ~0.974–0.980); a
  **fair tree-foil** — RF/HistGB on **raw** features (trees need no scaling) vs the MLP in its
  `StandardScaler` pipeline, same split/metric — *the preprocessing difference is the point*. **Honest verdict
  (measured):** MLP **fully competitive, NOT superior** (HistGB ~0.982 ≥ MLP; RF ~0.970, logistic ~0.972) —
  earns "learned features" as a *conceptual* virtue (ch 12 bridge), but trees need no scaling + stay
  interpretable; loss **non-convex** (a minimum, not *the*); **seed-sensitive**; UAT = existence. The genuine
  MLP accuracy win lives **elsewhere** (`breast_cancer` 0.9754 > RF 0.9649 / HistGB 0.9701 — the measured
  "MLP can win" fact). Arc (**≥6 figures, ~28–30 cells, visualization-first**): look at the digits → baseline
  → scaled MLP + loss curve → held-out metrics (confusion, per-class) → seed-variance → fair tree foil +
  honest limit → error gallery (which digits confuse). No `src/` change expected (reuse `viz`/`colors`;
  `load_digits`/`load_breast_cancer`; `MLPClassifier` + `Pipeline` + `StandardScaler`; RF/HistGB; pytest 20).
  NB-plan = **Rémy validates alone** (no reviewer gate; both reviewers return on the built notebook).
  **After it ships, close chapter 11 via PR `chapter/11_MLP → main` (`--no-ff`) on Rémy's explicit go.**
  **Plan APPROVED by Rémy via ExitPlanMode (2026-06-29) & persisted** (`docs/plans/11_MLP__05_digits_capstone.md`),
  with **two measured refinements approved** — (1) the digits cross-method is a **statistical TIE** (5-fold CV:
  MLP 0.9772±0.007 ≈ RF 0.9766±0.004 ≈ HistGB 0.9733±0.006), NOT the chapter-plan "HistGB ~0.982 ≥ MLP, RF
  ~0.970" (a single sealed split gave RF 0.984/MLP 0.978/HistGB 0.973 — split-luck); (2) the fair foil uses
  **5-fold CV** (not the noisy single split) while the MLP diagnostics stay on the sealed split. Anchors
  measured live (`measure_ch11_nb5.py`): baselines dummy 0.102 / logistic 0.969 / tree 0.856; scaled MLP(100,)
  sealed **0.9778** / n_iter 145 / loss 2.52→0.005; seed-variance (10 seeds) **0.971–0.982** mean 0.977;
  confusion **10 errors/450** (9→3, 8→3/1…); GridSearchCV picks the default (tuned==default 0.9778);
  breast_cancer CV MLP **0.9754** > RF 0.9631 / HistGB 0.9648. Next: build NB 5 from `build_ch11_nb5.py`.
  **BUILT (28 cells: 9 code / 19 md, 6 figures) & ff-merged into `chapter/11_MLP` — Rémy validated visually.
  CHAPTER 11 COMPLETE on the branch (5/5).** Figures: digit gallery; loss curve; confusion matrix; error
  gallery (all 10 mistakes, true→pred); seed-variance strip; 2-panel cross-method CV (digits tie /
  breast_cancer MLP-leads). **Anchors reproduced by nbconvert (exit 0, 6 figures):** baselines dummy 0.102 /
  logistic 0.969 / tree 0.856; scaled MLP(100,) sealed **0.9778** / n_iter 145 / loss 2.52→0.005; **10
  errors/450**; seed-variance 0.971–0.982 (mean 0.977, std 0.003); GridSearchCV→default 0.9778; **digits CV
  TIE** MLP 0.9772±0.007 ≈ RF 0.9766±0.004 ≈ HistGB 0.9733±0.006 (logistic 0.9694); **breast_cancer CV** MLP
  **0.9754** > HistGB 0.9701 > RF 0.9578 (MLP leads — shuffled `StratifiedKFold`). **Both reviewers PASS — no
  BLOCK** — `@ml-expert-reviewer` PASS (reproduced every anchor; ran unshown **paired t-tests** confirming the
  digits tie p≈0.49/0.92 and breast_cancer "a touch ahead" p=0.53; leakage-free paired CV; CV-over-single-split
  justified) + 1 MINOR / 1 NIT; `@pedagogy-reviewer` PASS (complete capstone arc; visualization-first backbone;
  empowering honest verdict; chapter synthesis) + 2 NIT. **Folded:** "8,000"→**7,400** weights (exact); dropped
  bare "clearly" before "ahead of RF" (±std bars touch). **Interruption recovery:** the prior process exited
  mid-gate; the **scratchpad build script was lost** (ephemeral `/private/tmp`), so it was **reconstructed from
  history and diff-verified** against the on-disk notebook (only the intended comment differed) — see
  [[scratchpad-build-scripts-ephemeral]]. Guards: 0 banned, hex clean, ruff clean (2 E501 fixed), output-free,
  nbconvert exit 0 (6 figures), pytest 20; no `ConvergenceWarning` (max_iter budgets sufficient). **No `src/`
  change** (reuse `viz.plot_confusion_matrix`/`colors`; `load_digits`/`load_breast_cancer`; `MLPClassifier` +
  `make_pipeline` + `StandardScaler`; RF/HistGB/logistic/dummy/tree; `GridSearchCV`/`cross_val_score`).
  End-of-NB checklist: rebuilt from `build_ch11_nb5.py`, `llms.txt` **86**, `course_map` §11 → **COMPLETE on
  branch**, `common_errors` **+3 MLP rows** (highest-accuracy-misleads / compare-on-CV-report-spread;
  no-universal-best / data-dependent / UAT-existence; accuracy-hides-error-structure-and-seed-wobble). **Last
  NB — chapter complete on the branch.** Next: close via PR `chapter/11_MLP → main` (`--no-ff`) on Rémy's
  explicit go.
- **NB 4 (the estimator `MLPClassifier`/`MLPRegressor` & its parameters — integrative) OPENED.** Branch
  `notebook/11_MLP__04_estimator_and_parameters` off `chapter/11_MLP` (@ `9ee1f9e`). Phase `notebook-plan`:
  measuring anchors live, then drafting the cell-by-cell plan. **Integrative (chapter plan §NB 4):** each knob
  from the concept that owns it — `hidden_layer_sizes` (depth×width = capacity); **`activation` — why ReLU is
  the default (no saturation); the measured sigmoid+adam stall from NB 2/3 is the concrete motivation**
  (depth-driven vanishing gradient deferred to ch 12); `solver` (sgd/adam/lbfgs — **Adam named** as "an
  adaptive-step upgrade of the GD you built"); `alpha` (L2); `learning_rate_init`;
  `early_stopping`+`validation_fraction`; `max_iter`; **`batch_size` — epoch / mini-batch / iteration
  introduced as vocabulary** (their genuine home; the loss-curve x-axis). **The K-class softmax output head —
  its explicit home** (turn the single sigmoid output into K softmax units; formula recap from ch 03 NB 5; one
  schematic + Read-the-figure). **Scaling mandatory**; the **loss curve** (`loss_curve_`) as the convergence
  diagnostic (reading recapped from ch 03 NB 4 fig (c)). Honest spine: unscaled-vs-scaled on a **synthetic
  2-feature mismatched-scale problem** (digits too homogeneous to break reliably — folded ml-expert MAJOR); a
  capacity sweep (width/depth) reading boundary + loss curve; `alpha` up/down; defaults-vs-`GridSearchCV` →
  **one sealed test**. `ConvergenceWarning` stays **visible**; `verbose=True` where per-iteration loss aids the
  lesson — never `verbose=False`-as-suppression. ~4 figures. No `src/` change expected (reuse `viz`/`colors`;
  `MLPClassifier` + `Pipeline` + `StandardScaler`; `make_classification`/`make_blobs`/`make_circles`/
  `make_moons`; `GridSearchCV`; pytest 20). NB-plan = **Rémy validates alone** (no reviewer gate; both
  reviewers return on the built notebook). **Plan APPROVED by Rémy via ExitPlanMode (2026-06-29) & persisted**
  (`docs/plans/11_MLP__04_estimator_and_parameters.md`), with the **figure-set swap approved** — 4 figures =
  ReLU-reveal · capacity-boundary · scaling-loss-curves · softmax-schematic (the chapter-plan alpha-path
  figure replaced by the activation/ReLU-reveal figure; alpha/lr/solver/batch/early-stopping/tuning become
  printed tables). Anchors measured live (`measure_ch11_nb4.py` + `_nb4b.py`): logistic+adam **0.373 stall**
  (loss≈ln2) vs relu **1.0** on circles (logistic+lbfgs 1.0 — the sigmoid+adam combo stalls); solver adam
  999it / sgd 2000it / **lbfgs 20it no loss_curve_**; **`len(loss_curve_)==n_iter_==epochs`**; capacity
  (1,)0.85 vs (50,)0.97; alpha sweet-spot **0.1→test 0.987** / 10→0.813 (underfit); early_stopping **n_iter 12
  vs 180** (costs acc here); scaling unscaled loss 7.2→0.116/test 0.94 vs scaled 0.71→0.073/test 0.99; softmax
  **n_outputs_ 3 / out_activation_ softmax / rows sum to 1**; defaults≈tuned (within noise). Next: build NB 4
  from `build_ch11_nb4.py`.
  **BUILT (27 cells: 9 code / 18 md, 4 figures) & ff-merged into `chapter/11_MLP` — Rémy validated visually.**
  Figures: the ReLU-reveal (sigmoid loss flat at ln2 vs ReLU descending); capacity→boundary (1 unit vs 50);
  scaling unscaled-vs-scaled loss curves; the softmax-head schematic. **Anchors reproduced by nbconvert (exit
  0, 4 figures):** logistic+adam **0.373 stall** (loss 0.693≈ln2) vs relu **1.0**; solver adam 999ep / sgd
  2000 / lbfgs 20ep **no loss_curve_**; `len(loss_curve_)==n_iter_==epochs`; capacity (1,) **0.853** /(50,)
  0.973; alpha sweet-spot 0.1→**0.987** / 10→**0.813** (underfit); early_stopping **12 vs 180 ep** (0.87 vs
  0.97); scaling unscaled 0.94 vs scaled 0.99; softmax n_outputs_ 3 / rows sum to 1; **tuned 0.987 vs default
  0.973**. **Both reviewers PASS-after-fold — no BLOCK.** `@ml-expert-reviewer` REVISE→folded **1 MAJOR** — my
  build-time "raw moons" refinement backfired on the capacity cell (the (1,)→0.50 was a **seed-fragile
  *unscaled* optimization stall**, not the representational ceiling ~0.85); **reverted to the approved plan's
  *scaled* moons** → (1,) **0.853** (honest single-fold underfit, robust) + 2 NIT (sklearn `n_iter_`=epochs
  clause; softmax shown to 4 decimals). `@pedagogy-reviewer` REVISE→folded **1 MAJOR** — the visible
  `ConvergenceWarning` was never *named* → added a cell-6 paragraph (it's information: a net hit `max_iter`
  before the loss flattened; raise `max_iter`, never hide it) + 1 MINOR (0.50/0.51, dissolved by the scaled
  revert). **Happy side effect:** the scaled revert restored alpha's sweet-spot story (0.1→0.987), per the
  plan. Guards: 0 banned, hex clean, ruff clean (2 E501 fixed), output-free, nbconvert exit 0 (4 figures),
  pytest 20. `ConvergenceWarning`s left **visible** (never silenced; now named). **No `src/` change** (reuse
  `viz`/`colors`; `MLPClassifier` + `make_pipeline` + `StandardScaler` + `GridSearchCV`; the four `make_*`
  generators). End-of-NB checklist: rebuilt from `build_ch11_nb4.py`, `llms.txt` **85**, `course_map` §11 → NB
  1–4 built, `common_errors` **+4 MLP rows** (ReLU-vs-sigmoid-saturation; scaling-mandatory; capacity-enough-
  then-regularize; early_stopping/GridSearchCV-not-free + sealed-test). Next: open & plan NB 5 (the digits
  capstone) — last NB, then close ch 11 via PR → main.
- **NB 3 (how a network learns — backpropagation, the chain rule) OPENED.** Branch
  `notebook/11_MLP__03_backpropagation` off `chapter/11_MLP` (@ `0cfc642`). Phase `notebook-plan`: measuring
  anchors live, then drafting the cell-by-cell plan. **One concept (chapter plan §NB 3):** backprop = the
  **chain rule applied layer by layer** — the forward pass **caches** activations, the backward pass carries
  the error back (`d_out`; `dH = d_out @ W2.T * H·(1−H)` for a sigmoid hidden layer; then `dW1`, `dW2`); the
  optimizer is **ch 03's GD loop, now multi-layer**; **init must not be zeros** (symmetry breaking). By hand
  on a `2-H-1` net: forward+backward, **gradient-check vs finite differences (rel_err ~2e-9)**, run full-batch
  GD, **demonstrate the zeros-init collapse**. The general L-layer algorithm as a **picture**, not a
  multi-index grind. Anchor: by-hand **2-4-1** sigmoid net trained by GD → **train acc 1.0 on circles**,
  matching `MLPClassifier((4,),'logistic','lbfgs')`; zeros-init — full zeros → all gradients **0** (frozen),
  W1=0/W2≠0 → identical-nonzero columns, random init breaks it. ~3 figures (forward-cache → backward-error
  flow; loss falling, by-hand vs sklearn overlaid; zeros vs random init). No `src/` change expected
  (notebook-local numpy by-hand net + `MLPClassifier`; `make_circles`; `viz.plot_decision_boundary`; pytest
  20). NB-plan = **Rémy validates alone** (no reviewer gate; both reviewers return on the built notebook).
  **Plan APPROVED by Rémy via ExitPlanMode (2026-06-29) & persisted** (`docs/plans/11_MLP__03_backpropagation.md`),
  with **two measured refinements approved** — (1) **parity is init-dependent**, not the chapter-plan
  "1.0==1.0": by-hand 2-4-1 GD reaches train/test **1.0** robustly (seeds 0–4), while
  `MLPClassifier((4,),'logistic','lbfgs')` lands **0.85–1.0 / 0.76–1.0** (rs=0 **0.98/0.95**) — the
  non-convexity the chapter flags; (2) the **symmetry sub-claim corrected** — W1=0/W2≠0 actually *breaks*
  symmetry (col-spread 0→3.37, acc 1.0), so the frozen/stuck demos are **full-zeros** (`|grad|=0`, loss
  ln2=0.6931, acc 0.50) + **fully-symmetric units** (identical W1 cols + W2 → col-spread 0→0, acc 0.68) vs
  **random** (col-spread 0.47→3.66, acc 1.0). Anchors measured live (`measure_ch11_nb3.py` + `_nb3b.py`):
  grad-check rel_err **6.0e-10**; sigmoid + adam/sgd **stall 0.48/0.50** → parity pins `lbfgs`.
  **BUILT (22 cells: 7 code / 15 md, 3 figures) & ff-merged into `chapter/11_MLP` — Rémy validated visually.**
  Figures: the backprop schematic (forward sky / backward rose); the GD loss curve + by-hand boundary with
  the lbfgs boundary overlaid (dashed, same ring); the symmetry experiment (4 weight vectors identical-on-one-
  point vs distinct; 3 loss curves). **Anchors reproduced by nbconvert (exit 0):** grad-check rel_err
  **2.51e-10** (seed=0; the plan's 6.0e-10 was seed=1 — same order; the prose commits only to "order of
  1e-10"); by-hand 2-4-1 GD **train/test 1.00**, final loss 0.0035; parity `MLPClassifier((4,),'logistic',
  'lbfgs')` **0.98/0.95**; **symmetry** — full-zeros `|grad|=0`/loss ln2/acc 0.50/spread 0→0, identical-units
  acc 0.68/spread 0→0, random acc 1.00/spread **0.342→5.269**. **Both reviewers PASS — no BLOCK** —
  `@ml-expert-reviewer` PASS (re-derived the backward pass to machine precision; confirmed the `1/n`↔mean-loss
  consistency, the gradient-check floor, the non-convexity attribution [adam 0.477/sgd 0.500 measured], the
  symmetry mechanism, the L-layer recursion, all DOIs; **1 MINOR folded** — the "every gradient exactly zero"
  claim scoped to *balanced* data, since on imbalanced data `db2≠0`); `@pedagogy-reviewer` PASS (progression on
  11.1/11.2/ch 03; epoch/mini-batch correctly deferred to NB 4; charter clean; every figure's "Read the
  figure" matches the pixels; exercises well-tiered; **1 MINOR + 1 NIT folded** — bridge `δ`↔`dH` in cell 19,
  "near machine precision" in cell 8). Guards: 0 banned, hex clean, ruff clean, output-free, nbconvert exit 0
  (3 figures), pytest **20**. **No `src/` change** (notebook-local numpy by-hand net + `MLPClassifier`;
  `make_circles`; `viz.plot_decision_boundary`). End-of-NB checklist: rebuilt from `build_ch11_nb3.py`
  (kernel-drift guard), `llms.txt` **84**, `course_map` §11 → NB 1–3 built, `common_errors` **+3 MLP rows**
  (backprop-is-the-gradient-not-the-optimizer; init-must-break-symmetry; non-convex-so-init-dependent). Next:
  open & plan NB 4 (the estimator `MLPClassifier` & its parameters).
- **NB 2 (why one neuron is not enough — the hidden layer) OPENED.** Branch
  `notebook/11_MLP__02_why_one_neuron_is_not_enough` off `chapter/11_MLP` (@ `c46a428`). Phase
  `notebook-plan`: measuring anchors live, then drafting the cell-by-cell plan. **One concept (chapter plan
  §NB 2):** a single neuron draws only a **straight** boundary; a **hidden layer + a non-linearity** carves
  curved ones; **why the non-linearity is essential** — linear∘linear = `Wx` collapses. Seat the
  universal-approximation *intuition* (Cybenko/Hornik, existence stated not proved). By hand: the `2→H→1`
  forward pass (numpy matrices); the linear-collapse (identity-MLP == one neuron); **hand-set** the classic
  ReLU **XOR** net (Goodfellow §6.1: W1=[[1,1],[1,1]], b1=[0,−1], w2=[1,−2]) → computes XOR exactly with
  *no training* (finding the weights is NB 3); an opaque `lbfgs`-fitted MLP for the circles **curved
  boundary**. Anchors to pin: XOR logistic 0.50 vs MLP 1.0; circles `identity` ~0.528 (chance, collapse) vs
  `tanh`/`relu` ~0.99. ~3 figures. Helpers: `viz.plot_decision_boundary`, `make_circles`/`make_moons`
  (sklearn). No `src/` change expected. **Plan APPROVED by Rémy via ExitPlanMode (2026-06-29) &
  persisted** (`docs/plans/11_MLP__02_why_one_neuron_is_not_enough.md`). **BUILT (21 cells: 6 code / 15 md,
  3 figures) & ff-merged into `chapter/11_MLP` — Rémy validated visually.** Anchors reproduced (nbconvert
  exit 0): XOR logistic 0.50 / hand-set ReLU net `[0,1,1,0]`; circles logistic 0.41 == identity-MLP 0.41;
  linear collapse `2.22e-16`; non-linear MLP 1.0 (XOR & circles). **Both reviewers PASS — no BLOCK** —
  ml-expert PASS (re-derived the Goodfellow XOR net 3 independent ways; collapse `2e-16`; UAT
  existence-framing exemplary; citations correct), pedagogy PASS (one-concept arc; training correctly
  deferred to NB 3; voice/exercises clean); **2 MINOR folded** (tanh refresher in the recap; a clause noting
  0.41 is within the coin-flip band on 100 points). Guards: 0 banned, hex clean, ruff clean (after an E501
  fix), output-free, pytest 20. **No `src/` change.** End-of-NB checklist: rebuilt from `build_ch11_nb2.py`,
  `llms.txt` **83**, `course_map` §11 → NB 1–2 built, `common_errors` **+2 MLP rows** (depth needs a
  non-linearity / the linear collapse; can-represent ≠ will-learn). Next: open & plan NB 3 (backpropagation).
- **NB 1 (the artificial neuron == the logistic unit you already built) OPENED.** Branch
  `notebook/11_MLP__01_the_neuron_is_logistic` off `chapter/11_MLP` (@ `c6ea3be`). Phase `notebook-plan`:
  measuring anchors live, then drafting the cell-by-cell plan. **One concept (chapter plan §NB 1):** the
  artificial neuron = weighted sum + bias + activation `a=φ(w·x+b)`; **a single sigmoid neuron is exactly
  logistic regression** (the ch 03 bridge — sigmoid / `z=w·x+b` / log-loss / GD all already built in ch 03);
  the activation is a *pluggable* φ — sigmoid (recap), tanh, ReLU. By hand: re-frame σ and `w·x+b` as "a
  neuron"; plot tanh/ReLU from scratch; one-unit forward pass; show `MLPClassifier(hidden_layer_sizes=())`
  reproduces the ch 03 logistic boundary. Anchor: empty-hidden MLP test == logistic (measure ~0.917 on
  `make_moons`) + the by-hand sigmoid neuron with logistic's `(w,b)` == `predict_proba` to machine
  precision. ~3 figures (neuron diagram; the three activations; empty-hidden MLP boundary == logistic).
  Helpers: `viz.plot_decision_boundary` (exists), `make_moons` (sklearn, called directly — no datasets
  loader). No `src/` change expected (notebook-local numpy + `MLPClassifier`/`LogisticRegression`; pytest
  20). **Plan APPROVED by Rémy via ExitPlanMode (2026-06-29) & persisted**
  (`docs/plans/11_MLP__01_the_neuron_is_logistic.md`). **BUILT (20 cells: 6 code / 14 md, 3 figures) &
  ff-merged into `chapter/11_MLP` — Rémy validated visually.** Anchors reproduced by nbconvert (exit 0):
  by-hand σ-neuron == `predict_proba` **0.00e+00**; logistic **0.9300** == MLP(()) **0.9300**. **Both
  reviewers no BLOCK** — ml-expert REVISE→folded 1 MAJOR (cell 16: the logistic↔MLP weight gap is the
  **default L2 penalty** `C=1` vs `alpha=1e-4`, *not* the optimizer — both `lbfgs`; matching it makes the
  weights coincide) + MINOR/NIT (ReLU "simplest non-linearity / single kink"; tanh-ReLU forward-motivation;
  drop "different optimizers"); pedagogy PASS (arc complete; *why*-non-linearity correctly teased to NB 2;
  voice/exercises clean). Guards: 0 banned, hex clean, ruff clean, output-free, nbconvert exit 0 (3
  figures), pytest 20. **No `src/` change** (notebook-local numpy + `MLPClassifier`/`LogisticRegression`;
  `viz.plot_decision_boundary`). End-of-NB checklist: rebuilt from `build_ch11_nb1.py` (kernel-drift guard),
  `llms.txt` **82**, `course_map` §11 → NB 1 built, `common_errors` **+2 MLP rows** (neuron==logistic;
  weight-gap-is-L2-penalty-not-optimizer). Next: open & plan NB 2 (the hidden layer).
- **CHAPTER 11 (MLP) plan APPROVED & persisted.** Phase `chapter-plan-approved`. Concept tour
  (`@concept-cartographer`) → draft → **two-reviewer gate** on the draft: `@ml-expert-reviewer` **REVISE →
  PASS-after-fold** (science verified to machine precision — the bridge `MLPClassifier(hidden_layer_sizes=())`
  == logistic 0.9167; backprop `dH=d_out@W2.T·H(1−H)` rel_err 2e-9; the 2-4-1 parity; non-convexity;
  UAT-as-existence; the `breast_cancer` win; DOIs; **one BLOCK folded** — the false "sigmoid/tanh/relu all
  ~0.99 on circles": sigmoid+`adam` actually stalls near chance, so the by-hand build is
  sigmoid+full-batch-GD/`lbfgs`, library sigmoid demos pin `lbfgs`, and the stall is reframed as the measured
  ReLU motivation; **two MAJORs folded** — the wild-scale scaling demo moved to a robust synthetic 2-feature
  problem, the digits capstone reframed "shines"→"fully competitive" with the foil that matches/beats it
  [HistGB ~0.982 ≥ MLP]); `@pedagogy-reviewer` **REVISE → no-BLOCK** (bridge load-bearing; 5-NB split
  justified; voice/charter clean; **two MAJORs folded** — the K-class softmax output head given an explicit
  home in NB 4, the prereq table's mini-batch/epoch framing reconciled [epoch absent in ch 03] with a home
  in NB 4). **5 notebooks:** (1) the neuron == the logistic unit · (2) the hidden layer + why non-linearity ·
  (3) backprop (chain rule) + init/symmetry · (4) `MLPClassifier` & params (+ the softmax head, epoch/batch,
  ReLU revealed) · (5) the digits capstone. Plan **APPROVED by Rémy via ExitPlanMode (2026-06-29) &
  persisted** (`docs/plans/chapter_11_MLP.md`); `course_map` §11 refined to the 5-NB decomposition. No `src/`
  change anticipated (reuse `viz`/`colors`/`datasets`; `MLPClassifier`; pytest 20). Next: open & plan NB 1 on
  Rémy's "go".
- **CHAPTER 11 (MLP) opened; chapter 10 (LightGBM) closed via PR #10** (merge commit `6609afb`,
  `gh pr merge --merge`; 5 NBs, per-notebook history preserved; remote
  `Ramdam17/QuickIntroToMachineLearning`). Branch `chapter/11_MLP` created off `main` (synced @
  `6609afb`). Phase `chapter-plan`: concept tour dispatched (`@concept-cartographer`, MLP scope),
  then drafting the chapter plan in plan mode per `course_map.md` §11 and the per-method arc, then the
  two-reviewer gate (`@ml-expert-reviewer` + `@pedagogy-reviewer`, no BLOCK), then Rémy approves via
  ExitPlanMode. The deferred ch-10-close housekeeping (`course_map` §10 → "merged via PR #10") is
  folded into this opening commit (on the chapter branch, not protected `main`). **MLP = the first
  method beyond trees (learned representations); it builds directly on ch 03 (the single sigmoid
  neuron IS logistic regression — sigmoid, log-odds, log-loss, gradient descent built by hand there)
  and re-uses ch 00 (scaling, over/underfitting, learning curves) — re-establish, never presuppose.**
  Chapter 12 (NeuralNetworks) follows — **the ch 11 / ch 12 scope boundary is the key planning
  decision** (ch 11 self-contained but not overloaded; leave real material for ch 12). No `src/`
  change anticipated yet (reuse `viz`/`colors`/`datasets`; pandas-first; pytest 20). Next: ingest the
  concept map → draft + gate the chapter 11 plan.
- **NB 5 (the demanding case — MiniBooNE, the visualization-first capstone) OPENED.** Branch
  `notebook/10_LightGBM__05_miniboone` off `chapter/10_LightGBM` (@ `1064a6c`). Phase `notebook-plan`:
  dataset verified + measuring anchors live, then drafting the capstone. **Dataset verified (fetch_openml
  MiniBooNE v1):** 130064×50, all float64, **0 NaN**, binary `signal` **72/28** (False 93565 / True 36499)
  — loads in <1s. The **last NB of ch 10**. Scope (chapter plan §NB 5): a larger tabular problem where
  speed can matter. **Matched-capacity comparison (pre-committed convention):** LightGBM / XGBoost-hist /
  HistGBR under ONE convention — num_leaves (= HistGBR `max_leaf_nodes`), leaf-wise, depth unbounded —
  report **fit time AND score**, reconcile in prose vs the spine's unmatched default-vs-default number;
  then **dial synthetic `n` up** (`make_classification` 300k–500k) to find where LightGBM crosses ahead.
  GOSS on/off as **efficiency** (accuracy vs fraction), not a flat speed bar. **Honesty axis:** speed
  measured & conditional, the three boosters close on accuracy, the winner depends on data/shape — "no
  universal best." Arc (≥6 figures, ~28–30 cells): look → baselines → tuned LightGBM + early stopping →
  held-out PR-AUC/threshold (ch 00) → matched speed/accuracy + dial-`n` crossover → error analysis →
  cross-method → importances (split vs gain vs **permutation**). Carry NB 4's output hygiene
  (`LGBM_VERBOSE` switch + named DataFrame). No `src/` change expected (reuse `viz`; `fetch_openml`;
  `LGBMClassifier`/`XGBClassifier`/`HistGradientBoostingClassifier`; pytest 20). **Plan APPROVED by Rémy
  (via ExitPlanMode, 2026-06-28) & persisted** (`docs/plans/10_LightGBM__05_miniboone.md`); ~30 cells / 7
  figures. **Measured refinement (Rémy signed off):** the chapter-plan "dial-`n` crossover" doesn't
  materialize — at **matched capacity** LightGBM is fastest at every `n` (50k→800k), all 3 boosters tie on
  PR-AUC (within 0.002); **default-vs-default** XGBoost-depth6 is faster at every `n` up to 4M but the gap
  narrows (2.4×@50k → 1.11×@4M), overtaking only beyond ~4M. The winner is set by the **convention (tree
  shape), slowly by scale** — "no universal best." Anchors (`measure_ch10_nb5.py`): MiniBooNE 130064×50/0
  NaN/72-28; baselines logistic 0.862 / RF 0.945; matched LightGBM 2.37s/0.9573 fastest / XGB-hist
  2.97s/0.9586 / HistGBR 3.23s/0.9569; default XGB 1.04s/0.9588 vs LightGBM 2.43s/0.9573; tuned early-stop
  @495 PR-AUC 0.9581, val-threshold 0.455 → test F1 0.9046; **GOSS 0.9579 > uniform 0.9556** (NB-2 edge
  appears); importances gain vs permutation top-1 differs (PID_0 vs PID_12). Carry NB 4's hygiene
  (`LGBM_VERBOSE` switch + named DataFrame); live dial-`n` ≤ ~600k, cite offline 1M–4M in prose.
  **BUILT (32 cells: 12 code / 20 md, 7 figures) & MERGED to `chapter/10_LightGBM` — Rémy validated
  visually. CHAPTER 10 COMPLETE on the branch (5/5).** **Anchors reproduced (nbconvert):** MiniBooNE
  130064×50 / 0 NaN / 468 -999 sentinels (0.36%) / 72-28; baselines logistic 0.862 / RF 0.945;
  **truly-matched** (L2 off all three, HistGBR early_stopping=False, 300 trees) LightGBM 2.33s/0.9573
  fastest / XGB-hist 2.99s/0.9578 / HistGBR 3.97s/0.9577 (**band 0.0005**); default XGB 1.04s/0.9588 vs
  LightGBM 2.40s/0.9573; dial-`n` matched LightGBM faster at all, default XGBoost faster at all (gap
  narrows, offline 1.11×@4M); **GOSS 0.9579 > uniform 0.9556**; tuned early-stop @495 PR-AUC 0.9581,
  val-threshold 0.455 → test F1 0.9046; gain top PID_0 vs permutation top PID_12. **Reviewers both PASS,
  no BLOCK** — ml-expert **REVISE→folded 2 MAJOR**: (1) "one convention/300 trees" was inaccurate (HistGBR
  auto-stopped ~270 + internal val; L2 posture differed) → made it **truly-matched** so the band tightened
  to 0.0005 and the thesis holds harder; (2) timings unpinned → **threading caveat** added (ordering is the
  portable lesson). + MINOR (statistically-a-tie → "within 0.001"; offline 4M marked synthetic/beyond-plot/
  reproducible; GOSS zoomed-axis note); pedagogy PASS. **Rémy-caught data-quality fix:** the PCA "look" was
  bizarre — diagnosed **−999 sentinels + a 16M outlier** wrecking StandardScaler+PCA (trees scale-invariant
  → the rest of the NB valid); fixed by **naming the sentinels** (dataset note + count; trees handle them
  natively — ch 09 NB 3 callback) and **setting those rows aside for the PCA projection only** + clipped
  axes → readable (signal left / background right / overlap). Guards: 0 banned, hex clean, ruff clean,
  output-free, nbconvert exit 0 (7 figures); **0 noise** (`[LightGBM] [Info]` 0 via `LGBM_VERBOSE`,
  feature-name warning 0 via named DataFrame, No-further-splits 0). **No `src/` change** (reuse `viz`;
  `fetch_openml`; the 4 estimators + `permutation_importance`; pytest **20**). End-of-NB checklist done:
  rebuilt from `build_ch10_nb5.py`, `llms.txt` **80**, `course_map` §10 → **COMPLETE**, `common_errors`
  **+2 rows** (library-X-faster = no-universal-best/convention; no-NaN≠clean / -999 sentinels handled
  natively but wreck PCA/linear). **Last NB — chapter 10 complete on the branch.** Next: close via PR
  `chapter/10_LightGBM → main` (`--no-ff`) on Rémy's explicit go.
- **NB 4 (the estimator `LGBMClassifier`/`LGBMRegressor` & its parameters — integrative) OPENED.** Branch
  `notebook/10_LightGBM__04_estimator_and_parameters` off `chapter/10_LightGBM` (@ `7df00ae`). Phase
  `notebook-plan`: measuring anchors live, then drafting the cell-by-cell plan. **Scope (chapter plan
  §NB 4):** the genuinely-new spine is **`num_leaves`/`min_child_samples` — the leaf-wise capacity dial +
  its floor, *tuned* here — which CLOSES NB 1's lopsided→overfit loop** (single tree test peaks ~64 leaves
  then falls as train→1.0; `num_leaves` is a *cap*; ensemble robust but the floor `min_child_samples`
  matters; rule `num_leaves < 2^max_depth`). Then knobs grouped by the concept that owns them:
  `learning_rate`×`n_estimators` (ch 08 NB 4 trade-off re-felt); `feature_fraction`/`bagging_fraction`
  (+`bagging_freq`) (stochasticity, ch 06/08); `reg_lambda`/`reg_alpha` **off by default** — the *posture*
  contrast with XGBoost's λ=1 (measure whether L2 lifts accuracy — expect not, like ch 09 NB 4);
  `data_sample_strategy='goss'` (NB 2); native categorical (NB 3); **early stopping via
  `callbacks=[lgb.early_stopping(N)]`+`eval_set`** (the 4.x API detail); importances `'split'` vs `'gain'`
  (MDI caveat, ch 06/08/09). Honest spine: the defaults vs a `GridSearchCV`-tuned model → **one sealed
  test** (`verbose` on so folds show — never `verbose=-1`). No `src/` change expected (reuse `viz`;
  `LGBMClassifier`; pytest 20). **Plan APPROVED by Rémy (via ExitPlanMode, 2026-06-28) & persisted**
  (`docs/plans/10_LightGBM__04_estimator_and_parameters.md`); ~24 cells / 4 figures. Anchors measured live
  (`measure_ch10_nb4.py`): defaults num_leaves31/max_depth−1/min_child_samples20/lr0.1/reg_lambda0; single
  tree test **peaks ~64 (0.878) then overfits** (255→168 leaves, a cap); ensemble plateau **0.92–0.927**;
  floor **mcs 1→0.858 / 20→0.927 / 300→0.906**; **reg_lambda flat** (0→0.919/1→0.921/100→0.906 — posture
  not lever); goss/bagging/feature ≈ flat; lr↓+trees↑ 0.9225→0.9267; **GridSearchCV tuned 0.9233 vs
  default 0.9225 (Δ+0.0008)**; importances split vs gain rankings differ (top-1 agrees).
  **BUILT (26 cells: 10 code / 16 md, 4 figures) & MERGED to `chapter/10_LightGBM` — Rémy validated
  visually.** **Two Rémy-arbitrated build decisions (his calls):** (1) the vivid `min_child_samples` floor
  lesson and zero "No further splits" warnings are **coupled** (vivid floor needs tiny leaves → over-requests
  splits → flood; can't silence without the banned `verbose=-1`) → Rémy chose **"gros jeu propre"**: the toy
  is **n=10000** (sweeps stay fillable, ~0 flood) and the floor is shown on a **single tree** (where it
  binds: leaves 256→18, depth 18→8, train 0.976→0.857, test peak@mcs20=0.902, underfit@300=0.858) — the
  ensemble being nearly inert to the floor on ample data; (2) a residual `[LightGBM] [Info]` banner flood
  (~6 lines × dozens of fits) + a spurious sklearn "feature names" warning → Rémy chose a documented
  **`LGBM_VERBOSE = 0` switch** (cell-exposed, explained, flip to 1 to see every banner; keeps real warnings
  — NOT the banned `verbose=-1`), and the feature-names warning fixed **at the root** by a named pandas
  DataFrame (`f0..f19`, pandas-first). Result: `[LightGBM] [Info]` **0**, feature-warn **0**, 4 benign "No
  further splits" left visible. **Final anchors (n=10000):** defaults nl31/md−1/mcs20/lr0.1/ne100/λ0; single
  tree num_leaves test peak@64 (0.910) then 128→0.904; ensemble plateau 0.918→0.951; single-tree floor as
  above; lr×n_est 0.945→0.950; reg_lambda flat 0.951/0.951/0.949 (posture); goss/bagging/feature ≈ 0.95;
  early stop best_iteration_ 139, test 0.948; **GridSearchCV (ne fixed 100) default 0.9493 / tuned 0.9483
  (Δ−0.0010, within noise — defaults strong)**, best {lr0.1,mcs10,nl63}; importances split [11,13,2,16,3] vs
  gain [13,3,4,10,11] (top-1 differs). **Reviewers both PASS/REVISE→folded, no BLOCK** — ml-expert PASS
  (re-mesured every anchor to the digit; verified single-tree-vs-ensemble floor claim, `predict` uses
  `best_iteration_`, LightGBM defaults genuinely gentler than XGBoost); pedagogy **REVISE→folded** the
  MAJOR (**missing "Your turn"** — added, 3 tiered exercises) + MINOR (predict/best_iteration_ note;
  Fig-1 wiggle/marker; `LGBMRegressor` aside; cell-24 read). Guards: 0 banned, hex clean, ruff clean,
  output-free, nbconvert exit 0 (4 figures), 225 KB. **No `src/` change** (notebook-local; pytest **20**).
  End-of-NB checklist done: rebuilt from `build_ch10_nb4.py`, `llms.txt` **79**, `course_map` §10 → NB 1–4
  built, `common_errors` **+3 rows** (num_leaves-vs-max_depth capacity dial; min_child_samples floor;
  tuning-within-noise). Next: open & plan NB 5 (the capstone) — last NB, then close ch 10 via PR → main.
- **NB 3 (the optimal categorical split, by hand — Fisher 1958) OPENED.** Branch
  `notebook/10_LightGBM__03_categorical_split` off `chapter/10_LightGBM` (@ `c08b6c0`). Phase
  `notebook-plan`: measuring parity anchors live, then drafting the cell-by-cell plan. **One concept
  (chapter plan §NB 3):** partitioning K categories into two groups is `2^(K−1)−1` ways (exponential);
  Fisher (1958) — for the convex structure-score gain `G²/(H+λ)` (ch 09 NB 2) — the optimal binary
  partition is **contiguous** once categories are sorted by their gradient statistic `G/H`, so only
  `K−1` candidates (linear). Build by hand on a binary toy: per-category `(G,H)`, sort by `G/H`, score
  the `K−1` contiguous splits, take the max → the category set going left; **match LightGBM exactly**
  (de-risked at the restructure: LEFT {1,3,5} ≡ LightGBM {0,2,4}) and brute-force-confirm it is the global
  optimum (vs all `2^(K−1)−1`). Genuinely new vs ch 09 (which *used* native categoricals but never built
  the split); contrast XGBoost's partition heuristic. **Build-time pins:** `min_data_per_group=1` (default
  100 breaks parity on a small toy), `min_data_in_leaf=1, min_sum_hessian_in_leaf=0, cat_l2=0, cat_smooth=0`
  — OR ≥100 rows/category. **Teaching note:** with `h=1` (regression toy) `G/H` = the per-category target
  mean; give `G/H` as the general key (carries to classification). Keep the toy **binary**. No `src/` change
  expected (reuse `viz`; `LGBMRegressor` + `dump_model`; pytest 20). **Plan APPROVED by Rémy (via
  ExitPlanMode, 2026-06-28) & persisted** (`docs/plans/10_LightGBM__03_categorical_split.md`); ~22 cells /
  3 figures. Anchors measured live (`measure_ch10_nb3_categorical.py`): per-category `G/H = F₀ − mean(y_c)`,
  sorted [1,3,5,0,2,4]; the `K−1=5` contiguous cuts (gains 448/718/**809**/721/450) → best LEFT={1,3,5};
  **brute-force all 31 (`2^(K−1)−1`) → global best {0,2,4}, same gain 808.78 = the identical partition**
  (contiguous == global, 6.2× fewer to check); **LightGBM single tree [[0,2,4],[1,3,5]] == by-hand**.
  **BUILT (22 cells: 8 code / 14 md, 3 figures) & MERGED to `chapter/10_LightGBM` — Rémy validated
  visually.** Figures: per-category `G/H` bars (scrambled index order); gain vs the `K−1` contiguous cuts
  (single peak at cut-3, annotated `K−1=5` vs `2^(K−1)−1=31`); LightGBM's two leaf values coloured by the
  by-hand side (== the partition). **Anchors reproduced by nbconvert (lightgbm 4.6.0, SEED=0):** F0 2.499;
  `G/H` sorted [1,3,5,0,2,4]; contiguous gains 448.08/718.45/**808.78**/720.59/449.99 → best {1,3,5};
  brute-force all 31 → global {0,2,4} gain 808.78 (**contiguous == global**); K=6/10/30 → 5/9/29 vs
  31/511/536,870,911; LightGBM [[0,2,4],[1,3,5]] == by-hand (identical). **Reviewers both PASS, no BLOCK**
  — ml-expert verified `G/H`≡Fisher between-groups (Δ 4.5e-13), contiguity on 400/400 configs (incl. λ>0),
  the multiclass break by counterexample (~37% — the caveat is earned), parity method valid; pedagogy
  verified the 3 Read-the-figure match the pixels, the brute-force honesty check lands, 0 banned, charter
  clean, exercises tiered. **Folds applied (markdown):** convexity argument tightened (sum of convex
  functions → max at a contiguous/extreme partition; + a pointer to the brute-force check) and **Fisher's
  exact scope** stated (least-squares / sort-by-mean; the convex generalization = CART/LightGBM); cell-14
  partition orientation aligned with the figure + "no loss of optimality" conditioned; a sign-chain
  reminder (`g=F₀−y` → high mean = negative `G/H` → `−G/H` positive) in Fig-3 read; dropped "you have
  seen". Guards: 0 banned, hex clean, ruff clean, output-free, nbconvert exit 0 (3 figures); LightGBM
  banner + the `min_data_in_leaf`/`min_sum_hessian_in_leaf` override warnings left visible. **No `src/`
  change** (notebook-local numpy/`itertools` + `LGBMRegressor`; pytest **20**). End-of-NB checklist done:
  rebuilt from `build_ch10_nb3.py` (kernel-drift guard), `llms.txt` **78**, `course_map` §10 → NB 1–3
  built, `common_errors` **+3 categorical rows** (native-not-one-hot; `2^(K−1)−1`→`K−1` Fisher;
  native-optimal-only-convex+1D / guards). Next: open & plan NB 4 (the estimator & its parameters).
- **NB 2 (GOSS built + EFB named — how LightGBM gets light) OPENED.** Branch
  `notebook/10_LightGBM__02_goss_efb` off `chapter/10_LightGBM` (@ `fed6560`). Phase `notebook-plan`:
  measuring GOSS anchors live, then drafting the cell-by-cell plan. **Scope (chapter plan §NB 2):** one
  built concept — **GOSS** (Gradient-based One-Side Sampling, Ke et al. 2017 §3.2) — + one named companion
  — **EFB** (Exclusive Feature Bundling, §4). Name the ch 09 NB 2 anchor: the split-gain is a **sum over
  rows** of `g,h`. GOSS: rank rows by `|gradient|`, keep the top `a`, sample `b` of the rest, **up-weight
  the sampled rest by `(1−a)/b`** so `G,H` stay ~unbiased (`H` exactly: `a·n + (1−a)/b · b·n = n`). The
  discriminating story: large-`|g|` rows are kept **exactly** (zero sampling variance) → the GOSS
  gain-estimate is **lower-variance** than a uniform subsample at the same fraction `a+b`, both ~centred on
  the full-data gain; this translates to **GOSS ≈ full-data held-out quality & beating a uniform subsample
  at matched fraction** (LightGBM ensemble). **Honesty bar:** GOSS's wall-clock benefit is
  **regime-dependent (~flat on dense moderate data — measured; bites on very large/wide data)**; the gain
  is `G²/(H+λ)`, a *function* of the row-sums, so it is *approximately* (not exactly) unbiased — GOSS lowers
  its **variance**, the honest claim (Ke Thm 3.2). EFB named (approximately-exclusive sparse features,
  tolerates a small conflict rate — more than concatenating one-hot columns). No `src/` change expected
  (reuse `viz`; `LGBMClassifier` `data_sample_strategy='goss'` / `bagging_fraction` for the uniform foil;
  pytest 20). **Plan APPROVED by Rémy (via ExitPlanMode, 2026-06-28) & persisted**
  (`docs/plans/10_LightGBM__02_goss_efb.md`); ~22 cells / 4 figures. **Measured-before-claim reframe (Rémy
  signed off):** the chapter-plan wording "GOSS beats a uniform subsample at matched fraction" proved
  **regime-dependent** — GOSS's variance edge over uniform is gated by **gradient concentration**
  (crossover ≈ 0.5; synthetic sweep GOSS/uniform std ratio 1.64/1.24/0.99/0.64/0.27 at conc
  0.20/0.36/0.49/0.67/0.86). On **moderate dense tabular data** concentration stays **0.21→0.47** even at
  200 boosting rounds (≤ crossover) → GOSS **≈ full quality on 30 % rows** (ensemble 0.9345 vs full 0.9313)
  but **~ties** uniform there; GOSS's real win is the large/wide/sparse concentrated regime (→ NB 5). Both
  unbiased (`H` exact `a·n+(1−a)/b·b·n=n`; `G` unbiased); the lever is **variance, only when gradients are
  imbalanced** (Ke et al. Thm 3.2). Wall-clock ~flat on dense (2.41 vs 2.45 s). NB teaches this honest
  version (a strict upgrade, consistent with the chapter's "GOSS = statistical efficiency, wall-clock
  regime-dependent" bar). **BUILT (23 cells: 8 code / 15 md, 4 figures) & MERGED to `chapter/10_LightGBM`
  — Rémy validated visually.** Figures: |g| distribution + GOSS's kept band; GOSS-vs-uniform gain-estimate
  distributions on a concentrated set (both centred on the truth, GOSS ~3× tighter); the concentration
  crossover sweep (variance ratio vs concentration, crossing 1.0 near ~0.5, with the measured dense-tabular
  band 0.21–0.47 shaded at/below it); the EFB one-hot exclusivity image. **Anchors reproduced by nbconvert
  (lightgbm 4.6.0, SEED=0):** round-1 toy |g| median 1.467/max 8.101, concentration 0.44; G_full +0.000 /
  H 2000.0; GOSS G mean +6.77 (std 161.7), **H exact 2000.000 (std 0)**; uniform G +10.4 (std 156.4);
  concentrated set (conc 0.66) true gain 50.42, GOSS std 1.88 vs uniform 5.54 (ratio 0.34); sweep
  1.49/1.42/1.06/0.56/0.35/0.27/0.28 at conc 0.20→0.86; ensemble (45k×30, 300 trees) full 0.9313 / GOSS
  0.9345 / uniform 0.9353 (fit ~2.0 s each — flat); EFB 15 one-hot cols → ~3 bundles (same-group conflict
  0.0000 / cross-group 0.0354). **Reviewers both PASS, no BLOCK** — ml-expert re-derived the algebra
  (`E[G_GOSS]=G_full`; `H_GOSS=n` exact every draw) and adversarially confirmed the synthetic feature is
  **conservative** for GOSS, not flattering; pedagogy verified all 4 Read-the-figure match the pixels, 0
  banned, charter clean, exercises tiered. **Folds applied:** Thm-3.2 attribution softened (it bounds GOSS's
  gain error by a term growing with the largest *sampled* gradient; the crossover-vs-uniform is *our*
  measurement); a feature-conservatism sentence + seeded into exercise 2; "Figure 3" → descriptive;
  cosmetic line-wrap. Guards: 0 banned, hex clean, ruff clean, output-free, nbconvert exit 0 (4 figures);
  LightGBM banner left visible (no `verbose=-1`). **No `src/` change** (notebook-local numpy + `LGBMClassifier`;
  pytest **20**). End-of-NB checklist done: rebuilt from `build_ch10_nb2.py` (kernel-drift guard), `llms.txt`
  **77**, `course_map` §10 → NB 1–2 built, `common_errors` **+4 LightGBM/GOSS/EFB rows** (GOSS-vs-uniform
  gated-by-concentration; unbiased-but-variance-is-the-point; GOSS-no-wall-clock-on-dense; EFB≠concatenation)
  **+ fixed a stale `num_leaves` NB-2→NB-4 ref**. Next: open & plan NB 3 (the optimal categorical split).
- **CHAPTER 10 RESTRUCTURED (Rémy-initiated) & re-gated — both reviewers no BLOCK.** Reaching NB 2,
  `num_leaves` proved **too light** for a standalone fundamental (a parameter sweep, no by-hand build,
  overlapping NB 4). Rémy chose to restructure. New arc: NB 1 leaf-wise (shipped) · **NB 2 GOSS+EFB** ·
  **NB 3 the optimal categorical split (NEW — built by hand, Fisher 1958: sort categories by `G/H`, best
  contiguous partition, `K−1` candidates; de-risked == LightGBM exactly, LEFT {1,3,5} ≡ {0,2,4})** · NB 4
  estimator (`num_leaves`/`min_child_samples` *tuned* here, closing NB 1's lopsided→overfit loop) · NB 5
  capstone. `num_leaves` standalone NB dropped → budget in NB 1 + dial in NB 4. All three fundamentals now
  **build a mechanism by hand** (leaf-wise · GOSS · categorical split). Re-gate folds: ml-expert (pin
  `min_data_per_group=1` for NB-3 parity; `G/H`↔mean note; structure-score criterion; toy binary);
  pedagogy (NB 1 had **5** forward-ref seams not 3 — repointed: overfit→NB 4, "Next"→NB 2 GOSS; NB 4 must
  close the NB-1 loop). **NB 1 forward-refs repointed** (cells 1/8/14/16 → NB 4; "Next" → NB 2 GOSS+EFB),
  rebuilt, guards green (0 residual seams, 0 banned, ruff clean, 3 figures, output-free). `chapter_10` plan
  + `course_map` §10 updated. Next: open & plan NB 2 (GOSS + EFB).
- **NB 1 (leaf-wise / best-first growth, by hand) OPENED.** Branch
  `notebook/10_LightGBM__01_leaf_wise_growth` off `chapter/10_LightGBM` (@ `b6cb63d`). Phase
  `notebook-plan`: measuring anchors live then drafting the cell-by-cell plan — one concept, **build
  leaf-wise (best-first) tree growth by hand** (a frontier of candidate leaves; repeatedly expand the
  one whose best split gives the largest loss reduction, to a `num_leaves` budget) and contrast with
  level-wise (ch 04/09 depthwise). Leaf-wise was *named* twice before (ch 08 NB 5 HistGBR
  `max_leaf_nodes`; ch 09 NB 4 `lossguide`) — **here built**. Anchor: by-hand leaf-wise == LightGBM's
  single tree on a toy; leaf-wise reaches lower **training** loss for the same #leaves, grows lopsided
  (qualitative; the depth-vs-num_leaves sweep is NB 2). No `src/` change expected (reuse `viz`;
  `dump_model`; pytest 20). **Plan APPROVED by Rémy (via ExitPlanMode, 2026-06-28) & persisted**
  (`docs/plans/10_LightGBM__01_leaf_wise_growth.md`); ~19 cells / 3 figures; anchors measured live
  (leaf-wise ≪ level-wise train SSE; lopsided sizes …109; **exact LightGBM parity** max|Δpred|=0,
  SSE 0.566==0.566). **BUILT (17 cells, 3 figures) & REVIEWED — both NO BLOCK, awaiting Rémy visual.**
  Each reviewer raised one MAJOR → folded: ml-expert caught a real bug — my `grow('level')` descended the
  left spine **depth-first** instead of breadth-first → rewrote it as a true FIFO (corrected level-wise
  curve 268→65@4 leaves→12@8; leaf-wise still wins at every budget — 6 leaves 0.57 vs 65.4); pedagogy
  caught Fig-2 subplot titles colliding → moved leaf sizes to a print + short titles + `tight_layout`.
  MINOR/NIT folded: named `LAMBDA=0.0`; one line that the leaf value `mean(y)` = `F0 + (−G/H)`; FIFO
  wording. Parity unchanged (leaf-wise logic untouched). Guards: ruff clean, 0 banned, output-free, hex
  clean, nbconvert exit 0 (3 figures); LightGBM banner left visible (no `verbose=-1`). **No `src/`
  change** (pytest 20). **BUILT & MERGED to `chapter/10_LightGBM` — Rémy validated visually.**
  End-of-NB checklist done: rebuilt from `build_ch10_nb1.py`, `llms.txt` **76**, `course_map` §10 → NB 1
  built, `common_errors` **+2 rows** (leaf-wise lower-train-loss-≠-better; leaf-wise-is-the-modern-default-
  not-LightGBM-only), pytest **20**. Next: open & plan NB 2 (`num_leaves`, the central dial).
- **Chapter 10 (LightGBM) opened.** Branch `chapter/10_LightGBM` created off `main` (synced @ `fe295aa`
  after PR #9). Phase `chapter-plan`: drafting the chapter plan in plan mode per `course_map.md` §10 and
  the per-method arc — **leaf-wise vs level-wise growth** (the XGBoost contrast; ch 09 NB 4 named
  `grow_policy=lossguide` as this bridge); histogram binning + **`num_leaves`** as the central dial;
  categorical handling & the overfitting traps; tuning / early stopping vs XGBoost; a demanding case on
  larger tabular data (speed/accuracy measured). LightGBM 4.6.0 lives in the `boosting` extra (+
  `libomp`); confirm the live API/behaviour before pinning anchors. Chapter plan is reviewer-gated (both
  reviewers, no BLOCK) before Rémy approves. The pending ch-09-close edits (STATE → idle + `course_map`
  §09 → "merged via PR #9") are folded into this opening commit (committed on the chapter branch, not on
  protected `main`). **Plan APPROVED by Rémy (via ExitPlanMode + the two-reviewer gate, 2026-06-28) &
  persisted** (`docs/plans/chapter_10_LightGBM.md`); both reviewers **REVISE → no BLOCK**, folded: the
  speed thesis → **matched-capacity both directions** (the winner flips with the convention; not "XGBoost
  faster"); GOSS → **statistical efficiency** (wall-clock regime-dependent, ~flat on dense data, measured);
  the **HistGBR-is-already-leaf-wise gift** (it defaults `max_leaf_nodes=31` = LightGBM's `num_leaves=31`)
  → used in the spine + as NB 5's matched sibling; NB 3 = "one built (GOSS) + one named (EFB)". Live
  anchors measured (num_leaves 31 / max_depth −1; num_leaves 127 → depth 15; GOSS API; categorical;
  early-stopping callbacks; speed/accuracy). `course_map` §10 refined to the 5-NB decomposition. Next:
  open & plan NB 1 (leaf-wise growth, by hand).
- **CHAPTER 09 (XGBoost) COMPLETE — merged to `main` via PR #9** (merge commit `fe295aa`,
  `gh pr merge --merge`; per-notebook history preserved; pushed to Ramdam17/QuickIntroToMachineLearning).
  **Five notebooks:** the second-order view (`w*=−G/H` unifies ch 08's two leaf rules) · the regularized
  objective (λ/γ, the structure-score gain, `Cover=Σh`, the measured 2×/½ convention) · sparsity-aware
  splits (a learned default direction for NaN) · the estimator & its parameters (each knob from its
  concept; **owns the histogram method**, ~5× faster at no accuracy cost; defaults overfit → honest
  tuning) · the Census-income capstone (8 figures: imbalance/PR-AUC, the **honest native-vs-imputed
  null**, cross-method, early stopping, validation-chosen threshold, MDI-vs-permutation, ethics /
  do-not-deploy). The **regularized, second-order, engineered refinement** of ch 08's engine — **not a
  new algorithm**, and **no universal best** (edges = speed / native missing+categorical handling /
  regularization, not accuracy). Two-reviewer gate (no BLOCK) + Rémy visual on every NB; every number
  measured live (xgboost 3.2.0 / lightgbm 4.6.0), seed-pinned; honest findings throughout (the 2×/½ gain
  convention; defaults memorize but L2-on is about calibration not accuracy; native-NaN ≈ imputed when
  missingness is redundant; convenience ≠ power; MDI-vs-permutation divergence). **No `src/` change**
  across the chapter (reused `viz`; `trees_to_dataframe`; `fetch_openml`; sklearn Pipelines) — pytest
  stays **20**. `main` synced @ `fe295aa`, ruff green, pytest 20, `llms.txt` 74. Env: `boosting` extra +
  `libomp`. STATE → `idle` (this edit + `course_map` §09 → "merged via PR #9" pending on `main`, fold
  into the ch-10 opening). Next: open chapter 10 (LightGBM).
- **NB 5 (the demanding case — Adult/Census Income capstone) BUILT & MERGED to `chapter/09_XGBoost` —
  Rémy validated visually. CHAPTER 09 COMPLETE on the branch (5/5).** 30 cells (9 code / 21 md), **8
  figures** (class balance; missingness-vs-target;
  native-vs-imputed PR-AUC [the honest null]; cross-method PR-AUC; early stopping; PR curve + threshold
  + confusion; gain-MDI vs permutation; base rates by sex/race). A full honest workflow mobilizing all
  of ch 09 + ch 00. **Anchors reproduced by nbconvert:** positive rate 0.2393; informative missingness
  (occupation missing 0.094 vs 0.248); **native-vs-imputed Δ −0.0011, within noise** — the honest null
  (the resampling band flips sign: mean +0.0008, range [−0.0011,+0.0035]; the two informative-missing
  cols are nearly the same flag, occupation∩workclass 2799/2809); cross-method HistGBR 0.8291 ≈ XGB
  0.8280 > GB 0.8140 > RF 0.7883 > Logistic 0.7679 > tree 0.6684 (LightGBM teaser 0.8278/2.2s); early
  stopping best_iteration 263/2000, test PR-AUC 0.8292/acc 0.8730/ROC-AUC 0.9280; threshold 0.5
  (P 0.782/R 0.650) → val-chosen 0.299 (P 0.649/R 0.825); gain-MDI relationship 0.34 vs permutation
  capital-gain 0.236; ethics base rates F 0.109/M 0.304, Black 0.121/White 0.254. Reviewers **both NO
  BLOCK** — pedagogy **PASS** (honest-null framed as the strongest lesson; 8/8 Read-the-figure match
  the pixels; ethics sober); ml-expert **REVISE→folded** the MAJOR (the cell-12 mechanism was too
  strong — "recovers the same information elsewhere" → re-aimed to **collinearity** [occ∩workclass
  2799/2809] + "largely carried elsewhere" + a **measured resampling band** that flips sign, justifying
  "within noise") + MINOR/NIT (OHE rare-pooling disclosed; native-country "marginally above"; exercise-1
  recompute hint). Notably ml-expert found the "convenience not power" framing **under**-claimed (at
  matched capacity on equal inputs XGB 0.810 vs GB 0.814 — the thesis holds harder). Threshold is
  **val-chosen** (leakage-free). Guards: ruff clean, **0 banned**, output-free, hex clean, nbconvert
  exit 0 (8 figures); LightGBM + early-stopping logs left **visible** (no silencing). **No `src/`
  change** (notebook-local; sklearn Pipelines; pytest 20). End-of-NB checklist done: rebuilt from
  `build_ch09_nb5.py` (kernel-drift guard), `llms.txt` **74**, `course_map` §09 → **COMPLETE**,
  `common_errors` **+3 rows** (informative-missingness-isn't-always-a-lever / measure-it; convenience-
  not-power / no-universal-best; threshold-on-validation + PR-AUC-on-imbalance). **Last NB of ch 09 —
  chapter complete on the branch.** Next: close the chapter via PR `chapter/09_XGBoost → main`
  (`--no-ff`).
- **NB 5 (the demanding case — Adult/Census Income, the visualization-first capstone) OPENED.** Branch
  `notebook/09_XGBoost__05_census_income` off `chapter/09_XGBoost` (@ `8b124e5`). Phase `notebook-plan`:
  measuring anchors live then drafting the capstone (~26 cells, ≥6 figures). The chapter's **capstone** —
  a full, honest tabular workflow mobilizing all of ch 09. **Verification owed at plan time** (chapter
  plan): does Adult's missingness actually move PR-AUC under native-NaN-vs-imputed, and is `?`→NaN; if
  the lever is negligible, lean the NB-3 callback on categorical native handling or fall back to Ames
  (rename branch). Arc: measured missingness-vs-target panel → baselines → tuned XGBoost with **early
  stopping** (`eval_set`) → held-out acc/precision/recall/PR-AUC + threshold → error analysis → honest
  cross-method comparison (XGB/GB/HistGBR/RF/linear, **native-NaN-vs-imputed a named axis**, XGB run both
  ways) → gain-MDI vs permutation → LightGBM teaser. **Last NB of ch 09 — after it ships, close the
  chapter via PR `chapter/09_XGBoost → main` (`--no-ff`).** **Verification done & plan APPROVED by Rémy
  (via ExitPlanMode + an AskUserQuestion fork, 2026-06-28) & persisted**
  (`docs/plans/09_XGBoost__05_census_income.md`); ~30 cells / 8 figures. The measured surprise:
  native-NaN-vs-imputed is **within noise** (Δ −0.001) because the informative missingness is redundant
  with correlated features → Rémy chose to **keep Adult and feature the honest null** ("measure the
  lever"), the NB-3 callback leaning on convenience + the cross-method axis. Anchors measured live (see
  the persisted plan). Building now from a `build_ch09_nb5.py` scratchpad script.
- **NB 4 (the estimator `XGBClassifier` & its parameters — integrative; owns the histogram method)
  BUILT & MERGED to `chapter/09_XGBoost` — Rémy validated visually.** 25 cells (8 code / 17 md), 4 figures (the
  depth dial — test peak@4, default@6 past it, leaves exploding; regularizers close the gap — gamma &
  min_child_weight, leaves collapse + train/test converge; eta×n_estimators staged, log-x; the
  histogram engine — binning idea + hist-vs-exact fit-time + test-acc bars). Each knob taught from the
  concept that owns it (depth=interaction order; reg_lambda/gamma/min_child_weight=NB 2 + Cover=Σh;
  subsample/colsample=stochasticity; eta×n_estimators=ch 08 NB 4). **Anchors reproduced exactly by
  nbconvert:** defaults train **1.0000**/test 0.9444 (gap 0.0556, 2608 leaves); depth peak@4 (0.9467);
  gamma leaves 2608→264 (gap→0.0165); reg_lambda **λ0 0.9456 / λ1 0.9444 / λ100 0.9433 / λ1000 0.9283**;
  eta0.3 plateaus ~0.944 / eta0.03→0.9511@600 (no collapse); **histogram exact 4.2s/0.9646 vs hist
  0.85s/0.9660 (~5×, no accuracy cost)**, max_bin64 0.70s; GridSearchCV→sealed **0.9472 vs default
  0.9444 (+0.0028)**. Reviewers **both NO BLOCK** — pedagogy **PASS** (gap-free knob→concept spine; all
  4 "Read the figure" match the pixels; +0.003 framed as the lesson; 5 MINOR/NIT optional); ml-expert
  **REVISE→folded** one **MAJOR** (I had wrongly credited `lambda=1` for the respectable test accuracy
  and the no-collapse — the counterfactual shows **λ=0 → 0.9456 > λ=1 → 0.9444**, so L2 at its default
  doesn't lift accuracy; re-aimed: L2's job is to **shrink leaf weights** `−G/(H+λ)`, and the
  non-collapse is because **accuracy is a coarse, bounded metric near the noise ceiling**, not L2) +
  MINOR/NIT (added **λ=0 to the printed reg_lambda sweep** so the claim is self-evidently measured;
  reach 3(a) reframed to the gap/magnitude question; histogram "+0.0014 = noise, bins mildly
  regularize"; `auto→hist` gloss; Fig-3 names the amber middle eta + best-marker subtlety; Fig-4
  tight_layout + shorter title). Guards: ruff clean, **0 banned** (JSON scan), output-free, hex clean,
  nbconvert exit 0 (4 figures). **No `src/` change** (notebook-local matplotlib; `trees_to_dataframe`;
  pytest 20). End-of-NB checklist done: rebuilt from `build_ch09_nb4.py` (kernel-drift guard), `llms.txt`
  **73**, `course_map` §09 → NB 1–4 built, `common_errors` **+3 XGBoost rows** (defaults-overfit-but-near-
  ceiling/depth-peak; L2-default-doesn't-lift-accuracy [the folded MAJOR]; hist-not-less-accurate).
  **NB 1–4 done.** Next: open & plan NB 5 — the demanding case (visualization-first capstone).
- **NB 4 (the estimator `XGBClassifier`/`XGBRegressor` & its parameters — the integrative NB; owns the
  histogram method) OPENED.** Branch `notebook/09_XGBoost__04_estimator_and_parameters` off
  `chapter/09_XGBoost` (@ `ad7c898`). Phase `notebook-plan`: drafting the cell-by-cell plan — recap the
  NB-2 by-hand parity, then the **one genuinely new mechanism**, **histogram / approximate split
  finding** (bin continuous features into ≤`max_bin` buckets → the threshold scan goes from every
  distinct value to ≤256 bin edges; `tree_method='hist'` is the 3.x default; **measure** hist-vs-`exact`
  speed + the negligible accuracy cost; the weighted-quantile sketch C&G §3.2–3.3 named & deferred to
  ch 10). Then the knobs grouped by the concept that owns them — objective regularizers
  `reg_lambda`/`reg_alpha`/`gamma` (NB 2), tree complexity `max_depth`(6)/`min_child_weight`(a Cover
  floor, NB 1/2)/`grow_policy` (depthwise vs lossguide — the ch-10 bridge, named-not-used), stochasticity
  `subsample` (Friedman 2002)/`colsample_*` (new vs sklearn GB), `eta`(0.3)×`n_estimators` (the ch 08
  NB 4 trade-off re-felt). Honest spine: the **aggressive defaults overfit** (eta 0.3 + depth 6) → show
  it → `GridSearchCV` on train → one sealed test (tuned vs default); `feature_importances_` gain-MDI
  caveat (honest reading deferred to NB 5). Anchors being measured at plan time (xgboost 3.2.0). No
  `src/` change expected (reuse `viz`; pytest 20). **Plan APPROVED by Rémy (via ExitPlanMode,
  2026-06-28) & persisted** (`docs/plans/09_XGBoost__04_estimator_and_parameters.md`); ~23 cells / 4
  figures; anchors measured live — defaults overfit **train 1.0000 / test 0.9444** (gap 0.0556, 2608
  leaves); **max_depth test peak @4** (default 6 past it); **gamma** prunes leaves 2608→264 (gap
  0.056→0.017); **min_child_weight** (Cover=ΣH floor) 2608→321; eta×n_estimators staged (eta0.3
  plateaus ~0.944, eta0.03 climbs to 0.951@600, no collapse); **histogram hist vs exact: 4.23s→0.85s
  (~5×), no accuracy cost** (0.9646→0.9660); GridSearchCV→sealed test **0.9472 vs default 0.9444
  (+0.003)**. Building now from a `build_ch09_nb4.py` scratchpad script.
- **NB 3 (sparsity-aware splits — a learned default direction for missing values, by hand) BUILT &
  MERGED to `chapter/09_XGBoost` — Rémy validated visually.** 19 cells (6 code / 13 md), 3 figures (the
  data + the missing rows in a side band; the gain-vs-threshold search, both directions; by-hand vs
  XGBoost). One concept: missing rows take a **learned default direction**, found by computing the NB-2
  gain both ways and keeping the larger (C&G §3.4) — no imputation. **Anchors (xgboost 3.2.0, reproduced
  exactly): by-hand search best (x<5, missing→right, half-gain 2.949); XGBoost split x<5, `Missing`→right
  (No), Gain 5.8985 = 2× (no-½ convention); both missing rows route to the dear leaf (Cover 5 = 3+2); GB
  rejects NaN (`ValueError`), HistGBR & XGBoost accept.** Reviewers **both PASS, no BLOCK** — ml-expert
  confirmed the single-default-direction semantics (both NaN rows predict 2.9146) + the exact
  direction/gain match; pedagogy verified the warm-up (flip the missing y → the direction flips to left).
  Folded: Fig-1 legend → y-based (cheap y≈1 / dear y≈3); cell-12 spells out `Cover=5 = 3+2`;
  `tree_method='exact'` commented. Guards: **0 banned** (JSON scan), hex clean, output-free; nbconvert
  exit 0 (3 figures / 0 errors); ruff clean; `llms.txt` **72**; `common_errors` +2 XGBoost rows
  (learned-default-direction; native-NaN-vs-impute + the GB/HistGBR/XGB contrast). **No `src/` change**
  (notebook-local matplotlib; `trees_to_dataframe`; pytest **20**). Rebuilt from `build_ch09_nb3.py` right
  before `git add` (kernel-drift guard). **Three fundamentals done.** Next: open & plan NB 4 (the
  estimator & its parameters + the histogram method).
- **NB 3 (sparsity-aware splits — a learned default direction for missing values, by hand) OPENED.**
  Branch `notebook/09_XGBoost__03_sparsity_aware_splits` off `chapter/09_XGBoost` (@ `307983f`). Phase
  `notebook-plan`: drafting the cell-by-cell plan — one concept, XGBoost handles **missing** values with
  no imputation by sending every missing row down a **learned default direction**, chosen by trying both
  ways and keeping the higher gain (Chen & Guestrin 2016 §3.4). Build by hand on a small NaN-bearing toy
  (enumerate thresholds on the non-missing values; per threshold compute the NB-2 gain with missing→left
  vs missing→right; take the global argmax) and check the chosen direction against XGBoost's `Missing`
  column in `trees_to_dataframe`. Contrast (measured): plain `GradientBoosting*` **rejects** NaN;
  `HistGradientBoosting*` (ch 08) & XGBoost **accept** it (XGBoost did it first, 2016). Anchors measured
  at plan time (xgboost 3.2.0). No `src/` change expected (notebook-local matplotlib; `trees_to_dataframe`;
  pytest 20). **Plan APPROVED by Rémy (via ExitPlanMode, 2026-06-27) & persisted**
  (`docs/plans/09_XGBoost__03_sparsity_aware_splits.md`); ~19 cells / 3 figures; anchors measured live
  (by-hand best `x<5, missing→right, 2.949`; XGBoost `Missing`→right, Gain `5.8985 = 2×`; GB rejects NaN,
  HistGBR & XGB accept). Building now from a `build_ch09_nb3.py` scratchpad script.
- **NB 2 (the regularized objective — λ, γ, and the gain that decides splits, by hand) BUILT & MERGED
  to `chapter/09_XGBoost` — Rémy validated visually.** 19 cells (5 code / 14 md), 3 figures (the
  regularized parabola λ∈{0,1,10}; the gain + γ threshold with a **measured** split/prune sweep; by-hand
  vs XGBoost parity). One concept: complexity priced **inside** the objective `Ω = γT + ½λΣw²`.
  **Anchors (xgboost 3.2.0, reproduced exactly): λ shrinks the leaf `w*=−G/(H+λ)` (+2.0/+1.5/+1.0/
  +0.4615/+0.0583); the structure-score split gain (C&G eq. 6→7) by-hand ½-gain = 9; XGBoost reports
  Gain = 18 (= 2×, the ½ dropped) and prunes at γ>18 — so γ is in no-½ units; leaf weights [−1.5,+1.5]
  == by-hand; Cover = ΣH (3/leaf); base_score learned = mean(y) when unpinned.** Reviewers **both PASS,
  no BLOCK** — ml-expert re-derived the gain sign and ran a fine γ-sweep (kept 18.00, pruned 18.01),
  confirming the no-½-units claim across several toys; pedagogy caught a real **Fig-1 colour collision**
  (`model`==`class_d` share a hex) and confirmed the one-concept build + the kind 2×/½ framing. Folded:
  Fig-1 colours → blue/amber/coral; the ½'s origin shown by the back-substitution cancellation (cell 8);
  γ pre-pruning softened + an NB-4 post-prune forward-note; "clearly"→"genuinely"; axis "2×9=18".
  Guards: **0 banned** (JSON scan), hex clean, output-free; nbconvert exit 0 (3 figures / 0 errors);
  ruff clean; `llms.txt` **71**; `common_errors` +3 XGBoost rows (regularizer-prices-complexity-inside;
  the gain sign-trap; the 2×/½-and-γ-units detail). **No `src/` change** (notebook-local matplotlib;
  `trees_to_dataframe`; pytest **20**). Rebuilt from `build_ch09_nb2.py` right before `git add`
  (kernel-drift guard). Next: open & plan NB 3 (sparsity-aware splits).
- **NB 2 (the regularized objective — λ, γ, and the gain that decides splits, by hand) OPENED.** Branch
  `notebook/09_XGBoost__02_regularized_objective` off `chapter/09_XGBoost` (@ `113ebb3`). Phase
  `notebook-plan`: drafting the cell-by-cell plan — one concept, put complexity **into the objective**
  `Obj = Σ loss + Ω(f)`, `Ω = γT + ½λΣw²`. Two consequences, both by hand: the leaf weight becomes
  `w* = −G/(H+λ)` (L2 shrinks it toward 0), and the **split gain** derived from the structure score
  (C&G eq. 6 → 7) `½[G_L²/(H_L+λ)+G_R²/(H_R+λ)−G²/(H+λ)] − γ` decides whether to split (γ = pre-pruning).
  The correctness anchor: by-hand leaf weights == XGBoost exactly, and the **measured 2×/½** detail
  (XGBoost reports 2× the textbook gain, dropping the ½ — argmax-invariant); `Cover = ΣH`; `base_score`
  honest (learned when unpinned). Anchors measured at plan time (xgboost 3.2.0). No `src/` change
  expected (notebook-local matplotlib; `trees_to_dataframe`; pytest 20). **Plan APPROVED by Rémy (via
  ExitPlanMode, 2026-06-27) & persisted** (`docs/plans/09_XGBoost__02_regularized_objective.md`); ~20
  cells / 3 figures; anchors measured live (λ-shrinkage ±2.0/±1.5/±0.46; ½-gain 9 vs XGBoost 18; γ prune
  at >18; `Cover=ΣH`; `base_score` learned = mean). Building now from a `build_ch09_nb2.py` scratchpad
  script.
- **NB 1 (the second-order view — gradients + curvature, by hand) BUILT & MERGED to
  `chapter/09_XGBoost` — Rémy validated visually.** 22 cells (7 code / 15 md), 3 figures (the parabola +
  its 2nd-order approximation; the curvature-blind vs Newton step; the two-losses-one-rule curvature
  panel). One concept: approximate any loss to second order around F → optimal leaf `w*=−G/H`.
  **Anchors (xgboost 3.2.0, reproduced exactly): the scalar step `w*=−g/h`; SE recovery on ch 08's sine
  (F0=−0.1199; per leaf −G/H == mean residual == tree leaf, +0.29451/+0.75892/−0.68149/−0.22921);
  log-loss recovery on ch 08's moons (F0=0; −G/H == Newton −2.0/+1.72881/−1.62590/+2.0); λ=0 XGBoost
  parity [−2.0,+2.0].** The unification: ch 08's two leaf rules are **one** second-order rule (SE h=1 →
  mean; log-loss h=p(1−p) → Newton); the loss only changes G,H. Reviewers **both PASS, no BLOCK** — each
  re-derived the math and re-ran every anchor (incl. the `trees_to_dataframe` leaf-in-`Gain` quirk, the
  ×4 rescale, all 3 rendered figures); folded their convergent MINOR polish (h>0 caveat + uniform
  "twice-differentiable"; recap now says *why* the mean fell short; the `−g=y−p` ↔ "mean residual" tie).
  Guards: **0 banned** (JSON scan), hex clean, output-free; nbconvert exit 0 (3 figures / 0 errors);
  ruff clean; `llms.txt` **70**; `common_errors` +2 XGBoost rows (two-leaf-rules-are-one-second-order-
  rule; gradient=direction / curvature=step-length). **No `src/` change** (notebook-local matplotlib;
  reused `viz`; XGBoost via `trees_to_dataframe`; pytest **20**). Rebuilt from `build_ch09_nb1.py` right
  before `git add` (kernel-drift guard). Next: open & plan NB 2 (the regularized objective).
- **NB 1 (the second-order view — gradients + curvature, by hand) OPENED.** Branch
  `notebook/09_XGBoost__01_second_order_view` off `chapter/09_XGBoost` (@ `9ebb1e1`). Phase
  `notebook-plan`: drafting the cell-by-cell plan — one concept, **approximate any loss to second order
  and read off the optimal leaf**: `L(F+w) ≈ L(F) + g·w + ½h·w²` (a parabola) → minimum `w*=−g/h` → over
  a leaf `w*=−G/H`. Pin the convention once (`g=∂L/∂F`, `h=∂²L/∂F²`, leaf `−G/H`), then recover **both**
  ch 08 leaf rules: SE (`g=F−y, h=1`) → mean residual (ch 08 NB 1), log-loss (`g=p−y, h=p(1−p)`) →
  `Σ(y−p)/Σp(1−p)` (ch 08 NB 3's Newton leaf) — equal billing, classification the climax. NB-1's **own
  λ=0 XGBoost parity** (`reg_lambda=0, gamma=0, eta=1, base_score` pinned, 1 tree / depth 1 → leaf ==
  by-hand `−G/H`). Anchors measured at plan time (xgboost 3.2.0). No `src/` change expected
  (notebook-local matplotlib; reuse `viz`; pytest 20). **Plan APPROVED by Rémy (via ExitPlanMode,
  2026-06-27) & persisted** (`docs/plans/09_XGBoost__01_second_order_view.md`); ~22 cells / 3 figures;
  anchors measured live (`w*=−G/H`; SE→mean & log-loss→Newton recoveries; λ=0 XGBoost parity `[−2.0,+2.0]`).
  Building now from a `build_ch09_nb1.py` scratchpad script.
- **Chapter 09 (XGBoost) plan APPROVED by Rémy & persisted** (`docs/plans/chapter_09_XGBoost.md`, this
  commit). **FIVE notebooks** on the arc — NB 1 the **second-order view** (gradients + curvature by hand;
  `w*=−G/H` unifies ch 08's SE leaf=mean and log-loss Newton leaf; its own λ=0 XGBoost parity) → NB 2 the
  **regularized objective** (`Ω=γT+½λΣw²`; the structure-score gain C&G eq. 6–7; the **measured 2×/½**
  parity detail + `Cover`=ΣH) → NB 3 **sparsity-aware splits** (a learned default direction for missing
  values, by hand; GB rejects NaN / HistGBR & XGBoost accept) → NB 4 the **estimator & its parameters**
  (owns the histogram method `tree_method='hist'`/`max_bin` measured; `reg_lambda`/`reg_alpha`/`gamma`,
  `max_depth`/`min_child_weight`/`grow_policy`, `subsample`/`colsample_*`, eta×n_estimators; defaults
  overfit) → NB 5 **demanding case** (Adult/Census visualization-first capstone; Ames fallback). The
  **regularized, second-order refinement** of ch 08's engine; **XGBoost is not a new algorithm** and does
  **not** reliably beat tuned GB/HistGBR on accuracy (edges = missing-handling/speed/regularization).
  **Env fixed & measured live:** `uv sync --extra dev --extra boosting` (xgboost 3.2.0 / lightgbm 4.6.0)
  + `brew install libomp` (macOS OpenMP runtime — xgboost could not load without it). Reviewer-gated on
  the live install: **both reviewers NO BLOCK**, each independently re-running the by-hand parity (leaf
  weights exact, gain 2× textbook, `Cover`=ΣH), the resolved defaults, NaN across GB/HistGBR/XGBoost,
  native categoricals, the early-stopping API, and Adult's informative missingness. All MAJOR/MINOR folded
  (NB 1 re-lays the Taylor move + own λ=0 parity + pinned signs + equal billing; NB 2 derives gain from
  the structure score + kind ½ framing + honest `base_score`; NB 3 kept pure + pinned anchor; NB 4 owns
  histogram-as-concept; NB 5 informative-missingness + native-vs-imputed axis + Adult/Ames verification).
  `course_map.md` §09 refined to the 5-NB decomposition. **No `src/` change expected** (reuse `viz`;
  trees via `trees_to_dataframe`; datasets via `fetch_openml`; pytest 20). Next: open & plan NB 1.
- **Chapter 09 (XGBoost) opened.** Branch `chapter/09_XGBoost` created off `main` (synced @ `4775fe2`
  after PR #8). Phase `chapter-plan`: drafting the chapter plan in plan mode per `course_map.md` §09 and
  the per-method arc — what XGBoost **adds** to gradient boosting (the regularized objective + the
  second-order / Newton view) → missing-value & sparsity handling + the histogram split → the key knobs
  (`eta`, `max_depth`, `subsample`, `colsample_*`, `lambda`/`alpha`) → early stopping & overfitting
  control → a demanding case tuning honestly vs the simpler boosters. Builds directly on ch 08's gradient
  boosting (XGBoost = a regularized, second-order, engineered refinement of the same engine); ch 10
  LightGBM is the sibling. `xgboost` lives in the `boosting` extra — confirm the live install/version
  before pinning any anchors. The pending `idle` STATE edit + `course_map.md` §08 → complete were folded
  into this opening commit (committed on the chapter branch, not on protected `main`).
- **CHAPTER 08 (Gradient Boosting) COMPLETE — merged to `main` via PR #8** (merge commit `4775fe2`,
  `gh pr merge --merge`; per-notebook history preserved; pushed to Ramdam17/QuickIntroToMachineLearning).
  **Six notebooks:** residuals by hand (regression) · the residual *was* the gradient · classification
  (Newton leaf; exp-loss = AdaBoost) · shrinkage & the trees (overfit-at-large-ν; RF contrast) · the
  estimator & early stopping · the California-housing capstone — plus a `style` commit making NB 1–3
  ruff-clean. The **general form** of boosting (AdaBoost = the exp-loss special case); the course's **first
  regression**; the launchpad for ch 09–10. **No `src/` change** across the chapter (reused
  `viz.plot_train_test_curve` / `plot_feature_importances`; `make_friedman1` / `fetch_california_housing`
  direct; pytest stays **20**). Two-reviewer gate + Rémy visual held on every NB; every number re-measured
  on sklearn 1.9.0, seed-pinned; honest findings throughout (the Newton-leaf trap; exp-GB ≠ AdaBoost
  predictor; overfit ν-dependent; OOB noisier than held-out; the dramatic MDI-vs-permutation location
  divergence + the censored $500k cap). NB 6's capstone took a **revise round** (ml-expert BLOCK on the cap
  mechanism + 2 MAJOR → folded → re-reviewed PASS). `main` synced @ `4775fe2`, ruff green, pytest 20.
  STATE → `idle` (this edit + `course_map` §08 heading pending, fold into the ch 09 opening). Next: open
  chapter 09 (XGBoost).
- **NB 6 (the demanding case — California housing, the visualization-first capstone) BUILT & MERGED to
  `chapter/08_GradientBoosting` — Rémy validated visually. CHAPTER 08 COMPLETE on the branch (6/6).**
  28 cells (9 code / 19 md), 7 figures (target histogram with the $500k cap; geographic price map; GB
  early-stopping learning curve; cross-method R²+MAE bars; predicted-vs-actual with the cap wall;
  MAE-by-price-bucket; MDI vs permutation). Real data `fetch_california_housing(as_frame=True)` (20640×8,
  target $100k, $500k cap 4.8%; split 16512/4128, seed 0; fetched direct, no `src/` change). **Anchors
  (sklearn 1.9.0, reproduced exactly): linear 0.594 ($53.5k) / tree(d3) 0.499; GB default 0.777 ($37.4k) →
  early-stop (453) 0.821 ($32.7k); RF 0.798 ($33.6k); HistGBR 0.837 ($31.0k); residual MAE by price
  $24.8k/$35.5k/$54.0k/$74.2k; importances MDI MedInc 0.576 / Lat 0.104 / Lon 0.113 vs permutation Lat
  3.382 / Lon 3.182 / MedInc 0.490 — the dramatic divergence.** Reviewers: **both PASS after a revise
  round** — ml-expert raised a **BLOCK** (the "$500k ceiling" mechanism: a tree-sum is NOT bounded by the
  training max — corrected to **censored labels**; Fig-5 axis widened to show the overshoots) + 2 MAJOR
  (early-stop figure honesty — validation-gain-<-tol, test creeps to ≈0.83, two-fit disclosure; the
  MDI-vs-perm **interaction** reconciliation with NB 5) → all folded → **re-reviewed PASS, no BLOCK**;
  pedagogy confirmed the reconciliation + correlated-pair caveat + the "deep blue" map wording. Guards:
  **0 banned** (JSON scan), hex clean, output-free; nbconvert exit 0 (7 figures / 0 errors); `llms.txt`
  **68**; `common_errors` +3 GB rows (censored cap; MDI-vs-perm dramatic divergence; one-metric-hides-
  segments). **No `src/` change** (`fetch_california_housing` direct; reused `viz.plot_feature_importances`;
  pytest **20**). Rebuilt from `build_ch08_nb6.py` right before `git add` (kernel-drift guard, after Rémy's
  `code .`). **Last NB — chapter 08 complete on the branch; next: close via PR `chapter → main` (`--no-ff`)
  on Rémy's explicit go.**
- **NB 6 (the demanding case — California housing, the visualization-first capstone) OPENED.** Branch
  `notebook/08_GradientBoosting__06_california_housing` off `chapter/08_GradientBoosting` (@ `f583d62`).
  Phase `notebook-plan`: drafting the cell-by-cell plan — the chapter's **capstone** (≥6 figures, ~24–26
  cells a floor): a full honest regression workflow on `fetch_california_housing(as_frame=True)`
  (20640×8, named columns). Look at the data (incl. the $500k target cap) → linear / shallow-tree baselines
  → a tuned GB **with early stopping** → held-out **R² and MAE in dollars** → **residual error analysis**
  (where it errs — high-value / coastal homes?) → cross-method foil (RF-regressor, linear) → a
  **`HistGradientBoostingRegressor` speed/score teaser** → the bridge to **ch 09 XGBoost / ch 10 LightGBM**.
  Anchors being measured at plan time (one-time ~14 MB fetch, visible logging; D2 estimates GBR ≈0.78 →
  early-stop ≈0.82 → HistGBR ≈0.84; RF ≈0.79; MAE ≈$37k). Regression-diagnostics helper added to `src/`
  **only** if 3× reuse emerges (→ pytest 20→21), else notebook-local matplotlib. **Last NB — after it
  ships, close chapter 08 via PR `chapter → main` (`--no-ff`).** **Plan APPROVED by Rémy (via ExitPlanMode,
  2026-06-27) & persisted** (`docs/plans/08_GradientBoosting__06_california_housing.md`) — after a detour
  through Ultraplan (twice; the refined plans were not teleported back, so the approved plan is the local
  draft). Building now from a `build_ch08_nb6.py` scratchpad script (anchors above; ~26 cells, 7 figures).
- **NB 5 (the estimator `GradientBoosting{Regressor,Classifier}` & its parameters — integrative, spine =
  early stopping) BUILT & MERGED to `chapter/08_GradientBoosting` — Rémy validated visually.** 21 cells
  (5 code / 16 md), 3 figures (early stopping: staged test R² vs trees, log-x, stop@142 of 2000 requested;
  subsample sweep train/test R²; MDI vs permutation importances). Regression on the NB-4
  `make_friedman1(2000, noise=1.0, seed 0)`. **Anchors (sklearn 1.9.0, reproduced exactly): early stopping
  2000→142, test R² 0.9299 ≥ full-2000 0.9271 (the full model gently overfits at ν=0.1); subsample 0.5–0.75
  (0.9363 / 0.9359) > full 0.9292; MDI sum x₀–x₄ 0.988 / noise 0.012, MDI & permutation agree on the
  informative ranking; GridSearchCV best = the default {lr0.1,depth3}, tuned sealed-test 0.9292 = default
  (tuning bought nothing); no `staged_score`; HistGB named (max_bins=255, max_leaf_nodes=31,
  early_stopping='auto').** Reviewers **both PASS, no BLOCK** — ml-expert verified the early-stopping
  mechanism in sklearn source + every number live + the ν=1 OOB-vs-test failure; pedagogy confirmed the
  early-stopping spine coheres and every figure read matches the pixels. Folded 3 MINOR/NIT (softened the
  **env-sensitive** seed-drift band — my env 142/163/133, the reviewer's 142/147/132, 133–199 over 6 seeds;
  literature-grounded the OOB caveat, dropping an off-page ν=1 claim; tightened the x₃/ranking wording).
  Guards: **0 banned** (JSON scan), hex clean, output-free; nbconvert exit 0 (3 figures / 0 errors);
  `llms.txt` **67**; `common_errors` +3 GB rows (early stopping; subsample-as-regularizer;
  MDI-vs-permutation on known structure). **No `src/` change** (reused `viz.plot_train_test_curve` +
  `viz.plot_feature_importances`; Fig J notebook-local; pytest **20**). Rebuilt from `build_ch08_nb5.py`
  right before `git add` (kernel-drift guard, after Rémy's `code .`). Next: open & plan NB 6 — the
  California-housing capstone (the last NB; then close the chapter via PR into `main`).
- **NB 5 (the estimator `GradientBoosting{Regressor,Classifier}` & its parameters) OPENED.** Branch
  `notebook/08_GradientBoosting__05_estimator_and_parameters` off `chapter/08_GradientBoosting`
  (@ `8556116`). Phase `notebook-plan`: drafting the cell-by-cell plan — **integrative**, anchored on the
  **early-stopping** story (the principled cure for NB 4's overfit). Covers parity recap (regression exact /
  classification Newton); `loss` names; `subsample` + `oob_improvement_`; early stopping
  (`n_iter_no_change`/`validation_fraction`); `feature_importances_` MDI vs permutation; `max_depth`/
  `max_features` as cross-refs to NB 4; the **no-`staged_score` API trap**; honest `GridSearchCV` → one
  sealed test; `HistGradientBoosting*` named as the fast modern default + the ch 09–10 bridge. Anchors being
  measured at plan time (sklearn 1.9.0, seed 0). On `make_friedman1` (the NB-4 set; continuity; known
  feature importances). No `src/` change expected (pytest 20). **Plan APPROVED by Rémy (via ExitPlanMode,
  2026-06-26) & persisted** (`docs/plans/08_GradientBoosting__05_estimator_and_parameters.md`); building now
  from a `build_ch08_nb5.py` scratchpad script. Anchors: early stopping 2000→142 (R² 0.930 ≥ full 0.927);
  subsample 0.5–0.75 > full; MDI sum x₀–x₄ 0.988 / noise 0.012; GridSearchCV best = default.
- **ch 08 NB 1–3 made ruff-clean (lint debt cleared).** `ruff check .` had flagged 8 pre-existing errors in
  the already-shipped NB 1–3 (`B007` / `E501` ×5 / `B905` ×2; NB 4 was already clean). Fixed in the build
  scripts (behaviour-preserving: `for m`→`for _`, line wraps, `zip(strict=False)`), notebooks rebuilt; ruff
  **All checks passed!**, all three nbconvert exit 0, diff = the 8 lines only. Committed on
  `chapter/08_GradientBoosting`. Repo-wide `ruff check .` now green.
- **NB 4 (shrinkage and the trees — ν, depth, n_estimators; the overfit-at-large-ν headline) BUILT &
  MERGED to `chapter/08_GradientBoosting` — Rémy validated visually.** 20 cells (5 code / 15 md), 3 figures
  (ν×n_estimators test R² vs trees, log-x — ν=1 peaks@18 then sags, ν=0.1 higher+flat, ν=0.01 still
  climbing; the overfit — ν=1 train MSE→0 / test MSE bottoms@18 then rises, + a flat RF reference; depth
  sweep test/train R²). Regression on `make_friedman1(2000, noise=1.0, seed 0)` (train 1400 / test 600; a
  real x₀·x₁ interaction + 5 noise features). One **declared** concept: how GB controls its complexity.
  **Anchors (sklearn 1.9.0, reproduced exactly by the notebook): ν=1.0 best test R² 0.8637@18 → 0.8130@1000
  (train→0 = overfit); ν=0.1 0.9300@308, flat to 0.9282@1000; ν=0.01 0.9213@1000 (climbing). depth1
  0.873/0.905 (no x₀·x₁) → depth2 0.931/0.966 → depth5 0.923/0.998 (memorizing). RF flat 0.858/0.862/0.862.
  By-hand ν: F0=14.13; one tree ν=1→9.63 vs ν=0.1→22.31; trees to train MSE≤2 = 13/12/48/496
  (ν=1/0.5/0.1/0.01).** Reviewers **both PASS, no BLOCK** — ml-expert: "mechanistically correct, reproducible
  to the digit, ν-dependence honest, cited" (verified ν=1 ΔR²=0.051 vs ν=0.1 0.0017; RF tree depth ≈19–21;
  the depth-1 additive ceiling holds to 5000 trees); pedagogy: "the three dials read as one story", every
  figure read exact, prerequisites re-laid. Folded small MINOR/NIT: ν-scope the headline + the
  What-you-built bullet; "depth-3 budget"; mark the 18-tree peak as a test-R² milestone; illustrative
  x₀·x₁ split; a Breiman-1996 note on make_friedman1. Guards: **0 banned** (JSON scan), hex clean,
  output-free; nbconvert exit 0 (3 figures / 0 errors); `llms.txt` **66**; `common_errors` +3 GB rows
  (more-trees-overfit-at-large-ν + the RF contrast; the ν×n_estimators trade-off; depth=interaction-order).
  **No `src/` change** (reused `viz.plot_train_test_curve`; Fig G/H notebook-local; pytest **20**). Rebuilt
  from `build_ch08_nb4.py` right before `git add` (kernel-drift guard, after Rémy's `code .`). **Flagged:**
  `ruff check .` surfaces 8 pre-existing errors in NB 1–3 (NB 4 itself ruff-clean) — see Notes/blockers;
  awaiting Rémy's decision before the chapter PR. Next: open & plan NB 5 (the estimator & its parameters).
- **NB 4 (shrinkage and the trees — ν, depth, n_estimators; the overfit-at-large-ν headline) OPENED.**
  Branch `notebook/08_GradientBoosting__04_shrinkage_and_trees` off `chapter/08_GradientBoosting`
  (@ `61a2b4d`). Phase `notebook-plan`: drafting the cell-by-cell plan — one **declared** concept
  (richer-scope, ch 07-NB 3 precedent): how GB controls its complexity. **Back to regression** on
  `make_friedman1(n=2000, noise=1.0, seed 0)` (train 1400 / test 600; a genuine `sin(π·x₀·x₁)`
  interaction + 5 noise features — the canvas that motivates depth). **Anchors re-measured at plan time
  (sklearn 1.9.0, seed 0):** ν=1.0 best test R² **0.864@18** → **0.813@1000** (test MSE 3.39→4.65 while
  train→0 — the overfit headline; peaks far earlier than the chapter-plan's ~130 estimate); ν=0.1 best
  **0.930@308**, flat to 0.928@1000 (lower floor, **no turn-up within budget**); ν=0.01 **still climbing
  0.921@1000** (underfit in budget). **depth = interaction order:** depth1 R² **0.873** (stumps can't
  represent x₀·x₁) → depth2 **0.931** (the pairwise jump) → depth3 0.929 → depth5 train 0.998 / test
  0.923 (memorizing). **RF contrast** flat R² 0.858/0.862/0.862 (B=50/200/1000) — more trees never hurts
  (each an independent variance-reduction draw). **By-hand ν** (NB1 recap): F0=14.13; one depth-3 tree
  drops train MSE 25.28 → **9.63 at ν=1.0** vs → **22.31 at ν=0.1** (a tenth of the way); trees to train
  MSE≤2 = **13/48/496** for ν=1.0/0.1/0.01 (the trade-off in single numbers). 3 figures (ν×trees test R²;
  ν=1.0 train-vs-test MSE overfit + RF reference; depth sweep). The overfit is **ν-dependent**, the RF
  contrast **mechanistic** — and *the* motivation for NB 5's early stopping. No `src/` change expected
  (notebook-local matplotlib; pytest 20). **Plan APPROVED by Rémy (via ExitPlanMode, 2026-06-26) & persisted**
  (`docs/plans/08_GradientBoosting__04_shrinkage_and_trees.md`); building now from a `build_ch08_nb4.py` scratchpad script.
- **NB 3 (gradient boosting for classification — the added notebook) BUILT & MERGED to
  `chapter/08_GradientBoosting` — Rémy validated visually.** 21 cells (7 code / 14 md), 3 figures
  (boundary sharpening n∈{1,10,50}; train log-loss by-hand-Newton == sklearn vs naive mean-leaf lagging;
  exp-GB vs AdaBoost boundaries). The chapter's **pivotal** NB: swap the loss → classification on ch 07's
  make_moons-0.20. **log-loss → pseudo-residual y − p**, fit a regression tree in log-odds space, with the
  **honest Newton leaf-step** `γ = Σ(y−p)/Σ p(1−p)`. F₀=log-odds(0.5)=0, round-1 residuals ±0.5; by-hand
  Newton **== `GradientBoostingClassifier` decision_function to 3.55e-15** (the chapter's correctness trap
  cleared), test acc **0.9417** (= ch 07 AdaBoost on this split); naive **mean-leaf = a different model**
  (train log-loss 0.035 vs 0.219 — config-dependent, shipped as direction + the machine-precision match).
  **Unifying reveal:** `loss='exponential'` = AdaBoost's *objective* (acc both 0.9417, agreement train
  0.982 / test 0.983, **not** bit-identical — different optimizers) → AdaBoost is the exponential-loss
  member of the GB family (the ch 07 bridge, crossed). Reviewers: **both PASS, no BLOCK** — ml-expert
  read sklearn's `_gb.py` and confirmed the by-hand Newton override mirrors it line-for-line (parity is
  honest machine precision), the gradient/curvature/Newton formula and the exercise-3 inequality all
  verified; pedagogy confirmed "density handled not crammed", the gentle intro + deferred derivation, and
  numpy-as-interface justified. Folded 2 MINOR (text): `p(1−p)` = variance of a **Bernoulli(p)** (not "of
  the prediction"); exercise-3 nudge `p(1−p)≤¼` **and** `¼<1`. Guards: **0 banned** (JSON scan), hex
  clean, output-free; nbconvert exit 0 (0 errors / 3 figures); `llms.txt` **65**; `common_errors` +2 GB
  rows (Newton-leaf-not-mean; exp-GB ≠ AdaBoost predictor). No `src/` change (pytest **20**). Rebuilt from
  `build_ch08_nb3.py` right before `git add` (no editor drift). Next: open & plan NB 4 (shrinkage & the
  trees).
- **NB 3 (gradient boosting for classification — the added notebook) OPENED.** Branch
  `notebook/08_GradientBoosting__03_classification` off `chapter/08_GradientBoosting` (@ `37d0ce8`).
  Phase `notebook-plan`: drafting the cell-by-cell plan — one concept, **swap the loss → classification**
  on ch 07's make_moons-0.20 (continuity + a head-to-head with AdaBoost). Recap ch 03 sigmoid/log-odds;
  **log-loss → pseudo-residual `y − p`**; fit a *regression* tree to it in log-odds space; the **honest
  Newton leaf-step** `Σr/Σp(1−p)` (the chapter's **correctness trap**, D4: by-hand Newton ==
  `GradientBoostingClassifier` to machine precision, naive mean-leaf gives a *different* model — ship the
  direction + the match, pin config & re-measure since the log-loss gap is config-dependent); and the
  **unifying reveal** `loss='exponential'` = AdaBoost's *objective* (identical test acc & ~95% pred
  agreement, **not** bit-identical — different optimizer). Build the mechanism by hand first; full Newton
  derivation in "Going further". Anchors being measured at plan time (sklearn 1.9.0, seed 0). No `src/`
  change expected (notebook-local matplotlib + `plot_decision_boundary`; pytest 20). Next: draft the plan
  → ExitPlanMode for Rémy → on approval persist + build.
- **NB 2 (the residual was the gradient — gradient descent in function space) BUILT & MERGED to
  `chapter/08_GradientBoosting` — Rémy validated visually.** 21 cells (6 code / 15 md), 2 figures (the
  step picture: negative gradient at round 10 + the tree's piecewise-constant approximation; gradient
  descent in function space: the 2-point loss bowl + the boosting trajectory into the minimum | the
  total loss vs trees). One concept: the residual we fit (NB 1) **is the negative gradient** of the
  squared-error loss, so boosting is **gradient descent in function space** (the n predictions are the
  variables; each tree an approximate tree-constrained step, ν the length). Re-illuminates NB 1's loop
  (no new data/estimator); ends with "a different loss → a different gradient → a different residual"
  (abs error → sign; log-loss → y−p forward to NB 3). Anchors: `−∂L/∂F = y−F` finite-diff **8.7e-11**
  (exact identity, quadratic loss → zero truncation); loss `L=½Σ(y−F)²` **30.12→0.44** (= (n/2)·MSE,
  strictly monotone); abs-error gradient = sign(y−F) ∈ {−1,+1}; the "fit the negative gradient" reframe
  == `GradientBoostingRegressor` **2.22e-16** (interpretive, same arithmetic). Reviewers: **both PASS, no
  BLOCK** — ml-expert verified the exact identity / monotone descent / slice legitimacy / parity;
  pedagogy confirmed "ré-illumination, not a rehash" and the gentle parameter→function-space lift.
  Folded 2 MINOR (text only): "about"→"under 1e-10"; an honest half-sentence on the slice trajectory's
  slight overshoot (the tree couples all points). Guards: **0 banned** (JSON scan), hex clean,
  output-free; nbconvert exit 0 (0 errors / 2 figures); `llms.txt` **64**; `common_errors` +2 GB rows
  (where-is-the-gradient; approximate-not-exact gradient descent). No `src/` change (pytest **20**).
  Rebuilt from `build_ch08_nb2.py` right before `git add` (no editor drift this time). Next: open & plan
  NB 3 (gradient boosting for classification — the added notebook).
- **NB 2 (the residual was the gradient — gradient descent in function space) OPENED.** Branch
  `notebook/08_GradientBoosting__02_residual_is_gradient` off `chapter/08_GradientBoosting` (@ `11319dc`).
  Phase `notebook-plan`: drafting the cell-by-cell plan — one concept, **the residual we fit is the
  negative gradient** of the squared-error loss, so NB 1's loop is **gradient descent in function
  space** (the ensemble F is a point in ℝⁿ, each tree an approximate downhill step, ν the step size).
  Re-lay ch 03 NB 4 gradient descent (parameter → function space); recompute NB 1's update as "fit the
  negative gradient" (identical sequence); name the generalisation "a different loss → a different
  gradient → a different residual" (absolute error → sign; sets up NB 3). Stays squared-error
  **regression** on NB 1's 1-D sine. Anchors being measured at plan time (sklearn 1.9.0, seed 0). No
  `src/` change expected (notebook-local matplotlib; pytest 20). Next: draft the plan → ExitPlanMode for
  Rémy → on approval persist + build.
- **NB 1 (boosting as fitting residuals — by hand, regression) BUILT & MERGED to
  `chapter/08_GradientBoosting` — Rémy validated visually.** 21 cells (7 code / 14 md), 3 figures
  (data + the flat F₀=mean; round-1 mechanics [residuals + the depth-2 step | the updated F₁]; the fit
  building up F₀→F₆₀ + the train-MSE-vs-trees curve with the single-tree reference). The chapter's
  **first regression**, re-laid honestly (continuous target, residual, MSE, and the **regression-tree
  leaf = mean** rule). Built **by hand**: F₀=mean → fit `DecisionTreeRegressor(max_depth=2)` to the
  residual → `F += ν·tree` (ν=0.3) → repeat; train MSE 0.502→0.299@1→0.135@3→0.101@4→0.081@5→0.0073@100;
  a single depth-2 tree (0.105) is passed at **round 4**. **Exact parity:** by-hand ==
  `GradientBoostingRegressor(loss='squared_error', subsample=1.0, …)` to **2.22e-16** (final & every
  staged round; verified by ml-expert across 24 configs), F₀=−0.1199=`init_.constant_` (DummyRegressor).
  **"Gradient" is deliberately NOT named** (NB 2's reveal). Reviewers: **both PASS, no BLOCK** —
  ml-expert verified parity + leaf=mean + citations; pedagogy praised the first-regression re-lay
  ("exemplary") and the gradient deferral ("rare skill"). Folded the convergent MINORs: crossover stated
  as **round 4** (round-4 MSE now printed); a one-line honest hook that the train MSE dipping below the
  injected-noise floor is a *training* phenomenon → NB 4; a gloss on `init_`; exercise 3 tied to the
  recap's leaf=mean. Guards: **0 banned** (JSON scan), hex clean, output-free; nbconvert exit 0 (0
  errors / 3 figures); `llms.txt` **63**; `common_errors` +3 GB rows (residuals≠reweighting;
  parity-exact-only-for-squared-error / leaf=mean; the `staged_score`-absent API trap). No `src/` change
  (pytest **20**). Rebuilt from `build_ch08_nb1.py` right before `git add` (kernel-drift habit; also
  `git restore`d a cosmetic editor drift on ch07/05's `language_info`). Next: open & plan NB 2 (the
  residual *was* the gradient).
- **NB 1 (boosting as fitting residuals — by hand, regression) OPENED.** Branch
  `notebook/08_GradientBoosting__01_fitting_residuals` off `chapter/08_GradientBoosting` (@ `0e6059c`).
  Phase `notebook-plan`: drafting the cell-by-cell plan — one concept, **fit a regression tree to the
  residuals of the current model, add a shrunken slice, repeat** (contrasted with AdaBoost's
  reweighting). **First regression in the course** (re-lay it, + the regression-tree-leaf=mean rule as
  the hinge to NB 3's Newton leaf); **"gradient" NOT named yet** (NB 2's reveal). **Anchors measured at
  plan time (sklearn 1.9.0, seed 0):** 1-D synthetic `y = sin(x) + N(0, 0.25²)`, n=120, x∈[0,2π];
  by-hand GB (F₀=mean → fit `DecisionTreeRegressor` to residual → F += ν·tree) **== `GradientBoostingRegressor`
  to 2.22e-16** (final & staged, all depth/ν configs), F₀ = −0.1199 = `init_.constant_`; chosen config
  **max_depth=2, ν=0.3** (visible per-round shrinkage, no train→0 spoiler); train MSE
  0.502→0.299@1→0.191@2→0.135@3→0.081@5→0.039@20→0.0073@100; single depth-2 tree 0.105 (the ensemble
  passes it ≈round 5). API verified: `staged_predict` present, **`staged_score` ABSENT** (use
  `staged_predict`), `init_`=DummyRegressor, default loss `squared_error`. 2 figures (residual-fitting
  story; train-MSE vs trees). No `src/` change expected (notebook-local matplotlib; pytest 20). Next:
  draft the plan → ExitPlanMode for Rémy → on approval persist + build.
- **Chapter 08 (Gradient Boosting) plan APPROVED by Rémy & persisted** (`docs/plans/chapter_08_GradientBoosting.md`,
  this commit). **SIX notebooks** (regression-first + an added classification notebook — Rémy's call;
  the 03_LogisticRegression six-NB precedent): NB 1 residuals by hand (regression; exact by-hand ==
  `GradientBoostingRegressor` to 1e-16) → NB 2 the residual *was* the gradient (gradient descent in
  function space) → NB 3 classification (log-loss, pseudo-residual y−p, the honest Newton leaf-step;
  `loss='exponential'` = AdaBoost's objective — the unifying reveal) → NB 4 ν × depth × n_estimators and
  the overfit-at-large-ν headline (the RF contrast) → NB 5 the estimator
  `GradientBoosting{Regressor,Classifier}` & its parameters (subsample/OOB, early stopping, importances;
  `HistGradientBoosting*` named) → NB 6 demanding case **California housing** (regression capstone,
  visualization-first). The **general form** of boosting (AdaBoost = the exponential-loss special case);
  the bridge ch 07 promised. Reviewer-gated on the live install (sklearn 1.9.0): **pedagogy PASS** (six
  NBs earned; regression-first honest, a coherent journey not whiplash — spine = "the loss is the dial";
  4 MINOR + 1 NIT noted for the NB-plan gates); **ml-expert REVISE → folded** — MAJOR: "reproduces
  AdaBoost" softened to **objective-level only** (GB `loss='exponential'` shares AdaBoost's loss but
  ~95% prediction agreement, identical test acc 0.9417, **not** bit-identical); MAJOR: NB 3 anchors
  re-pinned (moons-0.20 is balanced → round-1 residuals **±0.5000** not ±0.507; the Newton-vs-mean
  log-loss gap is **config-dependent** → ship the *direction* + the machine-precision Newton match, pin
  the config & re-measure at build); MINORs: overfit qualified to **large ν** with the mechanistic RF
  contrast, the **regression-tree-leaf=mean** rule re-laid in NB 1 as the hinge to NB 3's Newton leaf.
  **First regression in the course** (pays off ch 00's promise; by-hand parity exact only in
  regression). API verified: `loss='log_loss'` (`'deviance'` removed), **no `staged_score`** (use
  `staged_predict`), `subsample<1`→`oob_improvement_`, early stopping OFF by default. Capstone
  California housing (GBR ≈0.78 → early-stop ≈0.82 → HistGBR ≈0.84; RF ≈0.79; seed band). No `src/`
  change expected (one conditional `viz.plot_regression_diagnostics` at NB-plan time → pytest 20→21).
  `course_map.md` §08 refined 5→6. Next: open & plan NB 1.
- **Chapter 08 (Gradient Boosting) opened.** Branch `chapter/08_GradientBoosting` created off `main`
  (synced @ `b256580` after PR #7). Phase `chapter-plan`: drafting the chapter plan in plan mode per
  `course_map.md` §08 and the per-method arc — boosting as **fitting the residuals** / gradient descent
  in **function space** (by hand) → the **loss** and the role of **shrinkage** (learning rate) → **trees
  as the base learner** (depth × learning_rate × n_estimators interplay) → **parameters & early
  stopping** (the bias/variance trade-off) → a **demanding case** tuning a competitive tabular model
  honestly. The **general form** of the boosting family: ch 07's AdaBoost is the exponential-loss
  special case; gradient boosting generalises to **any differentiable loss** via gradient descent in
  function space — the bridge stated at the close of ch 07. Built on ch 04's trees (the base learner)
  and ch 07's sequential error-correction; the launchpad for ch 09 XGBoost / ch 10 LightGBM. The pending
  `idle` STATE edit + `course_map.md` §07 → complete were folded into this opening commit (committed on
  the chapter branch, not on protected `main`).
- **CHAPTER 07 (AdaBoost) COMPLETE — merged to `main` via PR #7** (merge commit `b256580`,
  `gh pr merge --merge`; per-notebook history preserved; pushed to
  Ramdam17/QuickIntroToMachineLearning). Five notebooks: reweighting by hand · the additive model &
  exponential loss · learning rate / rounds & overfitting behaviour · the estimator & its parameters · a
  demanding case (spam). The course's first **boosting** method — sequential error-correction, the
  contrast with ch 06's parallel bagging; the base learner of the boosting family (ch 08–10). **No
  `src/` change** across the chapter (reused the `viz` helpers; pytest stays **20**). The two-reviewer
  gate + Rémy's visual validation held on every notebook; every number re-measured on sklearn 1.9.0;
  honest findings surfaced throughout (the SAMME-α/margin-form reconciliation; resistance ≠ immunity;
  the spam noise reversal kept **internal**, not shipped as an RF-vs-AdaBoost law; importance not causal
  / corpus artifacts). Three gate catches fixed: NB 2's reconciliation *reason*, NB 4's transposed
  CV grid (BLOCK), NB 5's breast_cancer cross-reference (MAJOR). `main` synced locally to `b256580`,
  green (pytest 20). STATE set to `idle` (pending edit — folds into the chapter-08 opening). Next:
  chapter `08_GradientBoosting`.
- **NB 5 (a demanding case: spam / spambase — the chapter capstone) BUILT & MERGED to
  `chapter/07_AdaBoost` — Rémy validated visually. CHAPTER 07 built end to end (5/5).** The
  **visualization-first capstone**: 26 cells (8 code / 18 md), **7 figures** (class balance;
  cross-method accuracy; staged resistance; confusion; MDI vs permutation; noise 2-panel [by-hand
  weight-on-flipped + clean-vs-40% staged test]; AdaBoost-vs-RF degradation). On spambase (4601×57,
  openml): **AdaBoost shines** — test **0.949** ≈ RF **0.959** ≫ stump 0.782 (competitive, not a win;
  the tuned SVM led near-linear bc — no universal best); staged **resistance** (test bottoms 0.0485@279,
  flat to 0.0507@400; **train floor 0.045, never 0** — "train→0" is moons-only, real data has
  irreducible overlap); confusion [[810,27],[43,501]] (recall 0.921 / precision 0.949, false-alarm
  asymmetry); importance **MDI vs permutation** = spam-markers (`!`,`$`,`remove`) + **HP-Labs corpus
  artifacts** (`george`,`hp`), not causal; **noise from the inside** (by-hand SAMME: ~21% flipped points
  hoard ~40% of the weight; 40%-noise test bottoms 0.142@39 → climbs 0.177@400 = resistance is not
  immunity); and the **"RF beats AdaBoost under noise" folk claim REVERSES on spam** (AdaBoost more
  robust from 20% up; 40%: 0.823 vs RF 0.704) — shown but **NOT shipped as a law** (dataset-dependent:
  the weak-stump base can't memorize noise, deep RF trees can). Reviewers: **ml-expert PASS** (by-hand
  SAMME == sklearn to 5 dp, all anchors reproduced, 8 DOIs resolved, HP-Labs/`george` provenance
  verified; 1 MINOR + 1 NIT folded); **pedagogy MAJOR → fixed → re-confirmed PASS** — cell-8 "the
  ensembles trailed on breast_cancer" was false (ch 05's spine had no ensembles; 0.965 was the *tuned*
  SVM) → re-anchored to "the tuned SVM led the four methods compared there", with the noise counter-case
  (cell-22) carrying the ensemble-inclusive bc point. Guards: 0 banned (JSON scan), ruff/hex clean,
  output-free; nbconvert exit 0 (0 errors / 7 figures); `llms.txt` **61**; `common_errors` +3 AdaBoost
  rows (shines≠best; noise-not-a-law / mechanism; train-floor / irreducible-error). No `src/` change
  (pytest **20**). Rebuilt from `build_ch07_nb5.py` right before `git add`. **Last NB of chapter 07 —
  next: close the chapter via PR into `main` on Rémy's explicit go.**
- **NB 5 (a demanding case: spambase — the chapter capstone) OPENED.** Branch
  `notebook/07_AdaBoost__05_spambase` off `chapter/07_AdaBoost` (@ `802116b`). Phase `notebook-plan`:
  drafting the cell-by-cell plan (plan mode) — the **visualization-first capstone** (~24–26 cells a
  *floor*, ~7 figures): spambase (UCI, 4601×57, binary), **AdaBoost shines** (test acc ≈0.949 ≈ RF
  0.956, ≫ single stump 0.782; staged test curve resists — bottoms ≈round 280, plateaus, no overfit;
  spam *train* error never reaches 0, floor ≈0.045 — "train→0" is moons-only), and the **label-noise
  weakness framed honestly/internally** — exp-loss non-robustness: by-hand, mislabeled points hoard a
  disproportionate share of weight (~20% of points → ~45%), AdaBoost's **test error rises with rounds**
  on noisy train; the RF comparison is shown but **NOT generalised into a law** (dataset-dependent: RF
  wins on noisy breast_cancer but AdaBoost wins on noisy spam at 40% flip — the reversal). References
  the breast_cancer cross-method spine for continuity. **Anchors being re-measured on sklearn 1.9.0 at
  plan time.** No `src/` change expected (reuse `viz` helpers; pytest 20). Decision B (spambase, ESL ch
  10). **Plan APPROVED by Rémy (via ExitPlanMode) & persisted** (`docs/plans/07_AdaBoost__05_spambase.md`,
  ~27 cells / 7 figures); anchors re-measured (AdaBoost 0.949 ≈ RF 0.959 ≫ stump 0.782; staged clean
  bottoms 0.0485@279, train floor 0.045 never 0; weight-on-flipped 0.21→~0.40; noisy-40% bottoms
  0.142@39→0.177@400; degradation reversal AdaBoost > RF from 20% up); **building now.** **Last NB of
  chapter 07 — after it ships, close the chapter via PR into `main`.**
- **NB 4 (the estimator `AdaBoostClassifier` & its parameters) BUILT & MERGED to `chapter/07_AdaBoost`
  — Rémy validated visually.** 22 cells (9 code / 13 md), 3 figures (boundary stump-base vs depth-3
  base; the `n_estimators × learning_rate` CV heatmap; CV-vs-sealed-test bars, default vs tuned). The
  **integrative** notebook, de-overloaded: parity recap (sklearn `AdaBoost(50)` test **0.9417**,
  `estimator_weights_[:3]` = [1.6796, 1.1338, 1.3854] == by-hand NB 1/2); the **base-learner-strength**
  headline — every base depth memorises (train **1.000**) but TEST falls as the base deepens (stump
  **0.9417** → depth-5 **0.9167**), the **mirror image of a random forest** (boosting wants *weak*
  learners); `n_estimators × learning_rate` 5-fold CV (pandas-pivot grid, rows=lr / cols=n: bottom-left
  n50/lr0.1 **0.911** underfit → broad **0.95–0.96 plateau**, best lr0.5/n400 **0.9607**); **honest
  tuning** default (n50, lr1.0) CV **0.9536** / tuned CV 0.9607 / **both sealed-test 0.9417** (the +0.007
  CV gain did NOT transfer); current-API facts (**`algorithm` REMOVED** — SAMME only; `estimator` not
  `base_estimator`; default base = stump); multiclass SAMME `+ln(K−1)` named; `feature_importances_`
  ~[0.61, 0.39] (MDI, ch 06 caveat restated, honest reading deferred to NB 5). Reviewers: **pedagogy
  PASS** (1 MINOR / 2 NIT, optional); **ml-expert BLOCK → fixed → re-confirmed PASS** — the `n_est×lr` CV
  grid was **transposed** (GridSearchCV orders results by alphabetically-sorted keys; a naive `reshape`
  scrambled 10/12 cells and the heatmap contradicted `best_params_`), fixed via a **pandas pivot** keyed
  on the real param values (heatmap `origin="lower"`), with the consequential default-CV correction
  0.9429 → **0.9536** (`cross_val_score`-confirmed). Guards: 0 banned (JSON scan), ruff/black clean, hex
  clean, output-free; nbconvert exit 0 (0 errors / 3 figures / 8 streams); `llms.txt` **60**;
  `common_errors` gained 2 AdaBoost rows (base-must-stay-weak; the GridSearchCV alphabetical-key
  **transpose trap**). No `src/` change (pytest **20**). Rebuilt from `build_ch07_nb4.py` right before
  `git add` (kernel-drift habit; the display_name was already canonical). Next: open NB 5 (the spambase
  capstone) — the last of ch 07, then close the chapter via PR into `main`.
- **NB 4 BUILT; reviewer gate run; an ml-expert BLOCK found & fixed (awaiting re-confirm + Rémy
  visual).** Built `04_estimator_and_parameters.ipynb` (22 cells, 9 code / 13 md, 3 figs: boundary
  stump-vs-depth3; n_est×lr CV heatmap; CV-vs-test bars). pedagogy **PASS** (1 MINOR/2 NIT, optional).
  ml-expert **BLOCK**: the `n_estimators × learning_rate` CV grid was **transposed** — GridSearchCV
  orders results by *alphabetically-sorted* keys (`learning_rate` outer, `n_estimators` inner), so a
  naive `reshape(n, lr)` scrambled 10/12 cells and the heatmap contradicted the printed `best_params_`.
  (The bug was in the measurement script too — it slipped through.) **Fixed:** grid now built via a
  **pandas pivot** keyed on the real param values (order-independent); heatmap rows=`learning_rate`,
  cols=`n_estimators`, `origin="lower"` (bottom-left n=50/lr=0.1 = 0.911 underfit); the 0.961 best cell
  now sits at (lr=0.5, n=400) = `best_params_`. Consequential correction: **default (n50,lr1.0) CV =
  0.9536** (not 0.9429; `cross_val_score`-confirmed), tuned 0.9607, **both sealed-test 0.9417** — the
  "+0.007 CV gain didn't transfer" lesson, cleaner. Plan doc table + numbers corrected to match (+ a
  note on the transpose trap). Other anchors unchanged & verified (SAMME-only API, parity 0.9417,
  base-strength 0.9417→0.9167, importances ~[0.61,0.39]). Guards green; output-free; nbconvert exit 0.
  ml-expert re-confirming the fix in background (agent `a80d8427987ca88ce`). No `src/` change (pytest
  20). Next: reviewer PASS → Rémy visual → commit + ff-merge → NB 5 (spambase capstone).
- **NB 4 (the estimator `AdaBoostClassifier` & its parameters) OPENED.** Branch
  `notebook/07_AdaBoost__04_estimator_and_parameters` off `chapter/07_AdaBoost` (@ `c493293`). Phase
  `notebook-plan`: drafting the cell-by-cell plan — the **integrative** notebook (de-overloaded): parity
  recap (sklearn == by-hand, NB 1/2); the dials — **`estimator`** (the base-learner-strength knob:
  stump vs deeper → overfits faster), **`n_estimators` × `learning_rate`** interplay (CV heatmap); the
  current-API fact **`algorithm` is REMOVED** (SAMME only); multiclass SAMME named; `feature_importances_`
  (MDI, ch 06 caveat, honest reading deferred to NB 5); honest **`GridSearchCV`** on train → one sealed
  test. Anchors re-measured (moons-0.20): base-strength n=200 all train 1.000, test 0.9417(d1) →
  0.9167(d5); n_est×lr CV best lr0.5/n400 0.9607 vs default 0.9429; **tuned == default on sealed test
  0.9417** (CV gain didn't transfer — honest, echoes ch 06). 3 figures. No `src/` change planned (pytest
  20). **Plan APPROVED by Rémy & persisted** (`docs/plans/07_AdaBoost__04_estimator_and_parameters.md`);
  building now.
- **NB 3 (learning rate, rounds & overfitting behaviour) BUILT & MERGED to `chapter/07_AdaBoost` — Rémy
  validated visually.** 20 cells (5 code / 15 md), 3 figures (clean staged train/test; lr sweep
  {1.0,0.5,0.1}; a 2-panel noise figure — clean-vs-noisy test error on one clean test set + the
  contorted boundary with flipped points ringed). The **richer-scope** NB (Decision A), one declared
  concept: rounds × `learning_rate` & overfitting behaviour. Anchors: `learning_rate` scales α
  (estimator_weights_ = lr·ln((1−ε)/ε) → 1.68/0.84/0.168, **by-hand SAMME matched to 5 dp incl.
  estimator_errors_** — proving the *reweighting* uses ν·α); clean **resistance** train→0 @114, test
  holds 0.04–0.06 (+0.017 drift, margins/Schapire 1998); lr=1 plateaus ~10 rounds vs lr=0.1 ~400; noise
  **overfit** 25% flip test 0.067 @18 → 0.150 @400 (+0.083). **Mid-build correctness fix (measure-first):**
  AdaBoost does NOT memorize the noisy moons (train-vs-noisy floors ~0.21), and train-vs-noisy vs
  test-vs-clean aren't comparable → rebuilt fig C as clean-vs-noisy **test** (comparable), excised the
  false "train→0/memorize". Reviewers **both PASS (no BLOCK/MAJOR)** — ml-expert verified the fix
  complete + the ν-scales-reweighting subtlety + all anchors; folded their convergent MINORs (Fig B
  credits ν=0.5's lowest mid-band; clean +0.017 vs noisy +0.083 quantified; margin = hardest points;
  "stays there" blip noted; exercise-3 in error units; cell-13 ties noise to NB1's exp(α)). Guards: 0
  banned (JSON scan, caught 3 incl. a code comment, fixed), ruff/black clean, hex clean, output-free,
  `llms.txt` **59**; `common_errors` +2 AdaBoost rows ("never overfits" misconception; lr/rounds
  coupling). Canonical nbconvert exec (exit 0); 3 figures eyeballed. No `src/` change (pytest **20**).
  Next: open NB 4 (the estimator & its parameters).
- **NB 3 (learning rate, rounds & overfitting behaviour) OPENED.** Branch
  `notebook/07_AdaBoost__03_learning_rate_overfitting` off `chapter/07_AdaBoost` (@ `b1ae47b`). Phase
  `notebook-plan`: drafting the cell-by-cell plan — the **richer-scope** NB (Decision A), one declared
  concept: *how boosting controls its own complexity — the rounds × learning_rate trade-off and what it
  does to generalization*. Establish `learning_rate` ν by hand (it scales every α: estimator_weights_ =
  lr·ln((1−ε)/ε), measured 1.68/0.84/0.168), then BOTH faces on the **moons-0.20 through-line**:
  clean-data **resistance** (train→0 @ T=114, test bottoms 0.042 @ 35 then holds 0.04–0.06, no runaway —
  margins, Schapire 1998) AND the **noise overfit** (25% train-label flip: test bottoms 0.067 @ 18 →
  climbs 0.150 @ 400 while train still →0 — exp-loss non-robustness, Dietterich 2000); lr sweep
  {1.0,0.5,0.1} (lr=1 plateaus ~10 rounds, lr=0.1 needs ~400). **NB 3 stays all-moons** (2D boundary
  contortion visible; bc gives the same rise +0.088 but no picture; real-data noise deferred to NB 5).
  ~3 figures. No `src/` change planned (pytest 20). **Plan APPROVED by Rémy & persisted**
  (`docs/plans/07_AdaBoost__03_learning_rate_overfitting.md`); building now.
- **NB 2 (weak learners & the additive model) BUILT & MERGED to `chapter/07_AdaBoost` — Rémy validated
  visually.** 21 cells (6 code / 15 md), 3 figures (boundary sharpening triptych T=1/10/50 → an
  axis-aligned **staircase**; exponential loss vs margin, the smooth surrogate; the **L(α) bowl** with
  the minimiser dot at **0.84** and SAMME's dashed line at **1.68**). Built the **additive model**
  `F=sign(Σαₜhₜ)` (the same α plays reweighting *and* vote weight); **derived** α as the exp-loss
  minimiser **½ln((1−ε)/ε)=0.8398** (grid 0.8400, closed-form to 1e-15) by forward stagewise additive
  modelling; multiclass SAMME **+ln(K−1)** verified == sklearn (1.0788, 2e-16); boundary T=1 0.867 →
  T=50 0.942. Reviewers: **pedagogy PASS** (1 MINOR folded — "curve"→"staircase" exercise; hardest-maths
  judged accompanied); **ml-expert REVISE → fixed → re-confirmed PASS** — caught a real **MAJOR**: my
  SAMME-vs-classic reconciliation reached the right conclusion via a *wrong reason* ("a factor on α
  cancels in renorm" — false, α is in the exponent). **Verified** the correct margin-form story
  experimentally (classic reweight exp(−βyh)=exp(−β)·exp(2β·𝟙[miss]); common exp(−β) cancels → SAMME's
  update → identical 50-stump sequence & predictions; the indicator-β hybrid diverges 0.9333). Rewrote
  cell-15 to the margin-form derivation (turning the trap into an explicit learner warning), fixed the
  exercise-3 hint + the plan doc. Guards: 0 banned (JSON scan), ruff/black clean, hex clean,
  output-free, `llms.txt` **58**; `common_errors` gained 2 AdaBoost rows (SAMME/margin-form; α is
  derived/surrogate). Canonical nbconvert exec (exit 0); 3 figures eyeballed. No `src/` change (pytest
  **20**). Next: open NB 3 (learning rate, rounds & overfitting behaviour).
- **NB 2 (weak learners & the additive model) OPENED.** Branch `notebook/07_AdaBoost__02_additive_model`
  off `chapter/07_AdaBoost` (@ `e03be0b`). Phase `notebook-plan`: drafting the cell-by-cell plan — one
  concept, the **additive model** `F(x)=sign(Σ αₜ hₜ(x))` and **where α comes from**: the reveal that
  NB 1's reweighting α is the vote weight; weak learner = better than chance; the statistical view
  taught from scratch (re-lay ch 03 log-loss → **exponential loss** as a picture → forward-stagewise in
  words → grid-verify the minimiser). Anchors re-measured (moons-0.20, sklearn 1.9.0): boundary sharpens
  T=1 0.8667 / T=10 0.9417 / T=50 0.9417 (train 0.986); train err → **0 @ T=114**; exp-loss minimised at
  **α\*=½ln((1−ε)/ε)=0.8398** (grid argmin 0.8400, closed-form to 1e-15), SAMME α=1.6796=**2α\*** (same
  classifier — scale-invariance); multiclass K=3 by-hand `ln((1−ε)/ε)+ln(K−1)=1.0788` == sklearn (diff
  2e-16). 3 figures. No `src/` change planned (pytest 20). **Plan APPROVED by Rémy & persisted**
  (`docs/plans/07_AdaBoost__02_additive_model.md`); building now.
- **NB 1 (boosting intuition: reweighting by hand) BUILT & MERGED to `chapter/07_AdaBoost` — Rémy
  validated visually.** 24 cells (9 code / 15 md), 4 figures (training scatter; the weak stump's single
  cut, test **0.8667**; the 3-panel reweighting story — point size ∝ weight, the cut migrating round to
  round; the running-ensemble train-error curve 0.157 → **0.0143** @ 50). Built by hand: uniform
  weights → stump on weighted data → ε₁ **0.157** → **α = ln((1−ε)/ε) = 1.680** → reweight
  wᵢ·exp(α·𝟙[miss]) (the 44 misses' weight **0.157 → 0.500**, the SAMME "misclassified mass → ½"
  identity) → repeat; round-2 ε **0.244** > round-1 (the reshaped problem is deliberately harder).
  **Parity exact:** by-hand α == sklearn `estimator_weights_` (max diff **1.1e-15**), test acc
  **0.9417** both, predictions 120/120 identical. Reviewers **both PASS (no BLOCK)** — ml-expert
  verified parity + the ½-mass identity + every anchor; pedagogy confirmed one-concept + charter +
  figures-match-reads + answerable tiers. Folded 4 MINORs (train error "falls overall, non-monotone"
  not "round by round"; "about 1.4%" not "past 1%"; "~84%" ceiling framing; cell-4 "separate *without
  error*"; a comment flagging the running α-vote is NB 2's concept) + corrected the chapter-plan
  monotonicity anchor. Guards: 0 banned (JSON scan), ruff/black clean, hex clean, output-free,
  `llms.txt` **57**; `common_errors` gained 3 AdaBoost rows (bagging≠boosting; ε can rise by design;
  non-monotone train curve). Canonical nbconvert exec (exit 0); 4 figures eyeballed. No `src/` change
  (pytest **20**). Next: open NB 2 (weak learners & the additive model).
- **NB 1 (boosting intuition: reweighting by hand) OPENED.** Branch
  `notebook/07_AdaBoost__01_reweighting_by_hand` off `chapter/07_AdaBoost` (@ `08b5162`). Phase
  `notebook-plan`: drafting the cell-by-cell plan — one concept, the AdaBoost **reweighting loop by
  hand** on moons-0.20 (n_train 280): a single stump is weak (test **0.8667**, one cut); uniform
  weights → fit stump on weighted data → weighted error ε → learner weight **α=ln((1−ε)/ε)** →
  up-weight misclassified (wᵢ←wᵢ·exp(α·𝟙[miss]), renormalise) → repeat; the running weighted vote
  drives **train error 0.157→0.014 (T=50)**; **by-hand α == sklearn `estimator_weights_`** (max diff
  **1e-15**) and staged test acc **0.9417** both. Anchors re-measured at plan time, seed pinned
  (ε₁ 0.157 / α₁ 1.680; round-2 ε 0.244 > round-1 = the next problem is deliberately harder; weight on
  round-1's 44 misses jumps 0.157 → **0.500** after one reweight). Contrast ch 06 bagging
  (parallel/independent/equal vote) vs boosting (sequential/adaptive). **Plan APPROVED by Rémy &
  persisted** (`docs/plans/07_AdaBoost__01_reweighting_by_hand.md`); building now.
- **Chapter 07 (AdaBoost) plan APPROVED by Rémy & persisted** (`docs/plans/chapter_07_AdaBoost.md`,
  this commit). **FIVE notebooks** (standard arc), the first **boosting** method: NB 1 reweighting by
  hand (SAMME α=ln((1−ε)/ε); by-hand == sklearn `estimator_weights_` to 4 dp on moons-0.20) → NB 2 the
  additive model `F=sign(Σαₜhₜ)` + the exponential-loss / forward-stagewise view taught from scratch →
  NB 3 (**richer scope**, Decision A) rounds × learning-rate & overfitting behaviour (clean-data
  resistance AND the noise overfit) → NB 4 the estimator `AdaBoostClassifier` (`algorithm` REMOVED —
  SAMME only; the base must stay weak) → NB 5 demanding case **spambase** (Decision B; ESL ch 10):
  shines (≈0.949 ≈ RF 0.956), and where noise hurts framed **honestly/internally** (exp-loss
  non-robustness: mislabeled points hoard ~45 % of weight → test error rises with rounds — NOT the
  dataset-dependent "RF beats AdaBoost", which reverses on spam). API + all anchors re-measured on
  sklearn 1.9.0 (`estimator` not `base_estimator`; `algorithm` removed; default base = stump).
  Reviewer-gated, both **REVISE → all folded** (no BLOCK): **ml-expert** (MAJOR — "spam still improving
  at 400 rounds" overstated → it plateaus after ≈ round 280; MINORs — NB 1 moons-0.20/0.30 anchors
  disentangled, the α factor-2 rescale guard, spambase UCI DOI) — verdict the honesty framing "could
  not be broken", every reconciliation verified to 4 dp on the live install (incl. multiclass
  `+ln(K−1)`); **pedagogy** (MAJORs — NB 2's exp-loss derivation budgeted as taught-from-scratch with a
  ch 03 log-loss re-lay; NB 3 richer-scope declared as ONE concept with ν taught by hand before the lr
  sweep; NB 5 "Your turn" tiers sketched; MINORs — cross-method-spine continuity note, NB 1 ½-remark
  trimmed & scale-invariance deferred to NB 2). **No `src/` change** (reuse `viz.plot_decision_boundary`
  / `plot_train_test_curve` / `plot_confusion_matrix` / `plot_class_balance` / `plot_feature_importances`;
  pytest stays 20). `course_map.md` §07 annotated. Next: open NB 1.
- **Chapter 07 (AdaBoost) opened.** Branch `chapter/07_AdaBoost` created off `main` (synced @ `9f18507`
  after PR #6). Phase `chapter-plan`: drafting the chapter plan in plan mode per `course_map.md` §07 and
  the per-method arc — boosting = **focus on the mistakes** (reweight misclassified points, by hand) →
  weak learners (decision stumps) & the **additive model** (SAMME for multiclass) → **learning rate vs
  number of rounds** and overfitting behaviour → the estimator `AdaBoostClassifier` & its parameters
  (`n_estimators`, `learning_rate`, `estimator`) → a demanding case (where AdaBoost shines, and where
  **label noise** hurts it). The first **boosting** method — *sequential* error-correction, the direct
  contrast with ch 06's *parallel* bagging; built on ch 04's decision stumps as weak learners. The
  pending `idle` STATE edit was folded into this transition (committed on the chapter branch, not on
  protected `main`).
- **CHAPTER 06 (Random Forests) COMPLETE — merged to `main` via PR #6** (merge commit `9f18507`,
  `gh pr merge --merge`; per-notebook history preserved; pushed to Ramdam17/QuickIntroToMachineLearning).
  Five notebooks: averaging cuts variance (bagging) · the "random" — decorrelating the trees · out-of-bag
  estimation · the estimator & its parameters · a demanding case (covtype). The course's **first
  ensemble** method and the base learner for the boosting family (ch 07–10). **`src/` add:**
  `viz.plot_feature_importances` + smoke test (pytest 19 → 20). The two-reviewer gate + Rémy's visual
  validation held on every notebook; every number re-measured on sklearn 1.9.0 with each RF
  `random_state`-pinned; honest findings surfaced throughout (the ρσ² floor; OOB mildly optimistic;
  `max_features` forgiving; the covtype forest wins where breast_cancer's SVM did, the reversal stated;
  the cell-18 fabricated cross-reference caught by both reviewers and corrected). `main` synced locally
  to `9f18507`, green (pytest 20). STATE set to `idle` (pending edit, folds into the chapter-07
  opening). Next: chapter `07_AdaBoost`.
- **NB 5 (the demanding case — covtype) BUILT & MERGED to `chapter/06_RandomForest` — Rémy validated
  visually. CHAPTER 06 COMPLETE (5/5).** The **visualization-first capstone**: 25 cells (8 code / 17 md),
  7 figures (class balance; cross-method accuracy; aggregate metrics; per-class recall; 7×7 confusion;
  MDI vs permutation importance; fit-time vs n). On covtype (30k stratified subsample, 7 classes, 54
  features): **the forest wins** RF **0.844** / OOB **0.846** ≫ tree 0.770 ≫ LogReg 0.729 (+11 pts —
  the **reverse of breast_cancer**, where RF < SVM); **honest eval under imbalance** (accuracy 0.844 /
  weighted-F1 0.840 hide it, **macro-F1 0.737** reveals it; per-class recall **Aspen 0.279**; confusion
  shows Aspen→Lodgepole); **importance honestly** (Elevation MDI **0.233** ≈ perm **0.270** agree on
  rank; 40 one-hot Soil_* diluted, **combined 0.141/0.112** = 2nd-largest signal; permutation **put to
  work**); **fit-time ≈ n^0.99** (vs ch 05's SVM n^1.6 reference). Reviewers: **both REVISE → folded**
  (shared **MAJOR** — cell-18 wrongly claimed NB 4 *measured* an MDI-vs-permutation disagreement; NB 4
  only *named* permutation → reframed to NB 4's true MDI single-tree-spike→forest-spread story; MINORs
  — Soil group is 2nd not 3rd largest, "no soil col high" qualified for the perm panel, MDI/perm
  different scales → agree on *ranking*; added a "Going further" section). **No `src/` change**
  (`fetch_covtype` direct, names already descriptive, INFO logging shown; pytest **20**). Guards: 0
  banned (JSON scan), ruff/black clean, hex clean, output-free, `llms.txt` 55; `common_errors` gained 3
  rows (imbalance accuracy trap; one-hot dilution; no-universal-best). Canonical nbconvert exec (exit
  0); all 7 figures eyeballed. **Last NB of the chapter — next: close CHAPTER 06 via PR into `main`.**
- **NB 5 (the demanding case — covtype, the chapter capstone) OPENED.** Branch
  `notebook/06_RandomForest__05_covtype_strong_baseline` off `chapter/06_RandomForest` (@ `93857e1`).
  Phase `notebook-plan`: drafting the cell-by-cell plan (plan mode) — the **visualization-first
  capstone** (~24–26 cells a *floor*, figures may exceed six): forest cover type (`fetch_covtype`,
  30 000-row stratified subsample, 7 classes, 54 features); **the forest wins** (RF ≈ 0.846 ≫ tree ≈
  0.775 ≫ LogReg ≈ 0.728 — the reverse of breast_cancer); **honest eval under imbalance** (macro vs
  weighted re-laid; accuracy vs macro-F1 ≈ 0.733; per-class recall incl. Aspen ≈ 0.28; 7×7 confusion);
  **importance honestly** (Elevation dominates, MDI ≈ perm; 40 one-hot Soil_* diluted to ≈ 0.140;
  **permutation put to work**, NB 4's promise); OOB ≈ test at scale; **RF fit-time ≈ linear in n** (the
  counterpoint to ch 05's SVM n^1.6 wall); boosting bridge (ch 07–10). One-time ≈14 MB covtype fetch
  (visible INFO logging). Anchors **re-measured at plan time** on sklearn 1.9.0, every RF
  `random_state`-pinned (RF 0.844/OOB 0.846 ≫ tree 0.770 ≫ LogReg 0.729; macro-F1 0.737 vs accuracy
  0.844; Aspen recall 0.279; Elevation MDI 0.233 ≈ perm 0.270, 40 Soil one-hot 0.141/0.112; fit-time
  n^0.99). Build decisions: `fetch_covtype` direct (no loader/test, pytest stays 20); cross-method on
  fixed defaults + OOB (no test-set tuning). Plan **APPROVED** by Rémy & persisted
  (`docs/plans/06_RandomForest__05_covtype_strong_baseline.md`); building now.
  **Last NB of chapter 06 — after it ships, close the chapter via PR into `main`.**
- **NB 4 (the estimator `RandomForestClassifier` & its parameters) BUILT & MERGED to
  `chapter/06_RandomForest` — Rémy validated visually.** 24 cells (7 code / 17 md), 3 figures (OOB &
  test error vs `n_estimators`; single-tree vs forest test vs `max_depth`; single-tree MDI spike vs
  forest MDI spread). The integrative notebook: **honest parity** hand-bag **0.9357** ==
  `RF(max_features=None)` 0.9357 (accuracy match at B=200, tie-break-sensitive — rs=0 fixed gives
  0.9357, rs=b gives 0.9415; framed as "not a tree-for-tree clone"), `RF(sqrt)` **0.9415**;
  **`n_estimators`** OOB-err 0.271→0.040, **never overfits** (sklearn warns ≤10, surfaced not
  silenced); **`max_features`** OOB/CV **flat** 0.947–0.962 → forest is *forgiving*, `'sqrt'` robust,
  mechanism = NB 2's ρ (no test-acc ranking); **`max_depth`** single-tree test wobble 0.86–0.918 vs
  forest 0.918→0.942 plateau (both train **1.000**), run-to-run std **0.0163 vs 0.0043** (≈4×);
  knobs `bootstrap`/`class_weight`/`n_jobs` named; **feature importance** single-tree spike **0.740**
  vs forest peak **0.146** (spread over the correlated group; Strobl bias + dilution caveat;
  **permutation named** → NB 5); **`GridSearchCV`** `{None,'log2',1}` CV 0.957 → test **0.947** vs
  default 0.955/0.942 (**tuning barely beats the default**). Reviewers: **both PASS** (no BLOCK/MAJOR);
  ml-expert re-verified parity across 3 splits/2 B-values (diverges at B=50, converges at B=200) and
  endorsed deferring permutation to NB 5; folded 3 MINOR/nit (Fig C shared-x-scale note; cell-13
  train=1.000 print added; "variance" not "spread" for σ²/B). **`src/` add:**
  `viz.plot_feature_importances` + smoke test → **pytest 20**. Guards: 0 banned (JSON scan), ruff/black
  clean, hex clean, output-free, `llms.txt` 54. `common_errors` gained 3 RF rows (deep-on-purpose;
  forgiving/tuning; importance spread). Canonical nbconvert exec (exit 0); all 3 figures eyeballed.
  Next: open NB 5 (demanding case — covtype), the chapter capstone.
- **NB 4 (the estimator `RandomForestClassifier` & its parameters) OPENED.** Branch
  `notebook/06_RandomForest__04_estimator_and_parameters` off `chapter/06_RandomForest` (@ `4bb235a`).
  Phase `notebook-plan`: drafting the cell-by-cell plan (plan mode) — the **integrative** notebook
  (~22–24 cells, soft ceiling, ch 04's de-overload lesson): **honest parity first** (hand-bag (NB 1)
  == `RF(max_features=None)`; RF = that **plus** per-split subsampling, NB 2); then the knobs —
  **`n_estimators`** (OOB/test diminishing returns, never systematically overfits), **`max_features`**
  the central dial (NB 2's ρ trend as the hyperparameter, `'sqrt'` default), **`max_depth`/
  `min_samples_leaf`** (RF grows deep and tolerates it), **`bootstrap`/`class_weight`/`n_jobs`** named
  lightly; **feature importance introduced** (MDI over the forest **spreads** vs the single tree's
  ≈0.8 spike — leader read at build; bias caveat restated; **permutation named** → honest reading in
  NB 5); **`GridSearchCV` honest tuning** on TRAIN → one sealed test. `src/` add
  `viz.plot_feature_importances` (+ smoke test → pytest 19→20) **approved** (the recommended option).
  Anchors **re-measured at plan time** on sklearn 1.9.0, every RF `random_state`-pinned (parity
  hand-bag 0.9357 == RF(mf=None) 0.9357 / RF(sqrt) 0.9415; n_estimators OOB-err 0.271→0.040 no
  overfit, warns ≤10; max_features OOB/CV **flat** 0.947–0.962 = forgiving; max_depth single-tree
  wobble vs forest 0.918→0.942 plateau, run-to-run std 0.0163 vs 0.0043; MDI peak 0.146 vs single-tree
  0.740; GridSearch {'log2',1,None} CV 0.957 → test 0.947 vs default 0.955/0.942; raw==std 1.000). Plan
  **APPROVED** by Rémy & persisted (`docs/plans/06_RandomForest__04_estimator_and_parameters.md`);
  building now.
- **NB 3 (out-of-bag estimation) BUILT & MERGED to `chapter/06_RandomForest` — Rémy validated
  visually.** 20 cells (6 code / 14 md), 2 figures (in-bag/OOB schematic; OOB-error vs test-error vs
  `n_estimators`). One concept: OOB = the bootstrap's free validation set. Derived `(1−1/n)ⁿ → 1/e`
  (0.367 at n=398) + measured (0.368); **built the OOB vote by hand** (0.962, ~73 graders/point, 398/398
  covered); parity sklearn `oob_score_` **0.955** ≈ hand; OOB ≈ **sealed test 0.942**, mildly optimistic
  (~1–2 pts, parallel not converging); OOB unreliable < ~25 trees (sklearn **warns**, let through;
  P(never OOB)=0.63³≈0.25). Reviewers: **pedagogy PASS**; **ml-expert REVISE → folded** (MAJOR — the
  hand-vs-sklearn gap was wrongly blamed on hard-vs-soft vote; re-measured soft==hard (saturated leaf
  probs) → corrected to RNG (different bootstrap draws); MINORs — optimism quantified, n=10 0.349,
  `np.add.at` glossed). Guards: 0 banned, ruff/hex clean, output-free, `pytest` 19 (no `src/` change),
  `llms.txt` regenerated; `common_errors` gained an OOB row. Canonical nbconvert exec (exit 0); both
  figures eyeballed. Next: open NB 4 (the estimator & its parameters).
- **NB 3 (out-of-bag estimation) OPENED.** Branch `notebook/06_RandomForest__03_out_of_bag` off
  `chapter/06_RandomForest` (@ `1789474`). Phase `notebook-plan`: drafting the cell-by-cell plan (plan
  mode) — one concept, **OOB**: each bootstrap omits ~1/e ≈ 37 % of points (derive + measure); the trees
  that did not see a point grade it → the forest scores itself for free; **build the OOB vote by hand**
  and match sklearn `oob_score_` (parity); OOB ≈ sealed test (≈0.96 vs ≈0.94, mildly optimistic); OOB
  unreliable with too few trees (sklearn warns); OOB-error vs `n_estimators` → test error. Anchors
  re-measured at plan time, `random_state` pinned. Plan **APPROVED** by Rémy & persisted
  (`docs/plans/06_RandomForest__03_out_of_bag.md`); building now.
- **NB 2 (the "random" in the forest: decorrelating the trees) BUILT & MERGED to
  `chapter/06_RandomForest` — Rémy validated visually.** 22 cells (7 code / 15 md), 2 figures (ρ vs
  `max_features` rising→saturating; ensemble-CV vs mean-individual-tree across `max_features`). One
  concept: feature subsampling decorrelates the trees. On breast_cancer, ρ **0.822 → 0.797** (robust on
  every seed) at **equal individual-tree accuracy** (0.910 ≈ 0.909); the **Var = ρσ² + (1−ρ)σ²/B** law
  **derived from scratch** + Monte-Carlo-verified (the ρσ² floor); `max_features` the decorrelation dial
  (ρ 0.70→0.82, saturating); moons puzzle resolved (RF sqrt 0.900 < bag 0.933 on 2 features). Reviewers:
  **pedagogy PASS**; **ml-expert REVISE → folded** (MAJOR — the gem's CV gain 0.947→0.957 is seed-fragile
  (flips on 2/6 seeds) → re-anchored on the robust ρ-drop + individual-tree equality, *by elimination*,
  CV framed within the ±0.01 seed band; MINORs — ρ "saturates" not "monotone", ρ = proxy for
  error-correlation, cell-12↔16 fence, moons reframed, exercise-2 enriched). Guards: 0 banned, ruff
  clean, hex clean, output-free, `pytest` 19 (no `src/` change), `llms.txt` regenerated; `common_errors`
  gained two rows (the ρσ² floor; subsampling needs many features). Canonical nbconvert exec (exit 0);
  both figures eyeballed. Next: open NB 3 (out-of-bag estimation).
- **NB 2 (the "random" in the forest: decorrelating the trees) OPENED.** Branch
  `notebook/06_RandomForest__02_decorrelating_trees` off `chapter/06_RandomForest` (@ `065c84f`).
  Phase `notebook-plan`: drafting the cell-by-cell plan (plan mode) — one concept, **feature
  subsampling decorrelates the trees**: on breast_cancer, ρ (pairwise tree correlation) drops 0.82→0.80,
  the ensemble rises (CV 0.945→0.955) while individual trees stay equal (the gain is decorrelation);
  the **Var = ρσ² + (1−ρ)σ²/B** law derived from scratch (the ρσ² floor bagging cannot pass); `max_features`
  the decorrelation dial (ρ monotone 0.70→0.82); resolves NB 1's moons puzzle. Anchors re-measured at
  plan time, `random_state` pinned. Plan **APPROVED** by Rémy & persisted
  (`docs/plans/06_RandomForest__02_decorrelating_trees.md`); building now.
- **NB 1 (the wisdom of trees: averaging cuts variance / bagging) BUILT & MERGED to
  `chapter/06_RandomForest` — Rémy validated visually.** 22 cells (7 code / 15 md), 2 figures (five
  jagged single bootstrap-tree boundaries vs the smooth bagged-100 boundary; test-accuracy & run-to-run
  std vs number of trees). Built entirely by hand: a single deep tree is high-variance (test **0.878**,
  bootstrap std **0.031**) → bootstrap (the ~37 % left-out fraction, n=10 → 0.349 vs the n→∞ limit
  1/e ≈ 0.368) → majority vote (`HandBag` estimator) → **0.933**, run-to-run std **0.0465→0.0053
  (÷8.8)**; honest parity hand-bag(200) == `RandomForestClassifier(max_features=None)` = **0.9333**,
  `RF(default sqrt)` **0.900** a deliberate hook for NB 2. Reviewers: **pedagogy PASS** ("cleanest
  concept-boundary I've reviewed in this course"); **ml-expert REVISE → folded** (MAJOR — the honest
  anchor "averaging cancels variance, not bias" was missing → added cell 16; MINORs — empirical-vs-
  formula n=10 wording, std-non-monotone clause, even-B tie comment, ch 04 back-refs corrected to
  NB 4/5). Guards: 0 banned (JSON scan), ruff clean, hex clean, output-free, `pytest` 19 (no `src/`
  change), `llms.txt` regenerated; `common_errors` gained a "more trees ≠ better / variance-not-bias"
  row. Canonical nbconvert exec end-to-end (exit 0); both figures eyeballed. Next: open NB 2
  (decorrelating the trees).
- **NB 1 (the wisdom of trees: averaging cuts variance / bagging) OPENED.** Branch
  `notebook/06_RandomForest__01_averaging_cuts_variance` off `chapter/06_RandomForest` (@ `413cc4a`).
  Phase `notebook-plan`: drafting the cell-by-cell plan (plan mode) — one concept, **bagging by hand
  on `make_moons`**: a single deep tree is high-variance (test 0.878, bootstrap std 0.031) → bootstrap
  resampling + majority vote → 0.933, run-to-run std ÷9 (0.0465→0.0053); the σ²/B variance-reduction
  intuition; honest parity **hand-bag == `RF(max_features=None)`** (0.9333), with `RF(default sqrt)`
  0.900 a deliberate hook for NB 2. ~22 cells, 2 figures (single jagged trees vs the smooth averaged
  boundary; test-acc & run-to-run std vs B). No `src/` change (pytest stays 19). Plan **APPROVED** by
  Rémy & persisted (`docs/plans/06_RandomForest__01_averaging_cuts_variance.md`); building now.
- **Chapter 06 (Random Forests) plan APPROVED & persisted** (`docs/plans/chapter_06_RandomForest.md`,
  this commit). **FIVE notebooks** (standard arc): NB 1 averaging cuts variance / bagging (by hand on
  `make_moons`: single tree 0.878 → vote 0.933, run-to-run std ÷9; hand-bag == `RF(max_features=None)`)
  → NB 2 decorrelating the trees (feature subsampling on breast_cancer: ρ 0.82→0.80, ensemble
  0.924→0.95 while individual trees stay equal; the **Var = ρσ² + (1−ρ)σ²/B** law derived from scratch)
  → NB 3 out-of-bag estimation (the ~1/e left out per tree = free validation; OOB ≈ test) → NB 4 the
  estimator `RandomForestClassifier` & its parameters (`n_estimators` diminishing returns, `max_features`
  the decorrelation dial, depth; feature importance introduced) → NB 5 demanding case **covtype** (forest
  cover type — the forest **wins** on non-linear: RF 0.846 ≫ LogReg 0.728; honest eval under imbalance
  macro-F1 0.733; reading importances honestly: Elevation 0.231 vs 40 one-hot Soil cols combined 0.140;
  RF scales ~linearly vs ch 05's SVM n^1.6). First **ensemble** method; base learner of ch 07–10.
  **Refinement of `course_map.md` §06:** NB 3 = OOB only, feature importance → NB 4 (intro) + NB 5
  (honest reading), mirroring ch 04's importance arc; §06 aligned. Reviewer-gated, both **REVISE → all
  folded** (every number re-measured on sklearn 1.9.0): **ml-expert** (MAJORs — SVM foil `n^1.67`→`n^1.6`
  matching shipped ch 05; RF scaling `n^1.18`→"roughly linear ≈ n^1.0–1.2"; `max_features` decorrelation
  headline now the **monotone ρ trend**, not the seed-fragile per-mf test ranking; MINORs — MDI leader
  read at build not hard-coded, RF `random_state` pinned, Aspen n, covtype cache ≈ 14 MB) — praised the
  ρ-law (Monte-Carlo verified), the decorrelation gem, the exact OOB fraction, the covtype section
  reproducing to three decimals, the honest reversal. **pedagogy** (MAJORs — the ch 04→NB 1 bridge
  conflated two datasets, now states the breast_cancer-hand-bag/moons-variance split plainly; the
  variance law is now **derived** before the NB 2 exercise leans on it; MINORs — NB 5 cell count a
  *floor*, "clearly wins" softened, macro-vs-weighted re-laid in NB 5) — praised the first-contact
  fencing, NB 1 vs NB 2 distinctness, the sound NB 3 refinement. **Rémy chose covtype for NB 5.** **No
  `src/` change forced** (`viz.plot_feature_importances` possible at NB 4, → pytest 19→20). Next: open NB 1.
- **Chapter 06 (Random Forests) opened.** Branch `chapter/06_RandomForest` created off `main` (synced
  @ `b5c00f7` after PR #5). Phase `chapter-plan`: drafting the chapter plan in plan mode per
  `course_map.md` §06 and the per-method arc (why averaging many trees reduces variance — bagging, by
  hand → bootstrap samples + feature subsampling that decorrelate the trees → out-of-bag estimation &
  feature-importance caveats → parameters `n_estimators`/`max_features`/depth, diminishing returns →
  demanding case: a strong tabular baseline, reading importances honestly). The first **ensemble**
  method and the direct answer to the single tree's variance (ch 04 NB 5's hand-bagged 25-tree bar was
  a first taste). The pending `idle` STATE edit was folded into this transition (committed on the
  chapter branch, not on protected `main`).
- **CHAPTER 05 (Support Vector Machines) COMPLETE — merged to `main` via PR #5** (merge commit
  `b5c00f7`, `gh pr merge --merge`; per-notebook history preserved; pushed to
  Ramdam17/QuickIntroToMachineLearning). Five notebooks: the maximum margin · the soft margin & cost C ·
  the kernel trick · the estimator `SVC` & its parameters · breast_cancer (scaling, limits). The first
  **margin-based** method and the home of the **kernel trick**. **`src/` add:** `viz.plot_svm_decision`
  + 2 tests (pytest 17 → 19). The two-reviewer gate + Rémy's visual validation held on every notebook;
  every number re-measured on sklearn 1.9.0; honest findings surfaced (the threshold cannot rescue a
  confident miss; the measured large-`n` ceiling). `main` synced locally to `b5c00f7`, green (pytest
  19). STATE set to `idle` (pending edit, folds into the chapter-06 opening). Next: chapter
  `06_RandomForest`.
- **NB 5 (the demanding case: breast cancer) BUILT & MERGED to `chapter/05_SVM` — Rémy validated
  visually. Part of CHAPTER 05, merged to `main` via PR #5.** The chapter **capstone**,
  visualization-first: 26 cells,
  6 figures (class balance; raw-vs-std scaling bar; `C × gamma` heatmap; cross-method spine bar;
  confusion; fit-time-vs-`n` curve). Scaling raw CV 0.9095 → std 0.9648; GridSearch `{C100,γ0.001,rbf}`
  CV 0.982 / sealed test 0.9649 / 42 SVs; spine KNN 0.9415 / tree 0.9064 / LogReg 0.9532 / **SVM
  0.9649**; confusion `[[104,3],[3,61]]` recall 0.953; **honest threshold surprise** (the 3 misses sit
  at calibrated proba 0.06/0.13/0.19 — confidently wrong; lowering the cut only adds false alarms);
  measured fit-time ≈n^1.67 (worst case O(n³)), 2.68 s vs LinearSVC 0.018 s at n=32 000. Reviewers:
  **both PASS**; ml-expert 3 MINOR folded (the ch-03 contrast made precise — the lever reaches
  *borderline* misses, not confident ones, in either model; "lowering only adds positives" stated;
  exponent flagged this-run), pedagogy 2 MINOR (course_map §05 → mark complete at chapter close;
  Going-further optional, omitted). Guards: 0 banned, ruff/hex clean, output-free, `llms.txt` 49. No
  `src/` change (pytest 19). **Last NB of chapter 05.** Next: Rémy visual → commit + merge → PR to `main`.
- **NB 4 (the estimator `SVC` & its parameters) BUILT & MERGED to `chapter/05_SVM` — Rémy validated
  visually.** 21 cells (≤24 ceiling), 4 figures (the `C × gamma` CV heatmap;
  the gamma boundary grid under→good→over with SV counts 167/88/163; the OvO 3-class regions;
  calibration reliability). Parity `SVC(linear,C=1e6)` == NB-1 (‖w‖ 1.1612, SVs [23,26]); OvO
  penguins_full 3 pairwise / CV 0.956 / decision_function `(5,3)`; GridSearch best `{C=10,γ=1}` CV 0.919
  / sealed test 0.944. Reviewers: **pedagogy PASS** (cell budget exemplary; 2 MINOR); **ml-expert
  REVISE → folded** (MAJOR — calibration prose said "held-out" but `FrozenEstimator` fit the sigmoid
  in-sample → switched to **`CalibratedClassifierCV(SVC(), method="sigmoid", ensemble=False)`**, now
  leak-free and matching the printed deprecation idiom, Brier 0.106→0.072; MINOR — decision_function
  shape `(5,3)` to disambiguate). Guards: 0 banned, ruff/hex clean, output-free, `llms.txt` 48. No
  `src/` change (pytest 19). Next: Rémy visual → commit + merge.
- **NB 3 (the kernel trick) BUILT & MERGED to `chapter/05_SVM` — Rémy validated visually.** 21 cells,
  4 figures (2-D→3-D `r²` lift with a separating plane; the RBF circular
  boundary; poly degree-2 vs degree-3; RBF on moons). By hand on `make_circles`: linear CV 0.557 → `r²`
  lift separates (inner [0.05,1.48] vs outer [1.96,5.26], threshold acc 1.000) → RBF **0.997** (38 SVs)
  without forming `r²`; poly **deg-2 1.000 / deg-3 0.613** (degree must match the geometry); moons
  0.840→0.970. Reviewers: **both PASS**; 3 MINOR polish folded (named the poly default `coef0=0` as the
  reason odd degrees miss the radial form; noted RBF default `gamma='scale'`; flagged `make_circles` as
  new vocabulary). Guards: 0 banned, ruff/hex clean, output-free, `llms.txt` 47. No `src/` change
  (pytest 19). Next: Rémy visual → commit + merge.
- **NB 2 (the soft margin & the cost `C`) BUILT & MERGED to `chapter/05_SVM` — Rémy validated
  visually.** 22 cells, 3 figures (hinge-vs-log-loss; small-`C` vs large-`C`
  street; margin & #SV vs `C`). By hand on penguins: hard margin infeasible (1 error, idx 128) → slack;
  hinge `max(0,1−m)` at C=1 = 0 / 0.40 / 1.31; `C`-sweep margin 2.28→0.35, SVs 124→6, accuracy ~flat
  (`C` sets the geometry). Reviewers: **pedagogy PASS** (2 MINOR folded — Fig-B right y-label, "all of
  them" wording); **ml-expert REVISE → folded** (MAJOR — "support vector = pays slack" was wrong: SVs
  are points with m≤1, the on-edge ones pay zero slack → at C=1, **17 SVs vs 15 slack-payers**; cells
  6/8/14 corrected + reconnected to NB 1; MINOR — singular "point(s)"). Guards: 0 banned, ruff/hex
  clean, output-free, `llms.txt` 46. No `src/` change (pytest 19). Next: Rémy visual → commit + merge.
- **NB 2 (the soft margin & the cost `C`) OPENED.** Branch `notebook/05_SVM__02_soft_margin_C` off
  `chapter/05_SVM` (@ `0383cd3`). Phase `notebook-plan`: drafting the cell-by-cell plan in plan mode —
  one concept, **slack & the cost `C`**, by hand on penguins (real, near-separable: a hard margin is
  infeasible → slack). Sweep `C` (margin 2.28→0.35, support vectors 124→6, accuracy ~flat = `C` sets the
  geometry); the **hinge loss** `max(0,1−y·f(x))` in `y∈{−1,+1}`, tied to ch-03 log-loss. Anchors in the
  chapter plan §NB 2; re-measured at build. Next: Rémy validates the NB-2 plan → build.
- **NB 1 (the maximum margin) BUILT & MERGED to `chapter/05_SVM` — Rémy validated visually.** 22 cells,
  4 figures (candidate lines + margins; the max-margin street via the new
  `viz.plot_svm_decision`; support-vector invariance delete/move; LogReg contrast). By-hand → `SVC(linear,
  C=1e6)` parity **exact**: street 1.7224 = 2/‖w‖, ‖w‖ 1.1612, SVs [23,26], cos 1.0, functional margins
  ±1; LogReg nearest-point 0.774 < SVM 0.861. **`src/` add:** `viz.plot_svm_decision` (street ±1 contours
  + ringed SVs) + 2 tests → **pytest 19**. Reviewers: **pedagogy PASS** (3 MINOR folded — ±1-scaling
  sentence, exercise-3 panel ref, exercise-1 figure anchor); **ml-expert REVISE → folded** (MAJOR — the
  closest-pair/perpendicular-bisector recipe is a *special case* → added the **convex-hull scope caveat**
  in cells 7/9/20/21; MINORs — Figure-A tilted-band note, `C=1e6 ≈ hard margin` flagged in prose).
  Guards: 0 banned, ruff clean, hex clean, output-free, `llms.txt` 45. Next: Rémy visual → commit + merge.
- **NB 1 (the maximum margin) OPENED.** Branch `notebook/05_SVM__01_maximum_margin` off
  `chapter/05_SVM` (@ `8f1f982`). Phase `notebook-plan`: drafting the cell-by-cell plan in plan mode —
  one concept, **the widest margin & support vectors**, by hand on a separable blob set (measure
  several separating lines' margins → the widest; the 2 support vectors; `margin = 2/‖w‖`) →
  `SVC(kernel="linear", C=1e6)` parity (‖w‖≈1.16 / margin≈1.72 / 2 SVs). Introduces the
  `viz.plot_svm_decision` helper (street ±1 contours + ringed SVs) with a smoke test (pytest 17→18).
  Anchors in the chapter plan §NB 1; re-measured at build. Next: Rémy validates the NB-1 plan → build.
- **Chapter 05 (Support Vector Machines) plan APPROVED & persisted** (`docs/plans/chapter_05_SVM.md`,
  this commit). **FIVE notebooks** (standard arc): NB 1 the maximum margin & support vectors (by hand
  on separable blobs → `SVC(linear)` parity, ‖w‖≈1.16 / margin≈1.72 / 2 SVs) → NB 2 the soft margin &
  cost `C` (penguins; margin 2.28→0.35, SVs 124→6; hinge loss tied to ch-03 log-loss) → NB 3 the kernel
  trick (`make_circles`: linear CV 0.557 → `r²` lift → RBF 0.997; poly degree must match the geometry —
  deg-2 1.000 / deg-3 0.613) → NB 4 the estimator `SVC` & its parameters (the `C×gamma` bias-variance
  map; `kernel`; OvO; `decision_function`→calibration, `probability=True` deprecation pinned) → NB 5
  demanding case **breast_cancer** (scaling headline raw 0.910→std 0.965; GridSearch test 0.965; spine
  KNN 0.942 / tree 0.906 / LogReg 0.953 / **SVM 0.965**; measured fit-time ~n^1.6 = the large-data
  limit). First **margin-based** method; the **kernel trick**. Reviewer-gated: **pedagogy PASS** (3
  build-MINORs folded); **ml-expert REVISE → all folded** (MAJOR: default poly degree-3 fails on circles
  CV 0.613 → pin degree-2 + the *degree-must-match-geometry* beat; MINORs: calibration provenance, hinge
  `{−1,+1}`, n^1.6 framing). **21/22 anchors reproduced** on sklearn 1.9.0; API facts (`probability=True`
  deprecation, `gamma='scale'`, OvO) **verified on the live install**. **`src/` addition planned:**
  `viz.plot_svm_decision` (NB 1, reused NB 1–4) + test → pytest 17→18. `course_map.md` §05 annotated.
  Next: open NB 1.
- **Chapter 05 (Support Vector Machines) opened.** Branch `chapter/05_SVM` created off `main` (synced
  @ `5f61e56` after PR #4). Phase `chapter-plan`: drafting the chapter plan in plan mode per
  `course_map.md` §05 and the per-method arc (the widest-margin idea by hand on separable 2-D → soft
  margin / cost `C` → the kernel trick → parameters `C`/`kernel`/`gamma` and the bias/variance picture
  they control → demanding case: scaling matters, CV model selection, honest limits on large data). The
  fifth method — the first built on the **maximum-margin** principle, the bridge from linear boundaries
  (ch 03) to kernels. The pending `idle` STATE edit was folded into this transition (committed on the
  chapter branch, not on protected `main`).
- **CHAPTER 04 (Decision Trees) COMPLETE — merged to `main` via PR #4** (merge commit `5f61e56`,
  `gh pr merge --merge`; per-notebook history preserved; pushed to Ramdam17/QuickIntroToMachineLearning).
  Five notebooks: impurity & the best split · growing & reading a tree · overfitting & pruning · the
  estimator & its parameters · breast_cancer (interpretability vs accuracy). The first **non-linear**,
  rule-based method and the **base learner** for the ensemble half of the course. **No `src/` change**
  (pytest stays 17; `load_breast_cancer` reused from ch 03). The two-reviewer gate + Rémy's visual
  validation held on every notebook; sklearn-1.9 anchors re-measured throughout; Rémy's spot-checks
  caught real issues that were fixed (NB 3 thin-band rendering → verified real tree regions; NB 4
  leaf-count read; NB 5 threshold conflation). `main` synced locally to `5f61e56`, green (pytest 17).
  STATE set to `idle` (pending edit, folds into the chapter-05 opening). Next: chapter `05_SVM`.
- **NB 5 (demanding case: breast cancer) BUILT & MERGED to `chapter/04_DecisionTree` — Rémy validated
  visually. CHAPTER 04 COMPLETE (5/5).** The chapter
  **capstone**, visualization-first: 26 cells, 6 figures (class balance; depth-3 tree rules via
  `plot_tree`; cross-method accuracy bar KNN 0.942 / LogReg 0.953 / single tree 0.906 / bagged-25
  0.930; root-feature flips; Gini-vs-permutation importance; confusion matrix). Full honest workflow on
  breast_cancer (malignant=1): tree CV-on-train 0.940 < LogReg 0.985; tuned tree test 0.906 < 0.953;
  depth-3 readable rules (test 0.918); single-tree variance (root feature flips: concave points 15× …,
  std 0.021); **Gini (concave points 0.74) vs permutation (worst area 0.27) disagree** (NB 4's caveat
  made real); confusion `[[95,12],[4,60]]` (4/64 cancers missed); **hand-bagged 25 trees → 0.930**
  (the ensemble bridge to ch 06). **Both reviewers folded:** pedagogy PASS (2 MINORs); ml-expert
  REVISE → 1 MAJOR fixed (cell-22 "LogReg caught more" conflated the default vs lowered threshold — at
  0.5 both miss the same 4 cancers, LogReg's edge is fewer false alarms; reframed) + MINORs ("recovers
  most"→"about half the gap", majority-vote comment, `worst fractal dimension` named). Guards: 0
  banned, ruff clean, hex clean, pytest 17, output-free, `llms.txt` 43. **Last NB of chapter 04** —
  next: Rémy visual → commit + merge → close chapter via PR into `main`.
- **NB 5 (demanding case: breast cancer — interpretability vs accuracy; where a single tree fails)
  OPENED.** Branch `notebook/04_DecisionTree__05_breast_cancer_interpretability` off
  `chapter/04_DecisionTree` (@ `e9447f4`). Phase `notebook-plan`: drafting the cell-by-cell plan in
  plan mode — the chapter **capstone**, **visualization-first**. Full honest workflow on breast_cancer
  (569×30, malignant=1): the readable depth-3 rule set vs the tree's accuracy cost (tree < LogReg), the
  single tree's high variance (root-feature flips), Gini + permutation importance, the cross-method
  test spine, and the bridge to ensembles (ch 06). Anchors in the chapter plan §NB 5; re-measured at
  plan time. **This is the last notebook of chapter 04** — after it ships, the chapter closes via PR
  into `main`. Next: Rémy validates the NB-5 plan → build.
- **NB 4 (the estimator & its parameters) BUILT & MERGED to `chapter/04_DecisionTree` — Rémy validated
  visually.** The integrative
  notebook, 23 cells, 3 figures (min_samples_leaf 1-vs-5 boundaries; **two bootstrap trees side by
  side = the variance headline**; Gini importance bar on penguins_full). Parity by-hand depth-2 ==
  `DecisionTreeClassifier(max_depth=2)` (train 0.9964); 4 dials shown (`min_samples_leaf` 0.878/0.933/
  0.800/0.744, `criterion` 0.910/0.914/0.914, `max_depth`/`ccp_alpha` recapped) + 2 named (`max_features`
  None 0.910/1 0.886 = RF seed, `class_weight`); variance full std 0.032/6.3 % vs depth-3 0.022/5.6 %;
  scale-invariance raw==std identical predictions; penguins_full 3-class + 2 NaN rows CV 0.9535;
  importance flipper 0.55/bill 0.36 + Strobl bias caveat (permutation → NB 5); GridSearchCV → max_depth
  6 / sealed test 0.889. **Both reviewers REVISE → folded:** MAJOR (both) — cell-13 read claimed the two
  bootstrap trees' leaf counts differ, but both grow 17 → reframed to "same 17 leaves, different
  boundary, test 0.900 vs 0.833 = variance"; MINOR — added a leakage note (whole-set standardization
  before CV would leak for a scale-sensitive model, not a tree). Guards: 0 banned, ruff clean, hex
  clean, pytest 17, output-free, `llms.txt` 42. Next: Rémy visual → commit + merge.
- **NB 4 (the estimator & its parameters) OPENED.** Branch
  `notebook/04_DecisionTree__04_estimator_and_parameters` off `chapter/04_DecisionTree` (@ `fb607f8`).
  Phase `notebook-plan`: drafting the cell-by-cell plan in plan mode — the **integrative** notebook
  (~24-cell ceiling, dé-surcharged at chapter-plan): parity vs NB 2; 4 shown dials (`max_depth`,
  `min_samples_leaf`, `ccp_alpha`, `criterion`) + 2 named (`max_features`, `class_weight`); the
  **variance/instability headline** (the ensemble bridge); trees' native strengths (scale-invariance,
  multiclass + NaN on `penguins_full`); Gini importance + bias caveat; `GridSearchCV`. Anchors from
  the chapter plan; specifics re-measured at plan time. Next: Rémy validates the NB-4 plan → build.
- **NB 3 (overfitting & pruning) BUILT & MERGED to `chapter/04_DecisionTree` — Rémy validated
  visually.** Rémy flagged the thin horizontal/vertical bands in the deep-tree boundaries as
  surprising; **re-verified they are real tree regions** (unpruned tree: 13 x1-cuts min gap 0.0044, 9
  x2-cuts min gap 0.0124 — fences around individual noise points; identical at render res 300 vs 800,
  so not an aliasing artefact) and **tightened both "Read the figure" cells** to name them as real
  overfitting, not a glitch (`common_errors` gained a matching row). 21 cells, 4 figures on
  `make_moons(300, 0.30, 0)`: 3 boundaries (depth 1 underfit / 6 good / unlimited jagged); train/test
  **error U-curve** vs depth (train→0 by depth 8, test U min ~depth 6–7) with the CV-best-depth line;
  **cost-complexity pruning path** (test acc + #leaves vs `ccp_alpha`); **unpruned (23-leaf, jagged,
  test 0.878) vs CV-pruned (8-leaf, smooth, test 0.900)** boundary. Honest selection: CV picks depth 6
  (0.919) on train → sealed test 0.889 (deliberately *not* the test max 0.900 at depth 7); CV-best
  `ccp_alpha` ≈ 0.0087 → 8 leaves / test 0.900. **Both reviewers PASS (no BLOCK).** ml-expert
  reproduced every number, confirmed no leakage, `ccp_alphas[:-1]` root-drop correct; pedagogy
  confirmed one-concept + charter + figure-read accuracy. MINORs folded (dangling "Figure A/C" →
  content refs; "training error"→"training impurity"; a guard that CV-best=test-max here is the seed
  being kind, not the rule; plateau "8–13 leaves"). Guards: 0 banned, ruff clean, hex clean, pytest 17,
  output-free, `llms.txt` 41. Next: Rémy visual → commit + merge.
- **NB 3 (overfitting & pruning: depth is the complexity dial) OPENED.** Branch
  `notebook/04_DecisionTree__03_overfitting_and_pruning` off `chapter/04_DecisionTree` (@ `774c1b2`).
  Phase `notebook-plan`: drafting the cell-by-cell plan in plan mode — one concept, **overfitting &
  pruning**, on `make_moons(300, 0.30, 0)` (the chapter's first move to the non-linear set; depth
  boundaries 1/6/unlimited; train/test U-curve; cost-complexity `ccp_alpha`; CV choice). Anchors
  (sklearn 1.9): depth 1 test 0.744 → depth 6 0.889 (CV-best 0.919; test peak depth 7 = 0.900) →
  unlimited train 1.000 / test 0.878; `ccp_alpha` 0.01 → 8 leaves / test 0.900. Next: Rémy validates
  the NB-3 plan → build.
- **NB 2 (growing a tree, and reading it) BUILT & MERGED to `chapter/04_DecisionTree` — Rémy
  validated visually.** 20 cells, 2
  figures (a **custom charter-coloured flowchart** of the depth-2 tree — drawn by hand so Adélie/Gentoo
  colours stay consistent, not sklearn's clashing blue/orange; decision regions depth-1 (2 boxes) vs
  depth-2 (4 boxes)). By hand: recurse NB 1's `flipper ≤ 206` → left `bill ≤ 47.20`, right `bill ≤
  40.85` → 4 leaves (149/0, 0/1, 1/0, 1/122); **by-hand == `DecisionTreeClassifier(max_depth=2)`**
  (train 0.9964); CV depth-2 0.9855 > full 0.9818 (overfitting hook → NB 3); the one error = row 128 (a
  long-flippered Adélie in the Gentoo box); greedy ≠ optimal (NP-hard); depth-3 adds a leaf but no
  accuracy. **Both reviewers PASS (no BLOCK).** ml-expert verified parity, the depth-3 reason, the CV
  reproduction, all DOIs; pedagogy verified one-concept + charter + figure-read accuracy. MINORs folded
  (both flagged the "box-counting in NB 1" back-ref → reworded self-contained; CV got a "here"
  single-seed qualifier + a one-line gloss + added to header prereqs; flowchart 47.2→47.20 for
  symmetry). Guards: 0 banned (JSON scan), ruff clean, hex clean, pytest 17, output-free, `llms.txt` 40.
  Next: Rémy visual → commit + merge.
- **NB 2 (growing a tree, and reading it) OPENED.** Branch
  `notebook/04_DecisionTree__02_growing_and_reading` off `chapter/04_DecisionTree` (@ `3ba6499`).
  Phase `notebook-plan`: drafting the cell-by-cell plan in plan mode — one concept, **recursive greedy
  growth + reading the tree**, by hand on penguins (recurse NB 1's `flipper ≤ 206` to depth 2 → 4
  leaves; read it as a flowchart; trace a penguin; parity vs `DecisionTreeClassifier(max_depth=2)`).
  Anchors (sklearn 1.9): depth-2 train 0.9964 / 4 leaves / CV 0.9855 > full 0.9818; rules root
  `flipper ≤ 206`, then `bill ≤ 47.20` (left child) / `bill ≤ 40.85` (right child). Next: Rémy
  validates the NB-2 plan → build.
- **NB 1 (a question that splits the data: impurity) BUILT & MERGED to `chapter/04_DecisionTree` —
  Rémy validated visually.** 23 cells, 4 figures (feature histograms; impurity-vs-p shapes;
  decrease-vs-threshold, 2 panels sharing y so flipper's higher peak shows; the chosen split on the
  cloud + child class-mix bars). One concept,
  **by hand before the library** (the only sklearn call is the depth-1 parity at the end). All anchors
  reproduce exactly: root Gini 0.4948 / entropy 0.9925; best split `flipper ≤ 206` Gini decrease 0.4732
  vs `bill ≤ 43.25` 0.4044 (entropy picks the same thresholds); children 149A/1G (0.0132) & 2A/122G
  (0.0317) → weighted 0.0216 → decrease 0.4732; stump root `flipper ≤ 206`, acc 0.9891; raw == std
  (scale-invariance). Both reviewers folded: **pedagogy PASS** (2 MINORs — softened "single clean hump"
  for the ragged bill curve; kept the "Read the result" twin convention); **ml-expert REVISE → 1 MAJOR
  fixed** (the ch-03 `bill`-vs-`flipper` callback fabricated a false contrast — ch 03 introduced the
  sigmoid on `bill` by *narrative* choice and a linear fit also favours `flipper` here → reframed to "we
  *measure* which feature cuts best, not choose by eye"). Guards: 0 banned words (JSON-real-text scan),
  ruff clean, hex clean, pytest 17, output-free, `llms.txt` 39 lines. Next: Rémy visual → commit + merge.
- **NB 1 (a question that splits the data: impurity) OPENED.** Branch
  `notebook/04_DecisionTree__01_impurity_and_splits` off `chapter/04_DecisionTree` (@ `ee99b25`).
  Phase `notebook-plan`: drafting the cell-by-cell plan in plan mode — one concept, **impurity & the
  best split**, by hand on the binary penguins subset (raw units). Anchors (sklearn 1.9.0): root
  Gini 0.4948 / entropy 0.9925 bits; best split `flipper ≤ 206` (Gini decrease 0.4732) beats
  `bill ≤ 43.25` (0.4044) — *the "best feature" depends on the criterion*; depth-1 stump 0.9891; the
  split is a threshold ⇒ scale-invariant. Next: Rémy validates the NB-1 plan → build.
- **Chapter 04 (Decision Trees) plan APPROVED & persisted** (`docs/plans/chapter_04_DecisionTree.md`,
  commit `b2c9308`). **FIVE notebooks** (standard arc): NB 1 impurity & the best split (Gini/entropy,
  by hand on penguins) → NB 2 greedy growth + reading the tree → NB 3 overfitting & pruning (the depth
  dial, on `make_moons`) → NB 4 the estimator `DecisionTreeClassifier` & its parameters
  (**variance/instability the headline**) → NB 5 demanding case **breast_cancer** (interpretability vs
  accuracy; where a single tree fails → the bridge to ensembles). First **non-linear** method; the
  **base learner** of ch 06–10. Reviewer-gated, both **REVISE → all folded** (every number re-measured
  by Claude on sklearn 1.9.0): **ml-expert** (MAJOR — NB 5 CV → **CV-on-train** tree 0.940 / LogReg
  0.985 matching shipped ch 03 NB 6; MAJOR — `criterion` re-measured at **default depth** 0.910 / 0.914
  + the no-logarithm-cost argument; MINORs — depth 6 = CV-best not test-peak (peak is depth 7 / 0.900),
  NaN = 2 numeric rows, variance recipe **pinned** `default_rng(0)`/20/`rs=0`/150² grid). **pedagogy**
  (MAJOR — banned words cleaned; MAJOR — **NB 4 de-overloaded** to 4 shown knobs + 2 named, ~24-cell
  ceiling; MINORs — box-count beat in NB 2, KNN spine re-measured 0.942 on the pinned split, charter
  close named). Rémy chose **breast_cancer** for NB 5 (over `penguins_full` / a Titanic loader).
  `course_map.md` §04 already aligned. Next: open NB 1.
- **Chapter 04 (Decision Trees) opened.** Branch `chapter/04_DecisionTree` created off `main` (synced
  @ `8cdcc73` after PR #3). Phase `chapter-plan`: drafting the chapter plan in plan mode per
  `course_map.md` §04 and the per-method arc (a split & impurity by hand → grow & read a tree →
  overfitting/pruning & the depth dial; NB 4 `DecisionTreeClassifier` & its parameters; NB 5 a
  demanding case — interpretability vs accuracy, where a single tree fails). The first **non-linear,
  axis-aligned partition** method, and the **base learner** the ensemble half of the course
  (06 Random Forest → the boosting family) is built on. The pending `idle` STATE edit was folded
  into this transition (committed on the chapter branch, not on protected `main`).
- **CHAPTER 03 (Logistic Regression) COMPLETE — merged to `main` via PR #3** (merge commit `8cdcc73`,
  `gh pr merge --merge`; per-notebook history preserved; pushed to Ramdam17/QuickIntroToMachineLearning).
  Six notebooks: score→probability · boundary & weights · log-loss · gradient descent · estimator &
  parameters · breast_cancer (calibration/threshold). Added `datasets.load_breast_cancer()` + schema test
  (`pytest` 17). The two-reviewer gate + Rémy's visual validation held on every notebook; the sklearn-1.9
  API was pinned throughout; every number re-measured. `main` synced locally to `8cdcc73`, green. STATE
  set to `idle` (pending edit, folds into the chapter-04 opening). Next: chapter `04_DecisionTree`.
- **NB 6 (demanding case: breast cancer) built & merged — CHAPTER 03 COMPLETE (6/6).** The capstone,
  visualization-first (5 figures), on breast_cancer (569×30, malignant = positive). Honest workflow, no
  leakage: split → CV **on train** (LogReg **0.985** > GaussianNB **0.932**) → one sealed test (acc 0.953).
  **Calibration** closes ch 02's loop: LogReg Brier **0.033** vs GaussianNB **0.098** (~3×); a pile-up
  histogram makes GaussianNB's over-confidence visible (166 vs 119/171 at the extremes); reliability diagram
  with the y=x reference. **Threshold = clinical policy** (malignant the costly miss): 0.5 → recall 0.938
  (4/64 missed) vs 0.1 → 0.984 (1 missed, 14 false alarms). **L1** keeps 3/10/14 of 30; **coefficient story**
  (radius/concavity → malignant, clinically sensible). Bridge to trees (ch 04). **src/ add:**
  `datasets.load_breast_cancer()` pandas wrapper + schema test → **pytest 17**. Both reviewers folded
  (ml-expert REVISE→cell-8 truncated sentence + coef-read MINORs; pedagogy PASS→added the reliability
  diagonal, reworded ex-3 for the 1.9 `CalibratedClassifierCV`/`FrozenEstimator` API; several anchors
  re-measured vs the chapter plan — measured values used). `common_errors` gained a 0.5-threshold row;
  `llms.txt` regenerated; ruff/black/hex/banned clean. Rémy validated visually. **Next: PR chapter/03 →
  main, then chapter 04 (Decision Trees).**
- **NB 6 (demanding case: breast cancer — calibration, threshold, error analysis) OPENED.** Branch
  `notebook/03_LogisticRegression__06_breast_cancer_calibration_threshold` off `chapter/03` (@ `c2110e7`).
  Phase `notebook-plan`: drafting cell-by-cell — the chapter capstone (**visualization-first**). Anchors
  re-measured on sklearn 1.9 (breast_cancer 569×30, **malignant = positive** 212 / benign 357; 70/30 seed0
  stratify, one std `Pipeline`, StratifiedKFold5-shuffle-seed0): CV LogReg **0.979** vs GaussianNB
  **0.930**; test LogReg acc 0.953 / Brier **0.033** vs GaussianNB 0.895 / **0.098** (≈3×; pile-up 119 vs
  166/171 — GaussianNB over-confident, closes ch 02's loop); threshold 0.5 → recall 0.938 (4/64 missed)
  vs 0.1 → 0.984 (1 missed, 14 false alarms); L1 nonzero **3/10/14** of 30 at C=0.02/0.2/1.0; top
  malignant-driving coefs radius error / worst radius / mean concave points. **Several numbers differ
  from the chapter plan's preliminary figures** (Brier 0.033/0.098 vs 0.027/0.088; threshold 4/3-missed
  vs 3/2; L1 middle 10 vs 8) — qualitative stories intact, measured values used. Likely `src/` add
  `datasets.load_breast_cancer()` pandas wrapper + test (pytest 16→17). Next: Rémy validates the NB-6
  plan → build → chapter PR into `main`.
- **NB 5 (the estimator & its parameters) built & merged to `chapter/03_LogisticRegression`.**
  Role-4: the real `sklearn LogisticRegression` on the **1.9 API** (verified: `l1_ratio` not the deprecated
  `penalty`; no `multi_class`; `saga` for L1; `C=np.inf`=none). Parity vs by-hand (NB 4). Knobs *shown*:
  **`C`** reg-path ‖w‖₂ 0.84→6.80 → **separation→divergence** (1 overlap pt = finite ‖w‖≈11 vs separable
  slice ≈29); **`l1_ratio`** L1 zeroes the 4 injected noise cols (4/8) vs L2 8/8; **softmax vs OvR** (3
  species, CV 0.956/0.956, 0% disagreement, coef_ (3,2)); honest GridSearchCV (best C≈0.08, sealed test
  1.000 — flagged as easy-split, NB 6 is the real case). 24 cells, 4 figures. **ml-expert REVISE→fixed**
  (API exhaustively verified on 1.9.0, every number bit-for-bit; **1 MAJOR**: the reg-path/divergence
  plateau wrongly blamed on the iteration limit → it is **convergence/tolerance** (`n_iter_`≈14 ≪ 200000)
  → reworded cells 8/11 **+ added a `print(n_iter_)` that proves it**; MINORs: ≈8.5 not "nearly 7", OvR
  renormalizes in predict_proba), **pedagogy PASS** (added "defaults are regularized → parity uses
  `C=np.inf`"). `common_errors` gained a C-is-inverse row; `llms.txt` regenerated; ruff/hex/banned clean;
  pytest 16. Rémy validated visually. Next: open NB 6 (breast cancer) → then chapter PR into `main`.
- **NB 5 (the estimator & its parameters) OPENED.** Branch
  `notebook/03_LogisticRegression__05_estimator_and_parameters` off `chapter/03_LogisticRegression`
  (@ `1b68bc7`). Phase `notebook-plan`: drafting cell-by-cell in plan mode — the role-4 "method &
  parameters" notebook (first to use the real `sklearn LogisticRegression`). **sklearn 1.9 API verified
  at plan time:** `l1_ratio` present, **`penalty` deprecated** (FutureWarning 1.8→1.10: use l1_ratio=0/1,
  C=np.inf), **`multi_class` REMOVED**; `saga` for L1. Anchors measured on 1.9.0: L2 path ‖w‖₂ =
  0.84/1.91/3.28/6.80 (C=0.01/0.1/1/100, 4 std feats, plateau 8.46); separation→divergence — full 2-feat
  (1 overlap pt) MLE finite ‖w‖≈11 vs the slice with that point removed runs to ‖w‖≈29+; L1 (l1_ratio=1,
  saga) zeroes the 4 injected noise cols exactly (4/8 nonzero) while L2 keeps all 8; L1 on 4 real feats
  4/4 (1/4 at C=0.01); multinomial vs OvR (3 species) CV 0.956/0.956, **0.0% disagreement**, coef_ (3,2).
  4 figures planned (L2 path, separation→divergence, L1-vs-L2 noise bars, 3-class softmax boundaries) +
  honest GridSearchCV tuning. Next: Rémy validates the NB-5 plan → build.
- **NB 4 (Fitting II — gradient descent) built & merged to `chapter/03_LogisticRegression`.**
  One concept: **the optimizer** (the course's first), by hand on standardized 1-D `bill_length` (w & b).
  Gradient **(P−y)·x** stated & **verified** (finite-diff err 2e-11; σ′ cancels); update w←w−η∇L; descent
  on NB 3's convex bowl (figB surface+path → bottom; figC loss → floor 0.140). **Parity exact**: by-hand
  GD = `LogisticRegression(C=∞)` (6.29704 / −0.56139) — "the library is not magic". Learning-rate panel
  (figD): 0.1 crawls / 2 glides / 400 overshoots; raw-feature knife-edge (0.003 vs 0.005) = the
  "why standardize" tie-in. Convergence **shown, not proved** (leans on NB 3 convexity); SGD/backprop only
  named (→ ch 11–12). 22 cells, 4 figures. Both reviewers **PASS** (0 BLOCK/MAJOR): every number
  re-derived to machine precision; parity against C=∞ verified (default C=1 → w=4.25, different, so the
  choice is load-bearing). **MINORs folded:** softened "diverges/explodes/leaps past" → "overshoots /
  climbs the wrong way" (on this flat loss η=400 stays bounded, not →∞ — honest); **lr-panel η 90→400
  deviation from the approved plan** (90 did not visibly diverge on the well-conditioned loss — a
  correctness fix; title/legend/read all updated). `common_errors` gained a learning-rate row; `llms.txt`
  regenerated; ruff/hex/banned clean; pytest 16. Rémy validated visually. Next: open NB 5 (estimator &
  parameters).
- **NB 4 (Fitting II — gradient descent) OPENED.** Branch
  `notebook/03_LogisticRegression__04_gradient_descent` off `chapter/03_LogisticRegression` (@ `6940caf`).
  Phase `notebook-plan`: drafting cell-by-cell in plan mode — one concept, **the course's first
  optimizer**: gradient = steepest-ascent direction; step opposite by a learning rate; the weights roll to
  the bottom of NB 3's convex bowl. Gradient **(P−y)·x** (verified vs finite-diff to 2e-11). Anchors
  measured: by-hand full-batch GD on standardized 1-D `bill_length` (w,b) → `LogisticRegression(C=∞)`
  w*=6.297 / b*=−0.561 (gap 4e-4 at lr=1, 1e-7 at lr=2; ~1000 it at lr=0.5); learning-rate panel
  **standardized** (lr 0.1 crawls / 2 glides / 90 oscillates; surface flat, λ_max=0.041, stable to ~48) —
  divergence shown on **raw** bill as the knife-edge (0.003 crawls, 0.005 explodes → the "why standardize"
  tie-in). 4 figures planned (gradient-on-bowl, surface+path, loss-vs-iter, lr panel). Next: Rémy validates
  the NB-4 plan → build.
- **NB 3 (Fitting I — what we optimize: log-loss) built & merged to `chapter/03_LogisticRegression`.**
  One concept: **the training objective**, by hand, pre-fitting. **log-loss = cross-entropy = −log-
  likelihood** of the Bernoulli model (the bridge from ch 02's likelihood); punishes confident-and-wrong
  without bound (−log P; P=0.01→4.6) where squared error caps at 1 (Figure A); **convex** (one bottom,
  2nd-diff ≥ 0, min 0.146 at w≈6.2) vs **squared-error-on-sigmoid non-convex with stalling plateaus**
  (2nd-diff < 0, plateau slope ~3e-4) (Figure B); one number ranks weight choices (w=1/3/6.2 →
  0.39/0.19/0.146). 1-D std bill, **b held at 0**, no sklearn, nothing fitted (NB 4 minimizes). 19 cells,
  2 figures. **ml-expert REVISE→fixed** (every number verified to machine precision incl. gradient
  (P−y)·x and analytic convexity L''≥0; **1 BLOCK = banned word "simply" cell 18 → "exactly"**; MINOR
  bowl-ylim wording), **pedagogy PASS** (added a Bernoulli coin-flip gloss; the honest "single min, not
  bumps" framing praised). **Process fix:** the banned-word guard now parses the JSON real text — the old
  raw-`.ipynb` grep missed words glued after a literal `\n` (that is how "simply" slipped past); NB 1–2
  re-scanned **clean**. `common_errors` gained a "train with squared error" row; `llms.txt` regenerated;
  ruff/hex/banned clean; pytest 16. Rémy validated visually. Next: open NB 4 (gradient descent).
- **NB 3 (Fitting I — what we optimize: log-loss) OPENED.** Branch
  `notebook/03_LogisticRegression__03_logloss_objective` off `chapter/03_LogisticRegression` (@ `d15035d`).
  Phase `notebook-plan`: drafting cell-by-cell in plan mode — one concept, **the objective**: log-loss =
  cross-entropy = −log-likelihood of the Bernoulli model (bridge from ch 02's likelihood), punishes
  confident-and-wrong (−log P unbounded); **log-loss convex** (one bottom) vs **squared-error-on-sigmoid
  non-convex with stalling plateaus**. Anchors measured (1-D std bill: w*≈6.29 / b*≈−0.56, log-loss convex
  min 0.140; MSE 2nd-diff < 0, plateau slope ~3e-4 at w=20; per-example y=1/P=0.01 → log-loss 4.6 vs MSE
  0.98; hand weights w=1/3/6.3 → 0.398/0.188/0.140). **Note:** real 1-D data shows non-convex + plateaus
  (single min), not multiple "bumps" — framing adjusted from the chapter plan's "bumpy", flagged to Rémy.
  Next: Rémy validates the NB-3 plan → build.
- **NB 2 (decision boundary & reading the weights) built & merged to `chapter/03_LogisticRegression`.**
  One concept: **the weighted line & what its weights mean**, by hand, pre-fitting. On **standardized**
  bill+flipper: z=w₁x₁+w₂x₂+b, the **decision boundary** (z=0, P=½), **w ⟂ boundary** & ‖w‖=steepness,
  each **wⱼ = Δ log-odds per std unit** (×e^wⱼ to the odds: bill ×2.7, flipper ×7.4). Hand weights
  w=(1,2), b=0 (nothing fitted — "NB 3–4 find them"): acc **0.9891**, ‖w‖ 2.24, band ~37 % (the 3 errors
  all in-band). Contrast **nearest-centroid** unweighted bisector → **tilt 16.3°** = the weighting (NC acc
  0.9927). Figure C: weights rotate the line, b shifts it. 21 cells, 3 figures. Both reviewers folded:
  **ml-expert PASS** (every number re-measured exact; ‖w‖=steepness verified = ‖w‖/4 slope at z=0; no
  hidden `.fit`; 3 DOIs resolve), **pedagogy REVISE→all folded** (MAJOR: the w arrow didn't render
  perpendicular under unequal axes → `set_aspect("equal")` on figs A/B/C; MINORs: white→blue wording, and
  named that the borderline example is a real in-band error). Dropped the optional ~64° fit teaser (both
  reviewers preferred the clean no-fitting wall). `common_errors` gained a weight-magnitude/standardize
  row; `llms.txt` regenerated; ruff/hex/banned-word clean; pytest 16. Rémy validated visually. Next: open
  NB 3 (log-loss).
- **NB 2 (decision boundary & reading the weights) OPENED.** Branch
  `notebook/03_LogisticRegression__02_boundary_and_weights` created off `chapter/03_LogisticRegression`
  (@ `cbf90d0`). Phase `notebook-plan`: drafting the cell-by-cell plan in plan mode — one concept, on
  **standardized** bill+flipper: the weighted line z=w₁x₁+w₂x₂+b, the **decision boundary** (z=0, P=½),
  **w ⟂ the boundary** & sets steepness, each **wⱼ = Δ log-odds per standardized unit**; weights **set by
  hand** (rotate with w, shift with b), contrasted with module-00 nearest-centroid's *unweighted* bisector;
  **nothing fitted** (NB 3–4 find the weights). Anchors measured at plan time (scaler stats, fitted std
  coefs as the ballpark, NC-normal vs logistic-w angle). Next: Rémy validates the NB-2 plan → build.
- **NB 1 (From a linear score to a probability) built & merged to `chapter/03_LogisticRegression`.**
  One concept: **the sigmoid & log-odds**, fully by hand, pre-fitting. σ(z)=1/(1+e⁻ᶻ) coded from
  scratch & plotted → **p→odds→log-odds** table (the score *is* the log-odds; σ and logit are
  inverses) → σ applied to `bill_length` (**raw mm**) with **hand-chosen** weights (w=1.0, b=−43,
  ½-crossing **43 mm**; nothing fitted — "NB 3–4 find these") → ½-threshold prediction → borderline
  42.9 mm example (P=0.475). Build-measured: hand-rule acc **0.945** (≈ fitted 0.947, never called the
  optimum), transition band ~**21.5 %**. 19 cells, 2 figures. Both reviewers **PASS** (no BLOCK):
  ml-expert verified σ↔logit to 1e-14, the no-fitting promise airtight (no hidden `.fit`), all 3 DOIs
  resolve, calibration correctly **not** claimed; pedagogy confirmed one-concept + e/σ/odds-log-odds
  built from scratch. **2 MINORs folded** (log=natural-log base e; "a fifth" tied to the P∈[0.1,0.9]
  band); **skipped a 3rd** ("all 15 errors in the band") — measured 12/15 in band, 3 confidently-wrong
  → false, and that nuance belongs to NB 6. `common_errors` gained a score-vs-probability/log-odds row;
  `llms.txt` regenerated; ruff fixed (notebook import order I001); pytest 16. Rémy validated visually.
  Next: open NB 2.
- **Chapter 03 (Logistic Regression) plan APPROVED & persisted** (`docs/plans/chapter_03_LogisticRegression.md`).
  **SIX notebooks** (Rémy-approved exception to the 5-ceiling, like KNN's 6th): NB 1 sigmoid & log-odds
  → NB 2 decision boundary & reading weights → NB 3 **log-loss** (the objective) → NB 4 **gradient
  descent** (the optimizer — split from NB 3 on Rémy's go) → NB 5 estimator & parameters
  (`LogisticRegression`: C, `l1_ratio` L1/L2, softmax) → NB 6 demanding case **breast_cancer**
  (calibration/threshold/error analysis). First **discriminative** method; first trained by **iterative
  optimization**; closes ch 02's generative-vs-discriminative loop. Reviewer-gated, both **REVISE→all
  folded**: **ml-expert** (2 BLOCK — NB 1 sigmoid 30%/46mm self-contradiction → raw-mm acc 0.947 /
  crossing ≈43 mm / ~16 %; breast_cancer CV unreproducible → pinned StratifiedKFold5-shuffle LogReg
  **0.979** vs GaussianNB **0.930**; + GaussianNB calibration re-measured under one std pipeline Brier
  **0.088**/pile 167, GD parity vs `C=np.inf`, ‖w‖₂ over 4 std feats, OvR 0.952) — verified the
  **sklearn-1.9 API pivot** (`penalty` deprecated→`l1_ratio`; no `multi_class`→`OneVsRestClassifier`;
  `saga` for L1) and gradient ∝(P−y)·x to machine precision; **pedagogy** (1 BLOCK 2 banned words; 3
  MAJOR — split GD to its own NB, add odds/log-odds + gradient-as-slope first-contacts, give softmax its
  own section). Measured at plan time on sklearn **1.9.0**. `course_map.md` §03 aligned to six titles.
  Next: open NB 1.
- **Chapter 03 (Logistic Regression) opened.** Branch `chapter/03_LogisticRegression` created off
  `main` (synced @ `726d13e` after PR #2). Phase `chapter-plan`: drafting the chapter plan in plan
  mode per `course_map.md` §03 and the per-method arc (sigmoid → boundary/weights → log-loss fitting;
  NB 4 `LogisticRegression` C/L1-L2/multi-class; NB 5 calibration + threshold + error analysis —
  LogReg as the calibrated discriminative foil to NB's over-confidence). The pending `idle` STATE
  edit was folded into this transition (committed on the chapter branch, not on protected `main`).
- **NB 5 (Text classification — the demanding case) built & merged; CHAPTER 02 COMPLETE (5/5).** The
  capstone, on 20-newsgroups: **by-hand bag-of-words on-ramp** (toy sentences → vocab → dense count
  matrix) → `CountVectorizer` (12 384 words, density 0.0043, fit-on-train-only) → `MultinomialNB`
  (fit ≈ms, acc **0.887**, confusion → religion hardest) → **honest eval under imbalance** (one-vs-rest
  sci.med: acc **0.930** vs **baseline 0.724**, P/R/F1 0.887/0.854/0.870, PR AP 0.935) → **calibration**
  (MNB piles 1205/1433 at 0/1 = over-confident *in shape*; Brier 0.056 < LogReg 0.080 here because the
  task is easy → "trust the ranking, not the number"; cost shown on the confusable pair in Your turn) →
  Domingos-Pazzani at scale + **generative-vs-discriminative bridge to ch 03**. 27 cells, 5 figures.
  **2 `src/` additions with tests** (`datasets.load_newsgroups` fetch-and-cache + visible logging;
  `viz.plot_calibration_curve` reliability diagram) → **pytest 16**. Both reviewers **PASS** (no BLOCK):
  ml-expert verified every number + measured that keeping metadata leaks the label (0.887→0.955, so
  `remove=` is right); pedagogy confirmed the by-hand on-ramp + honest calibration framing. 5 MINORs
  folded (calibration wording, "crushes most", no-skill label value, multinomial pointer). `common_errors`
  + `course_map` §02 + `llms.txt` updated. Rémy validated visually. Next: PR `chapter/02_NaiveBayes` →
  `main`, then open chapter 03 (Logistic Regression).
- **NB 4 (The estimators & their parameters) built & merged to `chapter/02_NaiveBayes`.** The
  integration notebook: the NB family (Gaussian/Multinomial/Bernoulli) + each dial *measured* —
  **`var_smoothing`** (flat 0.9927 → 1.0:0.989 → 10:0.711, the flood), **`alpha`/Laplace** healing NB 1's
  penguins zero by hand (α=0→0; α=1→0.0065/post 0.018; verified by-hand == `MultinomialNB.feature_log_prob_`),
  **`priors`/`fit_prior`** tilting the boundary (124↔127 predicted Gentoo; borderline x=[40.8,208] flips
  Adélie→Gentoo at P(Gentoo) 0.15/0.5/0.85), and **honest tuning** (GridSearchCV on train, one sealed
  test). **Calibration named & deferred to NB 5** (penguins too separable, well-calibrated). 20 cells,
  3 figures. Both reviewers **PASS** (no BLOCK): ml-expert REVISE → the asserted Brier 0.0006 reframed as
  a NB-5 forward reference (split-dependent, unshown) + 2 MINORs; pedagogy PASS + 2 MINORs (flip sweep
  aligned to the panels). Rémy questioned whether NB 4 was a 4th concept notebook → confirmed it is the
  role-4 "method & parameters" notebook; chapter stays **5**. `common_errors.md` gained a
  "don't max out smoothing" row; `llms.txt` regenerated; pytest 14. Rémy validated visually.
- **NB 3 (The Gaussian likelihood, computed safely) built & merged to `chapter/02_NaiveBayes`.** One
  concept (chapter stays 5 — split not pulled): model P(feature∣class) with a continuous **density**,
  computed in **log-space**. Bins (NB 1) were crude → **density first-contact** taught from scratch
  (area not height; integrates to 1; can exceed 1) → per-class **Gaussian** fit (Adélie μ38.79/σ2.65,
  Gentoo μ47.50/σ3.07) overlaid on the density histogram (mass→density; the zero-frequency trap
  dissolves) → **by-hand Gaussian NB = sklearn `GaussianNB` 100 %** (acc 0.9927; curved boundary) →
  likelihood is a choice (multinomial/Bernoulli named → NB 5) → **underflow** (product → 0.0 at N=324)
  → **log-space** (sum of log-likelihoods, argmax unchanged). 20 cells, 4 figures, "Your turn" ×3. Both
  reviewers **PASS** (no BLOCK); folded: **3 banned words** ("simply" ×2, "just"), `var_smoothing`
  honesty (by-hand = GaussianNB with smoothing off; scores differ <1e-6 → NB 4 dial), the log-tie
  clause, σ gloss, and a NaN/3rd-species hint on exercise (a). `common_errors.md` gained underflow +
  density-height rows; `llms.txt` regenerated; pytest 14 (no `src/`). Rémy validated visually.
- **NB 2 (The "naive" assumption) built & merged to `chapter/02_NaiveBayes`.** One concept:
  **conditional independence**. Two features need the joint P(bill,flipper∣species); estimating it
  directly is expensive (5×5 grid, **18/25 cells empty** — curse echo). The naive shortcut: assume
  independence given the class → joint = product of 1-D marginals. **Where it breaks**, measured: real
  vs naive joint heatmaps + a **difference panel** (the discarded correlation), within-class corr
  **0.326/0.661** (vs overall 0.869 = mostly between-class). **Does the decision survive?** CV
  GaussianNB **0.9927** ties LDA, beats QDA **0.9890** (Domingos & Pazzani 1997). Honest: GaussianNB =
  QDA with diagonal covariance; the tie is an accuracy coincidence on near-separable data (NB/LDA
  boundaries differ on 11.5 %, NB/QDA on 18.2 % — verified by ml-expert), *not* a model identity; the
  probabilities suffer even when the decision holds (→ NB 5 calibration). 21 cells, 3 figures, "Your
  turn" ×3. Both reviewers **PASS** (no BLOCK); shared MINOR folded ("Read the table" for the count
  grid). `common_errors.md` gained an overall-vs-within-class-correlation row; `llms.txt` regenerated;
  pytest 14 (no `src/` change). Rémy validated visually.
- **NB 1 (Bayes' rule, from counts) built & merged to `chapter/02_NaiveBayes`.** By hand on the binary
  penguins subset: the **prior** P(species) 0.551/0.449 by counting → `bill_length` 3-bin contingency
  (Adélie [135,16,0] / Gentoo [3,67,53]) → **likelihood** P(bin∣species) by row-normalizing → **Bayes'
  rule** (4 terms named) → **posterior** P(species∣bin) (short→Adélie 0.978, medium→Gentoo 0.807,
  long→Gentoo 1.000) → predict by **argmax**, with "evidence cancels under argmax" *demonstrated*
  (`joint.idxmax == posterior.idxmax` True). The bin `long` surfaces the **live zero-frequency case**
  (no Adélie → P=0 → posterior exactly 0/1, overconfident) → foreshadows NB 4 smoothing. 21 cells, 3
  figures (prior bar / likelihood grouped bar / posterior stacked bar), "Your turn" ×3, vocabulary box.
  Both reviewers **PASS** (no BLOCK): ml-expert re-derived every number by exact fractions (45/46, 67/83);
  pedagogy confirmed one-concept + voice. 4 MINORs folded (conditional-probability gloss + drop "joint"
  cell 10; discrete-likelihood-normalization hedge cell 12; exhaustivity already in cell 6). Rémy
  validated visually. `common_errors.md` gained a zero-frequency row; `llms.txt` regenerated; pytest 14
  green (no `src/` change).
- **Chapter 02 (Naive Bayes) plan APPROVED & persisted** (`docs/plans/chapter_02_NaiveBayes.md`).
  5 notebooks, standard arc: NB 1 Bayes from counts (one feature, by hand) → NB 2 the naive
  (conditional-independence) assumption → NB 3 the Gaussian likelihood + log-space → NB 4 the
  estimators & parameters (var_smoothing, alpha/zero-frequency, priors, calibration-bridge) → NB 5
  demanding case = **text classification** (20newsgroups, MultinomialNB, honest eval under imbalance,
  the over-confidence limit, generative-vs-discriminative bridge to ch 03). Datasets: penguins (NB 1–4,
  Gaussian) + a 20newsgroups subset (NB 5, fetch-and-cache). Reviewer-gated: **ml-expert REVISE→1 BLOCK
  fixed** (the headline error: GaussianNB is **QDA with diagonal per-class covariance**, *not* LDA — the
  0.9927 NB/LDA/QDA tie is an accuracy coincidence on near-separable 2-D data; re-checked PASS at the
  parameter level), unfair NB(raw)-vs-LogReg(scaled) → both raw, α-curve marked version-indicative.
  **pedagogy REVISE→no BLOCK** (4 MAJORs folded: NB 3 re-scoped to one concept with mass→density as a
  taught first-contact; bag-of-words built by hand in NB 5; NB 4 calibration as a tight bridge; "Your
  turn" per NB). Measured at plan time: within-class corr 0.326/0.661/0.486; NB/QDA/LDA 0.9927/0.9890/
  0.9927; text acc ≈0.89 (4-cat) / ≈0.70 (hard binary); α→0 = log(0) collapse. Next: open NB 1.
- **Chapter 02 (Naive Bayes) opened.** Branch `chapter/02_NaiveBayes` created off `main` (synced to
  `110c081` after PR #1). Phase `chapter-plan`: drafting the chapter plan in plan mode per
  `course_map.md` §02 and the per-method arc. The pending `idle` STATE edit was folded into this
  transition (committed on the chapter branch, not on protected `main`).
- **Chapter 01 (k-Nearest Neighbours) COMPLETE — 6 notebooks merged to `main` via PR #1** (merge commit
  `110c081`, `gh pr merge --merge` — per-notebook history preserved; pushed to
  `Ramdam17/QuickIntroToMachineLearning`). The arc: the vote → distance & the scale trap → the k dial →
  the estimator & its parameters → demanding case + the curse → advanced distances & nested CV. The
  two-reviewer gate (`@ml-expert-reviewer` + `@pedagogy-reviewer`) + Rémy's visual validation held on
  every notebook; NB 6 was rebuilt from scratch (v1 was visually thin). Next: chapter `02_NaiveBayes`.
- **NB 6 (advanced: distances & choosing k) built & merged — chapter complete (6/6).** Optional Advanced
  capstone, **rebuilt from scratch (v1 scrapped by Rémy as too table-heavy)** → visualization-first, 28
  cells, **9 figures**: unit balls L1/L2/L∞; metric decision boundaries on moons; Mahalanobis ellipse +
  Euclidean-vs-Mahalanobis boundary on penguins; distance-concentration histogram (d=2/50/1000);
  near/far ratio by p; accuracy-vs-noise curve (penguins wash vs breast_cancer p=1>p=2, averaged over 8
  draws); nested-CV schematic + fold-score strip (naive 0.967 vs nested 0.960). Both datasets featured;
  silhouette dropped (it concerns k-means cluster count, not k-NN's k). Pedagogy PASS; ml-expert
  REVISE→fixed (L1 unit-ball geometry — vertices on axes, not sides; "staircase"→tendency; LDA→incise).
  Rémy validated. `feat(01_knn): notebook 06 — advanced: distances & choosing k`. **Note:** a stray
  `/tmp/struct.py` shadows stdlib `struct` for `python /tmp/*.py` → build via `uv run python - < file`.
- **NB 5 (demanding case: breast cancer & the curse) built & merged to `chapter/01_KNN`.** Full honest
  workflow on `load_breast_cancer` (569×30): pandas look → `Pipeline(StandardScaler, KNN)` (raw CV
  0.935 vs scaled **0.970**, NB 2 scale trap on real data) → CV picks **k=7** → one held-out eval (test
  **0.947**) → error analysis: confusion `[[57,7],[2,105]]`, **malignant recall 0.891 = 7 of 64 cancers
  missed** (accuracy hid it; threshold → NB 8) → when to/not use k-NN → **the curse, felt**: CV acc
  **0.970→0.771** as noise dims grow, near/far ratio **0.121→0.909** (→1, the *why*, Beyer 1999). Curse
  via `cross_val_score` on TRAIN (test stays used-once); near/far = pure geometry. Both reviewers PASS
  (re-derived leak-free, near/far = Beyer's inverse relevant-contrast confirmed; applied MINORs:
  plan-table→CV numbers, curse-is-CV framing + Pipeline-rescales note, asymptotic wording, prereqs
  +06/+08). Rémy validated. `feat(01_knn): notebook 05 — demanding case: breast cancer & the curse`.
  **Build note:** a stray `/tmp/struct.py` (a notebook cell-lister) shadows stdlib `struct` for `python
  /tmp/*.py` (sys.path[0]=/tmp) → run builders via `uv run python - < /tmp/x.py` (stdin) instead.
- **NB 4 (the estimator & its parameters) built & merged to `chapter/01_KNN`.** Introduced
  `KNeighborsClassifier`: parity with by-hand (0 diff, 0.956) and `cross_val_score` == NB 3's **0.919**;
  walked `n_neighbors` / `weights` / `metric`. weights uniform vs distance (k=151: 0.678 vs 0.833,
  boundary panels) — **mechanism corrected** after ml review: it is a graded 1/d *tilt* (effective
  ~73/151, nearest-15 hold ~34 %), NOT "near dominates", and 0.833 stays well below uniform@15's 0.956
  (degrades gracefully, doesn't recover). metric small effect (Manhattan p=1 0.967); chose weights by
  `cross_val_score` (distance 0.924 → test 0.967); even-k tie → lowest-label argmax (== by-hand
  `bincount().argmax()` convention) → odd k for binary (binary-only; 3-class can still tie). Pedagogy
  PASS; ml-expert REVISE→fixed (1 MAJOR: false distance-weight mechanism, re-measured & corrected; +
  MINORs: parity edge-case, argmax convention, figure wording). Rémy validated visually. `feat(01_knn):
  notebook 04 — the estimator & its parameters`.
- **NB 3 (the k dial) built & merged to `chapter/01_KNN`.** By-hand k-NN on `make_moons`: k as the
  bias–variance dial — boundaries k=1 (jagged/memorize) / 15 / 151 (over-smooth), contrasted with NB
  05's single bisector; train/test error vs k (log axis, the U; k=1 train err **0.000**); then the
  honest selection — show the test curve, **refuse to use it**, **5-fold CV on TRAIN picks k=15** (cv
  0.919), one sealed test eval → **0.956** (vs k=1 0.933, k=151 0.678). Surfaces (not hides) the
  test-min=k3 vs CV=k15 disagreement. CV by hand (`StratifiedKFold`+loop); `KNeighborsClassifier` /
  `cross_val_score` deferred to NB 4. Both reviewers **PASS** (re-derived CV leak-free; applied 6 fixes:
  variance-inferred-not-measured nod to NB 09, gap *indicates* overfit, CV-fold-seed wobble, odd-k
  explained, boundaries-on-train named, k15/0.956 seed-caveat). Rémy validated visually. `feat(01_knn):
  notebook 03 — the k dial`.
- **NB 2 (distance & the scale trap) built & merged to `chapter/01_KNN`.** By-hand k-NN on
  `make_moons`: Euclidean vs Manhattan (a genuine L1/L2 ranking flip — q=(0,0): L2 picks B=(0.7,0.7),
  L1 picks A=(1,0)), then the scale trap — feature 2 ×50 collapses test acc **0.956 → 0.733**,
  standardizing on **train stats** recovers it **exactly** (caveat stated: exact recovery is special to
  a pure rescale); raw-vs-standardized **decision boundary** (horizontal bands slicing the crescents →
  the two-moon S recovered, both PNGs eyeballed). Honest scoping: metric (0.956 vs 0.944 = **one** test
  point of 90) ≪ scale (~20 pts). Tiny in-notebook `ByHandKNN` wrapper reuses
  `viz.plot_decision_boundary` (single split; CV deferred to NB 3). Both reviewers **PASS**
  (re-measured; applied 5 fixes: "34× linear → ~34²≈1000× in the squared sum", metric-gap-is-one-point,
  ISLR §2.2.3 **& ch. 4**, a `ByHandKNN` framing cell naming the fit/predict interface, forced axis
  labels). Rémy validated visually. `feat(01_knn): notebook 02 — distance & the scale trap`.
- **NB 1 (predict by the neighbourhood vote) built & merged to `chapter/01_KNN`.** k-NN by hand on
  `make_moons(300, 0.30, 0)` (210/90): the majority vote, *k* = neighbourhood size, and the lazy cost
  **felt live** (fit flat ~µs vs predict super-linear at n=300/3000/30000). At build, the running query
  was **re-measured and corrected**: the planned q=(0.60,0.14) sat on the Bayes boundary (~50/50, so
  "true class 0" was dishonest); replaced with **q=(-0.23,0.75)** — a region only the class-0 crescent
  reaches (~85% class 0 → class 0 is the right answer), nearest training labels [1,0,0,0,0], votes
  k1=1/k3=0/k5=0. Both reviewers **PASS** (each re-measured & confirmed truth/votes/timing; 3 MINOR
  polish applied — NB09 ref → module-00 callback, "nothing learned by fitting", Cover&Hart asymptotic
  1-NN qualifier). Rémy validated visually. `common_errors.md` gained a KNN k=1-noise row; `llms.txt`
  regenerated. Commit `feat(01_knn): notebook 01 — predict by the neighbourhood vote`.
- Chapter 01 (k-NN) plan **approved & persisted** (`docs/plans/chapter_01_KNN.md`) — 6 notebooks: vote
  → distance/scale trap → k-dial → estimator/params → demanding case (breast_cancer + the curse) → an
  optional **NB 6 Advanced** (metric geometry L1/L2/L∞ + Mahalanobis/cosine, metric×curse, nested CV,
  and the silhouette≠k-NN clarification — a deliberate, Rémy-approved exception to the 5-ceiling).
  `make_moons` for NB 1–4 (penguins too separable, measured: 0/69 flips), breast_cancer for NB 5; all
  offline. Reviewer-gated (both REVISE→incorporated). `WORKFLOW.md` updated: chapter close now via PR
  (`main` is protected by a global pre-push hook).
- **Chapter 00 (Getting Started) COMPLETE — 11 notebooks, merged to `main` (`--no-ff`).** The on-ramp:
  what ML is → features/feature space → EDA → split & leakage → nearest centroid → accuracy/baseline →
  confusion/precision-recall → scores/ROC/AUC → over/under-fitting → cross-validation → preprocessing &
  leakage. Plus a fetch-and-cache data layer and a "get your data" step in NB 01. Two-reviewer gate +
  Rémy's visual validation throughout.
- NB 11 (preprocessing & leakage) built — standardization paying off NB 05's scale-sensitivity (NC
  boundary rotates ~52.6° in mm coords; CV 0.989→0.9927), one-hot encoding of `island` (by hand +
  `OneHotEncoder` fit-on-train), `ColumnTransformer`+`Pipeline` under CV, and the ESL §7.10.2 leakage
  demo (1000 pure-noise features: select-on-all 0.85 vs select-in-fold 0.57). Reviewer-gated (pedagogy
  PASS; ml-expert REVISE→fixed the boundary-rotation angle — my coordinate-space error), Rémy validated.
- NB 01 "Getting the data" step added (Phase 2 of the data-layer change): a fetch+cache section
  (intro md + `load_penguins_full()` with visible `logging` + a "Read the output") inserted after
  "Learning from labelled examples"; teaches fetching/caching as a first ML step, then narrows to the
  2-feature binary slice. Reviewer-gated (ml-expert PASS; pedagogy REVISE→fixed: framed the logging,
  explained the repeated cached-read, trimmed the forward-ref roster, first-run note; +`datasets.py`
  `sex` docstring MALE/FEMALE). Rémy validated visually, merged.
- Data-layer refactor (prereq for NB 11): `datasets.py` switched from a committed binary CSV to
  **fetch-and-cache** the full Palmer penguins set (pinned seaborn-data URL →
  `src/ml_course/data/penguins_full.csv`, git-ignored, offline after first run); added
  `load_penguins_full()` (344×7: +island, sex, bill_depth, body_mass, Chinstrap, missing values) +
  `PENGUINS_FULL_*` constants; `load_penguins()` now **derives** the 2-feature binary subset (verified
  byte-identical to the old committed 274-row CSV → NB 01–10 behaviourally unchanged); removed the
  committed `penguins.csv` + the `.gitignore` exception; repurposed `vendor_penguins.py` to warm the
  cache (logs the download). Tests 14 green; all 10 notebooks re-execute; ruff/black clean. Rémy chose
  fetch-and-cache + a visible fetch step in NB 01 (Phase 2 next). Logging used in `datasets.py` (hook).
- NB 10 (validating honestly: cross-validation) built — single notebook (Rémy chose over a 10a/10b
  split), 30 cells; continues NB 09's make_moons + `flexible_boundary`. Arc: parameters vs
  hyperparameters → the validation set → single-split instability (degree 3,3,5,6,3,9) → stratified
  k-fold BY HAND → CV picks degree 3 → by-hand == `cross_val_score` (0.914286, exact) → one honest
  test estimate (0.9111) → tuning-on-test inflation (+0.014 over 100 splits). No new `src/` (reused
  `plot_train_test_curve`; k-fold scheme + inflation bar in-notebook). Reviewer-gated (pedagogy PASS;
  ml-expert REVISE→ stratification-exactness MAJOR + minors fixed), Rémy validated, merged. Alongside:
  NB 01–09 made ruff-clean (option B, `f84eec6`).
- NB 09 (over-/under-fitting, the generalization gap) built — make_moons + polynomial-degree dial
  (linear engine as plumbing), complexity curve U, generalization gap (≠ variance), bias-variance,
  learning curve, optional variance fan; added `viz.plot_train_test_curve` + test. Reviewer-gated
  (pedagogy PASS; ml-expert REVISE→fixed the "train error always falls" vs measured kink), Rémy
  validated, merged.
- NB 08 (scores, thresholds, ROC & AUC) built — signed-distance score (`s>0` = centroid), threshold
  trade-off, ROC/AUC 0.989, PR curve, two-feature contrast AUC 1.0; added `viz.plot_roc_curve` +
  `plot_score_threshold` + tests. Reviewer-gated (both PASS; 2 minor polish), Rémy validated, merged.
  Closes the evaluation trilogy (06-08).
- NB 07 (confusion matrix, precision & recall) built — bill-only weaker model for real errors, cm
  [[37,1],[2,29]], precision 0.967 / recall 0.935 / F1 0.951, asymmetric costs. Reviewer-gated (both
  PASS; 3 minor polish), Rémy validated, merged.
- NB 06 (accuracy + baseline) built — accuracy formalised, DummyClassifier baseline, the accuracy
  paradox on an imbalanced what-if (Dummy 95%/0-of-2 vs centroid 100%/2-of-2), positive = Gentoo.
  Reviewer-gated (both PASS; 3 minor polish), Rémy validated, merged.
- NB 05 (first classifier: nearest centroid) built — by-hand fit/predict class, decision boundary
  (extended `plot_decision_boundary` to be pandas-first + label-agnostic, with a test), honest loop
  (100% test vs 55% baseline, by-hand = sklearn). Reviewer-gated (ml-expert REVISE→fixed, pedagogy
  PASS), Rémy validated, merged.
- NB 04 (generalize, don't memorize — stratified split + rote-memorizer demo) built, reviewer-gated
  (both PASS; convergent honesty MINOR fixed), Rémy validated, merged.
- NB 03 (look before you model — EDA) built (+ `viz.plot_class_balance` / `plot_feature_histograms`
  + tests), reviewer-gated (both PASS; polish applied), Rémy validated, merged. Also fixed NB 01 c06's
  dangling "fuller dataset" forward-reference (flagged by pedagogy reviewer).
- NB 02 (features, labels, the feature space) built, reviewer-gated (both PASS; 2 minor polish
  applied), Rémy validated visually, committed and merged into `chapter/00_GettingStarted`.
- NB 01 (what is machine learning?) built, reviewer-gated (pedagogy PASS; ml-expert REVISE→fixed:
  corrected the figure reading, softened the line claim, added subset honesty), Rémy validated
  visually, committed and merged into `chapter/00_GettingStarted`.
- Chapter 00 plan approved and persisted; `course_map.md` 00 section rewritten to 11 notebooks.
- Chapter 00 plan reviewer-gated (both REVISE→incorporated).
- Course scaffold + build workflow set up on `main` (3 infra commits).
