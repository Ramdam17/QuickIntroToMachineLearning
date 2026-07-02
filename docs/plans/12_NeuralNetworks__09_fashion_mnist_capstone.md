# NB plan — 12_NeuralNetworks / 09_fashion_mnist_capstone

> Status: **APPROVED by Rémy via ExitPlanMode (no edits) — 2026-07-01.** NB-plan stage = **Rémy validates
> alone** (no reviewer gate; both reviewers return on the built notebook). **NB 9 of 10** (chapter 12 — the
> PyTorch finale). **The demanding practical case — the visualization-first capstone**, in PyTorch. Anchors
> measured **on-box** (torch 2.12.1, CPU, determinism contract from NB 7; `measure_ch12_nb9_{a,b,c}.py`).
> Chapter plan: `docs/plans/chapter_12_NeuralNetworks.md` (APPROVED, §NB 9). **2nd `src/` change of the
> chapter** (the Fashion-MNIST loader + vendor script + tests → pytest rises from 22).

## Context
Chapter 12, **NB 9 of 10 — the demanding case.** The finale's capstone: a full, honest, end-to-end workflow on
**Fashion-MNIST**, mobilizing everything the chapter built (a deep, He-initialized, dropout-regularized torch
net in a scaled pipeline) and — the point of a capstone — **evaluated with rigor**: baselines, a read loss
curve, a held-out confusion matrix + error gallery, seed-variance, and a **fair cross-method foil** against
trees on raw pixels. **The workflow beats are applied, NOT re-taught** — baseline → loss curve → confusion →
error gallery → seed-variance → CV foil are the learner's *reflex* by the finale (ch 00 + every capstone since
KNN). NB 9's genuinely-new content is **the torch deep net as the thing under test, in a real pipeline**, and
the **honest verdict**: on this data a **tree even wins**, and the reason is the finale's punchline — **a dense
net flattens the image and throws away the spatial structure; that is a CNN's job** (the bridge to NB 10).
**Visualization-first** (≥6 figures). **Torch, shown not assigned** — the "Your turn" tiers *modify* the shown
pipeline.

## Recap vs new (the boundary)
- **RECAP (applied, not re-derived):** the whole evaluation workflow — train/test split, baselines, the loss
  curve as a convergence/optimization-failure diagnostic (NB 3/8), the confusion matrix + per-class error
  analysis (ch 11 capstone), seed-variance (ch 11 NB 5), a fair cross-method foil on CV (ch 11 capstone), the
  "report a number *with* its spread / no-universal-best" discipline (ch 10/11). The net's knobs — He (NB 4),
  dropout (NB 5), Adam (NB 8), mandatory scaling for a net (ch 11) — each named from the concept that owns it,
  not re-taught. The `nn.Module` / canonical loop / `.train()`/`.eval()` idiom (NB 7/8).
- **NEW (this NB owns):** the **deep torch net dropped into a real, scaled pipeline on a real image dataset**;
  the **honest capstone verdict** (competitive, not superior — a tree wins ~1.3 pp); and the **spatial-structure
  reading** — the flattened-image argument + a first-layer-weight gallery → **why images want a CNN** (c09, the
  NB-10 bridge). Fashion-MNIST itself (the loader is the chapter's 2nd `src/` change).
- **No forward references built:** CNN / RNN / transformer are **named as the horizon** (NB 10), never built.

## Live anchors (measured ON-BOX; re-pinned at build) — `measure_ch12_nb9_{a,b,c}.py`, determinism contract, SEED=0
Fashion-MNIST via `fetch_openml('Fashion-MNIST', v1)` → **70000×784, balanced (7000/class)**; fetch **~2 s**
(cached thereafter). Stratified **10k-train / 5k-test** subset (balanced 1000/class per split), pixels scaled
to [0,1] then `StandardScaler` (fit on train) for the net. **The net (SHOWN harness):** `784→256→128→10`,
**He init**, **Dropout 0.2**, ReLU, **Adam lr 1e-3**, mini-batch 128, 30 epochs — **1.4 s / training** on CPU
under the determinism contract (so the whole capstone, ~16 trainings, is trivial on the torch side; the tree
foil — HistGB especially — is the only slow part, ~90 s for its 5 folds).

- **(A) Baselines** (sealed split): **dummy (most-frequent) 0.100** (= chance, 10 balanced classes);
  **logistic on scaled pixels 0.803** (converged, `max_iter=2000`, n_iter 492) — a linear model already
  clears 0.80, so the bar for "deep learning earns its keep" is beating **0.80**, not chance.
- **(B) The deep net** (sealed 5k test): **test acc 0.8646**; loss **0.797 → 0.109** over 30 epochs (smooth,
  monotone — converged, no flat-curve optimization failure). Beats the linear baseline by ~6 pp.
- **(C) Per-class accuracy + confusion** (sealed 5k test, 677/5000 = 13.5 % errors): easy classes —
  **Trouser 0.978, Bag 0.958, Sneaker 0.950, Sandal 0.926, Ankle boot 0.914**; hard classes — **Shirt 0.614
  (worst), Pullover 0.772, T-shirt/top 0.834, Coat 0.842, Dress 0.858.** Top confusions (true→pred):
  **Shirt→T-shirt 72, Pullover→Coat 66, Shirt→Coat 55, Shirt→Pullover 51, T-shirt→Shirt 45, Coat→Pullover 43,
  Ankle boot→Sneaker 37.** Every error is *semantically honest* — the upper-body garments look alike at 28×28;
  footwear/bag/trouser are near-perfect.
- **(D) Seed-variance** (10 seeds, sealed test): **mean 0.8629, std 0.0042, range [0.8576, 0.8714]** — a
  ~1.4 pp band. The single-split 0.8646 sits inside it; any comparison closer than ~0.01 is within noise.
- **(E) Fair cross-method foil — 5-fold CV on the 10k subset** (torch net scaled per fold; RF / HistGB on
  **raw** pixels — trees need no scaling): **MLP (torch, scaled) 0.863 ± 0.005** (5.2 s / 5 folds);
  **RF (raw) 0.855 ± 0.005** (1.8 s); **HistGB (raw) 0.876 ± 0.009** (87 s). **Verdict: HistGB wins by
  ~1.3 pp; MLP and RF are tied within the ~0.01 noise band.** *Honest compute nuance (measured):* the
  **winning** tree (HistGB) is the **most expensive** of the three; RF — cheapest, no scaling — merely
  **ties** the net. So "a tree wins" is not "a cheap tree wins": the honest reading is **no free lunch on
  either accuracy or compute**, the finale's "no universal best".
- **(F) Tabular-humility aside** (`breast_cancer` 569×30, 5-fold CV — the ch-11 fact, re-verified): **MLP
  (scaled) 0.979**, **RF 0.965**, **HistGB 0.970** → an MLP *does* win on clean homogeneous tabular. "No
  universal best" lands **both ways**: match the tool to the data's structure.
- **(G) First-layer weights** `W1 (256, 784)` → each row a **28×28 filter** (renderable, |W1| ∈ [−0.32, 0.33]);
  a dense net rediscovers crude local strokes by brute force — a CNN builds locality in by design (the c09 bridge).

## Cell-by-cell (~31 cells, 7 figures) — visualization-first; every figure has a "Read the figure"; ends with "Your turn"
1. **(md) Header** `# 09 — Capstone: Fashion-MNIST, end to end`. **Prerequisites:** the whole chapter (NB 1–8);
   the evaluation workflow (ch 00 train/test + CV; the ch-11 capstone reflex). **What you'll do:** run a full
   honest workflow on a real image dataset — look at the data → baselines → a deep torch net in a scaled
   pipeline → loss curve → confusion matrix + error gallery → seed-variance → a fair foil against trees → the
   honest verdict. **The workflow is your reflex now; the new thing under test is the deep net — and the
   verdict may surprise you.** You *read and turn* the pipeline; it is shown.
2. **(md) The dataset, and the honest question.** Fashion-MNIST — 70 000 grayscale 28×28 images, 10 clothing
   classes, a drop-in **harder** replacement for MNIST digits. The capstone question is not "can a neural net
   do this" but the honest one: **is a deep net the right tool for this data — and how do we know?**
3. **(code) Setup + load.** The determinism contract (seeds + `use_deterministic_algorithms(True)` +
   `set_num_threads(1)`); **`load_fashion_mnist()`** (the new loader — fetch-and-cache, INFO-logged, never
   silenced); the stratified **10k / 5k** subset; class names; print shapes + class balance (7000/class full,
   1000/class per split).
4. **(code) Fig 1 — image gallery.** A grid (10 classes × a few examples), each labelled. Grayscale, charter
   frame. (Inline gallery — no `viz` helper, as ch 11 did.)
5. **(md) Read Fig 1.** What the data is; **why it is hard** — shirt / T-shirt / pullover / coat are genuinely
   ambiguous at 28×28 in grayscale; footwear and bags are distinctive. And a seed of the punchline: **each
   image is a 2-D grid — the pixels have spatial structure.**
6. **(md) Baselines first — always anchor against the trivial.** The reflex (ch 00): a number means nothing
   without a floor. Two floors — **majority-class** (chance, 0.10 for 10 balanced classes) and a **linear model
   on scaled pixels** (logistic regression). Whatever the deep net scores, it must clear *these*.
7. **(code) Baselines.** `DummyClassifier(most_frequent)` → 0.100; `LogisticRegression` on scaled pixels →
   0.803. Print both.
8. **(md) Read the baselines.** A *linear* model already reaches 0.80 — the pixels are far from random. So the
   bar for "deep learning earns its keep" is beating **0.80 by a real margin**, not beating chance. Honest
   framing set.
9. **(md) The deep net — the thing under test.** A deep, **He-initialized** (NB 4), **dropout-regularized**
   (NB 5) net, trained with **Adam** (NB 8) in a **scaled pipeline** (a net needs scaling — ch 11). Every knob
   comes from a concept you built. **Shown, not assigned.**
10. **(code) `build_net` + `train` (SHOWN).** `784→256→128→10`, He init, `nn.Dropout(0.2)`, ReLU, Adam lr 1e-3,
    batch 128, 30 epochs; `.train()`/`.eval()`. Fit `StandardScaler` on train, train on 10k, record the loss
    curve, evaluate on the sealed 5k. Print **test acc 0.865**.
11. **(code) Fig 2 — the loss curve.** Training loss 0.797 → 0.109 over 30 epochs. Charter `model` colour.
12. **(md) Read Fig 2.** A smooth, monotone descent that flattens — **converged, not stalled** (contrast the
    flat-at-`ln K` optimization failure of NB 3/8). 0.865 clears the 0.80 linear baseline: the depth bought a
    real ~6 pp. But accuracy is one number — *where* does it err?
13. **(md) Held-out evaluation — the confusion matrix.** One number hides structure; the confusion matrix (ch-11
    reflex) shows which classes the net confuses.
14. **(code) Fig 3 — confusion matrix** (`viz.plot_confusion_matrix`, 10×10, class names) on the sealed 5k.
15. **(md) Read Fig 3.** The diagonal dominates; the off-diagonal mass concentrates in the **upper-body
    garments** — Shirt (0.614, the worst) bleeds into T-shirt / Coat / Pullover; footwear, bag and trouser are
    near-perfect (0.91–0.98). The errors are **structured, not random.**
16. **(code) Fig 4 — error gallery.** A grid of *misclassified* test images with `true → pred` captions, drawn
    from the shirt/top confusions (charter `error` accent on the caption).
17. **(md) Read Fig 4.** The mistakes are **honest** — most are images a person would hesitate on at this
    resolution. The model is not broken; the discriminating signal is thin in 28×28 grayscale. (First hint that
    the *data representation*, not the model size, is the ceiling.)
18. **(md) One split is one draw — seed-variance.** The sealed 0.865 is a single initialization. Retrain with
    different seeds to see the spread (the ch-11-NB-5 lesson: report a number *with* its uncertainty).
19. **(code) Seed-variance.** 10 seeds on the sealed test → mean 0.863, std 0.004, range [0.858, 0.871]. Print.
20. **(code) Fig 5 — seed-variance strip.** 10 per-seed dots + the mean/±std band. Charter colours.
21. **(md) Read Fig 5.** A ~1.4 pp band around 0.863 from initialization alone. **Any two methods closer than
    ~0.01 are tied within noise** — the ruler we take into the foil.
22. **(md) The fair foil — is a deep net even the right tool?** Compare against **Random Forest and HistGB on
    RAW pixels** — trees need **no scaling** (the preprocessing difference *is* the point) — under the **same
    5-fold CV, same metric**. A capstone earns its verdict by comparing honestly.
23. **(code) CV foil.** 5-fold CV on the 10k subset: MLP (scaled, per-fold) vs RF (raw) vs HistGB (raw). Print
    mean ± std for each (0.863 / 0.855 / 0.876).
24. **(code) Fig 6 — cross-method CV bars.** MLP / RF / HistGB with ±std error bars. Charter colours; the
    winner (HistGB) in `highlight`.
25. **(md) Read Fig 6 + THE VERDICT.** **MLP 0.863 ≈ RF 0.855 < HistGB 0.876** — HistGB wins by ~1.3 pp; MLP
    and RF tie within the ~0.01 noise band. The deep net is **competitive but not superior; a tree wins**, and
    the trees need no scaling. The honest twist: the *winning* tree (HistGB) is also the *slowest* to train,
    while RF ties the net for a fraction of the cost — so more machinery buys neither guaranteed accuracy nor
    guaranteed speed. This is the finale's "**no universal best**".
26. **(md) Why a tree wins here — the punchline.** A dense net **flattens** the 28×28 image into 784 independent
    inputs — it **throws away spatial structure** (which pixels are neighbours). It has to *rediscover* locality
    from scratch, from 10k examples. The right inductive bias for images — **weight sharing + locality** — is a
    **convolutional network** (c09). *Optional evidence:* the first-layer filters.
27. **(code) Fig 7 — first-layer weight gallery** (optional). A few rows of `W1` reshaped to 28×28 — blurry
    stroke/edge-like patterns. Charter diverging map.
28. **(md) Read Fig 7.** Some filters resemble crude edges/strokes — the dense net **stumbles onto** local
    features by brute force; a CNN **builds them in** and shares them across the image. **This is the bridge to
    NB 10.**
29. **(md) Tabular-humility aside — "no universal best" both ways.** The lesson is not "trees beat nets". On
    **clean, homogeneous tabular** data an MLP *does* win — `breast_cancer` 5-fold CV: **MLP 0.979 > HistGB
    0.970 > RF 0.965** (ch 11, re-verified). The point is structural: **match the tool to the data's structure**
    — dense nets shine on homogeneous vectors, trees on heterogeneous tabular, CNNs on images.
30. **(md) What you built + Your turn.** You ran a **complete, honest** deep-learning workflow and reached a
    verdict that respects the data. **Where next: NB 10** — where ML goes (CNN / RNN / transformer) and the
    whole-course synthesis. **Your turn** (each *modifies the shown pipeline*): *(warm-up)* raise
    `nn.Dropout` to 0.5 and retrain — does stronger regularization move the ~0.865, or is the ceiling
    elsewhere?; *(core)* add a third hidden layer (e.g. `256→128→64`) **or** train 60 epochs and watch the
    train-vs-test gap — does more capacity/training beat 0.865, or just overfit?; *(reach)* rebuild the
    confusion matrix **restricted to the four upper-body classes** (T-shirt, Pullover, Coat, Shirt) — is the
    confusion *intrinsic* to 28×28 pixels (low accuracy even head-to-head)? Connect your answer to the CNN
    argument.
31. **(md) References.** Xiao, H., Rasul, K., & Vollgraf, R. (2017). Fashion-MNIST: a Novel Image Dataset for
    Benchmarking Machine Learning Algorithms. *arXiv:1708.07747*. LeCun, Y., Bottou, L., Bengio, Y., & Haffner,
    P. (1998). Gradient-based learning applied to document recognition. *Proc. IEEE* 86(11):2278–2324
    (DOI 10.1109/5.726791). Recap: He et al. 2015 (init; arXiv:1502.01852), Srivastava et al. 2014 (dropout;
    JMLR 15:1929–1958), Kingma & Ba 2015 (Adam; arXiv:1412.6980), Paszke et al. 2019 (PyTorch). *Previous:*
    **12.8 — the model and its parameters.** *Next:* **12.10 — where ML goes next, and the whole course.**

## `src/` change (the 2nd of ch 12) — validated by new tests → pytest rises from 22
- **`datasets.py`:** add **`load_fashion_mnist()`** (and **`load_mnist()`** — same pattern, near-free, and it
  lets a curious learner swap datasets) mirroring `_ensure_full_csv`: `_ensure_fashion_mnist()` fetches via
  `fetch_openml('Fashion-MNIST', version=1)`, **caches as `.npz`** under `src/ml_course/data/` (git-ignored),
  **INFO-logged (never silenced)**; a **subset helper** `fashion_mnist_subset(n_train, n_test, *, seed)`
  (stratified split); a **`FASHION_MNIST_CLASSES`** constant (the 10 names). **Decision (validated):** for
  **image pixels** the loader returns **numpy arrays** `(X float32 [n,784] in [0,1], y int [n])`, *not* a
  DataFrame — this is the CLAUDE.md "numpy under the hood where numpy is right" carve-out (784 pixel columns
  have no meaningful names; images are arrays). The docstring states this explicitly.
- **`scripts/vendor_fashion_mnist.py`:** mirror `vendor_penguins.py` — warm the offline cache and print
  shape + class balance. `uv run python scripts/vendor_fashion_mnist.py`.
- **`tests/test_datasets.py`:** add tests for `load_fashion_mnist` (shape 70000×784, 10 balanced classes, dtype
  / range, subset helper returns the requested balanced sizes) and a light `load_mnist` smoke test → **pytest
  rises from 22** (state the exact new total at build). The cache is git-ignored (`*.npz`), as `penguins_full.csv`
  already is (confirmed untracked).
- **`viz.py`:** **no change** — reuse `plot_confusion_matrix`; the image gallery / error gallery / weight
  gallery / seed strip / CV bars are **inline** (single-use, as ch 11 did).

## Decisions baked in (validated by Rémy)
1. **Capstone = Fashion-MNIST**, 10k/5k stratified subset (offline, CPU; the net trains in ~1.4 s). [chapter ✅]
2. **Torch, shown not assigned** — one shown `build_net`/`train` pipeline; the 3 "Your turn" tiers *modify* it.
3. **7 figures** — image gallery, loss curve, confusion matrix, error gallery, seed-variance strip, cross-method
   CV bars, (optional) first-layer-weight gallery. **Visualization-first** (≥6 required; capstone floor is ~20
   cells, not a ceiling).
4. **The workflow beats are applied, not re-taught** (the finale reflex); the genuinely-new content is the
   torch net in the pipeline + the spatial-structure verdict.
5. **Honest verdict — measured, not hidden:** **MLP 0.863 ≈ RF 0.855 < HistGB 0.876** (HistGB wins ~1.3 pp;
   re-pinned at build); the punchline is **"images want a CNN"** (NB 10 bridge), not an MLP victory lap. The
   **tabular-humility aside** makes "no universal best" cut both ways (`breast_cancer` MLP 0.979 wins).
6. **Diagnostics on the sealed 10k/5k split** (confusion, error gallery, seed-variance, first-layer); the
   **cross-method comparison on 5-fold CV** (the ch-11 capstone protocol — CV for the honest ordering).
7. **`src/`:** the Fashion-MNIST (+ MNIST) loader returns **arrays** (image carve-out); vendor script + tests
   → **pytest rises from 22**.

## Verification (end-to-end, at build)
1. `src/` first: write `load_fashion_mnist`/`load_mnist` + subset helper + `FASHION_MNIST_CLASSES`, the vendor
   script, and the tests; **`uv run pytest`** green at the new total; `uv run ruff check .` clean.
2. nbconvert a scratchpad copy → **exit 0, 7 figures**, anchors reproduce: baselines 0.100 / 0.803; net 0.865,
   loss 0.797→0.109; per-class (Shirt 0.614 … Trouser 0.978) + top confusions; seed-variance 0.863 ± 0.004;
   **CV foil MLP 0.863 ≈ RF 0.855 < HistGB 0.876**; breast_cancer MLP 0.979 > 0.970 / 0.965.
3. hex clean; **banned-word scan 0**; ruff/black clean (black skips `.ipynb`); output-free; determinism contract
   in-notebook; **"Read the figure" after every figure** (7/7); the one-time fetch **logged, never silenced**.
4. **pytest at the new total** (state it). 5. **Rémy validates this NB plan** (no reviewer gate; both reviewers
   return on the built notebook).
