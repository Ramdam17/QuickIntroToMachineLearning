# NB plan — 12_NeuralNetworks / 08_model_and_parameters

> Status: **APPROVED by Rémy (via ExitPlanMode, 2026-06-30, no edits).** NB-plan stage = **Rémy validates
> alone** (no reviewer gate; both reviewers return on the built notebook). **NB 8 of 10** (chapter 12 — the
> PyTorch finale). The **estimator notebook** (integrative), in PyTorch — **torch, SHOWN not assigned**.
> Anchors measured **on-box** (torch 2.12.1, CPU, determinism contract from NB 7; `measure_ch12_nb8.py`).
> Chapter plan: `docs/plans/chapter_12_NeuralNetworks.md` (APPROVED, §NB 8).

## Context
Chapter 12, **NB 8 of 10 — the model and its parameters, in PyTorch.** This is the finale's version of the
per-method "estimator" notebook (like ch 11 NB 4): we take the real torch net and **turn every knob, each
introduced from the concept that owns it**, reading honestly what each does and how it fails. The knobs and
their homes: **capacity** (depth/width — here); **`activation`** as a *tunable* depth-pathology (sigmoid on a
deep net → the c02 stall, NB 3); **initialization** via `torch.nn.init` (He/Xavier — c03, NB 4); the
**optimizer menu** (SGD → momentum → Adam — c07, why depth makes them matter); **learning rate**; real
**`nn.Dropout`** (c05, NB 5) and **`nn.BatchNorm`** (c04, NB 6) as one-line layers (now `.train()/.eval()`
matters — NB 7); **epochs / batch size** (ch 11 recap). Then the diagnostic spine: the **train-vs-validation
loss curve** separates the three failure modes — underfitting, overfitting, and **optimization failure** (the
flat curve — c06) — and honest **tuning on validation → one sealed test**. Everything is **shown to read**;
the "Your turn" tasks *modify* the shown harness, never author it from blank.

## Recap vs new (the boundary)
- **RECAP (named, not re-derived):** the by-hand meaning of each knob — depth-as-hierarchy (NB 2),
  vanishing/exploding (NB 3), He/Xavier (NB 4), dropout (NB 5), batch norm (NB 6); `nn.Module`/the canonical
  loop/autograd/`.train()`/`.eval()` (NB 7); epoch/mini-batch/learning-rate/loss-curve/over-underfitting/CV
  (ch 11 NB 4, ch 00). `alpha`(L2)/early-stopping/Adam were *named* in ch 11 → one tight recap.
- **NEW (this NB owns):** the **real torch layers/knobs** (`nn.Dropout`, `nn.BatchNorm1d`, `torch.nn.init`,
  `optim.SGD(momentum=)`, `optim.Adam`) as one-liners on NB 7's loop; **`activation` as a tunable failure**;
  the **optimizer menu** (c07, a deepened recap — kept short); the **optimization-failure reading** of a flat
  loss curve (c06); the honest **tune-on-val → sealed-test** workflow in torch.
- **No forward references:** the Fashion-MNIST capstone is **NB 9**; CNN/RNN/transformer are **NB 10** horizons.

## Live anchors (measured ON-BOX; re-pinned at build) — `measure_ch12_nb8.py`, determinism contract, SEED=0
A flexible `build_net(width, depth, activation, dropout, batchnorm, init)` + a `train(...)` loop recording
train+val loss. Two datasets: **moons** (2-D, noise 0.2, 400 pts, split 0.3, scaled) for capacity / activation
/ init / optimizer / lr; **make_classification** (50 feat / 10 informative, high-dim) for the regularizer &
tuning demos (dropout/BN bite in high dimensions — NB 5/6).

- **(A) Capacity** (moons, Adam lr 0.05, 300 ep): tiny `w2·d1` (12 params) train **0.868** / val 0.925
  (**underfits**); `w8·d1` 0.971/0.958; wide `w64·d1` (322 p) 0.982/**0.967** (best); deep `w32·d3` (2274 p)
  **1.000**/0.950 (train saturates, val dips — **mild overfit**). Capacity did the lifting; past enough, the
  gap opens.
- **(B) `activation` as a depth-pathology knob** (deep `w16·d8`, **SGD lr 0.1, He init**): relu loss
  0.758→**0.068** / val 0.942; tanh 0.507→0.076 / 0.950; **sigmoid 0.790→0.693 (flat at ln 2) / val 0.800 —
  STALLS.** *Honesty:* with **Adam** the deep sigmoid partly recovers (loss→0.324) — a robust optimizer masks
  it, but the **structural fix (ReLU) is the real answer** (the NB-3 "depth, not the optimizer" lesson).
- **(C) Optimizer menu** (moons `w32·d2`, lr 0.05, 150 ep — epochs to val-loss < 0.35): **SGD 42** (val 0.917);
  **momentum 10** (0.942); **Adam 1** (0.958). Same net; Adam converges fastest.
- **(init)** (moons deep `w16·d8`, SGD lr 0.1): **small train 0.500** (loss flat 0.693); **default 0.550**
  (0.704→0.693 — even torch's default stalls this deep with plain SGD); **He 0.942** (0.722→**0.074**).
- **(D) Regularizers at fixed capacity** (high-dim, train 200 / val 500, `w256·d2`, Adam lr 0.01, 400 ep):
  none train 1.000 / val **0.712** / gap 0.288; **dropout 0.5 → val 0.748 / gap 0.252** (best); weight-decay
  1e-2 → 0.738 / 0.262; **batch norm → 0.722 / 0.278 (gentle)**. All shrink the gap; dropout most here, BN
  least — the NB-5/6 honesty (BN's win was robustness, not raw accuracy on a small net).
- **(F) Learning rate** (moons `w32·d2`, Adam, 150 ep): **1e-4 → val 0.900** (crawls, loss stuck 0.548);
  **1e-2 → 0.967** (best, loss 0.055); 0.1 → 0.942; **1.0 → 0.850** (loss 0.349, unstable).
- **(G) Tune on validation → ONE sealed test** (high-dim, 3-way split train 200 / val 500 / **test 500 sealed**,
  Adam lr 0.01, 300 ep; grid width∈{64,256} × dropout∈{0,0.5}): val — w64/0.0 **0.718**, w64/0.5 0.700,
  w256/0.0 0.712, **w256/0.5 0.738 (best)** → **sealed test 0.756** (val ≈ test → the choice generalizes; no
  optimism from picking on val). *Honesty:* which config wins is **data-dependent / within split noise** (the
  ch-11-NB-5 lesson) — the discipline (tune on val, confirm once on a sealed test) is the transferable part.

## Cell-by-cell (~30 cells, 4 figures) — intuition → SHOW → "Read the figure"; ends with "Your turn"
1. **(md) Header** `# 08 — The model and its parameters, in PyTorch`. **Prerequisites** (NB 1–7; ch 11 NB 4 =
   the sklearn estimator notebook; ch 00 over/underfitting + CV). **What you'll do:** turn every knob of the
   real torch net — capacity, activation, init, optimizer, learning rate, dropout/batch norm — read what each
   does and how it fails, and finish with honest tuning → a sealed test. **You read and *turn* the knobs; the
   harness is shown.**
2. **(md) The knobs, and where each comes from.** A short map tying every knob to the concept that owns it:
   capacity (this NB), `activation` (c02 / NB 3), init (c03 / NB 4), dropout (c05 / NB 5), batch norm (c04 /
   NB 6), the optimizer menu (c07 — mostly new), learning rate / epochs / batch size (ch 11). One reusable
   harness, many knobs.
3. **(code)** setup: the **determinism contract** (seeds + `use_deterministic_algorithms(True)` +
   `set_num_threads(1)`); the two datasets (moons 2-D + high-dim `make_classification`, scaled); a flexible
   **`build_net(width, depth, activation, dropout, batchnorm, init)`** and **`train(...)`** (records train+val
   loss, returns accuracies) — **SHOWN** as the reusable harness. Print shapes.
4. **(md) Capacity — too little underfits, too much overfits.** Depth×width is the model's capacity; the honest
   version of NB 2 (depth helps *modestly* on flat data).
5. **(code)** capacity sweep → table (tiny `w2·d1` / small `w8·d1` / wide `w64·d1` / deep `w32·d3`: train, val,
   #params) — tiny 0.868/0.925, wide 0.982/0.967, deep 1.000/0.950.
6. **(code) Fig 1 — capacity**: grouped **train vs val** accuracy bars across the four nets (tiny: train low =
   underfit; wide: both high; deep: train 1.0, val dips = the gap opens). Charter `train`/`test` colours.
7. **(md) Read Fig 1** — capacity buys fit up to a point; past "enough", training accuracy hits 1.0 while
   validation dips — the overfitting gap. More layers is not free (NB 2's honest lesson, in torch).
8. **(md) `activation` — a tunable depth-pathology.** Recap c02/NB 3: on a deep net, sigmoid saturates → the
   gradient vanishes → training stalls. In torch this is one argument. Flip it and watch.
9. **(code)** deep net (`w16·d8`, SGD lr 0.1, He init), `activation` ∈ {relu, tanh, sigmoid} → table + recorded
   loss histories (relu 0.758→0.068 / tanh 0.507→0.076 / **sigmoid 0.790→0.693 flat**).
10. **(code) Fig 2 — the activation stall**: training-loss curves — relu & tanh descend to ~0, **sigmoid flat
    at `ln 2`** (stuck at chance). Charter colours + a dotted `ln 2` guide.
11. **(md) Read Fig 2 + honesty** — the deep sigmoid net never leaves `ln 2`: its gradient vanished (NB 3),
    now surfaced as a *knob*. **A robust optimizer (Adam) partly rescues it** (loss → 0.324), but the real fix
    is **structural** — use ReLU (and good init / normalization). "Depth, not the optimizer" (NB 3), restated.
12. **(md) Initialization — `torch.nn.init`.** Recap c03/NB 4: the initial weight *scale* must match depth and
    activation. He for ReLU, Xavier for tanh. We set it explicitly with `torch.nn.init` on a deep net.
13. **(code)** deep ReLU net (`w16·d8`, SGD lr 0.1), init ∈ {small, torch-default, He} → table + histories
    (small **0.500** flat / default **0.550** flat / **He 0.942**, 0.722→0.074).
14. **(code) Fig 3 — init**: training-loss curves — small and torch-**default** flat at `ln 2`, **He descends**.
    Charter colours + `ln 2` guide.
15. **(md) Read Fig 3** — even torch's *default* init stalls this eight-layer net under plain SGD; He rescues
    it (NB 4). Init is not a formality at depth; `torch.nn.init.kaiming_normal_` is the one-liner for it.
16. **(md) The optimizer menu (c07).** A short, honest recap: **SGD** (the loop you built) → **+ momentum** (a
    running average of past gradients — rolls through small bumps) → **Adam** (per-parameter adaptive step
    sizes). They matter more with depth. (Kingma & Ba 2015.)
17. **(code)** optimizer table (same net, moons `w32·d2`, lr 0.05): SGD / momentum / Adam — final train loss,
    val acc, **epochs to val-loss < 0.35** (42 / 10 / 1). Adam converges fastest here.
18. **(md) Learning rate — the single most important knob** (and batch size, one line: mini-batches you met
    in ch 11; `train(..., batch=)` is the argument).
19. **(code)** lr sweep table (Adam, moons): 1e-4 crawls (val 0.900) / **1e-2 best (0.967)** / 0.1 (0.942) /
    1.0 unstable (0.850). Too small crawls; too big overshoots.
20. **(md) Regularizers as one-line layers.** `nn.Dropout(p)` (NB 5) and `nn.BatchNorm1d` (NB 6) drop into the
    `nn.Sequential` — the by-hand layers you built, now one line each. **And now `.train()`/`.eval()` matters**
    (NB 7): dropout is off and batch norm uses its running statistics at eval.
21. **(code)** fixed big net (`w256·d2`) on the high-dim overfit problem: none / dropout 0.5 / batch norm /
    weight-decay 1e-2 → train/val/gap table (none gap 0.288; dropout 0.252; wd 0.262; BN 0.278).
22. **(code) Fig 4 — regularizers**: grouped **train vs val** bars for none / dropout / BN / weight-decay — all
    keep train at 1.0 but lift validation / shrink the gap. Charter colours.
23. **(md) Read Fig 4 + honesty** — every regularizer narrows the train–val gap; **dropout helps most here,
    batch norm least** (its win was robustness, not raw accuracy on a small net — NB 6). *Which* wins is
    data- and capacity-dependent; the robust fact is that they trade a little training fit for generalization.
24. **(md) Reading the loss curve: three failure modes (c06).** The train-vs-validation curve diagnoses which
    problem you have: **underfitting** (train accuracy never gets high — Fig 1 tiny; fix: capacity),
    **overfitting** (train high, validation lags — Fig 4 none; fix: a regularizer / less capacity),
    **optimization failure** (loss flat from the start — Fig 2 sigmoid, Fig 3 small init; fix: activation /
    init / normalization). A flat loss is **not** convergence — it is a net that never started (NB 3).
25. **(md) Honest tuning → one sealed test.** Search knobs on a **validation** set; touch the **test** set
    exactly once, at the end. Never tune on test.
26. **(code)** 3-way split (train 200 / val 500 / **test 500 sealed**); small grid width∈{64,256} ×
    dropout∈{0,0.5} scored on validation → pick the best (**w256/0.5, val 0.738**) → evaluate **once** on the
    sealed test (**0.756**).
27. **(md) Read the tuning result + honesty** — validation ≈ test (0.738 ≈ 0.756), so the choice generalizes
    and picking on validation did not inflate it. **Which config wins is within split noise / data-dependent**
    (the ch-11-NB-5 lesson) — the transferable part is the *discipline*: tune on validation, confirm once on a
    sealed test.
28. **(md) What you built** — turned every knob of the real torch estimator, each from the concept that owns
    it: capacity, `activation` (a tunable c02 stall), `torch.nn.init` (c03), the optimizer menu (c07), learning
    rate, `nn.Dropout` (c05) and `nn.BatchNorm` (c04); read the three failure modes off the loss curve (c06);
    and tuned honestly → a sealed test. You can now drive a PyTorch model, not just run one.
29. **(md) Where this goes next + Your turn.** **NB 9 — the capstone**: everything at once on **Fashion-MNIST**,
    end to end, with an honest verdict. *Your turn* (each **modifies the shown harness**): *(warm-up)* re-run
    the capacity sweep with `activation='tanh'` — does the picture change?; *(core)* add `nn.BatchNorm1d` to
    the deep net that stalled at small init (Fig 3) and retrain with SGD — does normalization rescue it where
    init did?; *(reach)* extend the tuning grid with a learning-rate axis {1e-3, 1e-2} and re-read the sealed
    test — did more search change the winner?
30. **(md) References** — Kingma, D. P., & Ba, J. (2015). Adam: A Method for Stochastic Optimization. *ICLR*
    (arXiv:1412.6980). He et al. 2015 (init), Glorot & Bengio 2010 (Xavier), Srivastava et al. 2014 (dropout),
    Ioffe & Szegedy 2015 (batch norm) — recap from NB 4/5/6. Paszke et al. 2019 (PyTorch). *Previous:* **12.7 —
    hello-world in PyTorch.** *Next:* **12.9 — capstone: Fashion-MNIST, end to end.**

## `src/` & guards
- **No `src/` change** (notebook-local torch harness; the `deep` extra already shipped in NB 7 [`08fbcf2`];
  sklearn `make_moons` / `make_classification` / `train_test_split` / `StandardScaler`; `viz`/`colors`; inline
  matplotlib). **pytest stays 22** (no new `src/`; the torch-determinism test from NB 7 still guards the
  contract).
- **Decisions baked in (Rémy validated):** (1) **torch, SHOWN not assigned** — one reusable harness, turned
  knob by knob; the 3 "Your turn" tiers *modify* it. (2) **4 figures** (capacity / activation stall / init /
  regularizer gap); optimizer, lr, and tuning as **printed tables** (the ch-11-NB-4 template — don't overload).
  (3) **Two datasets** — moons (2-D) for the trainable-knob demos, high-dim `make_classification` for
  dropout/BN + tuning (they bite in high dimensions — NB 5/6). (4) **Honesty:** the activation stall is stark
  under **SGD** and *partly masked by Adam* (structural fix is the answer); BN is the **gentlest** regularizer
  here; the tuning winner is **within split noise** — stated, not hidden. (5) **`.train()/.eval()` now matters**
  (dropout/BN) — the NB-7 beat realized.
- Determinism contract in-notebook (verified on-box — no op used [Linear / ReLU / Dropout / BatchNorm1d /
  SGD / Adam] errors under `use_deterministic_algorithms(True)`); **never silence output**; colours only from
  `ml_course.colors` (no hardcoded hex); `SEED=0`; **"Read the figure" after every figure**; banned-word scan 0;
  ruff/black clean; output-free.
- Build from `build_ch12_nb8.py` (source of truth in the **ephemeral** scratchpad; rebuild right before
  `git add` — kernel-drift guard).
- Exit guards: nbconvert exit 0 (4 figures) with the determinism set; anchors reproduce (capacity 0.868→0.967→
  overfit; sigmoid flat at ln 2 vs relu/tanh; small/default init flat vs He; optimizer 42/10/1; lr 1e-4/1e-2/1.0;
  regularizer gaps; tuning val 0.738 → sealed test 0.756); banned 0, hex clean, ruff clean, output-free; **pytest
  22**; **two-reviewer gate** (`@ml-expert-reviewer` + `@pedagogy-reviewer`, no BLOCK) → **Rémy visual** →
  end-of-NB checklist (`gen_llms_txt`, `common_errors` +rows, `course_map` §12 → NB 8 built, STATE) → commit
  `feat(12_neuralnetworks): notebook 08 — model and parameters` → `git merge --ff-only` into `chapter/12_NeuralNetworks`.

## Verification (end-to-end)
1. nbconvert a scratchpad copy → exit 0, 4 figures, anchors reproduce (capacity underfit→fit→overfit; activation
   sigmoid flat at ln 2; init small/default flat vs He; optimizer 42/10/1 epochs; lr sweep; regularizer gaps
   none 0.288 → dropout 0.252; tuning val 0.738 → sealed test 0.756).
2. hex + banned + ruff clean; output-free. 3. **pytest 22** (no `src/` change). 4. **Rémy validates this NB
   plan** (no reviewer gate; both reviewers return on the built notebook).
