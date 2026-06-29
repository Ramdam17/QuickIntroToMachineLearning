# NB plan — 11_MLP / 04_estimator_and_parameters — the MLPClassifier & its knobs

> Status: **APPROVED by Rémy (via ExitPlanMode, 2026-06-29).** No reviewer gate at the NB-plan stage —
> both reviewers return on the built notebook. All anchors measured live (scikit-learn 1.9.0, SEED=0,
> `measure_ch11_nb4.py` + `_nb4b.py`). The flagged figure-set swap (ReLU-reveal figure; alpha/tuning as
> tables) was presented and **approved**.

## Context
Chapter 11 (MLP), **NB 4 of 5** — the **integrative** notebook. NB 1–3 built the neuron, the hidden layer,
and backprop **by hand**. Now we drive the real `MLPClassifier`/`MLPRegressor` and meet **each knob from the
concept that owns it** — closing the loop from the by-hand net to the library. **ReLU is revealed here**
(the measured sigmoid+adam stall from NB 2/3 is the concrete motivation), **Adam is named** (an adaptive-step
upgrade of the GD we built), the **loss curve** becomes the convergence diagnostic, **epoch / mini-batch /
batch_size** get their genuine home, **scaling** is shown to be mandatory, and the **K-class softmax output
head** gets an explicit home. ~27 cells, 4 figures. NB-plan stage = **Rémy validates alone**.

## Figure-set decision (APPROVED)
The chapter plan sketched "~4 figures (capacity vs boundary & loss curve; softmax-head schematic; unscaled-
vs-scaled loss curves; alpha path / default-vs-tuned)". Approved swap: the **alpha-path figure** is replaced
by an **activation / ReLU-reveal figure** — "why ReLU is the default" is the §NB-4 headline and the measured
sigmoid+adam stall is its evidence. **alpha, learning_rate_init, solver, batch_size/epoch, early_stopping, and
defaults-vs-GridSearchCV become printed tables** (each with a "read it" paragraph). 4 figures = **ReLU-reveal ·
capacity-boundary · scaling-loss-curves · softmax-schematic**.

## Datasets (all sklearn generators; no `src/` change)
- **circles** = `make_circles(400, noise=0.10, factor=0.40, seed=0)`, 75/25 — activation (Fig 1) + solver
  (continuity with NB 2/3).
- **moons** = `make_moons(n=300, noise=0.20, seed=0)`, 75/25, **scaled** — capacity (Fig 2), alpha +
  early_stopping, lr + batch/epoch, defaults-vs-GridSearchCV (one consistent 2-D problem).
- **synthetic mismatched-scale** = `make_classification(600, n_features=2, n_informative=2, n_redundant=0,
  n_clusters_per_class=1, class_sep=1.2, seed=0)`, then `feature 1 ×100` — scaling (Fig 3).
- **blobs 3-class** = `make_blobs(600, centers=3, n_features=2, cluster_std=1.2, seed=0)` — softmax head (Fig 4).

## Live anchors (measured; re-pinned live at build)
- **Defaults:** `hidden_layer_sizes=(100,)`, `activation='relu'`, `solver='adam'`, `alpha=1e-4`,
  `learning_rate_init=1e-3`, `batch_size='auto'`, `max_iter=200`, `early_stopping=False`, `momentum=0.9`,
  `tol=1e-4`.
- **Activation / ReLU reveal** (circles, `(16,)`, **default adam**, max_iter 500): **logistic 0.373/0.390 —
  stalls at chance** (loss ≈ ln 2, n_iter 87); tanh 0.887/0.900 (loss 0.46); **relu 0.997/1.000** (loss 0.17).
  Contrast: logistic+**lbfgs** 1.0/1.0 — sigmoid *can* work full-batch; it is sigmoid+**adam** that stalls
  (the NB 2/3 deferral, now the ReLU motivation). Depth-driven vanishing gradients → ch 12.
- **Solver** (circles, relu, `(16,)`, max_iter 2000): sgd 0.997/1.0 (2000 iters), **adam 1.0/1.0 (999
  iters)**, lbfgs 1.0/0.99 (**20 iters, no `loss_curve_`**). Adam = adaptive per-weight step + momentum.
- **learning_rate_init** (moons, `(50,)`): 1e-4 → slow (n_iter↑, higher loss, lower acc); 1e-3 (default) good;
  1e-2 faster; 1e-1 fastest but **test degrades** (~0.86). The step size sets descent speed and stability.
- **batch_size / epoch** (moons, `(50,)`): **`len(loss_curve_) == n_iter_ == number of epochs`** (one point
  per full pass); smaller batch → more updates/epoch, fewer epochs to converge.
- **Capacity** (moons 0.20, n=300, scaled): `(1,)` **0.849/0.853** (one ReLU unit = one kink — underfits);
  `(3,)` 0.849/0.840; `(50,)` **0.982/0.973** (carves both moons). Boundary figure = `(1,)` vs `(50,)`.
- **alpha (L2)** (moons 0.20, n=300, `(200,200)`, scaled): 1e-6…1e-2 → train ~0.982 / test 0.973 (slight
  overfit); **alpha=0.1 → test 0.987 (sweet spot)**; alpha=1 → 0.876/0.907; **alpha=10 → 0.836/0.813
  (underfit)**.
- **early_stopping** (same setup): off → n_iter 180 / test **0.973**; on → **n_iter 12** / test 0.867 /
  best_val 0.826 — holds out `validation_fraction`, stops when validation plateaus; here it stops early and
  costs accuracy (a guard for large/long training, not free).
- **Scaling** (synthetic, feature1 ×100): **unscaled** loss curve starts at **7.2**, crawls to 0.116, test
  **0.940**; **scaled** starts at 0.71, converges to 0.073, test **0.993** (5-seed mean 0.923 vs 0.993).
- **Softmax head** (blobs, 3-class): `n_outputs_=3`, **`out_activation_='softmax'`**, `predict_proba` rows
  **sum to 1**, test 0.907.
- **defaults vs GridSearchCV** (moons 0.20, n=300, scaled, one sealed test): default ≈ tuned **within noise**
  (re-pinned at build; defaults are a strong baseline — consistent with ch 09/10 NB 4).

## Cell-by-cell (~27 cells, 4 figures) — intuition → implementation → "Read the figure"
1. **(md) Header** — `# 04 — The estimator: MLPClassifier and its parameters`; purpose; **Prerequisites**
   (11.1–11.3 the neuron/hidden-layer/backprop by hand; ch 03 the loss curve & gradient descent; ch 00
   scaling + `Pipeline` + fit-on-train-only, over/underfitting + generalization gap, CV / hyperparameters);
   **What you'll be able to do** (drive `MLPClassifier`; pick activation/solver; read the loss curve; size the
   network; regularize with `alpha`/early-stopping; scale; handle K classes; tune honestly).
2. **(code) Imports + defaults** — numpy, matplotlib; sklearn `make_*`, `train_test_split`, `MLPClassifier`,
   `GridSearchCV`, `make_pipeline`, `StandardScaler`; `from ml_course import colors, viz`;
   `viz.use_course_style()`; `SEED=0`; print the key defaults.
3. **(md) The estimator and its knobs** — NB 1–3 built this exact object by hand (forward, backprop, GD);
   sklearn ships it as `MLPClassifier` with knobs. We meet each from the concept that owns it. (Note: the
   by-hand net used sigmoid; the library defaults to ReLU — the next cell shows why.)
4. **(md) Intuition — activation: why ReLU is the default.** Sigmoid/tanh **saturate** — their flat tails
   send tiny gradients, and with the default mini-batch `adam` the network can stall before it learns. ReLU
   (`max(0,z)`) has slope 1 for positive inputs — no saturation there — so gradients keep flowing.
5. **(code) Fig 1 — the ReLU reveal** — on circles, fit `(16,)` with `logistic`/`tanh`/`relu` (default adam);
   plot the three `loss_curve_`s + print train/test. (Sigmoid flat near ln 2; relu descends.)
6. **(md) Read Fig 1** — sigmoid+adam stalls at chance (loss ≈ ln 2, acc 0.39); relu descends to ~0
   (acc 1.0). This is exactly why NB 2/3 pinned full-batch GD / lbfgs for the by-hand sigmoid net — and why
   **ReLU is the modern default**. (Sigmoid still works with lbfgs: 1.0. Depth-driven vanishing gradients →
   ch 12.)
7. **(md) Intuition — the optimizer, the loss curve, and epochs.** `solver`: **adam** (the default) is an
   adaptive-step, momentum-assisted upgrade of the GD we wrote; **sgd** is plain mini-batch GD; **lbfgs** is a
   full-batch optimizer, excellent on small data (but exposes no loss curve). The **loss curve**
   (`loss_curve_`) is the convergence diagnostic (ch 03 NB 4 fig (c)); it has **one point per epoch** (a full
   pass over the data). `max_iter` caps the epochs; `batch_size` sets the mini-batch; `learning_rate_init`
   sets the step.
8. **(code) solver table** — circles, relu: sgd/adam/lbfgs → n_iter, train/test, whether `loss_curve_` exists.
9. **(code) lr + batch/epoch tables** — moons: a `learning_rate_init` sweep (n_iter, final loss, test) and a
   `batch_size` table showing `len(loss_curve_) == n_iter_ == epochs`.
10. **(md) Read — the optimizer knobs.** adam converges in fewer epochs than sgd; lbfgs is fastest on this
    small set but has no per-epoch curve. lr too small → slow/underfit; too big → test degrades. Smaller
    batches take more steps per epoch and fewer epochs to converge.
11. **(md) Intuition — capacity (`hidden_layer_sizes`).** Width × depth set how intricate a boundary the
    network can draw. Too few units underfit; enough captures the shape; surplus capacity is reined in by
    `alpha` / early-stopping (next).
12. **(code) Fig 2 — capacity → boundary** — moons (scaled), `(1,)` vs `(50,)` decision boundaries
    (`viz.plot_decision_boundary`) + train/test in the titles.
13. **(md) Read Fig 2** — one ReLU unit draws a single kink and underfits (0.85); 50 units carve both moons
    (0.97). Depth/width is the capacity dial.
14. **(md) Intuition — regularization: `alpha` (L2) and `early_stopping`.** `alpha` penalizes large weights
    (weight decay) — more `alpha` = smoother. `early_stopping=True` holds out `validation_fraction` and stops
    when the validation score plateaus.
15. **(code) alpha sweep + early_stopping** — moons (scaled), `(200,200)`: `alpha` ∈ {1e-6…10} train/test;
    then `early_stopping` on vs off (n_iter, test, `best_validation_score_`).
16. **(md) Read — regularization.** Small `alpha` slightly overfits (train 0.98 / test 0.97); a touch of
    `alpha` (~0.1) is the sweet spot (test 0.99); too much underfits (0.81). early_stopping stopped at 12
    epochs here and cost accuracy — a guard for large/long training, not a free win on small data.
17. **(md) Intuition — scaling is mandatory.** When features live on wildly different scales, the loss starts
    enormous and gradient descent crawls. A `StandardScaler` (fit on train only — ch 00) fixes this.
18. **(code) Fig 3 — unscaled vs scaled loss curves** — synthetic (feature 1 ×100): plot `loss_curve_`
    unscaled vs `make_pipeline(StandardScaler, MLP)` + print test accuracies.
19. **(md) Read Fig 3** — unscaled starts at loss ~7 and limps (test 0.94); scaled starts near 0.7 and
    converges low and fast (test 0.99). Always put a `StandardScaler` before an MLP.
20. **(md) Intuition — from one output to K: the softmax head.** NB 1–3 had a single sigmoid output (binary).
    For K classes, the output layer has K units with **softmax** (probabilities that sum to 1), trained with
    K-class cross-entropy (ch 03 NB 5 recap). `MLPClassifier` builds this head automatically.
21. **(code) Fig 4 — softmax head schematic + 3-class demo** — schematic (1 sigmoid unit → K softmax units);
    fit on blobs (3-class), print `n_outputs_=3`, `out_activation_='softmax'`, `predict_proba` rows sum to 1.
22. **(md) Read Fig 4** — K output units, softmax-normalized to a probability vector summing to 1; the binary
    sigmoid is the K=2 special case.
23. **(code) defaults vs GridSearchCV** — moons (scaled): default pipeline test acc vs a `GridSearchCV` over
    `hidden_layer_sizes`/`alpha`/`learning_rate_init` → **one sealed test**.
24. **(md) Read — tuning honestly.** Defaults are a strong baseline; the tuned model matched/within-noise of
    the default here (CV picks fold-good params; the sealed test is the honest number). Tune when you have a
    reason and a budget.
25. **(md) Your turn** — (warm-up) switch the circles activation back to `logistic` with `adam`, predict the
    loss curve, then add `solver='lbfgs'` and watch it recover; (core) on moons, push `hidden_layer_sizes`
    to `(300,300)` with `alpha=1e-6` then raise `alpha` — find where test peaks; (reach) set
    `early_stopping=True` and plot `validation_scores_` against `loss_curve_` — what is each measuring?
26. **(md) What you built** — drove `MLPClassifier`; saw why ReLU is the default; read the loss curve and
    named epoch/batch/iteration; sized the network; regularized with alpha/early-stopping; made scaling
    mandatory; met the K-class softmax head; tuned honestly. Next: NB 5 — the handwritten-digits capstone.
27. **(md) References** — Kingma & Ba 2015 (Adam; arXiv:1412.6980); Nair & Hinton 2010 (ReLU; ICML); Glorot &
    Bengio 2010 (saturation/init; PMLR 9:249–256); Bishop 2006 *PRML* §5 (MLP training); scikit-learn
    `MLPClassifier` user guide. Previous: **11.3 — backpropagation**. Next: **11.5 — the digits capstone**.

## `src/` & guards
- **No `src/` change** (reuse `viz.plot_decision_boundary` / `colors`; `MLPClassifier` + `make_pipeline` +
  `StandardScaler` + `GridSearchCV`; `make_circles`/`make_moons`/`make_classification`/`make_blobs`; pytest
  20). Colours only from `ml_course.colors`; `SEED=0`; "Read the figure" after every figure; **never silence
  output** (`ConvergenceWarning` stays visible; `verbose=True` only where per-iteration loss aids a lesson —
  never `verbose=False`-as-suppression); banned-word scan 0; hex clean; ruff/black clean; output-free.
- Build from `build_ch11_nb4.py` (source of truth; rebuild right before `git add` — kernel-drift guard).
- Exit guards: nbconvert exit 0 (4 figures), banned 0, hex clean, ruff clean, output-free; **two-reviewer
  gate** (`@ml-expert-reviewer` + `@pedagogy-reviewer`, no BLOCK) → **Rémy visual** → end-of-NB checklist
  (`gen_llms_txt`, `common_errors` +rows, `course_map` §11 note, pytest 20, STATE) → commit
  `feat(11_mlp): notebook 04 — the estimator & its parameters` → `git merge --ff-only` into `chapter/11_MLP`.

## Verification (end-to-end)
1. nbconvert a scratchpad copy → exit 0, 4 figures, anchors reproduce (ReLU reveal logistic-stall/relu-1.0;
   solver table incl. no-loss_curve_ for lbfgs; capacity (1,) vs (50,); alpha sweet spot ~0.1 + early_stopping
   n_iter 12 vs 180; scaling unscaled 0.94 vs scaled 0.99; softmax n_outputs_ 3 / rows sum to 1; defaults vs
   GridSearchCV within noise).
2. hex + banned + ruff clean. 3. pytest 20. 4. **Rémy validates this NB plan (no reviewer gate); both
   reviewers return on the built notebook.**
