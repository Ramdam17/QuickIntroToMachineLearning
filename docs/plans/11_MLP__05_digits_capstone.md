# NB plan — 11_MLP / 05_digits_capstone — the demanding case (handwritten digits)

> Status: **APPROVED by Rémy (via ExitPlanMode, 2026-06-29).** No reviewer gate at the NB-plan stage —
> both reviewers return on the built notebook. All anchors measured live (scikit-learn 1.9.0, SEED=0,
> `measure_ch11_nb5.py`). The two measured refinements below were presented and **approved**. **The LAST
> notebook of chapter 11** — after it ships, close the chapter via PR `chapter/11_MLP → main` (`--no-ff`).

## Context
Chapter 11 (MLP), **NB 5 of 5 — the capstone.** A full, honest, **visualization-first** workflow on a real
**10-class** problem: `load_digits` (8×8, 1797×64, offline, CPU-fast). It mobilizes the whole chapter (the
network, ReLU, scaling, the loss curve, the softmax head) plus chapter 00's evaluation discipline (baseline,
held-out test, confusion matrix, CV). The capstone's job is **rigor**: an honest verdict with stated limits,
not a victory lap. ~28 cells, **6 figures** (≥6 is the floor for a capstone). NB-plan = **Rémy validates alone**.

## Measured refinements vs the chapter plan (APPROVED)
The thesis is intact (and stronger); two specifics changed under live measurement:

1. **The digits cross-method result is a statistical TIE, not "HistGB ≥ MLP".** The chapter plan predicted
   "HistGB ~0.982 ≥ MLP, RF ~0.970". Measured **5-fold CV (shuffled, seed 0): MLP(scaled) 0.9772 ± 0.007 ≈
   RF(raw) 0.9766 ± 0.004 ≈ HistGB(raw) 0.9733 ± 0.006**, logistic 0.9694 — the top three overlap within
   noise; **no method dominates.** (A single sealed split is noisy: it gave RF 0.984 / MLP 0.978 / HistGB
   0.973 — split-luck, which is itself a lesson.) Verdict unchanged and cleaner: MLP **fully competitive, not
   superior**; trees need no scaling and stay interpretable.
2. **Use CV for the fair foil, not the single split.** The chapter plan said "same split/metric"; a 450-point
   test set swings ±0.01, which crowned RF spuriously. The cross-method comparison runs on **5-fold CV
   (bars ± std)** for robustness, while the MLP's own diagnostics (loss curve, confusion matrix, error
   gallery) stay on the single sealed split. The "preprocessing difference is the point" still holds: MLP in
   a `StandardScaler` pipeline vs trees on raw features, under the same CV.

## Live anchors (measured; re-pinned live at build)
- **Dataset:** `load_digits` X (1797, 64), 10 classes, pixels [0, 16], balanced (~178–183/class). Split
  stratified 75/25 → train (1347, 64) / test (450, 64), SEED=0.
- **Baselines (sealed test):** Dummy(most_frequent) 0.102 (≈ chance); LogisticRegression(scaled) 0.969;
  single DecisionTree 0.856.
- **Scaled MLP** `Pipeline(StandardScaler, MLPClassifier(100,), max_iter=300)`: sealed test **0.9778**,
  converges **n_iter 145**, loss curve **2.52 → 0.005**. (GridSearchCV over hidden_layer_sizes/alpha → picks
  the default, tuned == default **0.9778** — defaults strong.)
- **Seed-variance (sealed test, 10 init seeds):** **0.971–0.982, mean 0.977, std 0.003** — init alone moves
  the number ±0.005 (non-convex).
- **Confusion (sealed test):** **10 errors / 450**; top confusions 9→3 (2), then 9→5, 8→3, 8→1, 6→1, 4→8 —
  visually ambiguous digits.
- **Fair cross-method (5-fold CV, shuffled seed 0):** MLP(scaled) **0.9772 ± 0.007**, RF(raw) **0.9766 ±
  0.004**, HistGB(raw) **0.9733 ± 0.006**, logistic(scaled) 0.9694 ± 0.010 — **statistical tie among the top
  three.**
- **`breast_cancer` (5-fold CV) — the genuine MLP win:** MLP(32) scaled **0.9754** > RF **0.9631** / HistGB
  **0.9648**. Same model family, different data, opposite verdict.

## Cell-by-cell (~28 cells, 6 figures) — visualization-first; intuition → implementation → "Read the figure"
1. **(md) Header** — `# 05 — A demanding case: handwritten digits, end to end`; the capstone framing;
   **Prerequisites** (all of ch 11 — the network, ReLU, scaling, loss curve, softmax head; ch 00 — baseline,
   held-out test, confusion matrix, CV); **What you'll do** (a full honest multi-class workflow + an honest
   verdict).
2. **(code) Imports + load + split** — `load_digits`; stratified 75/25; print shape, classes, balance.
3. **(md) The problem** — 10 classes, 8×8 = 64 grayscale pixels [0, 16]; a real, small, multi-class image
   task — and the bridge to ch 12 (learned features on images).
4. **(code) Fig 1 — a gallery of digits** — a grid of sample images (a few per class) via `digits.images`.
5. **(md) Read Fig 1** — what the pixels look like; some 9s/4s/8s are genuinely ambiguous even to us.
6. **(md) Baseline first (ch 00 discipline)** — never report a model's score without a floor to beat.
7. **(code) Baselines** — Dummy(most_frequent), LogisticRegression(scaled), single DecisionTree (sealed test).
8. **(md) Read baselines** — chance 0.10; a *linear* model already hits 0.97 (digits are nearly linearly
   separable after scaling); a single tree 0.86. The bar is high.
9. **(md) The MLP — scaled, with the loss curve.** Scaling is mandatory (NB 4); read convergence from the
   loss curve (ch 03 / NB 4).
10. **(code) Fig 2 — train the scaled MLP + loss curve** — `Pipeline(StandardScaler, MLPClassifier(100,))`;
    plot `loss_curve_`; print sealed-test accuracy and `n_iter_`.
11. **(md) Read Fig 2** — loss 2.52 → 0.005 over ~145 epochs (converged); sealed test 0.978. Above every
    baseline.
12. **(md) Where does it go wrong? (error analysis)** — accuracy alone hides the structure of mistakes.
13. **(code) Fig 3 — confusion matrix** (sealed test) via `viz.plot_confusion_matrix`.
14. **(md) Read Fig 3** — near-diagonal; only 10 errors / 450; the off-diagonal pairs (9→3, 8→3/1 …).
15. **(code) Fig 4 — the error gallery** — the ~10 misclassified digit images, titled `true → pred`.
16. **(md) Read Fig 4** — the mistakes are genuinely ambiguous strokes; a human would hesitate too. Honest,
    not hand-waved.
17. **(md) Is 0.978 reliable? (seed-variance)** — the loss is non-convex (NB 3): different inits land in
    different minima. Report the spread, never one lucky run.
18. **(code) Fig 5 — seed-variance** — sealed-test accuracy across 10 init seeds (a dot/strip plot).
19. **(md) Read Fig 5** — 0.971–0.982, ±0.005 from init alone. A single number is not the model's quality.
20. **(md) The honest question: is the MLP the *right* tool here?** A fair comparison — MLP scaled vs trees
    on raw features (trees are scale-invariant; the preprocessing difference is the point) — and on **CV**,
    because one split is noisy.
21. **(code) light tuning + the cross-method CV** — GridSearchCV (→ default, 0.978); then 5-fold CV for
    MLP(scaled) / RF(raw) / HistGB(raw) / logistic(scaled), and the same on `breast_cancer`.
22. **(code) Fig 6 — cross-method comparison (2 panels)** — *digits*: MLP ≈ RF ≈ HistGB (overlapping ± std
    bars, a tie) | *breast_cancer*: MLP > RF / HistGB (the genuine win). Charter colours.
23. **(md) Read Fig 6** — on digits the three are within noise: **the MLP is fully competitive, not
    superior**, and the trees need no scaling and stay interpretable. On clean homogeneous `breast_cancer`
    the **MLP genuinely wins** (0.975 vs 0.963/0.965). The right tool is **data-dependent — no universal
    best.**
24. **(md) Honest limits (the capstone's rigor)** — non-convex / seed-sensitive (Fig 5); universal
    approximation is an **existence** result (can-represent ≠ will-learn ≠ best); **not interpretable** like
    a tree or logistic weights; scaling mandatory; the "learned features" framing is a *conceptual* virtue
    that **pays off on raw high-dimensional data (images, audio) — the bridge to chapter 12**, not a promise
    of higher accuracy on small tabular problems.
25. **(md) Your turn** — (warm-up) change the MLP's `random_state` and re-run — does the *set* of
    misclassified digits change? (ties to non-convexity); (core) try `(100, 100)` or a different `alpha` —
    does the sealed test move beyond the ±0.005 seed band, and would you trust a 0.001 "win"?; (reach) add a
    little label noise (flip 2% of training labels) and compare how the MLP and RandomForest degrade.
26. **(md) What you built — and the whole chapter** — the neuron == logistic (11.1); the hidden layer +
    non-linearity (11.2); backprop by hand (11.3); the estimator and its knobs (11.4); and here, an honest
    end-to-end verdict. You can build, train, tune, evaluate, and *honestly judge* a neural network.
27. **(md) Where chapter 12 goes** — depth as a representation hierarchy, dropout, depth-driven gradient
    pathologies, normalization, modern init, the framework move — where "learned features" earns its keep.
28. **(md) References** — LeCun et al. 1998 (digits/MNIST lineage; DOI 10.1109/5.726791); Grinsztajn et al.
    2022 (trees vs deep nets on tabular; arXiv:2207.08815); Shwartz-Ziv & Armon 2022 (tabular; DOI
    10.1016/j.inffus.2021.11.011); Cybenko 1989 / Hornik 1991 (UAT); scikit-learn `load_digits` / MLP guide.
    Previous: **11.4 — the estimator & its parameters.** Next: **Chapter 12 — Neural Networks.**

## `src/` & guards
- **No `src/` change** (reuse `viz.plot_confusion_matrix` / `colors` / `use_course_style`; `load_digits` /
  `load_breast_cancer`; `MLPClassifier` + `make_pipeline` + `StandardScaler`; `RandomForestClassifier` /
  `HistGradientBoostingClassifier` / `LogisticRegression` / `DummyClassifier` / `DecisionTreeClassifier`;
  `GridSearchCV` / `cross_val_score`; pytest 20). Colours only from `ml_course.colors`; `SEED=0`; "Read the
  figure" after every figure; **never silence output** (`ConvergenceWarning` visible if any); banned-word
  scan 0; hex clean; ruff/black clean; output-free.
- Build from `build_ch11_nb5.py` (source of truth; rebuild right before `git add` — kernel-drift guard).
- Exit guards: nbconvert exit 0 (6 figures), banned 0, hex clean, ruff clean, output-free; **two-reviewer
  gate** (`@ml-expert-reviewer` + `@pedagogy-reviewer`, no BLOCK) → **Rémy visual** → end-of-NB checklist
  (`gen_llms_txt`, `common_errors` +rows, `course_map` §11 → **COMPLETE**, pytest 20, STATE) → commit
  `feat(11_mlp): notebook 05 — handwritten digits capstone` → `git merge --ff-only` into `chapter/11_MLP`.
- **Chapter close:** on Rémy's explicit go, PR `chapter/11_MLP → main` (`--no-ff`), `course_map` §11 →
  "merged via PR #11".

## Verification (end-to-end)
1. nbconvert a scratchpad copy → exit 0, 6 figures, anchors reproduce (baselines 0.10/0.97/0.86; MLP sealed
   0.978 / n_iter 145; seed-variance 0.971–0.982; confusion 10 errors; CV tie MLP 0.977 ≈ RF 0.977 ≈ HistGB
   0.973; breast_cancer MLP 0.975 > 0.963/0.965).
2. hex + banned + ruff clean. 3. pytest 20. 4. **Rémy validates this NB plan (no reviewer gate); both
   reviewers return on the built notebook.**
