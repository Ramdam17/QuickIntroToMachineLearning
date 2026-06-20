# Notebook plan — 03_LogisticRegression / 05_estimator_and_parameters

> Status: **APPROVED** (2026-06-20, by Rémy — validated alone; reviewers gate the *built* notebook).
> Build via `uv run python - < <scratchpad>/build_nb5.py` (stdin — avoids the `/tmp/struct.py` shadow).

## Context

NB **5 of 6** — the **role-4 "method & parameters" notebook**: the first to use the real
`sklearn.linear_model.LogisticRegression`. NB 1–4 built the method by hand (sigmoid → boundary → log-loss
→ gradient descent); NB 5 hands the job to the library and explores its **knobs**. This is the chapter's
**API-correctness pivot**: sklearn changed the regularization API, and the notebook teaches the **current
1.9** one, verified at plan time. Not "one concept" — by design this notebook covers the estimator's
parameters: **`C`** (regularization strength), **`l1_ratio`** (L2 vs L1 vs elastic-net), **`solver`**,
and **multi-class softmax vs one-vs-rest** — each *shown*, then honest tuning (CV on train, one sealed
test).

## sklearn 1.9 API — VERIFIED at plan time (the correctness pivot)

Introspected on the installed **sklearn 1.9.0**: `LogisticRegression.__init__` has **`l1_ratio`** and
**`C`**; **`penalty` is present but DEPRECATED** (FutureWarning: "deprecated in 1.8, removed in 1.10");
**`multi_class` is REMOVED** (not a parameter). The official warning states the mapping explicitly:
**`l1_ratio=0` = L2 (ridge, default)**, **`l1_ratio=1` = L1 (lasso)**, a float between = elastic-net,
**`C=np.inf` = no penalty**. **`solver`**: `lbfgs` (default) is L2-only; **L1 needs `saga`**. Multiclass:
**softmax (multinomial) by default** for ≥3 classes; **one-vs-rest is the explicit
`sklearn.multiclass.OneVsRestClassifier` wrapper** (no `multi_class=` argument any more).

## Datasets & measured anchors (penguins; sklearn 1.9.0; re-measured at build)

- **2-feature binary** Adélie/Gentoo (`load_penguins`, std bill+flipper) — parity, separation→divergence,
  honest tuning. `C=np.inf` fit: coef **(4.97, 10.18)**, b −2.81, **‖w‖ 11.33, acc 0.9964** (1 misclassified
  → the **finite** MLE; matches NB 2/NB 4).
- **4-feature binary** Adélie/Gentoo (`load_penguins_full` → bill_length, bill_depth, flipper_length,
  body_mass; std; 274 rows) — the regularization path and the L1-vs-L2 noise demo. **The 4 measurements
  separate the two species perfectly (acc 1.0)**, which is *why* the weights keep climbing with C.
  - **L2 path (l1_ratio=0, lbfgs): ‖w‖₂ = 0.84 / 1.91 / 3.28 / 6.80** for C = 0.01 / 0.1 / 1 / 100;
    plateau **8.46** at C ≥ 1e4 (solver-limited on separable data — *not* a finite optimum).
- **separation → divergence (2-feature):** the full set (with its 1 overlapping penguin) has a **finite**
  MLE (‖w‖ ≈ 11). Remove that **one** misclassified point → 273 perfectly-separable points → ‖w‖ **runs
  away with C**: 4.3 (C=1) → 14.8 (C=100) → 28.5 (C=1e4) → 29.0 (C≥1e6, solver stop). No finite best
  weight; regularization is what tames it.
- **L1 vs L2 with 4 injected pure-noise columns** (4 real + 4 noise, std, 8 features): **L1**
  (`l1_ratio=1`, `saga`, C=1) drives the **4 noise weights to exactly 0** (4/8 nonzero); **L2** (default)
  keeps **all 8** (noise weights small: 0.08/0.18/0.16/−0.005). L1 = automatic feature selection; L2 =
  smooth shrinkage. On the **4 real features only**, L1 keeps **4/4** for reasonable C (**1/4 at C=0.01**).
- **multinomial (softmax) vs OvR — 3 species** (Adélie 151 / Gentoo 123 / Chinstrap 68; std bill+flipper):
  multinomial CV **0.956**, OvR CV **0.956** (pinned `StratifiedKFold(5, shuffle, random_state=0)`),
  **0.0 % disagreement** on predictions; multinomial `coef_` shape **(3, 2)**. (Re-measured; the chapter
  plan's earlier 0.955/0.952 reconciles to 0.956/0.956 here — they agree on predictions; the difference is
  in *how probabilities are formed*, not the labels on this separable data.)
- **Honest tuning:** `GridSearchCV` over `C` on a train split of the **2-feature** binary (it has real
  errors, so `C` matters), best `C`, **one sealed test** score — exact numbers re-measured at build.

## Library / figures

`from sklearn.linear_model import LogisticRegression`; `from sklearn.multiclass import
OneVsRestClassifier`; `GridSearchCV`, `StratifiedKFold`, `train_test_split`, `StandardScaler`. **Reuse
`viz.plot_decision_boundary`** for the 3-class softmax regions (it is pandas-first, label-agnostic, and
supports ≥3 classes per the chapter plan). Other figures one-off in-notebook (charter colours; numpy under
the hood). **No `src/` change → `pytest` stays 16.**

- **Figure A — L2 regularization path:** ‖w‖₂ (and each of the 4 coefs) vs `C` on a log axis.
- **Figure B — separation → divergence:** ‖w‖ vs `C` for the full 2-feature set (finite plateau ≈ 11) vs
  the perfectly-separable slice (runs to ≈ 29) — two curves.
- **Figure C — L1 vs L2 with noise:** grouped/paired coefficient bars over the 8 features (4 real + 4
  noise), L1 (noise = 0) beside L2 (noise small-but-nonzero).
- **Figure D — softmax decision regions:** the 3-species boundaries via `viz.plot_decision_boundary`
  (multinomial), with the per-class coefficients noted.

## Cell-by-cell (~24 cells; the estimator & its parameters; "Read the figure" after every figure)

1. (md) **Header** — `# 05 — The estimator & its parameters`; *notebook 5 of 6*; purpose; warm welcome.
   **Prerequisites:** NB 1–4 (we built sigmoid → boundary → log-loss → gradient descent by hand); module
   00 cross-validation / tune-on-train (NB 10); chapter 01 standardization. **What you'll be able to do:**
   fit `LogisticRegression`; set **`C`** and read a **regularization path**; choose **L1 vs L2** with
   `l1_ratio` (and know `saga` is needed for L1); use **softmax** vs **one-vs-rest** for >2 classes; tune
   honestly with cross-validation and a sealed test.
2. (code) **Imports + seed + style + data** — sklearn (`LogisticRegression`, `OneVsRestClassifier`,
   `GridSearchCV`, `StratifiedKFold`, `train_test_split`, `StandardScaler`); `ml_course`. Load the
   2-feature binary (std bill+flipper) for the parity check.
3. (md) **Recap & footing** — across NB 1–4 we built logistic regression from nothing: a score, a
   boundary, the log-loss, and gradient descent. The library packages all of it in one object. First we
   confirm it is the **same** thing, then we open its panel of knobs.
4. (code) **Parity** — `LogisticRegression(C=np.inf)` on std bill+flipper; print `coef_`/`intercept_`
   (≈ (4.97, 10.18), −2.81). Note NB 4 proved by-hand gradient descent lands on these exact weights — so
   this is our model, now with knobs.
5. (md) **The sklearn 1.9 API, in one box** — the knobs and the **current** spelling (this changed
   recently): **`C`** = inverse regularization strength (`C=np.inf` → none); **`l1_ratio`** sets the
   *type* — **0 = L2 (ridge, default), 1 = L1 (lasso)**, between = elastic-net (the old **`penalty=` is
   deprecated**, removed in 1.10); **`solver`** — `lbfgs` (default, L2-only), **`saga` for L1**;
   multi-class — **softmax by default**, `OneVsRestClassifier` for one-vs-rest (the old `multi_class=` is
   **gone**). A small table.
6. (md) **Intuition — regularization & `C`** — big weights make a confident, steep boundary that can
   overfit (and, on separable data, run away). A **penalty** on weight size keeps them modest; **`C`** is
   the dial — *small `C` = strong penalty = small weights; large `C` = weak penalty = weights free to
   grow*. (Counter-intuitive: `C` is *inverse* strength.)
7. (code) **Figure A — the L2 regularization path** — 4-feature binary (std); fit `LogisticRegression`
   (default L2) for C across a log grid; plot ‖w‖₂ and each coefficient vs `C`; print the
   0.84/1.91/3.28/6.80 table.
8. (md) **Read the figure (A)** — as `C` grows (penalty weakens) every weight grows together and ‖w‖₂
   climbs (0.84 → 6.8), then appears to plateau. Small `C` shrinks them smoothly toward 0. On *this* data
   the four measurements separate the species perfectly — which is why the weights never truly settle.
   Why that happens is the next section.
9. (md) **Intuition — separation → divergence (why regularization exists)** — if a line can separate the
   classes with **no** mistakes, the model can always cut the loss a little more by making the weights
   **bigger** (a sharper boundary) — so they run to infinity. There is **no finite best weight**.
   Regularization (a penalty, i.e. a finite `C`) is what stops the runaway.
10. (code) **Figure B — separation → divergence** — 2-feature: the **full** set (1 overlapping penguin)
    vs the set with **that one point removed** (now perfectly separable); plot ‖w‖ vs `C` for both.
11. (md) **Read the figure (B)** — one penguin makes the difference: keep the overlap and the weights
    settle to a finite fit (‖w‖ ≈ 11); remove it and they **run away** with `C` (≈ 29 and still climbing).
    A real pipeline always keeps a little regularization so this never bites.
12. (md) **Intuition — L1 vs L2 (`l1_ratio`)** — **L2** (`l1_ratio=0`) shrinks **all** weights smoothly
    toward 0 (none exactly 0). **L1** (`l1_ratio=1`) drives **some weights to exactly 0** — it *selects*
    features. (L1 needs the **`saga`** solver.)
13. (code) **Figure C — L1 vs L2 with injected noise** — 4 real features + 4 pure-noise columns (std);
    fit L1 (`l1_ratio=1`, `saga`) and L2 (default); paired coefficient bars; print nonzero counts (L1 4/8,
    L2 8/8).
14. (md) **Read the figure (C)** — L1 sets the **four noise weights to exactly 0** — a sparse,
    self-selecting model — while L2 keeps them all, merely shrunk. L1 = feature selection; L2 = smooth
    shrinkage. (On the 4 *real* features L1 keeps all four — each carries signal; only very strong L1,
    `C=0.01`, drops to one.)
15. (md) **Intuition — more than two classes: softmax** — two classes used one sigmoid. With three+,
    `LogisticRegression` fits the **softmax (multinomial)** model by default: one score per class,
    normalized so the probabilities sum to 1. The alternative, **one-vs-rest**, fits one binary classifier
    per class and renormalizes — the explicit `OneVsRestClassifier`.
16. (code) **Figure D — the three-species boundaries** — 3 species (std bill+flipper); fit multinomial;
    `viz.plot_decision_boundary` for the 3 regions; print multinomial CV (0.956), OvR CV (0.956),
    prediction disagreement (0.0 %), and `coef_` shape (3, 2).
17. (md) **Read the figure (D)** — three regions meeting at the boundaries. Softmax and one-vs-rest agree
    on **every** prediction here (0 %), because the species are well separated; they differ in **how the
    probabilities are formed** (softmax normalizes all three jointly; OvR fits three independent sigmoids,
    then renormalizes). On overlapping classes the two would part ways.
18. (md) **Intuition — choosing `C` honestly** — `C` and `l1_ratio` are **hyperparameters**: pick them by
    **cross-validation on the training data**, never on the test set (module 00 NB 10), then read the test
    set **once**.
19. (code) **Honest tuning** — 2-feature binary; `train_test_split`; `GridSearchCV` over `C` on the
    **train** split (CV); report the chosen `C` and the **one** sealed-test accuracy (re-measured at
    build).
20. (md) **Read the result** — cross-validation chose `C = …`; the sealed-test accuracy is …; the
    discipline is the point — tune on train, touch the test set once.
21. (md) **Your turn** (3 tiered) — *easy*: trace the regularization path — which `C` gives the smallest
    weights, and is that the most accurate? *medium*: with L1 (`saga`), lower `C` until a **real** feature
    is zeroed — which goes first, and why? *harder*: take one borderline 3-species penguin and compare its
    **softmax** vs **OvR** class probabilities (do they rank the classes the same? sum the same?).
22. (md) **What you built** — `LogisticRegression` and its knobs: **`C`** (regularization path,
    separation→divergence), **`l1_ratio`** (L1 selection vs L2 shrinkage; `saga` for L1), **softmax vs
    one-vs-rest** for >2 classes, and **honest tuning** (CV on train, sealed test). **Vocabulary box:**
    regularization · `C` (inverse strength) · L1 / lasso · L2 / ridge · `l1_ratio` · solver (`saga`) ·
    softmax / multinomial · one-vs-rest · hyperparameter.
23. (md) **Going further (optional)** — **elastic-net** (`l1_ratio` strictly between 0 and 1) blends
    shrinkage and selection; **`class_weight`** reweights an imbalanced problem; the **solver** table
    (`lbfgs`/`saga`/`liblinear`) trades penalties for speed. Forward pointer: **NB 6** is the demanding
    case — breast cancer — where calibrated probabilities and a chosen threshold decide real outcomes.
24. (md) **References** — Cox 1958 (DOI 10.1111/j.2517-6161.1958.tb00292.x); Hoerl & Kennard (1970), ridge,
    DOI 10.1080/00401706.1970.10488634; Tibshirani (1996), lasso, DOI 10.1111/j.2517-6161.1996.tb02080.x;
    ESL §3.4 & §4.4 (DOI 10.1007/978-0-387-84858-7); ISLR §4.3 (DOI 10.1007/978-1-0716-1418-1).
    `Previous: 04 — Fitting II: gradient descent` · `Next: 06 — A demanding case: breast cancer`.

## Honest scoping (stated in the notebook)

- **The sklearn 1.9 API is taught as current and verified** (`l1_ratio` not `penalty`; no `multi_class`;
  `saga` for L1; `C=np.inf` for none). The deprecation is stated so the notebook does not age badly.
- **The L2-path "plateau" is solver-limited, not a finite optimum** — the 4-feature data is perfectly
  separable; this is named and is exactly what the separation→divergence section explains.
- **Penguins cannot show L1 sparsity on its own real features** (all four are informative) → the sparsity
  demo **injects pure-noise columns** (controlled), and the real-feature behaviour (4/4, 1/4 at C=0.01) is
  stated separately so nothing is overclaimed.
- **softmax vs OvR agree on predictions here only because the species are well separated** — a measured
  property of this data; the genuine difference is in how probabilities are formed.
- **Honest tuning**: CV on train, one sealed test — the full rigor lands in NB 6; here it is the procedure.

## Verification

Build via `uv run python - < <scratchpad>/build_nb5.py` (stdin). Re-measure at build on sklearn 1.9:
the API introspection (l1_ratio/penalty-deprecated/no multi_class), L2 path (0.84/1.91/3.28/6.80, plateau
8.46), separation→divergence (full ≈11 vs slice ≈29), L1 noise (4/8 vs 8/8; real 4/4, 1/4 at C=0.01),
multinomial/OvR (0.956/0.956, 0.0 % disagreement, coef_ (3,2)), and the GridSearchCV best-C + sealed test.
Runs top-to-bottom (nbconvert to scratchpad; tracked file **output-free**, `--clear-output --inplace`);
**banned-word scan over the JSON real text** = 0; `check_no_hardcoded_hex` passes; **no `src/` change**
(`pytest` stays 16); `gen_llms_txt` re-run; `ruff`/`black` clean. Both reviewers PASS (no BLOCK); Rémy
validates visually; commit `feat(03_logistic_regression): notebook 05 — the estimator & its parameters`;
merge `notebook → chapter`.
