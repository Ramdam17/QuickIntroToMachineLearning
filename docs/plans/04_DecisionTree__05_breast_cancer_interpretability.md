# Notebook plan — 04_DecisionTree / 05_breast_cancer_interpretability

> Status: **APPROVED** (2026-06-21, by Rémy; notebook plan validated by Rémy alone — the two reviewers
> gate the *built* notebook). Numbers re-measured at build. Drives the NB-5 build in `docs/WORKFLOW.md`.
> **Final notebook of chapter 04; after it ships, the chapter closes via PR into `main`.**

## Context

NB **5 of 5** — the chapter's **demanding practical case and capstone**: a single decision tree on
**breast-cancer diagnosis** (569 × 30), the full honest workflow where the chapter's payoff
(**readable rules**) and its honest limit (**accuracy & high variance**) both bite. The same dataset
KNN met in chapter 01 NB 5 (felt the curse) and logistic regression read calibrated probabilities from
in chapter 03 NB 6 — now read it as a **tree**. Two questions drive the notebook: *how interpretable*
is a tree, and *how accurate* — and the honest answer (interpretable but less accurate, and unstable)
is exactly what motivates **ensembles** (ch 06+). Capstone rule: **visualization-first**, ~20 cells a
floor not a ceiling.

## Anchors (measured at plan time, sklearn **1.9.0**; re-measured at build)

**Positive class = MALIGNANT** (sklearn target 0 = malignant → `y = (target == 0)`). 569 patients,
malignant 212 (37.3 %). 70/30 stratified split (seed 0): train 398 / test 171 (malignant test 64).
CV = `StratifiedKFold(5, shuffle=True, random_state=0)` on train. **No standardization** (a tree is
scale-invariant — established NB 4).

- **Interpretability vs accuracy (CV-on-train):** tree default **CV 0.940** vs LogReg(std pipeline)
  **CV 0.985**; **tuned tree** (`GridSearchCV` over depth / min_samples_leaf / criterion / ccp_alpha)
  best `{max_depth None, gini, min_samples_leaf 1, ccp_alpha 0}` → **test 0.906** (the grid keeps the
  full tree; pruning did not help on this split). LogReg **test 0.953**.
- **The readable model:** a **depth-3** tree — train 0.975 / **test 0.918**, **7 leaves** — a short,
  clinically sensible rule set: root `mean concave points ≤ 0.05`, then `worst area`, `area error`,
  `worst concave points` thresholds. The kind of explanation KNN / NB / LogReg could not give.
- **Cross-method TEST spine (same pinned split):** **KNN (k = 5) 0.942 → LogReg 0.953 → single tree
  0.906**; and **bagged 25 trees (hand-built, majority vote) 0.930** — averaging recovers ground.
- **Where a single tree fails — variance:** across 25 bootstraps the **root split feature flips** —
  `mean concave points` 15× / `worst perimeter` 6× / `worst concave points` 3× / `worst radius` 1×;
  full-tree bootstrap **test std 0.021** (mean 0.916). Even the first question is unstable.
- **Reading importance honestly — Gini vs permutation disagree:** Gini importance (full tree)
  concentrates on **`mean concave points` 0.74** (then worst area 0.08, worst concave points 0.07);
  **permutation importance** (tuned tree, on the test set) ranks **`worst area` 0.27 / `worst concave
  points` 0.14 / `worst concavity` 0.13** first and does **not** put `mean concave points` on top. The
  two measures *disagree* — the NB-4 Gini-bias caveat made concrete; permutation is the honest read.
- **Confusion (tuned tree, test):** `[[95, 12], [4, 60]]` (rows true benign/malignant) — **4 of 64
  cancers missed**, 12 false alarms, malignant recall **0.94**.
- **The ensemble bridge, measured:** single bootstrap trees test mean 0.912 (std 0.020) vs **bagged-25
  majority vote test 0.930** — averaging many high-variance trees lifts accuracy and steadies it. A
  random forest in miniature (ch 06).

## Library / figures

- **No `src/` change** (`load_breast_cancer` exists since ch 03; `pytest` stays 17). Reuse
  `viz.use_course_style`, **`viz.plot_class_balance`** (Fig A), **`viz.plot_confusion_matrix`** (Fig F);
  `ml_course.colors`. One-off charter bars for the cross-method spine (Fig C), the root-feature-flip
  counts (Fig D), and Gini-vs-permutation importance (Fig E). The depth-3 tree (Fig B) via
  `sklearn.tree.plot_tree(filled=False, …)` — **charter-neutral** (no off-palette node fills; NB 5 has
  no 2-D class scatter to clash with), with the rules also printed via `export_text`. Sklearn:
  `DecisionTreeClassifier`, `GridSearchCV`, `StratifiedKFold`, `cross_val_score`, `LogisticRegression`,
  `KNeighborsClassifier`, `StandardScaler`, `Pipeline`, `permutation_importance`, `confusion_matrix`,
  `train_test_split`.
- **Six figures** (visualization-first): **A** class balance; **B** the depth-3 tree (rules);
  **C** cross-method accuracy bar (KNN / LogReg / single tree / bagged-25); **D** root-feature flips
  across bootstraps; **E** Gini vs permutation importance; **F** confusion matrix. Each + "Read the
  figure".

## Cell-by-cell (~26 cells; visualization-first; "Read the figure" after every figure)

1. (md) **Header** — `# 05 — A demanding case: breast cancer`; *notebook 5 of 5*; warm welcome; this
   is where the chapter comes together. **Prerequisites:** NB 1–4 (the whole method); chapter 01 NB 5
   (KNN on this same dataset — the curse), chapter 03 NB 6 (logistic regression's calibrated
   probabilities on it); module 00 — confusion / precision / recall (NB 07), cross-validation (NB 10),
   the train/test split (NB 04). **What you'll be able to do:** run an honest tree workflow end to end;
   read a tree as a clinical rule set; weigh interpretability against accuracy; see a single tree's
   variance and read importance honestly; know why ensembles come next.
2. (code) **Imports + seed + style + data** — `df = datasets.load_breast_cancer()`;
   `X = df.drop(columns="target")`, **`y = (df["target"] == 0).astype(int)`** with the comment
   *malignant = 1 = the costly miss*; 70/30 stratified split (seed 0); print shape & class counts.
3. (md) **Where we are & the stakes** — 569 patients, 30 measurements, malignant 212 / benign 357. The
   dataset KNN felt the curse on and logistic regression read calibrated probabilities from; now we
   read it as a **tree**. Two questions: how *accurate*, and how *interpretable*? No standardization —
   a tree does not need it (NB 4).
4. (code) **Fig A — class balance** (`plot_class_balance`) + a `df.describe()` glance at two features.
5. (md) **Read the figure (A)** — 37 % malignant: not balanced, so we watch **recall on malignant**
   (a missed cancer is the costly error), not accuracy alone.
6. (md) **Intuition — the honest workflow** — cross-validate **on train** to compare a tree against the
   ch-03 logistic-regression baseline; tune the tree on train; read the **sealed test** once.
7. (code) **CV + tune + test** — `cross_val_score` (pinned CV): tree **0.940** vs LogReg(std) **0.985**;
   `GridSearchCV` the tree on train → best params, **test 0.906**; LogReg **test 0.953**.
8. (md) **Read the result** — the tree trails the linear model by ~4–5 points, in CV and on the sealed
   test. Accuracy is not the whole story, though — the next cell asks what the tree can *tell* us.
9. (md) **Intuition — interpretability: a tree is a rule set** — unlike weights or distances, a shallow
   tree is a flowchart of yes/no questions a person can read and check.
10. (code) **Fig B — the depth-3 tree** (`plot_tree(filled=False)`) + `export_text` rules; print test
    0.918 / 7 leaves.
11. (md) **Read the figure (B)** — the rules read like clinical criteria: split on `mean concave
    points`, then tumour size (`worst area`) and shape (`area error`, `worst concave points`). Larger,
    more irregular tumours fall to the malignant leaves. A clinician could read and challenge this —
    interpretability KNN / NB / logistic regression could not offer.
12. (md) **Intuition — interpretability vs accuracy** — line the methods up on the same sealed test.
13. (code) **Fig C — cross-method accuracy bar** — KNN 0.942 / LogReg 0.953 / single tree 0.906 /
    **bagged-25 trees 0.930** (all on the pinned test split).
14. (md) **Read the figure (C)** — the single tree is the **most interpretable but the least accurate**
    of the three methods. The last bar is a preview: average 25 trees and accuracy climbs back toward
    the linear model — the ensemble idea, which the rest of the chapter-block (ch 06+) is built on.
15. (md) **Intuition — where a single tree fails: variance** — a tree is high-variance (NB 4). On real
    30-feature data, watch even the **first question** move as the data is resampled.
16. (code) **Fig D — root-feature flips** — across 25 bootstraps, a bar of which feature the tree puts
    at its root (`mean concave points` 15× / `worst perimeter` 6× / `worst concave points` 3× / `worst
    radius` 1×); print the bootstrap test std (0.021).
17. (md) **Read the figure (D)** — the very first split is not stable: resample the patients and the
    tree often opens on a *different* measurement. That instability (test std 0.021) is the single
    tree's defining weakness — and the precise thing ensembles fix.
18. (md) **Intuition — reading importance honestly (Gini vs permutation)** — NB 4 warned Gini
    importance is biased; here is the warning made real.
19. (code) **Fig E — Gini vs permutation importance** (two sorted bars, top features): Gini (full tree)
    vs permutation (tuned tree, on test).
20. (md) **Read the figure (E)** — they **disagree**: Gini crowns `mean concave points` (0.74) — the
    root it happened to pick — while permutation, which measures actual reliance on held-out data,
    ranks `worst area` / `worst concave points` first. With correlated features, Gini over-credits the
    one chosen first; trust the **permutation** read, or at least cross-check (NB 4's caveat, lived).
21. (code) **Fig F — confusion matrix** (`plot_confusion_matrix`, tuned tree, test): `[[95,12],[4,60]]`.
22. (md) **Read the figure (F)** — the tree misses **4 of 64 cancers** (recall 0.94) and raises 12
    false alarms. The misses are where the rules are too coarse for a borderline tumour — and a more
    accurate, calibrated alternative existed (ch 03's logistic regression), at the cost of the
    readable rules.
23. (md) **Error analysis & honest limits + the bridge** — a single tree here is **interpretable,
    scale/missing-value-friendly (NB 4), but less accurate than the linear model and high-variance**.
    **When to use:** you need a model a human can read and audit, mixed/looking-glass features, a fast
    baseline. **When not:** you need the last few points of accuracy or stable importances → average
    trees (**random forests**, ch 06) or boost them (ch 07–10). The bagged-25 bar (Fig C) is that fix
    in miniature.
24. (md) **Your turn** (3 tiered) — *easy*: trace the depth-3 rules for a patient with `mean concave
    points = 0.08, worst concave points = 0.15, worst area = 800` and give the prediction; *medium*:
    compare the Gini and permutation top-3 and explain, in two sentences, why they differ; *harder*:
    hand-bag K bootstrap trees with a majority vote, sweep K = 1, 5, 25, and plot test accuracy — you
    have built a random forest's core idea.
25. (md) **What you built** — an honest tree workflow on real diagnostic data: CV model choice, a
    sealed test, the **readable rule set**, the **interpretability-vs-accuracy** trade, the single
    tree's **variance**, **Gini-vs-permutation** importance, and the **ensemble bridge**.
    **Vocabulary:** interpretability vs accuracy · readable rule set · variance / instability ·
    Gini vs permutation importance · bagging · ensemble.
26. (md) **Chapter wrap — decision trees, end to end** — from one split (NB 1) to a grown, pruned,
    tuned, *read* tree on real data (NB 5). The first **non-linear**, rule-based method; **scale-free**
    and native to multiclass/missing data; interpretable — but **unstable**, the weakness the
    **ensemble** chapters are built to fix. The course turns next to **margins** (Module 05 — Support
    Vector Machines); the fix for the variance you saw here lands in **Module 06 — Random Forests**,
    which averages many trees (the bagged-25 bar was a first taste). **References:** Breiman et al. 1984
    (CART); Breiman 1996 (bagging, DOI 10.1007/BF00058655); Strobl et al. 2007 (importance bias, DOI
    10.1186/1471-2105-8-25); Street/Wolberg/Mangasarian 1993 (the WDBC dataset, DOI 10.1117/12.148698);
    ESL §9.2 (DOI 10.1007/978-0-387-84858-7); ISLR §8.1 (DOI 10.1007/978-1-0716-1418-1). `Previous: 04 —
    The estimator & its parameters` · `Next: Module 05 — Support Vector Machines`.

## Honest scoping (stated in the notebook)

- **Positive = malignant** stated up front; sklearn's 0 = malignant convention flagged where `y` is built.
- **Interpretability has a real cost** — the tree trails LogReg in CV (0.940 vs 0.985) and on the sealed
  test (0.906 vs 0.953); stated with the numbers, not hidden.
- **The single tree is high-variance** — the root feature flips across resamples; named as the precise
  motivation for ensembles, with the bagged-25 lift (0.930) shown as the fix in miniature.
- **Gini importance is biased** — Gini and permutation *disagree* here; trust permutation / cross-check.
- **CV-on-train, one sealed test** — no leakage; the tree is tuned on train, the test read once.
- **No standardization** (scale-invariance, NB 4) — and the recall-not-accuracy framing under imbalance.

## Verification

Build via `uv run python - < <scratchpad>/build_ch04_nb5.py` (stdin). Re-measure at build: tree CV
0.940 vs LogReg 0.985; tuned tree test 0.906; depth-3 test 0.918 / 7 leaves + rules; spine KNN 0.942 /
LogReg 0.953 / tree 0.906 / bagged-25 0.930; root-feature flips (concave points 15× …), bootstrap std
0.021; Gini (concave points 0.74) vs permutation (worst area 0.27); confusion `[[95,12],[4,60]]`
(4 cancers missed). Runs top-to-bottom (nbconvert to scratchpad; tracked file **output-free**,
`--clear-output --inplace`); **banned-word scan over the JSON real text** = 0; `check_no_hardcoded_hex`
passes; `gen_llms_txt` re-run; `pytest` 17 (no `src/` change); `ruff` clean. Both reviewers PASS (no
BLOCK); Rémy validates visually; commit `feat(04_decision_tree): notebook 05 — a demanding case: breast
cancer`; merge `notebook → chapter`; **then close CHAPTER 04 via PR into `main`** (`--no-ff`,
per-notebook history preserved).
