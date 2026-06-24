# Notebook plan — 06_RandomForest / 05_covtype_strong_baseline

> Status: **APPROVED** (2026-06-23, by Rémy). Built next; both reviewers (`@ml-expert-reviewer` +
> `@pedagogy-reviewer`) gate the **built** notebook (no gate at the plan stage). Anchors re-measured
> at build on sklearn **1.9.0**, every random forest `random_state`-pinned. **Last NB of chapter 06 —
> after it ships, close the chapter via PR into `main` (`--no-ff`).** Build decisions: use
> `fetch_covtype(as_frame=True)` directly (names already descriptive; INFO logging shown; no loader,
> no new test → pytest stays 20); cross-method comparison on fixed defaults + OOB (no test-set tuning).

## Context

NB **5 of 5** — the chapter's **demanding practical case and capstone**, **visualization-first**. A
random forest on **forest cover type** (covtype): predict which of 7 tree species covers a 30 m × 30 m
patch from 54 cartographic features. Thematically apt (a random *forest* on a *forest*), genuinely
**non-linear**, large, and **severely imbalanced** — everything the chapter built now bites at once.
Two questions drive it: does the forest **win** where the linear models won on breast_cancer, and can
we evaluate it **honestly** under imbalance? Capstone rule: ~24–26 cells a **floor**, **7 figures**.

## Anchors (re-measured at plan time, sklearn **1.9.0**; every RF `random_state`-pinned)

`fetch_covtype(as_frame=True)` (581 012 × 54), **30 000-row stratified subsample (seed 0)**, 70/30
stratified split (seed 0) → train 21 000 / test 9 000. Columns already descriptively named by sklearn
(`Elevation`, `Aspect`, …, `Wilderness_Area_0..3`, `Soil_Type_0..39`). Cover types (Blackard 1999):
1 Spruce/Fir, 2 Lodgepole Pine, 3 Ponderosa Pine, 4 Cottonwood/Willow, 5 Aspen, 6 Douglas-fir,
7 Krummholz. **No scaling for the forest** (tree-based, NB 4); LogReg gets a StandardScaler.

- **Class balance (30k):** Lodgepole **48.8 %**, Spruce/Fir **36.5 %**, Ponderosa 6.2 %, Krummholz
  3.5 %, Douglas-fir 3.0 %, Aspen **1.6 %**, Cottonwood **0.5 %**. Severe imbalance.
- **The forest wins:** `RF(300, oob_score=True)` test **0.844**, OOB **0.846**; single tree **0.770**;
  `LogReg` (StandardScaler, max_iter=2000) **0.729**. RF beats the linear model by **+11 points** —
  the **reverse of breast_cancer** (there RF ≈ 0.94 < SVM 0.965). OOB ≈ test → not overfit.
- **Honest eval under imbalance:** accuracy **0.844**, weighted-F1 **0.840** (both dominated by the two
  big classes), **macro-F1 0.737** (unweighted over classes). Per-class **recall** (RF, test):
  Lodgepole **0.901**, Ponderosa 0.872, Spruce/Fir 0.821, Krummholz 0.767, Douglas-fir 0.569,
  Cottonwood 0.558, **Aspen 0.279** (test n = 147). The big classes are easy; the rare ones are hard —
  accuracy hides it, macro-F1 reveals it.
- **Confusion (7×7, test):** Aspen (5) is mostly misread as Lodgepole (2) — the rare class collapses
  into its common look-alike; Douglas-fir (6) ↔ Ponderosa (3) confusion. Big-class diagonal strong.
- **Reading importance honestly:** **Elevation dominates** — MDI **0.233**, permutation **0.270**
  (they **agree**, because one feature genuinely dominates — unlike NB 4's correlated-twins case). The
  **40 one-hot `Soil_*` columns combined** reach only **0.141 (MDI) / 0.112 (perm)** — each near-zero
  alone: the **dilution caveat** made vivid (a single dominant continuous feature outweighs forty
  diluted dummies). **Permutation put to work**, as promised in NB 4.
- **Fit-time ≈ linear in n:** `RF(100, n_jobs=1)` single-thread, n = 1k→64k: 0.11 / 0.18 / 0.36 /
  0.74 / 1.50 / 3.00 / 6.25 s → empirical **n^0.99** (theory O(n log n · trees · √p)); embarrassingly
  parallel (`n_jobs=-1`). The counterpoint to ch 05's SVM RBF **≈ n^1.6** wall (worst-case O(n³)).

## Library / figures

- **No `src/` change** — `fetch_covtype(as_frame=True)` already returns a named DataFrame + Series, so
  no loader wrapper is needed (cf. NB 4 using sklearn's breast_cancer directly); **INFO logging shown**
  in the notebook (`logging.basicConfig(level=logging.INFO)`) so the one-time ≈14 MB fetch is visible,
  never silenced. `pytest` stays **20**.
- **Reused:** `viz.use_course_style`; **`viz.plot_feature_importances`** (NB 4 — Fig F, MDI vs perm);
  **`viz.plot_confusion_matrix`** (Fig E); **`viz.plot_class_balance`** (Fig A); `ml_course.colors`.
  Sklearn: `fetch_covtype`, `RandomForestClassifier`, `DecisionTreeClassifier`, `LogisticRegression`,
  `StandardScaler`, `make_pipeline`, `permutation_importance`, `train_test_split`, `accuracy_score`,
  `f1_score`, `recall_score`, `confusion_matrix`.
- **Seven figures** (visualization-first capstone, each + "Read the figure"): **A** class balance;
  **B** cross-method accuracy (RF / tree / LogReg); **C** aggregate metrics (accuracy / weighted-F1 /
  macro-F1 — the imbalance gap); **D** per-class recall (RF); **E** 7×7 confusion matrix; **F** MDI vs
  permutation importance (two panels); **G** fit-time vs n (RF ≈ n^1.0 vs an n^1.6 SVM reference).

## Cell-by-cell (~25 cells; visualization-first; "Read the figure" after every figure)

1. (md) **Header** — `# 05 — A demanding case: forest cover type`; *Chapter 06 · Notebook 5 of 5*;
   warm welcome (the capstone — the whole chapter comes together). **Prerequisites:** NB 1–4 (the whole
   forest: bagging, the "random"/ρ, OOB, the estimator & its dials, MDI vs permutation); the
   cross-method spine (ch 01 KNN, 03 LogReg, 05 SVM); module 00 — confusion / precision / recall /
   **macro vs weighted** (NB 07), cross-validation (NB 10), the split (NB 04). **What you'll be able to
   do:** run an honest forest workflow on a large, imbalanced, non-linear problem; see the forest win
   where a linear model loses; evaluate honestly under imbalance (macro-F1, per-class recall,
   confusion); read importances honestly (MDI vs permutation, one-hot dilution); see a forest scale
   gently; know when to reach for boosting.
2. (code) **Imports + INFO logging + fetch + subsample + split** — `logging.basicConfig(INFO)` (the
   one-time ≈14 MB fetch is visible); `fetch_covtype(as_frame=True)`; map cover-type names; stratified
   30k subsample (seed 0); 70/30 split (seed 0); print shapes & feature groups (10 continuous + 4
   wilderness + 40 soil one-hot). **No scaling for the forest.**
3. (md) **Where we are & the stakes** — 581k forest patches, 54 features, 7 species, severely
   imbalanced; we subsample 30k to stay brisk. A genuinely **non-linear** problem — the opposite shape
   to breast_cancer (near-linear), where SVM/LogReg won. Two questions: does the forest **win** here,
   and can we read it **honestly** under imbalance?
4. (code) **Fig A — class balance** (`plot_class_balance`, 7 bars).
5. (md) **Read the figure (A)** — Lodgepole 49 % + Spruce/Fir 36 % dominate; Aspen 1.6 %, Cottonwood
   0.5 % are rare. Under this imbalance, accuracy is dominated by the two big classes — we will watch
   **per-class recall and macro-F1**, not accuracy alone.
6. (md) **Intuition — the honest workflow & the cross-method question.** Fit a forest (no scaling,
   NB 4), and a single tree and a linear model (LogReg, scaled), and compare on the **sealed test**;
   use the forest's **OOB** as the free in-train check. The real question: which model *family* fits
   this problem's shape?
7. (code) **Fig B — cross-method accuracy** — fit `RF(300, oob_score=True)`, tree, LogReg(scaled);
   bar of test accuracy RF **0.844** / tree **0.770** / LogReg **0.729**; print OOB **0.846**.
8. (md) **Read the figure (B)** — the forest **wins decisively**: 0.844 vs LogReg 0.729 (+11 pts). The
   **reverse of breast_cancer** (RF ≈ 0.94 < SVM 0.965 on near-linear data). covtype's species are
   carved by **non-linear interactions** of elevation, distances and soil — a committee of trees
   captures that; a single linear boundary cannot. OOB 0.846 ≈ test 0.844: not overfit. (The forest
   needed no scaling; LogReg did — NB 4's scale-invariance, lived.) *The right tool tracks the
   problem's shape.*
9. (md) **Intuition — accuracy lies under imbalance (re-lay macro vs weighted).** Accuracy and
   weighted-F1 average over **samples**, so the big classes dominate. **Macro**-F1/recall average over
   **classes** (unweighted), so a rare class counts as much as a common one. Under imbalance, report
   macro.
10. (code) **Fig C — aggregate metrics** — bar of accuracy **0.844** / weighted-F1 **0.840** /
    macro-F1 **0.737**.
11. (md) **Read the figure (C)** — accuracy and weighted-F1 sit together near 0.84 (both ruled by the
    two big classes); macro-F1 drops to **0.737**. That **10-point gap** is the rare classes being
    missed — invisible to accuracy, surfaced by macro-F1.
12. (code) **Fig D — per-class recall** (RF, 7 bars).
13. (md) **Read the figure (D)** — the big classes are easy (Lodgepole 0.90, Ponderosa 0.87,
    Spruce/Fir 0.82) but the rare ones are hard (**Aspen 0.28**, Cottonwood 0.56, Douglas-fir 0.57). A
    forest does **not** fix imbalance — it inherits it; `class_weight='balanced'` / resampling are the
    levers (you try one in *Your turn*), not more trees.
14. (code) **Fig E — confusion matrix** (`plot_confusion_matrix`, 7×7, cover-type names).
15. (md) **Read the figure (E)** — the misses are structured, not random: **Aspen (5) → Lodgepole
    (2)** (the rare class collapses into its common look-alike), Douglas-fir (6) ↔ Ponderosa (3). The
    big-class diagonal is strong. The matrix shows imbalance + species similarity.
16. (md) **Intuition — reading importance honestly (MDI vs permutation; one-hot dilution).** NB 4
    *named* permutation importance; here we put it to work. MDI (impurity) is biased and **dilutes**
    across one-hot groups; permutation (shuffle a column, measure the held-out accuracy drop) is the
    honest read. When **one** feature truly dominates, the two **agree**.
17. (code) **Fig F — MDI vs permutation** (two panels via `plot_feature_importances`, top-10 each);
    print **Soil-40 combined** MDI 0.141 / perm 0.112, and Elevation MDI 0.233 ≈ perm 0.270.
18. (md) **Read the figure (F)** — **Elevation dominates both** (MDI 0.23, perm 0.27): when one
    feature genuinely carries the signal, MDI and permutation **agree** (unlike NB 4's correlated
    twins, where they disagreed). The **40 one-hot `Soil_*` columns combined** reach only 0.141 / 0.112
    — each near-zero alone. That is the **dilution caveat** made vivid: soil is not unimportant, its
    signal is split across forty thin dummies (and MDI's bias compounds it). **Read importance at the
    group level**, never one dummy at a time.
19. (md) **Intuition — a forest at scale.** Two practical virtues for large data: OOB gives the
    generalization estimate **for free** (no validation split), and fit-time grows **gently** with n —
    the opposite of ch 05's SVM wall.
20. (code) **Fig G — fit-time vs n** — `RF(100, n_jobs=1)` single-thread, n = 1k→64k; plot measured
    time with an **n^1.0** guide and an **n^1.6** reference (ch 05's SVM exponent); print the empirical
    **n^0.99**.
21. (md) **Read the figure (G)** — RF fit-time tracks the **n^1.0** line (measured ≈ n^0.99): double
    the data, double the time. The dashed **n^1.6** curve is ch 05's SVM RBF scaling (worst-case
    O(n³)) — by 64k it towers above RF. And RF is **embarrassingly parallel** (`n_jobs=-1`). Gentle
    scaling + no scaling + free OOB is why a forest is the go-to first model on large tabular data.
22. (md) **Error analysis & honest limits + the boosting bridge.** RF here is **strong** (0.844, wins
    decisively), scales ~linearly, needs no scaling, gives OOB free. But: (1) it won **because** the
    problem is non-linear — on near-linear breast_cancer it **lost** to SVM (no universal best); (2) it
    **inherits imbalance** (Aspen recall 0.28) — `class_weight`/resampling, not more trees, are the
    fix; (3) it is **no longer one readable tree** (the ch 04 payoff, traded for accuracy & stability);
    (4) importance is **not causal** and dilutes across one-hot. **When to push further:** boosting
    (ch 07–10) often edges RF with more tuning. The forest is the strong, low-effort baseline you reach
    for **first**.
23. (md) **Your turn** (3 tiered) — *easy:* from the confusion matrix, name the class Aspen is most
    confused with and explain it from the class balance; *medium:* refit `RF(class_weight='balanced')`
    and compare macro-F1 and Aspen recall to the default — does balancing help the rare classes, and at
    what cost to accuracy?; *harder:* rank features by MDI and by permutation, sum the 40 `Soil_*`
    columns under each, and explain in two sentences why the soil dummies look unimportant either way
    (dilution vs bias).
24. (md) **What you built** — an honest forest workflow on a large, imbalanced, non-linear problem: the
    forest **wins** where the linear model loses (the shape of the problem); honest evaluation under
    imbalance (accuracy vs macro-F1; per-class recall; confusion); reading importance honestly (MDI vs
    permutation; one-hot dilution); OOB at scale; near-linear fit-time. **Vocabulary:** macro vs
    weighted F1 · per-class recall · class imbalance · MDI vs permutation importance · one-hot dilution
    · embarrassingly parallel · right-tool-for-the-shape.
25. (md) **Chapter wrap — Random Forests, end to end** + **References.** From averaging cuts variance
    (NB 1) → the "random" decorrelates (NB 2) → OOB for free (NB 3) → the estimator & its dials (NB 4)
    → a demanding case where the forest wins (NB 5). The first **ensemble**: many high-variance trees
    averaged into one strong, low-variance, scale-free, parallel model — the fix for ch 04's
    single-tree variance, and the strong baseline you reach for first on tabular data. Its limits (not
    always best; not one readable tree; importance not causal) set up the **boosting** family
    (ch 07–10), which builds trees that fix each other's mistakes. **Going further (optional):**
    `class_weight`/resampling for imbalance; `ExtraTrees`; OOB-guided selection. **References:** Breiman
    2001 (DOI 10.1023/A:1010933404324); Breiman 1996 (DOI 10.1007/BF00058655); Ho 1998 (DOI
    10.1109/34.709601); Strobl 2007 (DOI 10.1186/1471-2105-8-25); Blackard & Dean 1999 (covtype, DOI
    10.1016/S0168-1699(99)00046-0); ESL §15 (DOI 10.1007/978-0-387-84858-7); ISLR §8.2 (DOI
    10.1007/978-1-0716-1418-1). `Previous: 04 — The estimator & its parameters` · `Next: Module 07 —
    AdaBoost`.

## Honest scoping (stated in the notebook)

- **The forest wins *because* the problem is non-linear** — explicitly contrasted with breast_cancer
  (RF lost to SVM there); no method is universally best, the right tool tracks the problem's shape.
- **Accuracy is misleading under imbalance** — accuracy 0.844 / weighted-F1 0.840 vs macro-F1 0.737;
  per-class recall (Aspen 0.28) is the honest read; the forest does not fix imbalance.
- **Importance is not causal and MDI dilutes across one-hot** — Elevation dominates (MDI ≈ perm,
  agreeing), 40 Soil dummies diluted; read at the group level; permutation is the cross-check.
- **A forest is no longer one readable tree** — ch 04's interpretability payoff is traded for accuracy.
- **Boosting often edges RF** with more tuning (ch 07–10); RF's claim is *strong with little effort*.
- **One sealed test; OOB on train** (no leakage); the cross-method comparison uses fixed defaults (no
  hyperparameter tuning on the test set).
- **Scaling exponent is this-machine, single-thread** (≈ n^0.99); the n^1.6 SVM curve is ch 05's
  measured exponent shown as a reference, not re-measured here.

## Verification

Build via `uv run python - < <scratchpad>/build_ch06_nb5.py` (stdin). Re-measure at build: class
balance (Lodgepole 48.8 % … Cottonwood 0.5 %); RF 0.844 / OOB 0.846 / tree 0.770 / LogReg 0.729;
accuracy 0.844 / weighted-F1 0.840 / macro-F1 0.737; per-class recall (Aspen 0.279 … Lodgepole 0.901);
confusion 7×7; MDI Elevation 0.233 / Soil-40 0.141, permutation Elevation 0.270 / Soil-40 0.112;
fit-time n^0.99. Runs top-to-bottom (nbconvert from project cwd to a scratchpad copy; tracked file
**output-free**); **banned-word scan over the JSON real cell text** = 0; `check_no_hardcoded_hex`
passes; `gen_llms_txt` re-run; `pytest` 20 (no `src/` change); `ruff`/`black` clean; `course_map.md`
§06 marked complete; `common_errors.md` extended (imbalance / one-hot dilution). Both reviewers PASS
(no BLOCK); Rémy validates visually; commit `feat(06_random_forest): notebook 05 — a demanding case:
forest cover type`; merge `notebook → chapter`; **then close CHAPTER 06 via PR into `main`
(`--no-ff`)**.
