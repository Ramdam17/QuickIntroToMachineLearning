# Notebook plan — 09_XGBoost / 05_census_income (the demanding case, capstone)

> Status: **APPROVED by Rémy (2026-06-28)** (no reviewer gate at the per-NB plan stage — both reviewers
> return on the built notebook). All anchors measured live (xgboost 3.2.0 / lightgbm 4.6.0 / sklearn
> 1.9.0, SEED=0); re-measured at build. Build via `<scratchpad>/build_ch09_nb5.py`.

## Context

NB **5 of 5**, the chapter's **visualization-first capstone** — a full, honest tabular workflow on
**Adult / Census Income** mobilizing all of ch 09 and ch 00's evaluation discipline. ~30 cells, **8
figures** (capstone floor ~24–26 / ≥6 figures).

**The verification & the decision.** The chapter plan owed a check: does Adult's missingness move PR-AUC
under native-NaN vs imputed? **Measured: it does NOT** (XGB native 0.8280 vs mode-imputed 0.8291,
Δ −0.0011, within noise). Missingness IS informative univariately (P(>50K | occupation missing) 0.094
vs 0.248) but **redundant** with correlated features, so erasing it costs nothing measurable. **Rémy's
decision (2026-06-28): keep Adult and FEATURE the honest null** — the lesson is "measure the lever,
don't assume native handling wins." The NB-3 callback leans on (a) the informative-missingness panel,
(b) native handling's real value = **convenience** (XGB/HistGBR take NaN+categoricals with no
preprocessing; the NaN-blind models must impute+encode), shown in the cross-method comparison. This
reinforces the chapter thesis: **no universal best; XGBoost's edge is speed/native-handling/
regularization, not a PR-AUC jump.**

## Anchors (measured live; re-measured at build)

- **Dataset:** `fetch_openml('adult', version=2, as_frame=True)` → 48 842×14, target `>50K` **positive
  rate 0.2393**; 6 numeric + 8 `category` cols; **missing already NaN** in `workclass` (2799),
  `occupation` (2809), `native-country` (857). Stratified 80/20 → train 39 073 / test 9 769.
- **Informative missingness:** P(>50K | missing) vs present — workclass 0.0947 / 0.2481; occupation
  0.0943 / 0.2481; native-country 0.2567 / 0.2390; any-of-three 0.1323 / 0.2478.
- **Verification (XGB depth6/eta0.1/300):** native 0.8280 · mode-imputed 0.8291 · explicit-missing-cat
  0.8281 — **Δ −0.0011, within noise** (acc 0.8704 / 0.8718).
- **Cross-method (PR-AUC / acc):** HistGBR (native) **0.8291 / 0.8717** ≈ XGB (native) **0.8280 /
  0.8704** > GradientBoosting (imputed) 0.8140 / 0.8669 > RandomForest (imputed) 0.7883 / 0.8578 >
  Logistic (one-hot+scale) 0.7679 / 0.8536 > DecisionTree d4 0.6684 / 0.8477. **LightGBM teaser**
  (native, ch 10): PR-AUC 0.8278, fit 2.15 s.
- **Tuned XGB + early stopping** (n_estimators 2000, eta0.05, depth6, subsample/colsample 0.8,
  `early_stopping_rounds=50`, `eval_metric='aucpr'`, eval_set 20% of train): **best_iteration 263**,
  test **PR-AUC 0.8292 / acc 0.8730 / ROC-AUC 0.9280**; @0.5 **precision 0.782 / recall 0.650**;
  **F1-optimal threshold 0.358 → precision 0.692 / recall 0.779 / F1 0.733**.
- **gain-MDI vs permutation (sharp divergence):** MDI relationship 0.340 / capital-gain 0.157 /
  marital-status 0.144 …; permutation (PR-AUC drop) capital-gain 0.236 / age 0.050 / capital-loss 0.039
  / relationship 0.036 / occupation 0.035. (relationship ≈ marital-status redundant; capital-gain
  unique.)
- **Ethics (data base rates):** P(>50K) 0.2393 overall; **sex Female 0.109 vs Male 0.304**; race White
  0.254 / Asian-Pac 0.269 vs Black 0.121 / Amer-Indian 0.117.

## Cell-by-cell (~30 cells; 8 figures; "Read the figure" after each)

1. (md) **Header** — `# 05 — A demanding case: Census income, end to end`; *Ch 09 · NB 5 of 5*. The
   capstone mobilizing NB 1–4 + ch 00. What you'll do.
2. (md) **The problem.** Predict 1994 US income > $50K from demographic + work features. Why a good
   capstone: mixed numeric/categorical, genuine missing, ~24% imbalance, a real ethics dimension.
   Stated upfront: a teaching dataset with real social bias — not for deployment.
3. (code) **Setup & load.** imports (numpy, pandas, matplotlib; `XGBClassifier`; sklearn
   datasets/metrics/model_selection/ensemble/linear_model/tree/preprocessing/compose/pipeline/inspection;
   `LGBMClassifier`); `viz`/`COLORS`; `SEED=0`. `fetch_openml('adult', version=2, as_frame=True)`;
   `y=(target=='>50K')`; print shape, positive rate, #num/#cat, missing counts; stratified 80/20 split.
4. (md) **Look before you model — the imbalance.**
5. (code) **Fig 1 — class balance** (<=50K 76% / >50K 24%); print counts.
6. (md) **Read Fig 1.** ~24% positive → accuracy misleads (0.76 all-negative baseline); lead with
   **PR-AUC** + precision/recall/threshold (ch 00).
7. (md) **Missing values — where, and informative?** (NB 3 enters.)
8. (code) **Fig 2 — missingness vs target.** P(>50K | missing) vs present for the 3 NaN columns + base
   line; print.
9. (md) **Read Fig 2.** workclass/occupation missing → ≈0.09 vs 0.25; native-country barely differs. So
   missingness *looks* informative — the perfect case to test native handling (NB 3).
10. (md) **The honest test — native NaN vs imputed.** Same XGBoost; vary only NaN-kept (native) vs
    mode-filled (erased) vs explicit "missing" category.
11. (code) **Fig 3 — does native handling move the needle?** PR-AUC bars native / mode-imputed /
    explicit-missing (≈0.828); print Δ (−0.001). From-zero axis.
12. (md) **Read Fig 3.** The honest surprise: **within noise**. The missing signal is **redundant** with
    correlated features, so erasing one column costs nothing measurable. Lesson: informative-*looking*
    missingness ≠ native-handling-wins — **measure the lever**. Native handling's real value here =
    **convenience** (no preprocessing) — shown next.
13. (md) **The cross-method comparison — the honest axis.** What each model needs (XGB/HistGBR native;
    GB/RF ordinal+impute; linear one-hot+impute+scale). Disclose the preprocessing — the comparison is
    only fair if you do.
14. (code) **Fig 4 — PR-AUC across methods** (native vs must-impute, coloured); print table.
15. (md) **Read Fig 4.** HistGBR 0.829 ≈ XGB 0.828 lead; GB 0.814; RF 0.788; Logistic 0.768; tree 0.668.
    Boosters win, margin over GB small — part is **convenience**, not raw power. **No universal best**;
    XGBoost's edge = speed + native handling + regularization, not a PR-AUC jump (the chapter thesis,
    measured).
16. (md) **Tuning honestly with early stopping.** Carve a val slice; let early stopping pick #trees; open
    the test once. The 3.x API: `early_stopping_rounds` + `eval_metric` in the **constructor**,
    `eval_set` in `.fit()`.
17. (code) **Fig 5 — early stopping.** eval PR-AUC vs trees, mark `best_iteration` (263 of 2000); print
    best_iteration + test PR-AUC/acc.
18. (md) **Read Fig 5.** A 2000-tree request halted at **263**; tuned test PR-AUC 0.829. Set a generous
    budget, let the data choose the size (ch 08 NB 5, the XGBoost way).
19. (md) **Reading the classifier — precision, recall, threshold** (ch 00, real imbalanced data).
20. (code) **Fig 6 — PR curve & threshold.** (a) PR curve, mark 0.5 (0.78/0.65) and F1-optimal 0.358
    (0.69/0.78); (b) confusion at the chosen threshold; print both.
21. (md) **Read Fig 6.** 0.5 → 0.358 trades precision for recall; on a 24%-minority problem **the
    threshold is a choice** set by error costs — no single "accuracy" (ch 00).
22. (md) **What drives the model? gain-MDI vs permutation** (the ch 06/08 caveat, on real data).
23. (code) **Fig 7 — MDI vs permutation** (top-6 each); print both.
24. (md) **Read Fig 7.** Sharp disagreement: gain → `relationship` (0.34); permutation → `capital-gain`
    (0.24), relationship far lower (0.04). `relationship` ≈ `marital-status` redundant (permuting either
    barely hurts); `capital-gain` unique (permuting collapses PR-AUC). Trust held-out **permutation**;
    importance is *use*, not cause.
25. (md) **Limits and ethics — a substantive section.** The model predicts a 1994 income label that is
    itself demographically skewed.
26. (code) **Fig 8 — base rates by demographic.** P(>50K) by sex (F 0.109 / M 0.304) and by race; base
    line; print.
27. (md) **Read Fig 8 + the ethics argument.** The label encodes 1994 inequities; a model reproduces
    them; `relationship`/`marital-status`/`sex` are demographic **proxies**. Using it to gate decisions
    would **automate and perpetuate** bias — a teaching dataset, **do not deploy**. Honest scope: 1994
    US data, coarse binary target, no single technical "fairness" fix.
28. (md) **Your turn.** *easy:* change the threshold, report precision/recall (recall-first vs
    precision-first). *core:* add explicit "missing" indicator columns and check PR-AUC moves (predict
    first — it won't much; say why). *reach:* the XGB(0.828)-vs-GB(0.814) gap mixes model + preprocessing
    — design a comparison that isolates the **model** (same encoded+imputed inputs) and say what you'd
    expect.
29. (md) **What you built — and the whole chapter.** Capstone recap + **chapter recap**: XGBoost = ch 08's
    engine + (1) second-order view (NB 1), (2) regularized objective λ/γ (NB 2), (3) sparsity-aware splits
    (NB 3), + the histogram engine (NB 4), driven & tuned honestly (NB 4–5). **No universal best**; edges
    = speed / native handling / controllable regularization. Vocabulary. Next: **Ch 10 — LightGBM**, the
    leaf-wise sibling (teased: 0.828 in ~2 s).
30. (md) **References** — Chen & Guestrin 2016 (DOI 10.1145/2939672.2939785); Kohavi 1996 (Adult/Census
    income dataset); Friedman 2001/2002; Ke et al. 2017 (LightGBM, NeurIPS 2017); Pedregosa et al. 2011
    (scikit-learn — PR-AUC, permutation importance); ESL §10 (DOI 10.1007/978-0-387-84858-7).
    `Previous: 04 — The estimator and its parameters.` (Chapter 09 complete.)

## Figures (8, each followed by "Read the figure")
1. Class balance. 2. Missingness vs target. 3. Native vs imputed PR-AUC (the honest null). 4. Cross-method
PR-AUC. 5. Early stopping (263/2000). 6. PR curve + threshold + confusion. 7. gain-MDI vs permutation
(divergence). 8. Base rates by sex/race (ethics). From-zero axes on the metric bars (3, 4, 8) so
equal-looking bars are honestly equal.

## `src/` & guards
**No `src/` change** — notebook-local matplotlib + `viz.use_course_style`; data via `fetch_openml`;
sklearn Pipelines/ColumnTransformer for the NaN-blind baselines; **pytest stays 20**. Build via
`<scratchpad>/build_ch09_nb5.py`; re-measure anchors at build; nbconvert top-to-bottom from project cwd
on a scratchpad copy (output-free tracked file); **never silence output** (LightGBM: no `verbose=-1`);
banned-word JSON scan = 0; `check_no_hardcoded_hex`; ruff/black clean; `gen_llms_txt` re-run. Two-reviewer
gate (no BLOCK) + Rémy visual before commit; ff-merge `notebook → chapter`; then **close chapter via PR
`chapter/09_XGBoost → main` (`--no-ff`)** on Rémy's go.

## Honest scoping
A full classification capstone (imbalance/PR-AUC/threshold/early-stopping/error-analysis/importances/
ethics), all measured. The chapter-plan verification is **featured, not hidden**: native-vs-imputed is
**within noise** on Adult (informative missingness redundant with correlated features) — the lesson is
"measure the lever." The cross-method comparison discloses each model's preprocessing and frames
native-vs-impute as a named axis; the boosters lead by a small margin (convenience as much as power).
gain-MDI vs permutation **diverge sharply**. The ethics section is substantive and measured (demographic
base-rate disparities → do-not-deploy). LightGBM is a named teaser (ch 10 bridge). Every number measured
live, seed-pinned, re-measured at build.
