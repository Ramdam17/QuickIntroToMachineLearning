# Notebook plan — 00_GettingStarted / 11_preprocessing_leakage

> Status: **APPROVED** (2026-06-17, by Rémy). Notebook plans validated by Rémy alone; reviewers
> return on the built notebook. Drives `notebooks/00_GettingStarted/11_preprocessing_leakage.ipynb`.
> The **last notebook of chapter 00** — integrative (preprocessing + leakage), budget ~26-30 cells,
> no compression.

## Context

The on-ramp's closing notebook, paying off two debts the module deliberately left open: NB 05's
nearest-centroid measured **raw** Euclidean distance and was flagged **scale-sensitive** (NB 02/05),
and NB 04 warned about **leakage** without showing it bite. NB 11 standardizes (and watches the model
change), one-hot **encodes** a real categorical (now available via `load_penguins_full()` — `island`),
keeps every preprocessing step **fit-on-train-only** inside a **`Pipeline`**, and then *shows leakage
inflating a score* with the canonical **ESL §7.10.2** demo. Prereqs: 04 (split + leakage idea), 05
(nearest-centroid + its scale-sensitivity), 10 (cross-validation, `Pipeline`-under-CV).

## Design (measured — `uv run python`, not guessed)

- **Standardization payoff** (penguins 2-feature subset, nearest-centroid): standardizing **rotates the
  decision boundary ~52.6°** in the data's mm coordinates (boundary-normal 72.5°→19.9°) — the model
  genuinely changes. Single split:
  raw acc 1.00 vs standardized 0.9855 (1 of 69 flips — split luck). Fair 5-fold stratified **CV: raw
  0.989 vs standardized 0.9927** — the standardized version edges ahead. Honest: the accuracy effect is
  small here because the classes are far apart; the boundary rotation is the real evidence.
- **Encoding:** `island` has three values (Biscoe 168 / Dream 124 / Torgersen 52, no missing) → a clean
  one-hot into 3 columns. (`sex` = MALE/FEMALE with 11 missing — deferred to "Going further".)
- **ESL §7.10.2 leakage demo** (measured): 100 samples, **1000 pure-noise features**, random labels
  (honest accuracy ≈ 0.5). Selecting the top-20 features **on all data then CV** → **0.850** (logreg);
  selecting **inside each fold** (SelectKBest in a Pipeline) → **0.570** (≈ chance). 85% "accuracy" on
  pure noise *is* the leak.

## Library additions

**None.** Reuse `viz.plot_decision_boundary` for the raw-vs-standardized boundary panels. The leakage
bar (WRONG vs RIGHT) is a one-off in-notebook figure pulling colours from `ml_course.colors`
(`COLORS["model"]` / `COLORS["highlight"]`). New sklearn pieces (`OneHotEncoder`, `ColumnTransformer`,
`SelectKBest`/`f_classif`) need no course code. `pytest` stays 14.

## Cell-by-cell (~26-30 cells; intuition → implementation → "Read"; 2 figures)

1. (md) Header — `# 11 — Preprocessing & leakage`; purpose; `Prerequisites: 04, 05, 10`; 4 objectives.
2. (code) Imports + seed + `use_course_style()`; load penguins 2-feature subset `X, y` + stratified
   split (test_size=0.25, random_state=0) as in NB 05.
3. (md) Recap & the two debts — NB 05's raw-distance scale-sensitivity; NB 04's leakage warning.
4. (md) Why scale — bill (~5 mm) vs flipper (~15 mm); distance lets the larger-range feature dominate;
   standardization = (x − mean)/std per feature, computed on train.
5. (code) Print train mean/std; standardize **by hand**; show means ≈ 0, stds ≈ 1.
6. (code) **figure 1** — nearest-centroid boundary RAW vs STANDARDIZED (two `plot_decision_boundary`
   panels: `NearestCentroid()` and `make_pipeline(StandardScaler(), NearestCentroid())`, raw coords).
7. (md) Read figure 1 — the boundary rotates (~25°); model genuinely changed; accuracy barely moves on
   these separable penguins (split luck); the larger-range feature no longer dominates. NB 05's flag, seen.
8. (code) Fair comparison — 5-fold stratified CV: raw NC (0.989) vs standardized pipeline (0.9927).
9. (md) Read — standardized edges ahead over folds; principled default for distance-based models; small
   gain here (separable), decisive when scales differ wildly (forward-ref KNN/SVM).
10. (md) Why encode — models need numbers; full set has `island`/`sex` text; one-hot = one 0/1 column
    per category.
11. (code) `load_penguins_full()`; `island` value_counts; one-hot **by hand** with `pd.get_dummies`.
12. (code) `OneHotEncoder(handle_unknown="ignore")` **fit on train only**; show encoded columns.
13. (md) Read — one 0/1 column per category, learned from train; one-hot (not 1/2/3) because categories
    have no order.
14. (md) Putting it together — `ColumnTransformer` (scale numerics, one-hot categoricals) wrapped in a
    `Pipeline` so transforms fit on the training fold only.
15. (code) Build `ColumnTransformer` + classifier → `Pipeline`; `cross_val_score` on the Adélie/Gentoo
    subset (2 numeric + island).
16. (md) Read — one object scales+encodes+fits, refit cleanly inside each fold; the leakage-safe way.
17. (md) The rule — preprocessing fit on train only; the `Pipeline` enforces it; penguins too clean to
    show much, then a case where it bites.
18. (code) Penguins micro-check — scaler on ALL (leaky) vs train only (correct) → ≈ identical.
19. (md) Read — negligible here (mean/std barely move); danger grows with data-hungry preprocessing.
20. (md) The ESL example — 100 samples, 1000 pure-noise features, random labels (honest ≈ 50%).
21. (code) **figure 2** — WRONG select-on-all-then-CV (0.850) vs RIGHT select-in-fold (0.570); bar chart.
22. (md) Read figure 2 — 85% on pure noise = leakage (selection peeked at held-out rows); in-fold ≈
    chance (truth); only *when* preprocessing saw the data differs; the `Pipeline` makes the right way easy.
23. (md) Your turn — (a) standardize-before-split leak + fix; (b) one-hot vs 1/2/3; (c) 99% with
    preprocessing fit before the split — what do you say?
24. (md) What you built + vocabulary — standardization, one-hot, nominal vs ordinal, ColumnTransformer,
    Pipeline, fit-on-train-only, data leakage.
25. (md) Going further (optional) — target leakage; imputation/target-encoding fit on all data; nested
    CV; "fit on train, transform train and test."
26. (md) References — ISLR (ch. 5) DOI 10.1007/978-1-0716-1418-1; ESL §7.10.2 DOI
    10.1007/978-0-387-84858-7; scikit-learn Pipeline/ColumnTransformer/Common pitfalls; Géron ch. 2.
    `Previous: 10` · `Next: —` (end of module 00; `01_KNN` follows).

## Honest limits / no pre-emption

- Standardization payoff stated honestly (small accuracy effect here; boundary rotation + CV edge are
  the evidence); KNN/SVM forward-referenced, not pre-empted.
- ESL leakage demo is clearly-labelled synthetic pure-noise data (constructed cautionary example, not
  penguins); the lesson is the *when-you-preprocess* principle, citable to ESL.
- `ColumnTransformer`/`Pipeline` introduced as the practical leakage-safe tool; deep API not belaboured.
- Imputation only **mentioned** (Going further); the on-ramp does not do it.

## Verification

Measured numbers re-run + reconciled at build. Runs top-to-bottom (nbconvert to /tmp; output-free);
`check_no_hardcoded_hex` passes; `gen_llms_txt` re-run; `pytest` green (14); `course_map.md` NB 11 line
reflects what was built; both reviewers pass; Rémy validates; commit + merge `notebook → chapter`. Then
**chapter 00 close**: `git switch main && git merge --no-ff chapter/00_GettingStarted`.
