# NB plan — 10_LightGBM / 05_miniboone — the demanding case (visualization-first capstone)

> Status: **APPROVED by Rémy (via ExitPlanMode, 2026-06-28)**. The capstone & last NB of ch 10. No
> reviewer gate at the NB-plan stage (reviewers return on the built notebook). Reframed vs the chapter
> plan's "dial-`n` crossover" thesis after live measurement (see Context) — Rémy signed off.

## Context

Chapter 10, NB 5 of 5 — the **capstone** and the **last notebook** (then close ch 10 via PR → main). A
larger, real tabular problem (**MiniBooNE**, particle physics: signal vs background) where **speed can
matter**, mobilizing the whole chapter: leaf-wise (NB 1), GOSS (NB 2), the categorical split (NB 3, named —
MiniBooNE is all-numeric), the estimator/knobs/tuning (NB 4). The honesty thesis lands here: **the
histogram boosters are close on accuracy; the speed winner is set by the matching convention (tree shape),
not the library — "no universal best, measure on your data."** Visualization-first (≥6 figures, ~30 cells).

### Measured refinement of the chapter-plan "dial-`n` crossover" thesis (Rémy-approved)

The chapter plan anticipated "dial `n` up to find where LightGBM crosses ahead." Measured (50k→4M, offline):
**no crossover within the practical range — the winner is set by the convention, not `n`.**
- **Matched capacity** (leaf-wise, num_leaves=31): LightGBM fastest at every `n` (50k→800k); 3 boosters tie
  on PR-AUC (within 0.002).
- **Default-vs-default** (LightGBM nl31 leaf-wise vs XGBoost depth-6 depthwise): XGBoost faster at every `n`
  up to 4M (smaller default trees), gap narrowing monotonically (2.4×@50k → 1.4×@1M → 1.32×@2M → **1.11×@4M**)
  → LightGBM's default overtakes only **beyond ~4M**. The live NB dials to ~600k; prose cites the offline
  1M–4M run.

## Live anchors (measured, lightgbm 4.6.0 / xgboost 3.2.0 / sklearn 1.9.0, SEED=0 — `measure_ch10_nb5.py`)

- **MiniBooNE** (`fetch_openml("MiniBooNE", version=1, as_frame=True)`): 130064×50, all float64, **0 NaN**,
  `signal` 72/28 (False 93565 / True 36499). Split 97548 train / 32516 test (stratified).
- **Baselines:** Logistic(scaled) PR-AUC 0.862 / acc 0.882 (0.31s); RandomForest(200) 0.945 / 0.936 (7.2s).
- **Matched capacity** (num_leaves/max_leaf_nodes=31, depth unbounded, 300 trees, lr 0.1): **LightGBM 2.37s
  / 0.9573**, **XGBoost-hist(lossguide,31) 2.97s / 0.9586**, **HistGBR 3.23s / 0.9569** — LightGBM fastest,
  all within 0.002 PR-AUC (ROC-AUC ~0.985, acc ~0.944).
- **Unmatched default:** LightGBM 2.43s / 0.9573; XGBoost depth-6 **1.04s / 0.9588** (smaller default trees).
- **Tuned LightGBM + early stopping** (val slice, `average_precision`, `early_stopping(50)`): best_iteration
  **495**, PR-AUC 0.9581 / ROC-AUC 0.9853 / acc 0.9446; **val-chosen F1 threshold 0.455 → test F1 0.9046**
  (vs 0.5 → 0.9019).
- **GOSS efficiency:** full 0.9573 / **GOSS(.2,.1) 0.9579** / uniform-30% 0.9556 — **GOSS ≈ full AND >
  uniform** (the NB-2 edge appears on this larger real data; times ~flat 2.3–2.65s).
- **Dial-`n` matched** (LightGBM vs XGBoost-hist): LightGBM faster at all n — 50k 2.02/2.89, 150k 2.34/3.09,
  400k 2.91/3.62, 800k 3.70/4.42.
- **Dial-`n` default** (LightGBM vs XGBoost depth-6): XGBoost faster, gap narrowing — 50k 1.97/0.81, 200k
  2.40/1.16, 500k 3.10/1.82, 1M 4.24/3.02, 2M 6.90/5.23, **4M 11.43/10.33 (1.11×)**.
- **Importances:** gain (train) top ParticleID_0/16/2/12/31; **permutation (held-out PR-AUC)** top
  ParticleID_12/2/0/27/16 — **top-1 differs** (gain→PID_0, permutation→PID_12).

## Cell-by-cell (~30 cells, 7 figures) — visualization-first

1. (md) Header — capstone; MiniBooNE; mobilizes NB 1–4; the honest thesis. Prereqs ch 00, NB 1–4, ch 06/08/09.
2. (md) The dataset — MiniBooNE signal/background, 130k×50, 72/28, no NaN.
3. (code) Setup + load + split — `LGBM_VERBOSE=0` switch (carried), named DataFrame; `y=(target=='True')`;
   stratified train/test + val slice. Print shape/balance/NaN.
4. (code) Fig 1 class balance + (md) read (imbalance → lead with PR-AUC).
5. (code) Fig 2 PCA-2D scatter by class + (md) read (overlap but separable; non-linear signal).
6. (md) Baselines.  7. (code) logistic + RF → PR-AUC (0.862 / 0.945).
8. (md) The boosters & the matched-capacity question (the shared leaf-wise+histogram skeleton; one convention).
9. (code) Matched comparison (LightGBM/XGBoost-hist/HistGBR) → time + PR-AUC.
10. (code) Fig 3 matched scatter (time vs PR-AUC) + RF/logistic markers + (md) read (within 0.002; LightGBM fastest).
11. (md) Unmatched default-vs-default reconciliation (tree shapes differ).
12. (code) default-vs-default → print.  13. (md) read — gap is tree shape, not the library.
14. (md) Does scale change the winner? (dial `n`).
15. (code) Dial-`n` (live ≤600k): matched + default fit times → Fig 4 (speed vs n, 2 panels).
16. (md) read Fig 4 — matched: LightGBM always; default: XGBoost always but gap narrows (cite offline 4M
    1.11×); winner = convention, slowly scale; **no universal best**.
17. (md) GOSS on real data (NB 2's edge — does it appear?).
18. (code) GOSS full/GOSS/uniform → Fig 5 PR-AUC bars + (md) read (GOSS ≈ full & > uniform here; times flat).
19. (md) A model to ship: tuned LightGBM + early stopping + honest threshold (chosen on val).
20. (code) Tuned LightGBM (num_leaves=63, early_stopping(50), average_precision) → best_iter 495; held-out.
21. (code) Fig 6 PR curve + val-chosen threshold + confusion (0.455) + (md) read (PR-AUC 0.958, F1 0.905).
22. (md) Which features matter? gain vs permutation (the honest reading deferred from NB 4).
23. (code) gain vs permutation → Fig 7 + (md) read (disagree even on top-1; trust permutation).
24. (md) Honest verdict & when to reach for LightGBM (no universal best; convention + scale; large/wide/
    sparse/categorical regime). Closes boosting family; bridge to ch 11–12.
25. (md) Your turn — (easy) matched at num_leaves=63; (core) dial default `n` further, estimate crossover;
    (reach) add a categorical feature, native vs one-hot (NB 3 in action).
26. (md) What you built + References — Ke et al. 2017; Roe et al. 2005 (MiniBooNE, DOI
    10.1016/j.nima.2004.12.018); Chen & Guestrin 2016; sklearn HistGBR docs; ch 00, ch 06/08/09.

(The build expands each grouped "Read"/figure into its own md cell → ~30 cells, 7 figures.)

## `src/` & guards
- **No `src/` change** (reuse `viz`; `fetch_openml`; the four estimators + `permutation_importance`; pytest
  20). Colours only from `ml_course.colors`; seeds fixed.
- Build from `build_ch10_nb5.py` (rebuild right before `git add` — kernel-drift guard).
- **Output hygiene carried from NB 4:** `LGBM_VERBOSE=0` switch (documented, flip to 1); named DataFrame
  end-to-end; never `verbose=-1`. Live dial-`n` ≤ ~600k (nbconvert runtime); cite offline 1M–4M in prose.
- Exit guards: nbconvert exit 0 (7 figures), banned scan = 0, hex clean, ruff/black clean, output-free;
  **two-reviewer gate** (no BLOCK) → fold → **Rémy visual** → end-of-NB checklist (`gen_llms_txt.py`,
  `common_errors` +rows, `course_map` §10 → COMPLETE, pytest 20, STATE) → commit
  `feat(10_lightgbm): notebook 05 — MiniBooNE capstone` → `git merge --ff-only` into `chapter/10_LightGBM`
  → **close ch 10 via PR `chapter/10_LightGBM → main` (`--no-ff`)** on Rémy's explicit go.

## Verification (end-to-end)
1. nbconvert a copy → exit 0, 7 figures, anchors reproduce (matched LightGBM 2.37s/0.957 fastest, 3 within
   0.002; default XGBoost 1.04s faster; GOSS 0.958 > uniform 0.956; tuned early-stop ~495, threshold 0.455
   → F1 0.905; gain vs permutation top-1 differs). 2. hex + banned + ruff clean. 3. pytest 20.
   4. Two-reviewer gate, then Rémy visual. 5. Then PR to close the chapter.
