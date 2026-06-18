# Notebook plan — 01_KNN / 05_demanding_case_curse

> Status: **APPROVED** (2026-06-17, by Rémy). Notebook plans validated by Rémy alone; reviewers return
> on the built notebook. Drives `notebooks/01_KNN/05_demanding_case_curse.ipynb`.

## Context

NB 5 of the k-NN chapter — the **demanding case** (the notebook-5 role). The first four notebooks used
a 2-D toy (`make_moons`) so every step was visible. Now we run the **full honest workflow** on a real
30-D medical dataset — the Wisconsin breast cancer set — end to end: look at the data → standardize
inside a `Pipeline` (no leakage, NB 11) → choose k by cross-validation (NB 10) → evaluate **once** on a
sealed test set → read the confusion matrix and **recall on the malignant class** (NB 07) → state when
k-NN is and is not the right tool. Then we **feel the curse of dimensionality**: bury the signal under
pure-noise features and watch distances concentrate until "nearest" stops meaning anything. This is
where the chapter's rigor bites: an honest score, a clinically honest error, and a stated limit.
Prereqs: NB 1–4, plus 07 (confusion/precision/recall), 10 (CV), 11 (Pipeline/standardization).

## Design (measured — `sklearn.datasets.load_breast_cancer`, 569×30, malignant 212 / benign 357; stratified 70/30 → 398/171, seed 0)

- **Standardization matters on real data** (the NB 2 payoff): raw (unscaled) 5-fold CV **0.935** vs
  `Pipeline(StandardScaler, KNN)` CV **0.970** — the 30 features span wildly different ranges (areas in
  the hundreds, smoothness ~0.1), so unscaled distance is dominated by the large-range features.
- **Choose k by CV** on the training set: CV accuracy peaks at **k = 7** (0.970). Sequence (k:CV) 1:.952
  3:.962 5:.965 7:**.970** 9:.967 11:.962 15:.965 21:.957 31:.950 — a clear, gentle peak.
- **Honest held-out evaluation** at k = 7: **test accuracy 0.947**; confusion matrix `[[57, 7], [2,
  105]]` (rows = true malignant/benign, cols = predicted). The clinically important number: **recall on
  malignant = 0.891** — **7 of 64 malignant tumours are predicted benign** (false negatives). Accuracy
  0.947 hides that ~11 % of cancers are missed; in screening, a missed malignancy costs far more than a
  false alarm (the asymmetric-cost lesson of NB 07; thresholding is NB 08). Benign precision 0.938,
  recall 0.981, F1 0.959.
- **The curse of dimensionality, felt.** Append pure-noise features (N(0,1) — at the signal's unit
  scale after the Pipeline standardizes) and watch k-NN degrade. The accuracy curve uses
  **`cross_val_score` on the training set** (so the sealed test set is still used exactly once, in the
  held-out eval, cell 11); the mean **near/far ratio** (per test point, nearest-train distance /
  farthest) is pure geometry. Across 0/30/100/300/1000/2000 noise dims:

  | noise dims | CV accuracy | near/far ratio |
  |---|---|---|
  | 0 | 0.970 | 0.121 |
  | 30 | 0.922 | 0.311 |
  | 100 | 0.894 | 0.505 |
  | 300 | 0.887 | 0.711 |
  | 1000 | 0.792 | 0.856 |
  | 2000 | 0.771 | 0.909 |

  CV accuracy falls 0.970 → 0.771 **and the near/far ratio climbs toward 1** — that ratio is the *why*:
  when the nearest neighbour is nearly as far as the farthest, "nearest" carries almost no information
  and the vote degrades (distance concentration, Beyer 1999). The signal never changed; we buried it.

## Library additions / figures

**None to `src/`.** Uses `sklearn.datasets.load_breast_cancer`, `Pipeline` / `StandardScaler` (NB 11),
`cross_val_score` / `StratifiedKFold` (NB 10), `KNeighborsClassifier`, and `sklearn.metrics`
(`confusion_matrix`, `recall_score`, …) + `pairwise_distances` for the near/far ratio. Reuse
`viz.plot_confusion_matrix` (NB 07) and `viz.plot_train_test_curve` (or a one-off twin plot) for the
CV-vs-k and curse curves. `pytest` stays 14.

## Cell-by-cell (~21 cells; look → preprocess → select → evaluate → analyse → the curse → limits)

1. (md) **Header** — `# 05 — Demanding case: breast cancer & the curse`; *Module 01 · k-NN — notebook 5
   of 6*; purpose (first real dataset; the full honest workflow; then feel the curse);
   `Prerequisites: 01–04, plus 07 (confusion/recall), 10 (CV), 11 (Pipeline & scaling)`; objectives
   (run the full workflow on a real 30-D set; standardize without leakage via a Pipeline; choose k by
   CV; evaluate once on test and read recall on the costly class; feel the curse of dimensionality).
2. (code) imports + seed + `use_course_style()` + `load_breast_cancer`; print shape & class balance.
3. (md) **The dataset** — Wisconsin breast cancer: 569 tumours, 30 numeric features (cell-nucleus
   measurements), binary **malignant / benign**. A real screening task — and one where the *kind* of
   error matters. We look before we model (the NB 03 habit).
4. (code) a quick look — class balance; the min/max range of a few features to expose the scale spread.
5. (md) **Read the output** — 212 malignant / 357 benign (mild imbalance); feature ranges differ by
   orders of magnitude (area in the hundreds vs smoothness ~0.1) → pure-distance k-NN will need
   standardization (NB 2), and we must scale **inside** cross-validation to avoid leakage (NB 11).
6. (md) **The workflow** — `Pipeline(StandardScaler, KNeighborsClassifier)` so the scaler is re-fit on
   each training fold (no test/val statistics leak in); CV on the training set to choose k; **one**
   held-out evaluation at the end. Let us first confirm scaling earns its place.
7. (code) raw `KNeighborsClassifier` CV vs `Pipeline(StandardScaler, KNN)` CV on the training set.
8. (md) **Read the output** — scaling lifts CV from 0.935 to 0.970: NB 2's scale trap, now on a real
   30-D set, and the Pipeline applies the fix leak-free.
9. (code) CV accuracy across an odd-k grid (Pipeline) → pick best k; plot CV accuracy vs k.
10. (md) **Read the figure** — CV peaks at **k = 7** (0.970); a gentle plateau around it. That is our
    choice, made on the training data alone.
11. (code) fit the chosen Pipeline on all training data; evaluate **once** on the sealed test set —
    accuracy, `viz.plot_confusion_matrix`, and precision/recall/F1 (including malignant recall).
12. (md) **Read the figure/output** — test accuracy 0.947 looks strong, but read the confusion matrix
    honestly: **7 malignant tumours are predicted benign** (recall on malignant 0.891). In screening,
    that false-negative is the dangerous error — accuracy alone would have hidden it (NB 06–07). One
    would likely lower the threshold to trade some false alarms for fewer missed cancers (NB 08).
13. (md) **When k-NN is the right tool — and when it is not** — here it does well: moderate n,
    standardized, a meaningful distance. Its costs, from notebook 1: predict is slow on large n (it
    measures against every stored point) and it holds the whole dataset in memory. And one failure mode
    we have not yet felt — **high dimensionality**. Let us provoke it.
14. (md) **The curse of dimensionality** — intuition: as dimensions pile up, the distances from a point
    to its nearest and farthest neighbours grow closer together, until every point is about equally far
    from every other. "Nearest" stops being meaningful. We will bury the 30 real features under pure
    noise at the same scale, and watch.
15. (code) append D ∈ {0, 30, 100, 300, 1000, 2000} noise dims (N(0,1) in standardized space); for each,
    record accuracy and the mean near/far ratio (`pairwise_distances`); two-panel figure.
16. (md) **Read the figure** — left: accuracy slides from 0.947 to 0.789 as noise drowns the signal;
    right: the near/far ratio climbs from 0.12 toward 0.91. The right panel is the *why*: when the
    nearest neighbour is nearly as far as the farthest (ratio → 1), "nearest" carries little
    information and the vote decays (distance concentration, Beyer 1999). The data's signal never
    changed — we buried it under noise (a controlled illustration; real high-dimensional data rarely is
    pure noise, but real datasets do carry many weak/irrelevant features, which is the same pressure).
17. (md) **Your turn** — (a) why does the Pipeline standardize **inside** each CV fold rather than once
    on all the data beforehand? (b) the model misses 7 of 64 malignant tumours — what would you change
    to catch more, and what would it cost (NB 08)? (c) in your own words, why does the near/far ratio
    approaching 1 make k-NN useless?
18. (md) **What you built** + vocabulary — the honest workflow (look → Pipeline → CV → one test eval →
    error analysis → limits); confusion matrix & recall on the costly class; the **curse of
    dimensionality**, distance concentration, the near/far ratio; when to / not to use k-NN.
19. (md) **References** — W. Wolberg, W. Street, O. Mangasarian, *Breast Cancer Wisconsin (Diagnostic)*
    (UCI); ISLR §2.2.3; ESL §2.5 & §13.3; K. Beyer et al. (1999), *When is "nearest neighbor"
    meaningful?* (ICDT) — distance concentration; C. Aggarwal et al. (2001) on distance metrics in high
    dimensions. `Previous: 04 — The estimator & its parameters` · `Next: 06 — Advanced: distances &
    choosing k`.

*(Cells 11–12 and 15–16 may each split into an extra read cell at build to keep the
intuition→figure→"Read the figure" rhythm; target stays ~21 cells.)*

## Honest limits / no pre-emption

- The **demanding case** integrates everything; it does not introduce a *new* method concept. It uses
  metrics (NB 06–08), the Pipeline (NB 11), and CV (NB 10) as established tools.
- **No leakage:** the scaler is fit inside CV (Pipeline) and inside the final fit-on-train; the test
  set is evaluated exactly once (cell 11). The curse accuracy curve uses **`cross_val_score` on the
  training set** (not the test set), so the "test used once" promise holds; it is a controlled
  illustration of a geometric phenomenon (deliberately injected noise), not model selection. The
  near/far ratio is pure geometry (distances only, no labels, no fitting).
- The curse is shown with **pure-noise** dims for a clean signal-vs-noise contrast; the notebook states
  that real high-d data is rarely pure noise but routinely carries many weak/irrelevant features, which
  exert the same distance-concentration pressure (Beyer 1999; the deeper metric-in-high-d story is NB
  6, Aggarwal 2001).
- Numbers (k = 7, 0.947, the curse curve) are dataset/seed-specific; the *patterns* (scaling helps;
  CV picks a middling k; accuracy ≠ recall on the costly class; accuracy falls as the near/far ratio
  rises) are the lessons. Stated.

## Verification

Measured anchors (raw 0.935 vs scaled 0.970; CV-best k = 7; test 0.947; confusion `[[57,7],[2,105]]`;
malignant recall 0.891; curse CV 0.970→0.771 with near/far 0.121→0.909) re-run in the notebook and
reconciled into prose at build. Runs top-to-bottom (nbconvert to /tmp; output-free; **`--clear-output
--inplace` before commit**); `check_no_hardcoded_hex` passes; `gen_llms_txt` re-run; `pytest` green
(14, no `src/` change); both reviewers pass (no BLOCK); Rémy validates visually; commit + merge
`notebook → chapter`.
