# Notebook plan — 05_SVM / 05_breast_cancer_scaling_limits

> Status: **APPROVED** (2026-06-22, by Rémy; NB plans are validated by Rémy alone — the two reviewers
> gate the *built* notebook, not this plan). Drives the build. Numbers re-measured at build and
> reconciled into prose. **This is the final notebook of chapter 05; after it ships, the chapter
> closes via PR into `main`.**

## Context

NB **5 of 5** — the chapter's **demanding practical case and capstone**, **visualization-first** (~20
cells a floor). A kernel SVM on **breast-cancer diagnosis** (569 × 30) run the full honest way, where
the chapter's strength (a strong, well-tuned non-linear classifier) and its honest limits (scale
sensitivity, no probabilities for free, no scaling to large `n`, no native importances) all bite. The
same dataset KNN met (ch 01 NB 5, the curse), logistic regression read calibrated probabilities from
(ch 03 NB 6), and a decision tree turned into rules (ch 04 NB 5) — now meet it with the **widest
margin**. The honest payoff and the honest ceiling are both stated with numbers.

## Anchors (measured at plan time, sklearn **1.9.0**; re-measured at build)

**Positive class = MALIGNANT** (`y = (target == 0)`). 569 patients, malignant 212. 70/30 stratified
split (seed 0): train 398 / test 171 (malignant test 64). CV = `StratifiedKFold(5, shuffle=True,
random_state=0)` on train.

- **Scaling headline (course_map §05's "scaling matters"):** RBF `SVC` **raw CV 0.9095 → standardized
  CV 0.9648** inside a `Pipeline` (fit-on-train-only). A distance-based method, so the ch 01 scale trap
  is back; standardization is not optional.
- **CV-tune → one sealed test:** `GridSearchCV` over `C`/`gamma`/kernel (standardized pipeline) → best
  **`{C=100, gamma=0.001, kernel="rbf"}`**, CV **0.9824**, **sealed test 0.9649**, **42 support vectors
  of 398**.
- **Cross-method TEST spine (same pinned split):** **KNN(k=5) 0.9415 · DecisionTree 0.9064 · LogReg
  0.9532 · SVM(tuned) 0.9649** — the margin method is the strongest *here* (stated honestly: on this
  dataset and split, not universally).
- **Confusion (tuned SVM, test):** `[[104, 3], [3, 61]]` — **3 of 64 cancers missed**, recall
  **0.9531**, 3 false alarms.
- **Calibrated probability & the threshold — an honest surprise:** wrap the tuned pipeline in
  `CalibratedClassifierCV(..., method="sigmoid", ensemble=False)` (NB 4's pattern). Sliding the
  threshold **down** (0.5 → 0.3 → 0.2) keeps the **same 3 missed cancers** and only **adds false
  alarms** (1 → 7 → 12). The 3 misses sit at calibrated probability **below 0.2** — they are
  *confidently* misclassified, not borderline. So the threshold knob, which rescued ch 03's logistic
  regression (NB 6, where lowering it caught more cancers), **does not help here**. A real, measured
  lesson: the threshold trade works only when the misses are near the boundary.
- **The large-`n` limit, MEASURED:** kernel `SVC` `.fit` time on synthetic 20-feature data —
  `n` 500 → 32 000 grows **0.003 s → 2.67 s** (empirical **n^1.67**; worst-case `O(n³)`), while
  `LinearSVC` stays ~flat (0.001 → 0.019 s); the kernel-vs-linear ratio explodes **4× → 141×**. Kernel
  SVMs do not scale; `LinearSVC` / `SGDClassifier` are the large-`n` alternatives.

## Library / figures

- **No `src/` change** — reuse `viz.use_course_style`, `viz.plot_class_balance` (Fig A),
  `viz.plot_confusion_matrix` (Fig E); the scaling bar (B), the `C × gamma` heatmap (C, `CMAP_PROBA`,
  as in NB 4), the cross-method spine bar (D) and the fit-time curve (F) are one-off in-notebook
  figures in charter colours. No `plot_svm_decision` (30-D, no 2-D boundary to draw). Sklearn: `SVC`,
  `LinearSVC`, `GridSearchCV`, `CalibratedClassifierCV`, `KNeighborsClassifier`, `LogisticRegression`,
  `DecisionTreeClassifier`, `StandardScaler`, `Pipeline`, `StratifiedKFold`, `cross_val_score`,
  `train_test_split`, `confusion_matrix`, `make_classification`. (`pytest` stays 19.)
- **Six figures** (visualization-first; each + "Read the figure"): **A** class balance; **B** the
  **raw-vs-standardized** accuracy bar (the scaling headline); **C** the **`C × gamma` CV heatmap** on
  breast_cancer (the tuning surface); **D** the **cross-method test spine** bar (KNN / tree / LogReg /
  SVM); **E** the **confusion matrix**; **F** the **measured fit-time vs `n`** curve (kernel `SVC` vs
  `LinearSVC`, log–log).

## Cell-by-cell (~26 cells; visualization-first; "Read the figure" after every figure)

1. (md) **Header** — `# 05 — A demanding case: breast cancer`; *notebook 5 of 5*; warm welcome; this is
   where the chapter comes together. **Prerequisites:** NB 1–4 (the whole method); ch 01 NB 5 (KNN on
   this same dataset — the curse), ch 03 NB 6 (logistic regression's calibrated probabilities on it),
   ch 04 NB 5 (a tree on it — readable rules); module 00 — confusion / recall (NB 07), cross-validation
   (NB 10), the `Pipeline` / fit-on-train-only (NB 11). **What you'll be able to do:** run an honest
   SVM workflow end to end; avoid the scale trap a kernel SVM is acutely vulnerable to; tune `C`/`gamma`
   by CV and read a sealed test once; place the SVM on the cross-method spine; read a calibrated
   probability and see when the threshold knob *cannot* help; state the kernel SVM's large-`n` limit.
2. (code) **Imports + seed + style + data** — `df = datasets.load_breast_cancer()`;
   `X = df.drop(columns="target")`, **`y = (df["target"] == 0).astype(int)`** (*malignant = 1 = the
   costly miss*); 70/30 stratified split (seed 0); print shapes & class counts.
3. (md) **Where we are & the stakes** — 569 patients, 30 measurements, malignant 212. The dataset KNN
   felt the curse on, logistic regression read calibrated probabilities from, and a tree turned into
   rules; now meet it with the **widest margin**. No standardization yet — we show *why* it is needed.
   Malignant is the costly miss, so we watch **recall on malignant**, not accuracy alone.
4. (code) **Fig A — class balance** (`plot_class_balance`) + a `df.describe()` glance at two features'
   wildly different scales (e.g. `mean area` vs `mean smoothness`).
5. (md) **Read the figure (A)** — 37 % malignant: not balanced, so accuracy alone can mislead; we watch
   malignant recall. And the feature scales differ by orders of magnitude — a warning for a
   distance-based method.
6. (md) **Intuition — the scale trap returns** — an SVM measures distances (the RBF kernel is a
   function of `‖x − x′‖`), so a large-range feature drowns the rest, exactly as for KNN (ch 01).
7. (code) **Fig B — raw vs standardized** — RBF `SVC` CV on train, raw vs inside a
   `Pipeline(StandardScaler, SVC)`: **0.910 vs 0.965**.
8. (md) **Read the figure (B)** — standardizing lifts CV accuracy by ~5 points. This is the chapter's
   headline limit made concrete: for an SVM, scaling is mandatory, fit on the training data only.
9. (md) **Intuition — tune honestly** — choose `C`/`gamma` by cross-validation **on the training
   split**, then read the **sealed test once** (module 00 NB 10).
10. (code) **`GridSearchCV`** over `C`/`gamma`/kernel (standardized pipeline) → best
    `{C=100, gamma=0.001, rbf}`, CV **0.982**; print the **sealed test 0.965** and the 42 support
    vectors.
11. (code) **Fig C — the `C × gamma` CV heatmap** on breast_cancer (the tuning surface the grid
    searched).
12. (md) **Read the figure (C)** — the bright cell (`C=100, gamma=0.001`) is where the grid landed; the
    surface shows the same bias–variance geography as NB 4, now on a real 30-D problem.
13. (md) **Intuition — line the methods up** — the SVM is the chapter's payoff; place it on the same
    sealed test as the methods the course already met on this dataset.
14. (code) **Fig D — cross-method test spine** bar: KNN 0.942 / tree 0.906 / LogReg 0.953 / **SVM
    0.965** (all on the pinned split).
15. (md) **Read the figure (D)** — the margin method is the strongest **here** — but this is one
    dataset and one split, not a universal ranking; the honest claim is "competitive and, on this
    problem, best", not "SVMs win".
16. (code) **Fig E — confusion matrix** (`plot_confusion_matrix`, tuned SVM, test): `[[104,3],[3,61]]`.
17. (md) **Read the figure (E)** — the SVM misses **3 of 64 cancers** (recall 0.953) and raises 3 false
    alarms — fewer of both than the ch 04 tree (`[[95,12],[4,60]]`).
18. (md) **Intuition — a probability, and the threshold's limit** — `decision_function` is a score
    (NB 4); calibrate it (`CalibratedClassifierCV`) to get a probability, then ask whether lowering the
    threshold can catch the 3 missed cancers.
19. (code) **Calibrate + slide the threshold** — wrap the tuned pipeline (Platt / `ensemble=False`);
    print recall / missed / false-alarms at thresholds 0.5 / 0.3 / 0.2.
20. (md) **Read the result — an honest surprise** — lowering the threshold **does not** recover the 3
    misses (they stay missed) and only **adds false alarms** (1 → 7 → 12): those 3 tumours sit at
    calibrated probability **below 0.2** — *confidently* misclassified, not borderline. The threshold
    knob that rescued ch 03's logistic regression (NB 6) **cannot help here**; it trades precision for
    recall only when the misses are near the boundary. An honest limit, measured.
21. (md) **Intuition — the large-`n` limit** — a kernel SVM solves a dense problem in the number of
    samples; it does not scale. We *measure* it rather than quote it.
22. (code) **Fig F — fit-time vs `n`** (kernel `SVC` vs `LinearSVC`, log–log): `n` 500 → 32 000, kernel
    `SVC` 0.003 s → 2.67 s (≈ **n^1.67**), `LinearSVC` ~flat; the ratio explodes 4× → 141×.
23. (md) **Read the figure (F)** — kernel SVC training grows super-linearly (here ≈ `n^1.67`; worst
    case `O(n³)`), so at 10⁵–10⁶ rows it becomes impractical, while `LinearSVC` / `SGDClassifier` stay
    near-linear. For large tabular data, reach for the linear SVM (or another method).
24. (md) **Error analysis & honest limits + when to use SVM** — strengths: a strong, well-tuned
    non-linear classifier on small-to-medium, well-scaled data with a clear margin. Limits, all shown:
    **must standardize**; **no probability for free** (calibrate); **the threshold cannot rescue
    confident misses**; **no native feature importance** (unlike trees — a real interpretability cost);
    **does not scale to large `n`**. **When to use:** clean, scaled, small-to-medium data. **When not:**
    huge `n`, a need for built-in importances or probabilities, or raw unscaled features.
25. (md) **Your turn** (3 tiered) — *easy*: standardize vs not and report the CV gap; *medium*: read
    the `C × gamma` heatmap and name the cell you would ship and why; *harder*: subsample/replicate to
    grow `n`, time `SVC` vs `LinearSVC`, and say at what `n` you would switch.
26. (md) **What you built + chapter wrap** — an honest SVM workflow on real diagnostic data: the
    **scaling headline**, CV model selection, the cross-method spine, the calibrated-probability limit,
    and the measured large-`n` ceiling. **Vocabulary:** scale sensitivity · CV model selection ·
    cross-method spine · calibrated probability · the threshold's limit · the `O(n²–n³)` ceiling ·
    `LinearSVC`. **Chapter wrap — support vector machines, end to end:** from the widest street (NB 1)
    through the soft margin (NB 2), the kernel trick (NB 3), and the estimator's knobs (NB 4) to a
    real, honestly-evaluated case (NB 5). The first **margin-based** method and the home of the
    **kernel trick**; powerful on clean medium-sized data, but scale-sensitive and limited on large
    `n`. The course turns next to **ensembles** (Module 06 — Random Forests), which average many trees
    into a strong, scalable tabular baseline. **References** (with DOI); `Previous: 04 — The estimator
    and its parameters` · `Next: Module 06 — Random Forests`.

## Honest scoping (stated in the notebook)

- **Positive = malignant**, stated up front; the costly miss; recall-not-accuracy under imbalance.
- **Scaling is mandatory** — the headline, measured (raw 0.910 vs std 0.965), the ch 01 trap returning.
- **SVM wins *here*, not universally** — the spine framed as "competitive and best on this split", not
  a general ranking.
- **The threshold knob has a limit** — lowering it does not recover *confident* misses (measured: the 3
  cancers stay missed); it helps only borderline misses. An honest contrast with ch 03 NB 6.
- **No probability for free, no native importance, no scaling to large `n`** — each stated, the last
  one *measured* (`n^1.67`), with `LinearSVC`/`SGD` named as the large-`n` route.
- **CV-on-train, one sealed test** — no leakage; the test set is read once.
- **Bridge forward** — the variance/limits motivate the ensembles (ch 06+), the next module.

## Verification

Build via `uv run python - < <scratchpad>/build_ch05_nb5.py` (stdin). Re-measure at build: scaling raw
0.9095 / std 0.9648; GridSearch best `{C100, γ0.001, rbf}` CV 0.9824 / test 0.9649 / 42 SVs; spine KNN
0.9415 / tree 0.9064 / LogReg 0.9532 / SVM 0.9649; confusion `[[104,3],[3,61]]` recall 0.9531; threshold
0.5/0.3/0.2 → missed stays 3, false alarms 1/7/12; fit-time n 500→32000 (0.003→2.67 s, ≈ n^1.67, ratio
4→141×). Runs top-to-bottom (nbconvert to scratchpad; tracked file **output-free**, `--clear-output
--inplace`); **banned-word scan over the JSON real text** = 0; `check_no_hardcoded_hex` passes;
`gen_llms_txt` re-run; `ruff` / `black` clean; `pytest` 19 (no `src/` change); `course_map.md` §05
marked complete; `common_errors.md` extended (the threshold-cannot-rescue-confident-misses trap). Both
reviewers pass (no BLOCK); Rémy validates visually; commit `feat(05_svm): notebook 05 — a demanding
case: breast cancer`; merge `notebook → chapter`; **then close CHAPTER 05 via PR into `main`** (`--no-ff`
merge commit, per-notebook history preserved).
