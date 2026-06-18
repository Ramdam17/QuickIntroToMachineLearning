# STATE — where the course build is right now

> The compaction lifeline. Update this at **every** phase transition. On any session start / after a
> compaction, read this first, then `docs/WORKFLOW.md`, then the active plan. See the phase
> vocabulary in `docs/WORKFLOW.md`.

| Field | Value |
|---|---|
| Current chapter | `01_KNN` (plan APPROVED, 6 notebooks) |
| Current notebook | `06_advanced_distances_and_k` (planning — branch created) |
| Phase | `notebook-plan` |
| Active branch | `notebook/01_KNN__06_advanced_distances_and_k` |
| Active plan | `docs/plans/chapter_01_KNN.md` (NB 6) — per-notebook plan being drafted in plan mode |
| Next concrete action | **Draft & approve the NB 6 plan, then build.** Optional **Advanced** notebook answering Rémy's two questions. **Measured anchors:** **Q1 distance:** unit balls L1/L2/L∞ (geometry — diamond/circle/square); **Mahalanobis shown GEOMETRICALLY** (its unit ball = ellipse aligned to the covariance) — *not* an accuracy race (measured: maha ≈ euclid ≈ std ≈0.85 on correlated data; a clean win needs within-class cov = LDA territory, out of scope — stated honestly); **metric × curse** (Aggarwal 2001): near/far ratio smaller for smaller p (p=0.5<1<2 at every dim), and accuracy on breast_cancer+noise **p=1 0.877 vs p=2 0.842 at 1000 noise dims** (0.906 vs 0.860 at 200) → in high-d prefer smaller p (ties NB 5). Cosine named. **Q2 choosing k:** **nested CV** (inner tunes k, outer estimates) **0.960** honest vs naive/optimistic **0.967** (winner's curse); **silhouette** (Rousseeuw 1987) peaks at **3 clusters** on 3-blob data → it picks k-MEANS cluster count (unsupervised), NOT k-NN's k (supervised → CV). Figures: unit balls; Mahalanobis ellipse vs Euclidean circle; metric×curse (ratio/accuracy); silhouette-vs-clusters. From `chapter/01_KNN`: `git switch -c …` done; plan mode. Prereqs NB 1–5. On approval → persist plan, commit → build. **Then chapter close → PR into `main` (6/6).** |

## Notes / blockers

- **Resolved (lint debt):** Rémy chose option B — fix the notebooks. NB 01–09 made ruff-clean
  (explicit `zip(strict=False)`, unused `pandas` imports removed, long lines wrapped; all seven
  re-executed end-to-end), committed as `f84eec6`. `ruff check .` is now fully green across the repo.
- NB 01 shipped: `ml_course.datasets.load_penguins()` + vendored `penguins.csv` + tests are in place
  and reusable by later notebooks. pandas-first convention recorded in CLAUDE.md/AGENTS.md.
- Through-line reminder: Palmer penguins (binary, 2 features), nearest-centroid first classifier (NB
  05), polynomial-degree complexity dial (NB 09–10).
- Confirmed: notebooks all-English; maths re-established not presupposed; git history = preserve
  per-notebook, mark chapters (`--no-ff` chapter → main).

## Progress log (most recent first)

- **NB 5 (demanding case: breast cancer & the curse) built & merged to `chapter/01_KNN`.** Full honest
  workflow on `load_breast_cancer` (569×30): pandas look → `Pipeline(StandardScaler, KNN)` (raw CV
  0.935 vs scaled **0.970**, NB 2 scale trap on real data) → CV picks **k=7** → one held-out eval (test
  **0.947**) → error analysis: confusion `[[57,7],[2,105]]`, **malignant recall 0.891 = 7 of 64 cancers
  missed** (accuracy hid it; threshold → NB 8) → when to/not use k-NN → **the curse, felt**: CV acc
  **0.970→0.771** as noise dims grow, near/far ratio **0.121→0.909** (→1, the *why*, Beyer 1999). Curse
  via `cross_val_score` on TRAIN (test stays used-once); near/far = pure geometry. Both reviewers PASS
  (re-derived leak-free, near/far = Beyer's inverse relevant-contrast confirmed; applied MINORs:
  plan-table→CV numbers, curse-is-CV framing + Pipeline-rescales note, asymptotic wording, prereqs
  +06/+08). Rémy validated. `feat(01_knn): notebook 05 — demanding case: breast cancer & the curse`.
  **Build note:** a stray `/tmp/struct.py` (a notebook cell-lister) shadows stdlib `struct` for `python
  /tmp/*.py` (sys.path[0]=/tmp) → run builders via `uv run python - < /tmp/x.py` (stdin) instead.
- **NB 4 (the estimator & its parameters) built & merged to `chapter/01_KNN`.** Introduced
  `KNeighborsClassifier`: parity with by-hand (0 diff, 0.956) and `cross_val_score` == NB 3's **0.919**;
  walked `n_neighbors` / `weights` / `metric`. weights uniform vs distance (k=151: 0.678 vs 0.833,
  boundary panels) — **mechanism corrected** after ml review: it is a graded 1/d *tilt* (effective
  ~73/151, nearest-15 hold ~34 %), NOT "near dominates", and 0.833 stays well below uniform@15's 0.956
  (degrades gracefully, doesn't recover). metric small effect (Manhattan p=1 0.967); chose weights by
  `cross_val_score` (distance 0.924 → test 0.967); even-k tie → lowest-label argmax (== by-hand
  `bincount().argmax()` convention) → odd k for binary (binary-only; 3-class can still tie). Pedagogy
  PASS; ml-expert REVISE→fixed (1 MAJOR: false distance-weight mechanism, re-measured & corrected; +
  MINORs: parity edge-case, argmax convention, figure wording). Rémy validated visually. `feat(01_knn):
  notebook 04 — the estimator & its parameters`.
- **NB 3 (the k dial) built & merged to `chapter/01_KNN`.** By-hand k-NN on `make_moons`: k as the
  bias–variance dial — boundaries k=1 (jagged/memorize) / 15 / 151 (over-smooth), contrasted with NB
  05's single bisector; train/test error vs k (log axis, the U; k=1 train err **0.000**); then the
  honest selection — show the test curve, **refuse to use it**, **5-fold CV on TRAIN picks k=15** (cv
  0.919), one sealed test eval → **0.956** (vs k=1 0.933, k=151 0.678). Surfaces (not hides) the
  test-min=k3 vs CV=k15 disagreement. CV by hand (`StratifiedKFold`+loop); `KNeighborsClassifier` /
  `cross_val_score` deferred to NB 4. Both reviewers **PASS** (re-derived CV leak-free; applied 6 fixes:
  variance-inferred-not-measured nod to NB 09, gap *indicates* overfit, CV-fold-seed wobble, odd-k
  explained, boundaries-on-train named, k15/0.956 seed-caveat). Rémy validated visually. `feat(01_knn):
  notebook 03 — the k dial`.
- **NB 2 (distance & the scale trap) built & merged to `chapter/01_KNN`.** By-hand k-NN on
  `make_moons`: Euclidean vs Manhattan (a genuine L1/L2 ranking flip — q=(0,0): L2 picks B=(0.7,0.7),
  L1 picks A=(1,0)), then the scale trap — feature 2 ×50 collapses test acc **0.956 → 0.733**,
  standardizing on **train stats** recovers it **exactly** (caveat stated: exact recovery is special to
  a pure rescale); raw-vs-standardized **decision boundary** (horizontal bands slicing the crescents →
  the two-moon S recovered, both PNGs eyeballed). Honest scoping: metric (0.956 vs 0.944 = **one** test
  point of 90) ≪ scale (~20 pts). Tiny in-notebook `ByHandKNN` wrapper reuses
  `viz.plot_decision_boundary` (single split; CV deferred to NB 3). Both reviewers **PASS**
  (re-measured; applied 5 fixes: "34× linear → ~34²≈1000× in the squared sum", metric-gap-is-one-point,
  ISLR §2.2.3 **& ch. 4**, a `ByHandKNN` framing cell naming the fit/predict interface, forced axis
  labels). Rémy validated visually. `feat(01_knn): notebook 02 — distance & the scale trap`.
- **NB 1 (predict by the neighbourhood vote) built & merged to `chapter/01_KNN`.** k-NN by hand on
  `make_moons(300, 0.30, 0)` (210/90): the majority vote, *k* = neighbourhood size, and the lazy cost
  **felt live** (fit flat ~µs vs predict super-linear at n=300/3000/30000). At build, the running query
  was **re-measured and corrected**: the planned q=(0.60,0.14) sat on the Bayes boundary (~50/50, so
  "true class 0" was dishonest); replaced with **q=(-0.23,0.75)** — a region only the class-0 crescent
  reaches (~85% class 0 → class 0 is the right answer), nearest training labels [1,0,0,0,0], votes
  k1=1/k3=0/k5=0. Both reviewers **PASS** (each re-measured & confirmed truth/votes/timing; 3 MINOR
  polish applied — NB09 ref → module-00 callback, "nothing learned by fitting", Cover&Hart asymptotic
  1-NN qualifier). Rémy validated visually. `common_errors.md` gained a KNN k=1-noise row; `llms.txt`
  regenerated. Commit `feat(01_knn): notebook 01 — predict by the neighbourhood vote`.
- Chapter 01 (k-NN) plan **approved & persisted** (`docs/plans/chapter_01_KNN.md`) — 6 notebooks: vote
  → distance/scale trap → k-dial → estimator/params → demanding case (breast_cancer + the curse) → an
  optional **NB 6 Advanced** (metric geometry L1/L2/L∞ + Mahalanobis/cosine, metric×curse, nested CV,
  and the silhouette≠k-NN clarification — a deliberate, Rémy-approved exception to the 5-ceiling).
  `make_moons` for NB 1–4 (penguins too separable, measured: 0/69 flips), breast_cancer for NB 5; all
  offline. Reviewer-gated (both REVISE→incorporated). `WORKFLOW.md` updated: chapter close now via PR
  (`main` is protected by a global pre-push hook).
- **Chapter 00 (Getting Started) COMPLETE — 11 notebooks, merged to `main` (`--no-ff`).** The on-ramp:
  what ML is → features/feature space → EDA → split & leakage → nearest centroid → accuracy/baseline →
  confusion/precision-recall → scores/ROC/AUC → over/under-fitting → cross-validation → preprocessing &
  leakage. Plus a fetch-and-cache data layer and a "get your data" step in NB 01. Two-reviewer gate +
  Rémy's visual validation throughout.
- NB 11 (preprocessing & leakage) built — standardization paying off NB 05's scale-sensitivity (NC
  boundary rotates ~52.6° in mm coords; CV 0.989→0.9927), one-hot encoding of `island` (by hand +
  `OneHotEncoder` fit-on-train), `ColumnTransformer`+`Pipeline` under CV, and the ESL §7.10.2 leakage
  demo (1000 pure-noise features: select-on-all 0.85 vs select-in-fold 0.57). Reviewer-gated (pedagogy
  PASS; ml-expert REVISE→fixed the boundary-rotation angle — my coordinate-space error), Rémy validated.
- NB 01 "Getting the data" step added (Phase 2 of the data-layer change): a fetch+cache section
  (intro md + `load_penguins_full()` with visible `logging` + a "Read the output") inserted after
  "Learning from labelled examples"; teaches fetching/caching as a first ML step, then narrows to the
  2-feature binary slice. Reviewer-gated (ml-expert PASS; pedagogy REVISE→fixed: framed the logging,
  explained the repeated cached-read, trimmed the forward-ref roster, first-run note; +`datasets.py`
  `sex` docstring MALE/FEMALE). Rémy validated visually, merged.
- Data-layer refactor (prereq for NB 11): `datasets.py` switched from a committed binary CSV to
  **fetch-and-cache** the full Palmer penguins set (pinned seaborn-data URL →
  `src/ml_course/data/penguins_full.csv`, git-ignored, offline after first run); added
  `load_penguins_full()` (344×7: +island, sex, bill_depth, body_mass, Chinstrap, missing values) +
  `PENGUINS_FULL_*` constants; `load_penguins()` now **derives** the 2-feature binary subset (verified
  byte-identical to the old committed 274-row CSV → NB 01–10 behaviourally unchanged); removed the
  committed `penguins.csv` + the `.gitignore` exception; repurposed `vendor_penguins.py` to warm the
  cache (logs the download). Tests 14 green; all 10 notebooks re-execute; ruff/black clean. Rémy chose
  fetch-and-cache + a visible fetch step in NB 01 (Phase 2 next). Logging used in `datasets.py` (hook).
- NB 10 (validating honestly: cross-validation) built — single notebook (Rémy chose over a 10a/10b
  split), 30 cells; continues NB 09's make_moons + `flexible_boundary`. Arc: parameters vs
  hyperparameters → the validation set → single-split instability (degree 3,3,5,6,3,9) → stratified
  k-fold BY HAND → CV picks degree 3 → by-hand == `cross_val_score` (0.914286, exact) → one honest
  test estimate (0.9111) → tuning-on-test inflation (+0.014 over 100 splits). No new `src/` (reused
  `plot_train_test_curve`; k-fold scheme + inflation bar in-notebook). Reviewer-gated (pedagogy PASS;
  ml-expert REVISE→ stratification-exactness MAJOR + minors fixed), Rémy validated, merged. Alongside:
  NB 01–09 made ruff-clean (option B, `f84eec6`).
- NB 09 (over-/under-fitting, the generalization gap) built — make_moons + polynomial-degree dial
  (linear engine as plumbing), complexity curve U, generalization gap (≠ variance), bias-variance,
  learning curve, optional variance fan; added `viz.plot_train_test_curve` + test. Reviewer-gated
  (pedagogy PASS; ml-expert REVISE→fixed the "train error always falls" vs measured kink), Rémy
  validated, merged.
- NB 08 (scores, thresholds, ROC & AUC) built — signed-distance score (`s>0` = centroid), threshold
  trade-off, ROC/AUC 0.989, PR curve, two-feature contrast AUC 1.0; added `viz.plot_roc_curve` +
  `plot_score_threshold` + tests. Reviewer-gated (both PASS; 2 minor polish), Rémy validated, merged.
  Closes the evaluation trilogy (06-08).
- NB 07 (confusion matrix, precision & recall) built — bill-only weaker model for real errors, cm
  [[37,1],[2,29]], precision 0.967 / recall 0.935 / F1 0.951, asymmetric costs. Reviewer-gated (both
  PASS; 3 minor polish), Rémy validated, merged.
- NB 06 (accuracy + baseline) built — accuracy formalised, DummyClassifier baseline, the accuracy
  paradox on an imbalanced what-if (Dummy 95%/0-of-2 vs centroid 100%/2-of-2), positive = Gentoo.
  Reviewer-gated (both PASS; 3 minor polish), Rémy validated, merged.
- NB 05 (first classifier: nearest centroid) built — by-hand fit/predict class, decision boundary
  (extended `plot_decision_boundary` to be pandas-first + label-agnostic, with a test), honest loop
  (100% test vs 55% baseline, by-hand = sklearn). Reviewer-gated (ml-expert REVISE→fixed, pedagogy
  PASS), Rémy validated, merged.
- NB 04 (generalize, don't memorize — stratified split + rote-memorizer demo) built, reviewer-gated
  (both PASS; convergent honesty MINOR fixed), Rémy validated, merged.
- NB 03 (look before you model — EDA) built (+ `viz.plot_class_balance` / `plot_feature_histograms`
  + tests), reviewer-gated (both PASS; polish applied), Rémy validated, merged. Also fixed NB 01 c06's
  dangling "fuller dataset" forward-reference (flagged by pedagogy reviewer).
- NB 02 (features, labels, the feature space) built, reviewer-gated (both PASS; 2 minor polish
  applied), Rémy validated visually, committed and merged into `chapter/00_GettingStarted`.
- NB 01 (what is machine learning?) built, reviewer-gated (pedagogy PASS; ml-expert REVISE→fixed:
  corrected the figure reading, softened the line claim, added subset honesty), Rémy validated
  visually, committed and merged into `chapter/00_GettingStarted`.
- Chapter 00 plan approved and persisted; `course_map.md` 00 section rewritten to 11 notebooks.
- Chapter 00 plan reviewer-gated (both REVISE→incorporated).
- Course scaffold + build workflow set up on `main` (3 infra commits).
