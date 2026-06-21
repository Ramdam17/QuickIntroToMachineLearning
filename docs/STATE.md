# STATE — where the course build is right now

> The compaction lifeline. Update this at **every** phase transition. On any session start / after a
> compaction, read this first, then `docs/WORKFLOW.md`, then the active plan. See the phase
> vocabulary in `docs/WORKFLOW.md`.

| Field | Value |
|---|---|
| Current chapter | **04_DecisionTree** (5 NBs). Chapter plan **APPROVED** (`docs/plans/chapter_04_DecisionTree.md`, commit `b2c9308`); chapter 03 complete (merged to `main`, PR #3 `8cdcc73`). |
| Current notebook | — (chapter plan approved; opening NB 1). |
| Phase | `chapter-plan-approved` (ready to open NB 1) |
| Active branch | `chapter/04_DecisionTree` (off `main` @ `8cdcc73`) |
| Active plan | `docs/plans/chapter_04_DecisionTree.md` (**APPROVED** — 5 notebooks) |
| Next concrete action | **Open NB 1 — a question that splits the data: impurity.** `git switch -c notebook/04_DecisionTree__01_impurity_and_splits` off `chapter/04_DecisionTree`; set STATE notebook = 01, phase `notebook-plan`; in plan mode draft the NB-1 cell-by-cell plan (Gini/entropy by hand on penguins root: 0.4948 / 0.9925; the best split = max impurity decrease — `flipper ≤ 206` dec 0.4732 beats `bill ≤ 43.25` dec 0.4044; stump 0.9891; the split is a threshold ⇒ **scale-invariant**; figures: impurity-decrease-vs-threshold curve + the chosen split with child class-mix bars). Rémy validates the NB-1 plan via ExitPlanMode → build → both reviewers → visual check → commit → merge to chapter. |

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

- **Chapter 04 (Decision Trees) plan APPROVED & persisted** (`docs/plans/chapter_04_DecisionTree.md`,
  commit `b2c9308`). **FIVE notebooks** (standard arc): NB 1 impurity & the best split (Gini/entropy,
  by hand on penguins) → NB 2 greedy growth + reading the tree → NB 3 overfitting & pruning (the depth
  dial, on `make_moons`) → NB 4 the estimator `DecisionTreeClassifier` & its parameters
  (**variance/instability the headline**) → NB 5 demanding case **breast_cancer** (interpretability vs
  accuracy; where a single tree fails → the bridge to ensembles). First **non-linear** method; the
  **base learner** of ch 06–10. Reviewer-gated, both **REVISE → all folded** (every number re-measured
  by Claude on sklearn 1.9.0): **ml-expert** (MAJOR — NB 5 CV → **CV-on-train** tree 0.940 / LogReg
  0.985 matching shipped ch 03 NB 6; MAJOR — `criterion` re-measured at **default depth** 0.910 / 0.914
  + the no-logarithm-cost argument; MINORs — depth 6 = CV-best not test-peak (peak is depth 7 / 0.900),
  NaN = 2 numeric rows, variance recipe **pinned** `default_rng(0)`/20/`rs=0`/150² grid). **pedagogy**
  (MAJOR — banned words cleaned; MAJOR — **NB 4 de-overloaded** to 4 shown knobs + 2 named, ~24-cell
  ceiling; MINORs — box-count beat in NB 2, KNN spine re-measured 0.942 on the pinned split, charter
  close named). Rémy chose **breast_cancer** for NB 5 (over `penguins_full` / a Titanic loader).
  `course_map.md` §04 already aligned. Next: open NB 1.
- **Chapter 04 (Decision Trees) opened.** Branch `chapter/04_DecisionTree` created off `main` (synced
  @ `8cdcc73` after PR #3). Phase `chapter-plan`: drafting the chapter plan in plan mode per
  `course_map.md` §04 and the per-method arc (a split & impurity by hand → grow & read a tree →
  overfitting/pruning & the depth dial; NB 4 `DecisionTreeClassifier` & its parameters; NB 5 a
  demanding case — interpretability vs accuracy, where a single tree fails). The first **non-linear,
  axis-aligned partition** method, and the **base learner** the ensemble half of the course
  (06 Random Forest → the boosting family) is built on. The pending `idle` STATE edit was folded
  into this transition (committed on the chapter branch, not on protected `main`).
- **CHAPTER 03 (Logistic Regression) COMPLETE — merged to `main` via PR #3** (merge commit `8cdcc73`,
  `gh pr merge --merge`; per-notebook history preserved; pushed to Ramdam17/QuickIntroToMachineLearning).
  Six notebooks: score→probability · boundary & weights · log-loss · gradient descent · estimator &
  parameters · breast_cancer (calibration/threshold). Added `datasets.load_breast_cancer()` + schema test
  (`pytest` 17). The two-reviewer gate + Rémy's visual validation held on every notebook; the sklearn-1.9
  API was pinned throughout; every number re-measured. `main` synced locally to `8cdcc73`, green. STATE
  set to `idle` (pending edit, folds into the chapter-04 opening). Next: chapter `04_DecisionTree`.
- **NB 6 (demanding case: breast cancer) built & merged — CHAPTER 03 COMPLETE (6/6).** The capstone,
  visualization-first (5 figures), on breast_cancer (569×30, malignant = positive). Honest workflow, no
  leakage: split → CV **on train** (LogReg **0.985** > GaussianNB **0.932**) → one sealed test (acc 0.953).
  **Calibration** closes ch 02's loop: LogReg Brier **0.033** vs GaussianNB **0.098** (~3×); a pile-up
  histogram makes GaussianNB's over-confidence visible (166 vs 119/171 at the extremes); reliability diagram
  with the y=x reference. **Threshold = clinical policy** (malignant the costly miss): 0.5 → recall 0.938
  (4/64 missed) vs 0.1 → 0.984 (1 missed, 14 false alarms). **L1** keeps 3/10/14 of 30; **coefficient story**
  (radius/concavity → malignant, clinically sensible). Bridge to trees (ch 04). **src/ add:**
  `datasets.load_breast_cancer()` pandas wrapper + schema test → **pytest 17**. Both reviewers folded
  (ml-expert REVISE→cell-8 truncated sentence + coef-read MINORs; pedagogy PASS→added the reliability
  diagonal, reworded ex-3 for the 1.9 `CalibratedClassifierCV`/`FrozenEstimator` API; several anchors
  re-measured vs the chapter plan — measured values used). `common_errors` gained a 0.5-threshold row;
  `llms.txt` regenerated; ruff/black/hex/banned clean. Rémy validated visually. **Next: PR chapter/03 →
  main, then chapter 04 (Decision Trees).**
- **NB 6 (demanding case: breast cancer — calibration, threshold, error analysis) OPENED.** Branch
  `notebook/03_LogisticRegression__06_breast_cancer_calibration_threshold` off `chapter/03` (@ `c2110e7`).
  Phase `notebook-plan`: drafting cell-by-cell — the chapter capstone (**visualization-first**). Anchors
  re-measured on sklearn 1.9 (breast_cancer 569×30, **malignant = positive** 212 / benign 357; 70/30 seed0
  stratify, one std `Pipeline`, StratifiedKFold5-shuffle-seed0): CV LogReg **0.979** vs GaussianNB
  **0.930**; test LogReg acc 0.953 / Brier **0.033** vs GaussianNB 0.895 / **0.098** (≈3×; pile-up 119 vs
  166/171 — GaussianNB over-confident, closes ch 02's loop); threshold 0.5 → recall 0.938 (4/64 missed)
  vs 0.1 → 0.984 (1 missed, 14 false alarms); L1 nonzero **3/10/14** of 30 at C=0.02/0.2/1.0; top
  malignant-driving coefs radius error / worst radius / mean concave points. **Several numbers differ
  from the chapter plan's preliminary figures** (Brier 0.033/0.098 vs 0.027/0.088; threshold 4/3-missed
  vs 3/2; L1 middle 10 vs 8) — qualitative stories intact, measured values used. Likely `src/` add
  `datasets.load_breast_cancer()` pandas wrapper + test (pytest 16→17). Next: Rémy validates the NB-6
  plan → build → chapter PR into `main`.
- **NB 5 (the estimator & its parameters) built & merged to `chapter/03_LogisticRegression`.**
  Role-4: the real `sklearn LogisticRegression` on the **1.9 API** (verified: `l1_ratio` not the deprecated
  `penalty`; no `multi_class`; `saga` for L1; `C=np.inf`=none). Parity vs by-hand (NB 4). Knobs *shown*:
  **`C`** reg-path ‖w‖₂ 0.84→6.80 → **separation→divergence** (1 overlap pt = finite ‖w‖≈11 vs separable
  slice ≈29); **`l1_ratio`** L1 zeroes the 4 injected noise cols (4/8) vs L2 8/8; **softmax vs OvR** (3
  species, CV 0.956/0.956, 0% disagreement, coef_ (3,2)); honest GridSearchCV (best C≈0.08, sealed test
  1.000 — flagged as easy-split, NB 6 is the real case). 24 cells, 4 figures. **ml-expert REVISE→fixed**
  (API exhaustively verified on 1.9.0, every number bit-for-bit; **1 MAJOR**: the reg-path/divergence
  plateau wrongly blamed on the iteration limit → it is **convergence/tolerance** (`n_iter_`≈14 ≪ 200000)
  → reworded cells 8/11 **+ added a `print(n_iter_)` that proves it**; MINORs: ≈8.5 not "nearly 7", OvR
  renormalizes in predict_proba), **pedagogy PASS** (added "defaults are regularized → parity uses
  `C=np.inf`"). `common_errors` gained a C-is-inverse row; `llms.txt` regenerated; ruff/hex/banned clean;
  pytest 16. Rémy validated visually. Next: open NB 6 (breast cancer) → then chapter PR into `main`.
- **NB 5 (the estimator & its parameters) OPENED.** Branch
  `notebook/03_LogisticRegression__05_estimator_and_parameters` off `chapter/03_LogisticRegression`
  (@ `1b68bc7`). Phase `notebook-plan`: drafting cell-by-cell in plan mode — the role-4 "method &
  parameters" notebook (first to use the real `sklearn LogisticRegression`). **sklearn 1.9 API verified
  at plan time:** `l1_ratio` present, **`penalty` deprecated** (FutureWarning 1.8→1.10: use l1_ratio=0/1,
  C=np.inf), **`multi_class` REMOVED**; `saga` for L1. Anchors measured on 1.9.0: L2 path ‖w‖₂ =
  0.84/1.91/3.28/6.80 (C=0.01/0.1/1/100, 4 std feats, plateau 8.46); separation→divergence — full 2-feat
  (1 overlap pt) MLE finite ‖w‖≈11 vs the slice with that point removed runs to ‖w‖≈29+; L1 (l1_ratio=1,
  saga) zeroes the 4 injected noise cols exactly (4/8 nonzero) while L2 keeps all 8; L1 on 4 real feats
  4/4 (1/4 at C=0.01); multinomial vs OvR (3 species) CV 0.956/0.956, **0.0% disagreement**, coef_ (3,2).
  4 figures planned (L2 path, separation→divergence, L1-vs-L2 noise bars, 3-class softmax boundaries) +
  honest GridSearchCV tuning. Next: Rémy validates the NB-5 plan → build.
- **NB 4 (Fitting II — gradient descent) built & merged to `chapter/03_LogisticRegression`.**
  One concept: **the optimizer** (the course's first), by hand on standardized 1-D `bill_length` (w & b).
  Gradient **(P−y)·x** stated & **verified** (finite-diff err 2e-11; σ′ cancels); update w←w−η∇L; descent
  on NB 3's convex bowl (figB surface+path → bottom; figC loss → floor 0.140). **Parity exact**: by-hand
  GD = `LogisticRegression(C=∞)` (6.29704 / −0.56139) — "the library is not magic". Learning-rate panel
  (figD): 0.1 crawls / 2 glides / 400 overshoots; raw-feature knife-edge (0.003 vs 0.005) = the
  "why standardize" tie-in. Convergence **shown, not proved** (leans on NB 3 convexity); SGD/backprop only
  named (→ ch 11–12). 22 cells, 4 figures. Both reviewers **PASS** (0 BLOCK/MAJOR): every number
  re-derived to machine precision; parity against C=∞ verified (default C=1 → w=4.25, different, so the
  choice is load-bearing). **MINORs folded:** softened "diverges/explodes/leaps past" → "overshoots /
  climbs the wrong way" (on this flat loss η=400 stays bounded, not →∞ — honest); **lr-panel η 90→400
  deviation from the approved plan** (90 did not visibly diverge on the well-conditioned loss — a
  correctness fix; title/legend/read all updated). `common_errors` gained a learning-rate row; `llms.txt`
  regenerated; ruff/hex/banned clean; pytest 16. Rémy validated visually. Next: open NB 5 (estimator &
  parameters).
- **NB 4 (Fitting II — gradient descent) OPENED.** Branch
  `notebook/03_LogisticRegression__04_gradient_descent` off `chapter/03_LogisticRegression` (@ `6940caf`).
  Phase `notebook-plan`: drafting cell-by-cell in plan mode — one concept, **the course's first
  optimizer**: gradient = steepest-ascent direction; step opposite by a learning rate; the weights roll to
  the bottom of NB 3's convex bowl. Gradient **(P−y)·x** (verified vs finite-diff to 2e-11). Anchors
  measured: by-hand full-batch GD on standardized 1-D `bill_length` (w,b) → `LogisticRegression(C=∞)`
  w*=6.297 / b*=−0.561 (gap 4e-4 at lr=1, 1e-7 at lr=2; ~1000 it at lr=0.5); learning-rate panel
  **standardized** (lr 0.1 crawls / 2 glides / 90 oscillates; surface flat, λ_max=0.041, stable to ~48) —
  divergence shown on **raw** bill as the knife-edge (0.003 crawls, 0.005 explodes → the "why standardize"
  tie-in). 4 figures planned (gradient-on-bowl, surface+path, loss-vs-iter, lr panel). Next: Rémy validates
  the NB-4 plan → build.
- **NB 3 (Fitting I — what we optimize: log-loss) built & merged to `chapter/03_LogisticRegression`.**
  One concept: **the training objective**, by hand, pre-fitting. **log-loss = cross-entropy = −log-
  likelihood** of the Bernoulli model (the bridge from ch 02's likelihood); punishes confident-and-wrong
  without bound (−log P; P=0.01→4.6) where squared error caps at 1 (Figure A); **convex** (one bottom,
  2nd-diff ≥ 0, min 0.146 at w≈6.2) vs **squared-error-on-sigmoid non-convex with stalling plateaus**
  (2nd-diff < 0, plateau slope ~3e-4) (Figure B); one number ranks weight choices (w=1/3/6.2 →
  0.39/0.19/0.146). 1-D std bill, **b held at 0**, no sklearn, nothing fitted (NB 4 minimizes). 19 cells,
  2 figures. **ml-expert REVISE→fixed** (every number verified to machine precision incl. gradient
  (P−y)·x and analytic convexity L''≥0; **1 BLOCK = banned word "simply" cell 18 → "exactly"**; MINOR
  bowl-ylim wording), **pedagogy PASS** (added a Bernoulli coin-flip gloss; the honest "single min, not
  bumps" framing praised). **Process fix:** the banned-word guard now parses the JSON real text — the old
  raw-`.ipynb` grep missed words glued after a literal `\n` (that is how "simply" slipped past); NB 1–2
  re-scanned **clean**. `common_errors` gained a "train with squared error" row; `llms.txt` regenerated;
  ruff/hex/banned clean; pytest 16. Rémy validated visually. Next: open NB 4 (gradient descent).
- **NB 3 (Fitting I — what we optimize: log-loss) OPENED.** Branch
  `notebook/03_LogisticRegression__03_logloss_objective` off `chapter/03_LogisticRegression` (@ `d15035d`).
  Phase `notebook-plan`: drafting cell-by-cell in plan mode — one concept, **the objective**: log-loss =
  cross-entropy = −log-likelihood of the Bernoulli model (bridge from ch 02's likelihood), punishes
  confident-and-wrong (−log P unbounded); **log-loss convex** (one bottom) vs **squared-error-on-sigmoid
  non-convex with stalling plateaus**. Anchors measured (1-D std bill: w*≈6.29 / b*≈−0.56, log-loss convex
  min 0.140; MSE 2nd-diff < 0, plateau slope ~3e-4 at w=20; per-example y=1/P=0.01 → log-loss 4.6 vs MSE
  0.98; hand weights w=1/3/6.3 → 0.398/0.188/0.140). **Note:** real 1-D data shows non-convex + plateaus
  (single min), not multiple "bumps" — framing adjusted from the chapter plan's "bumpy", flagged to Rémy.
  Next: Rémy validates the NB-3 plan → build.
- **NB 2 (decision boundary & reading the weights) built & merged to `chapter/03_LogisticRegression`.**
  One concept: **the weighted line & what its weights mean**, by hand, pre-fitting. On **standardized**
  bill+flipper: z=w₁x₁+w₂x₂+b, the **decision boundary** (z=0, P=½), **w ⟂ boundary** & ‖w‖=steepness,
  each **wⱼ = Δ log-odds per std unit** (×e^wⱼ to the odds: bill ×2.7, flipper ×7.4). Hand weights
  w=(1,2), b=0 (nothing fitted — "NB 3–4 find them"): acc **0.9891**, ‖w‖ 2.24, band ~37 % (the 3 errors
  all in-band). Contrast **nearest-centroid** unweighted bisector → **tilt 16.3°** = the weighting (NC acc
  0.9927). Figure C: weights rotate the line, b shifts it. 21 cells, 3 figures. Both reviewers folded:
  **ml-expert PASS** (every number re-measured exact; ‖w‖=steepness verified = ‖w‖/4 slope at z=0; no
  hidden `.fit`; 3 DOIs resolve), **pedagogy REVISE→all folded** (MAJOR: the w arrow didn't render
  perpendicular under unequal axes → `set_aspect("equal")` on figs A/B/C; MINORs: white→blue wording, and
  named that the borderline example is a real in-band error). Dropped the optional ~64° fit teaser (both
  reviewers preferred the clean no-fitting wall). `common_errors` gained a weight-magnitude/standardize
  row; `llms.txt` regenerated; ruff/hex/banned-word clean; pytest 16. Rémy validated visually. Next: open
  NB 3 (log-loss).
- **NB 2 (decision boundary & reading the weights) OPENED.** Branch
  `notebook/03_LogisticRegression__02_boundary_and_weights` created off `chapter/03_LogisticRegression`
  (@ `cbf90d0`). Phase `notebook-plan`: drafting the cell-by-cell plan in plan mode — one concept, on
  **standardized** bill+flipper: the weighted line z=w₁x₁+w₂x₂+b, the **decision boundary** (z=0, P=½),
  **w ⟂ the boundary** & sets steepness, each **wⱼ = Δ log-odds per standardized unit**; weights **set by
  hand** (rotate with w, shift with b), contrasted with module-00 nearest-centroid's *unweighted* bisector;
  **nothing fitted** (NB 3–4 find the weights). Anchors measured at plan time (scaler stats, fitted std
  coefs as the ballpark, NC-normal vs logistic-w angle). Next: Rémy validates the NB-2 plan → build.
- **NB 1 (From a linear score to a probability) built & merged to `chapter/03_LogisticRegression`.**
  One concept: **the sigmoid & log-odds**, fully by hand, pre-fitting. σ(z)=1/(1+e⁻ᶻ) coded from
  scratch & plotted → **p→odds→log-odds** table (the score *is* the log-odds; σ and logit are
  inverses) → σ applied to `bill_length` (**raw mm**) with **hand-chosen** weights (w=1.0, b=−43,
  ½-crossing **43 mm**; nothing fitted — "NB 3–4 find these") → ½-threshold prediction → borderline
  42.9 mm example (P=0.475). Build-measured: hand-rule acc **0.945** (≈ fitted 0.947, never called the
  optimum), transition band ~**21.5 %**. 19 cells, 2 figures. Both reviewers **PASS** (no BLOCK):
  ml-expert verified σ↔logit to 1e-14, the no-fitting promise airtight (no hidden `.fit`), all 3 DOIs
  resolve, calibration correctly **not** claimed; pedagogy confirmed one-concept + e/σ/odds-log-odds
  built from scratch. **2 MINORs folded** (log=natural-log base e; "a fifth" tied to the P∈[0.1,0.9]
  band); **skipped a 3rd** ("all 15 errors in the band") — measured 12/15 in band, 3 confidently-wrong
  → false, and that nuance belongs to NB 6. `common_errors` gained a score-vs-probability/log-odds row;
  `llms.txt` regenerated; ruff fixed (notebook import order I001); pytest 16. Rémy validated visually.
  Next: open NB 2.
- **Chapter 03 (Logistic Regression) plan APPROVED & persisted** (`docs/plans/chapter_03_LogisticRegression.md`).
  **SIX notebooks** (Rémy-approved exception to the 5-ceiling, like KNN's 6th): NB 1 sigmoid & log-odds
  → NB 2 decision boundary & reading weights → NB 3 **log-loss** (the objective) → NB 4 **gradient
  descent** (the optimizer — split from NB 3 on Rémy's go) → NB 5 estimator & parameters
  (`LogisticRegression`: C, `l1_ratio` L1/L2, softmax) → NB 6 demanding case **breast_cancer**
  (calibration/threshold/error analysis). First **discriminative** method; first trained by **iterative
  optimization**; closes ch 02's generative-vs-discriminative loop. Reviewer-gated, both **REVISE→all
  folded**: **ml-expert** (2 BLOCK — NB 1 sigmoid 30%/46mm self-contradiction → raw-mm acc 0.947 /
  crossing ≈43 mm / ~16 %; breast_cancer CV unreproducible → pinned StratifiedKFold5-shuffle LogReg
  **0.979** vs GaussianNB **0.930**; + GaussianNB calibration re-measured under one std pipeline Brier
  **0.088**/pile 167, GD parity vs `C=np.inf`, ‖w‖₂ over 4 std feats, OvR 0.952) — verified the
  **sklearn-1.9 API pivot** (`penalty` deprecated→`l1_ratio`; no `multi_class`→`OneVsRestClassifier`;
  `saga` for L1) and gradient ∝(P−y)·x to machine precision; **pedagogy** (1 BLOCK 2 banned words; 3
  MAJOR — split GD to its own NB, add odds/log-odds + gradient-as-slope first-contacts, give softmax its
  own section). Measured at plan time on sklearn **1.9.0**. `course_map.md` §03 aligned to six titles.
  Next: open NB 1.
- **Chapter 03 (Logistic Regression) opened.** Branch `chapter/03_LogisticRegression` created off
  `main` (synced @ `726d13e` after PR #2). Phase `chapter-plan`: drafting the chapter plan in plan
  mode per `course_map.md` §03 and the per-method arc (sigmoid → boundary/weights → log-loss fitting;
  NB 4 `LogisticRegression` C/L1-L2/multi-class; NB 5 calibration + threshold + error analysis —
  LogReg as the calibrated discriminative foil to NB's over-confidence). The pending `idle` STATE
  edit was folded into this transition (committed on the chapter branch, not on protected `main`).
- **NB 5 (Text classification — the demanding case) built & merged; CHAPTER 02 COMPLETE (5/5).** The
  capstone, on 20-newsgroups: **by-hand bag-of-words on-ramp** (toy sentences → vocab → dense count
  matrix) → `CountVectorizer` (12 384 words, density 0.0043, fit-on-train-only) → `MultinomialNB`
  (fit ≈ms, acc **0.887**, confusion → religion hardest) → **honest eval under imbalance** (one-vs-rest
  sci.med: acc **0.930** vs **baseline 0.724**, P/R/F1 0.887/0.854/0.870, PR AP 0.935) → **calibration**
  (MNB piles 1205/1433 at 0/1 = over-confident *in shape*; Brier 0.056 < LogReg 0.080 here because the
  task is easy → "trust the ranking, not the number"; cost shown on the confusable pair in Your turn) →
  Domingos-Pazzani at scale + **generative-vs-discriminative bridge to ch 03**. 27 cells, 5 figures.
  **2 `src/` additions with tests** (`datasets.load_newsgroups` fetch-and-cache + visible logging;
  `viz.plot_calibration_curve` reliability diagram) → **pytest 16**. Both reviewers **PASS** (no BLOCK):
  ml-expert verified every number + measured that keeping metadata leaks the label (0.887→0.955, so
  `remove=` is right); pedagogy confirmed the by-hand on-ramp + honest calibration framing. 5 MINORs
  folded (calibration wording, "crushes most", no-skill label value, multinomial pointer). `common_errors`
  + `course_map` §02 + `llms.txt` updated. Rémy validated visually. Next: PR `chapter/02_NaiveBayes` →
  `main`, then open chapter 03 (Logistic Regression).
- **NB 4 (The estimators & their parameters) built & merged to `chapter/02_NaiveBayes`.** The
  integration notebook: the NB family (Gaussian/Multinomial/Bernoulli) + each dial *measured* —
  **`var_smoothing`** (flat 0.9927 → 1.0:0.989 → 10:0.711, the flood), **`alpha`/Laplace** healing NB 1's
  penguins zero by hand (α=0→0; α=1→0.0065/post 0.018; verified by-hand == `MultinomialNB.feature_log_prob_`),
  **`priors`/`fit_prior`** tilting the boundary (124↔127 predicted Gentoo; borderline x=[40.8,208] flips
  Adélie→Gentoo at P(Gentoo) 0.15/0.5/0.85), and **honest tuning** (GridSearchCV on train, one sealed
  test). **Calibration named & deferred to NB 5** (penguins too separable, well-calibrated). 20 cells,
  3 figures. Both reviewers **PASS** (no BLOCK): ml-expert REVISE → the asserted Brier 0.0006 reframed as
  a NB-5 forward reference (split-dependent, unshown) + 2 MINORs; pedagogy PASS + 2 MINORs (flip sweep
  aligned to the panels). Rémy questioned whether NB 4 was a 4th concept notebook → confirmed it is the
  role-4 "method & parameters" notebook; chapter stays **5**. `common_errors.md` gained a
  "don't max out smoothing" row; `llms.txt` regenerated; pytest 14. Rémy validated visually.
- **NB 3 (The Gaussian likelihood, computed safely) built & merged to `chapter/02_NaiveBayes`.** One
  concept (chapter stays 5 — split not pulled): model P(feature∣class) with a continuous **density**,
  computed in **log-space**. Bins (NB 1) were crude → **density first-contact** taught from scratch
  (area not height; integrates to 1; can exceed 1) → per-class **Gaussian** fit (Adélie μ38.79/σ2.65,
  Gentoo μ47.50/σ3.07) overlaid on the density histogram (mass→density; the zero-frequency trap
  dissolves) → **by-hand Gaussian NB = sklearn `GaussianNB` 100 %** (acc 0.9927; curved boundary) →
  likelihood is a choice (multinomial/Bernoulli named → NB 5) → **underflow** (product → 0.0 at N=324)
  → **log-space** (sum of log-likelihoods, argmax unchanged). 20 cells, 4 figures, "Your turn" ×3. Both
  reviewers **PASS** (no BLOCK); folded: **3 banned words** ("simply" ×2, "just"), `var_smoothing`
  honesty (by-hand = GaussianNB with smoothing off; scores differ <1e-6 → NB 4 dial), the log-tie
  clause, σ gloss, and a NaN/3rd-species hint on exercise (a). `common_errors.md` gained underflow +
  density-height rows; `llms.txt` regenerated; pytest 14 (no `src/`). Rémy validated visually.
- **NB 2 (The "naive" assumption) built & merged to `chapter/02_NaiveBayes`.** One concept:
  **conditional independence**. Two features need the joint P(bill,flipper∣species); estimating it
  directly is expensive (5×5 grid, **18/25 cells empty** — curse echo). The naive shortcut: assume
  independence given the class → joint = product of 1-D marginals. **Where it breaks**, measured: real
  vs naive joint heatmaps + a **difference panel** (the discarded correlation), within-class corr
  **0.326/0.661** (vs overall 0.869 = mostly between-class). **Does the decision survive?** CV
  GaussianNB **0.9927** ties LDA, beats QDA **0.9890** (Domingos & Pazzani 1997). Honest: GaussianNB =
  QDA with diagonal covariance; the tie is an accuracy coincidence on near-separable data (NB/LDA
  boundaries differ on 11.5 %, NB/QDA on 18.2 % — verified by ml-expert), *not* a model identity; the
  probabilities suffer even when the decision holds (→ NB 5 calibration). 21 cells, 3 figures, "Your
  turn" ×3. Both reviewers **PASS** (no BLOCK); shared MINOR folded ("Read the table" for the count
  grid). `common_errors.md` gained an overall-vs-within-class-correlation row; `llms.txt` regenerated;
  pytest 14 (no `src/` change). Rémy validated visually.
- **NB 1 (Bayes' rule, from counts) built & merged to `chapter/02_NaiveBayes`.** By hand on the binary
  penguins subset: the **prior** P(species) 0.551/0.449 by counting → `bill_length` 3-bin contingency
  (Adélie [135,16,0] / Gentoo [3,67,53]) → **likelihood** P(bin∣species) by row-normalizing → **Bayes'
  rule** (4 terms named) → **posterior** P(species∣bin) (short→Adélie 0.978, medium→Gentoo 0.807,
  long→Gentoo 1.000) → predict by **argmax**, with "evidence cancels under argmax" *demonstrated*
  (`joint.idxmax == posterior.idxmax` True). The bin `long` surfaces the **live zero-frequency case**
  (no Adélie → P=0 → posterior exactly 0/1, overconfident) → foreshadows NB 4 smoothing. 21 cells, 3
  figures (prior bar / likelihood grouped bar / posterior stacked bar), "Your turn" ×3, vocabulary box.
  Both reviewers **PASS** (no BLOCK): ml-expert re-derived every number by exact fractions (45/46, 67/83);
  pedagogy confirmed one-concept + voice. 4 MINORs folded (conditional-probability gloss + drop "joint"
  cell 10; discrete-likelihood-normalization hedge cell 12; exhaustivity already in cell 6). Rémy
  validated visually. `common_errors.md` gained a zero-frequency row; `llms.txt` regenerated; pytest 14
  green (no `src/` change).
- **Chapter 02 (Naive Bayes) plan APPROVED & persisted** (`docs/plans/chapter_02_NaiveBayes.md`).
  5 notebooks, standard arc: NB 1 Bayes from counts (one feature, by hand) → NB 2 the naive
  (conditional-independence) assumption → NB 3 the Gaussian likelihood + log-space → NB 4 the
  estimators & parameters (var_smoothing, alpha/zero-frequency, priors, calibration-bridge) → NB 5
  demanding case = **text classification** (20newsgroups, MultinomialNB, honest eval under imbalance,
  the over-confidence limit, generative-vs-discriminative bridge to ch 03). Datasets: penguins (NB 1–4,
  Gaussian) + a 20newsgroups subset (NB 5, fetch-and-cache). Reviewer-gated: **ml-expert REVISE→1 BLOCK
  fixed** (the headline error: GaussianNB is **QDA with diagonal per-class covariance**, *not* LDA — the
  0.9927 NB/LDA/QDA tie is an accuracy coincidence on near-separable 2-D data; re-checked PASS at the
  parameter level), unfair NB(raw)-vs-LogReg(scaled) → both raw, α-curve marked version-indicative.
  **pedagogy REVISE→no BLOCK** (4 MAJORs folded: NB 3 re-scoped to one concept with mass→density as a
  taught first-contact; bag-of-words built by hand in NB 5; NB 4 calibration as a tight bridge; "Your
  turn" per NB). Measured at plan time: within-class corr 0.326/0.661/0.486; NB/QDA/LDA 0.9927/0.9890/
  0.9927; text acc ≈0.89 (4-cat) / ≈0.70 (hard binary); α→0 = log(0) collapse. Next: open NB 1.
- **Chapter 02 (Naive Bayes) opened.** Branch `chapter/02_NaiveBayes` created off `main` (synced to
  `110c081` after PR #1). Phase `chapter-plan`: drafting the chapter plan in plan mode per
  `course_map.md` §02 and the per-method arc. The pending `idle` STATE edit was folded into this
  transition (committed on the chapter branch, not on protected `main`).
- **Chapter 01 (k-Nearest Neighbours) COMPLETE — 6 notebooks merged to `main` via PR #1** (merge commit
  `110c081`, `gh pr merge --merge` — per-notebook history preserved; pushed to
  `Ramdam17/QuickIntroToMachineLearning`). The arc: the vote → distance & the scale trap → the k dial →
  the estimator & its parameters → demanding case + the curse → advanced distances & nested CV. The
  two-reviewer gate (`@ml-expert-reviewer` + `@pedagogy-reviewer`) + Rémy's visual validation held on
  every notebook; NB 6 was rebuilt from scratch (v1 was visually thin). Next: chapter `02_NaiveBayes`.
- **NB 6 (advanced: distances & choosing k) built & merged — chapter complete (6/6).** Optional Advanced
  capstone, **rebuilt from scratch (v1 scrapped by Rémy as too table-heavy)** → visualization-first, 28
  cells, **9 figures**: unit balls L1/L2/L∞; metric decision boundaries on moons; Mahalanobis ellipse +
  Euclidean-vs-Mahalanobis boundary on penguins; distance-concentration histogram (d=2/50/1000);
  near/far ratio by p; accuracy-vs-noise curve (penguins wash vs breast_cancer p=1>p=2, averaged over 8
  draws); nested-CV schematic + fold-score strip (naive 0.967 vs nested 0.960). Both datasets featured;
  silhouette dropped (it concerns k-means cluster count, not k-NN's k). Pedagogy PASS; ml-expert
  REVISE→fixed (L1 unit-ball geometry — vertices on axes, not sides; "staircase"→tendency; LDA→incise).
  Rémy validated. `feat(01_knn): notebook 06 — advanced: distances & choosing k`. **Note:** a stray
  `/tmp/struct.py` shadows stdlib `struct` for `python /tmp/*.py` → build via `uv run python - < file`.
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
