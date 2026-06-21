# Chapter plan — 04_DecisionTree (Decision Trees)

> Status: **APPROVED** (2026-06-20, by Rémy; reviewer-gated — both `@ml-expert-reviewer` +
> `@pedagogy-reviewer` REVISE → all BLOCK / MAJOR / MINOR folded, every number re-measured on sklearn
> **1.9.0**). Drives the notebook loop in `docs/WORKFLOW.md`. Numbers re-measured at each notebook's
> build and reconciled into prose.
>
> **Five notebooks** (the standard per-method arc — no 6th needed; the fundamentals split cleanly
> into three one-concept notebooks). The course's **fourth method**: the first **non-linear**
> classifier, the first expressed as **human-readable rules**, and the **base learner** the entire
> ensemble half of the course (06 Random Forest → 07–10 the boosting family) is built on.

## Context

Logistic regression (ch 03) drew **one straight line** and read a calibrated probability off it — and
its closing notebook named its own ceiling: a **linear** boundary **underfits** when the truth is
curved (it cannot fit `make_moons`). Decision trees are the answer the course has been pointing to:
instead of one global line, **ask a sequence of simple yes/no questions**, each splitting the data on
one feature at a threshold, carving the feature space into **axis-aligned boxes**. A curved or
interaction-shaped boundary a line cannot touch becomes a staircase of rectangles a tree builds with
ease.

Three things make this chapter pivotal for everything after it:

1. **The first non-linear model — recursive partitioning.** KNN voted by distance, NB multiplied
   probabilities, logistic regression drew a weighted line. A tree **recursively partitions**: pick
   the question that most "unmixes" the classes, split, and repeat on each side. The decision regions
   are unions of axis-aligned boxes — a genuinely non-linear classifier built from very simple
   parts.
2. **The first model that is a set of rules a human can read.** A fitted tree is a **flowchart**:
   "if flipper ≤ 206 mm and bill > 47 mm → Gentoo." No weights to standardize, no distances — the
   model *is* its explanation. This is interpretability of a kind KNN / NB / logistic regression
   could not offer, and it is the chapter's first payoff.
3. **The base learner of the back half of the course.** Random forests (ch 06) average many trees;
   AdaBoost / gradient boosting / XGBoost / LightGBM (ch 07–10) add trees in sequence. Understanding
   **one** tree — how it splits, why it overfits, why it is unstable — is the prerequisite for all of
   them. The chapter ends by *feeling* the weakness (a single tree is **high-variance**) that the next
   five chapters exist to fix.

Two more properties, each a deliberate contrast with what came before, are surfaced and measured:
trees are **scale-invariant** (splits are thresholds, so standardization changes nothing — the ch 01
scale trap and ch 03 standardization no longer apply), and they handle **multiclass and missing
values natively** (no one-vs-rest / softmax machinery, no imputation). The cost of all this is the
chapter's honest spine: an unpruned tree **overfits** (it memorizes — training accuracy → 1.0) and a
single tree is **high-variance** (a small change in the data → a very different tree).

## Prerequisites (re-established briefly; **first-contacts flagged, not mislabelled as recaps**)

- **Module 00 (genuine recaps):** train/test split & leakage (NB 04); accuracy + a baseline (NB 06);
  the confusion matrix, precision/recall (NB 07); **over-/under-fitting and the generalization gap,
  on `make_moons`** (NB 09) — the **depth dial is a complexity dial** and the train/test **U-curve**
  returns here, now with depth on the x-axis; **cross-validation** (NB 10) — used to choose depth and
  the pruning strength. **Note the absence:** standardization / `Pipeline` (NB 11) is *not* needed —
  a deliberate, stated contrast (trees are scale-invariant).
- **Chapter 00 NB 05 (nearest centroid)** — a single *linear* bisector; the tree's axis-aligned boxes
  are the visual counterpoint.
- **Chapter 01 (KNN):** the **scale trap** (NB 2) — trees are its counterexample (scale-invariant, no
  standardization); **`make_moons`** (NB 1–4) is reused as the non-linear playground; **breast_cancer**
  (NB 5) is reused in NB 5 here — the cross-method demanding-case spine (KNN felt the curse → logistic
  regression read calibrated probabilities → a single tree gives readable rules but is high-variance).
- **Chapter 03 (Logistic Regression) — now genuine recaps:** the **linear boundary that underfits
  curved truth** (the NB 6 bridge that opened the door to trees); the **complexity dial** (C ↔ depth)
  and the U-curve; **CV-on-train, one sealed test** tuning discipline; the breast_cancer **LogReg
  baseline** (CV-on-train 0.985 / test 0.953) that NB 5 measures the tree against.
- **Re-established from scratch (assumed from nobody) — the FIRST CONTACTS, each budgeted as a
  taught-from-scratch concept, never called a recap:**
  - **node impurity** — *how class-mixed* a group is: **Gini** = 1 − Σ pₖ² and **entropy** =
    − Σ pₖ log₂ pₖ, built and plotted from scratch — **NB 1**;
  - **the split score = impurity decrease** (parent impurity minus the sample-weighted impurity of the
    two children; a.k.a. information gain for entropy) — **NB 1**;
  - **recursive / greedy partitioning** — apply the best-split rule to each child, repeat; **greedy ≠
    globally optimal** (the optimal tree is NP-hard, Hyafil & Rivest 1976) — **NB 2**;
  - **a tree as a flowchart of rules; a leaf predicts the majority class** — **NB 2**;
  - **overfitting as memorization** (an unpruned tree reaches 0 training error) and **pruning** —
    **pre-pruning** (`max_depth`, `min_samples_leaf`) and **post-pruning** (cost-complexity,
    `ccp_alpha`) — **NB 3**;
  - **variance / instability of a single tree** (resample the data → a different tree) — the property
    that motivates ensembles — **NB 4**;
  - **Gini feature importance** and its **caveat** (biased toward high-cardinality / continuous
    features; permutation importance as the honest alternative) — **NB 4 / NB 5**.

## Datasets (measured at plan time, sklearn **1.9.0**; seeds fixed; re-measured at build)

- **NB 1–2: Palmer penguins** — the binary 2-feature subset (`load_penguins` / `penguins_xy`:
  Adélie / Gentoo, `bill_length_mm`, `flipper_length_mm`), the course fil-rouge, in **raw units**
  (trees are scale-invariant, so no standardization — a teachable contrast). Measured facts the
  fundamentals rest on:
  - **Root node:** counts Adélie 151 / Gentoo 123 (n = 274), p(Gentoo) = 0.4489 → **Gini = 0.4948**,
    **entropy = 0.9925 bits** (computed by hand, then matched to sklearn).
  - **Best single split (scan thresholds, maximize impurity decrease):** **`flipper_length_mm` ≤ 206.0
    → Gini decrease 0.4732** *beats* **`bill_length_mm` ≤ 43.25 → 0.4044**. A clean teaching beat and a
    callback: ch 03's sigmoid favoured `bill_length`; the impurity criterion favours `flipper` — *the
    "best feature" depends on the question you ask.* The depth-1 **stump** scores **train acc 0.9891**.
  - **Greedy growth (NB 2):** depth-2 tree **train 0.9964** (4 leaves), depth-3 0.9964 (5 leaves),
    unlimited **1.0000** (6 leaves, depth 4 — it **memorizes**). **CV: depth-2 0.9855 > full 0.9818**
    (the unpruned tree already generalizes slightly *worse* — the NB 3 hook). The depth-2 rules read
    cleanly (`flipper ≤ 206` then a `bill_length` split on each side) → a 4-box decision region.
  - **By-hand == `DecisionTreeClassifier(max_depth=2)`** (identical splits and predictions) — the
    parity that proves the mechanism, the same move every chapter makes.
- **NB 3–4: `make_moons(n_samples=300, noise=0.30, random_state=0)`**, 210 / 90 stratified split
  (seed 0) — the same non-linear set KNN (ch 01) and the over/underfitting notebook (00 NB 09) used,
  and the one **logistic regression could not fit**. Trees carve it.
  - **Depth dial (overfitting, NB 3):** depth 1 → train 0.833 / test 0.744 (**underfit**, one stripe);
    depth 6 → train 0.981 / test 0.889 (**CV-best**; the raw test optimum is depth 7 at 0.900 — but we
    never tune on test); depth ≥ 8 → **train 1.000** / test 0.878 (**memorizes**, jagged single-point
    boxes). **CV-best depth = 6** (CV 0.919). The train→1.0 climb and the widening train−test gap are
    the overfitting signature (the test dip is mild but real).
  - **Cost-complexity post-pruning (NB 3):** `ccp_alpha` 0.0 → 23 leaves / test 0.878; **0.01 → 8
    leaves / test 0.900 (best)**; 0.05 → 3 leaves / test 0.833. Pruning a memorizing tree *improves*
    held-out accuracy — the lesson, measured.
  - **Parameters (NB 4):** `criterion` gini / entropy / log_loss → at the **default depth** CV
    **0.910 / 0.914 / 0.914** (a near-tie; entropy edges Gini *here*, and the ranking flips to
    0.871 / 0.867 at `max_depth=4` — it is noise-level, so Gini is the default for being **cheaper to
    compute** (no logarithm), not for scoring higher). `min_samples_leaf` 1 / 5 / 20 / 50 → test
    0.878 / **0.933** / 0.800 / 0.744 (a clean pre-pruning dial).
  - **Variance / instability (NB 4 headline):** 20 bootstrap resamples of the training set, under a
    **pinned recipe** (`np.random.default_rng(0)`, 20 resamples, `random_state=0` per tree,
    decision-region disagreement on a 150×150 grid) → **full trees: test mean 0.877, std 0.032,
    pairwise disagreement 6.3 %**; depth-3 trees: std 0.022, disagreement 5.6 % (shallower = steadier).
    Two trees fit on two resamples draw **visibly different boundaries** — the motivation for averaging
    (ch 06). *(penguins is too separable to show overfitting — its full tree still CVs 0.982 — which is
    exactly why the overfitting/variance story lives on moons.)*
- **NB 5: breast cancer** (`datasets.load_breast_cancer`, 569 × 30, **malignant = 1 = positive**,
  37.3 %), reused from ch 01 NB 5 / ch 03 NB 6 — the cross-method spine, now read as **rules**.
  Pinned like ch 03: 70/30 stratified split (seed 0), `StratifiedKFold(5, shuffle=True,
  random_state=0)`.
  - **Interpretability vs accuracy, measured (CV-on-train, the ch-03 protocol):** a single tree
    **underperforms the linear model** — tree **CV 0.940** vs LogReg **CV 0.985** (cross-validated on
    the training split, matching the shipped ch 03 NB 6); tuned tree (GridSearchCV) **test 0.906** vs
    LogReg **0.953**. Yet the **depth-3 tree is readable** (train 0.975 / test 0.918, 7 leaves) —
    clinically sensible rules (`mean concave points ≤ 0.05` → benign branch; size/concavity
    thresholds), the kind of explanation KNN / NB / LogReg could not give.
  - **Where a single tree fails — variance:** across 25 bootstraps, **test std 0.021** and the **root
    split feature itself changes** — `mean concave points` 15×, `worst perimeter` 6×, `worst concave
    points` 3×, `worst radius` 1× (of 25). Even the *first question* is unstable. → the bridge to
    **ensembles** (ch 06: average many trees, variance falls).
  - **Feature importance + caveat:** Gini importance concentrates hard (`mean concave points` 0.74,
    `worst area` 0.08, `worst concave points` 0.07) — useful but **biased** toward continuous /
    high-cardinality features (Strobl 2007); permutation importance named as the honest cross-check.
- **Trees' native strengths (shown in NB 4, measured):** **scale-invariant** (raw == standardized,
  identical tree — the anti-scale-trap demo); **multiclass native** (no OvR/softmax) and **missing
  values native** (sklearn ≥ 1.3) — `penguins_full` (3 species) trees to **CV 0.9535** with its **2
  numeric-feature-NaN rows kept** untouched (11 rows miss `sex`, a column the numeric-feature tree
  never sees). Honest limit stated: sklearn's implementation still needs **numeric encoding for string
  categoricals** (CART theory handles categoricals; the library wants numbers).
- All datasets offline (sklearn-bundled / generated); colours from `ml_course.colors`; "Read the
  figure" after every figure; a **"Your turn" (2–3 tiered)** in every notebook; every notebook closes
  on the charter arc (**Your turn → What you built → optional Going further → References**), and **NB 5
  closes by framing the single tree's variance as the *door to ensembles* (ch 06), never as a wall**;
  a running decision-tree vocabulary box.

## Primordial concepts → notebooks 1–3 (one concept each; by hand before any library)

| NB | Title | The one concept | Done by hand | Key figure(s) → "Read the figure" (one per figure) | Your turn (tiered sketch) |
|----|-------|-----------------|--------------|----------------------------------------------------|---------------------------|
| 1 | A question that splits the data: **impurity** | **Impurity & the best split.** A node's **impurity** measures how class-mixed it is (**Gini** 1−Σpₖ², **entropy** −Σpₖlog₂pₖ); the **best yes/no question** is the threshold split that most **decreases** impurity (parent − sample-weighted children = information gain) | compute Gini & entropy of the root by hand (penguins: 0.4948 / 0.9925) and match sklearn; **scan thresholds** on a feature, compute each split's weighted child impurity, find the best — `flipper ≤ 206` (decrease 0.4732) edges out `bill ≤ 43.25` (0.4044); note the split is a **threshold ⇒ scale-invariant** | (a) **impurity-decrease vs threshold** for a feature (the best split = the peak); (b) the cloud with the chosen split line + the two children's class-mix bars — *"the split that most unmixes the two sides is the question the tree asks first; the curve's height = how much purer the split makes things"* | easy — compute a node's Gini from its counts; medium — scan the other feature and find its best split; harder — argue why a pure node has impurity 0 and a 50/50 node is maximal |
| 2 | Growing a tree, and reading it | **Recursive greedy growth → a readable flowchart.** Apply the best-split rule to each child, and **repeat** — that is the whole tree. A **leaf** predicts its **majority class**. Greedy (locally best split each time) is **not** the globally optimal tree (NP-hard) but works well in practice | grow depth-2 **by hand** (re-split each child of NB 1's root) → 4 leaves; **trace a penguin** down the questions to its leaf; confirm **by-hand == `DecisionTreeClassifier(max_depth=2)`**; plot the **axis-aligned boxes** | (a) the **tree as a flowchart** (`plot_tree` / charter schematic) with the rules + leaf classes; (b) the **4-box decision region** over the cloud — *"each split is a horizontal or vertical cut; follow the questions top-to-bottom and you land in one box, whose colour is the prediction; depth 2 → 4 leaves → 4 boxes, and one more level can split a box in two — the seed of the depth dial"* | easy — trace a given penguin to its leaf and predict; medium — name the box a point falls in from the rules; harder — say what one more level of depth would add, and why train accuracy can only rise |
| 3 | Overfitting & pruning: **depth is the complexity dial** | **Control complexity or memorize.** An unpruned tree drives **training error to 0** by isolating points (overfit); **depth** is the complexity dial (00 NB 09's U-curve returns). **Prune** to generalize — **pre-prune** (`max_depth`, `min_samples_leaf`) or **post-prune** (cost-complexity `ccp_alpha`); choose by **CV** | on moons: grow boundaries at depth 1 / 6 / unlimited (underfit → good → jagged); plot the **train/test U-curve vs depth** (train→1.0, test peaks at 6); run the **cost-complexity path** and pick `ccp_alpha` by CV (0.01 → 8 leaves, test 0.900) | (a) **three boundaries** depth 1 / 6 / unlimited side by side; (b) **train vs test accuracy vs depth** (the U / the gap); (c) **pruning**: test accuracy & #leaves vs `ccp_alpha` (or the boundary before/after) — *"depth 1 is too coarse, an unlimited tree carves a box around every noisy point; the best tree is in between, and pruning finds it"* | easy — from the U-curve pick the depth you'd ship; medium — predict train & test behaviour at depth 1 vs unlimited; harder — prune at a given `ccp_alpha`, report leaves and test accuracy |

## Notebook 4 — the estimator & its parameters (~24 cells; the integrative notebook)

`sklearn.tree.DecisionTreeClassifier` for real (sklearn **1.9.0**). **Parity first:** the by-hand
depth-2 tree (NB 2) == `DecisionTreeClassifier(max_depth=2)` on penguins. Then walk the knobs — **four
core dials shown**, two more **named** — keeping the notebook to a **soft ~24-cell ceiling** so the
*headline* (the variance section) gets room, not knob-completeness:
- **Shown — the four core dials:** **`max_depth`** (the main complexity dial; NB 3's U-curve as the
  knob), **`min_samples_leaf`** (pre-pruning by node size — moons 1→5→20→50: test 0.878 → **0.933** →
  0.800 → 0.744), **`ccp_alpha`** (cost-complexity post-pruning; NB 3 as the knob), and **`criterion`**
  — at the default depth a measured **near-tie** on moons (`gini` 0.910 / `entropy` 0.914 / `log_loss`
  0.914; the ranking is noise-level and **flips with depth** — at `max_depth=4` Gini edges entropy
  0.871 / 0.867). Treat it as a tie and pick **Gini as the default because it skips the logarithm
  (cheaper to compute), not because it scores higher** (ESL §9.2.3).
- **Named, one line each (no figure):** **`max_features`** — random split-feature subsets, a deliberate
  **foreshadow of random forests** (ch 06) — and **`class_weight`** — reweighting under imbalance (used
  lightly in NB 5).
- **Instability & variance — THE headline (course_map §04's "instability and variance"):** resample
  the training data → a **different tree**. Two bootstrap trees' boundaries side by side, plus the
  20-bootstrap numbers under the **pinned recipe** (`np.random.default_rng(0)`, 20 resamples,
  `random_state=0` per tree, disagreement on a 150×150 grid): **full trees std 0.032 / disagreement
  6.3 %**, depth-3 steadier at **0.022 / 5.6 %**. This is the weakness ensembles fix — the explicit
  bridge to ch 06.
- **What trees handle that earlier methods needed machinery for (one compact section):**
  **scale-invariance** — a measured raw-vs-standardized demo → an **identical tree** (the ch 01 scale
  trap disappears and the ch 03 `StandardScaler` step is unnecessary) — plus a one-line aside that
  trees are **natively multiclass** (no OvR/softmax) and handle **missing values natively** (sklearn
  ≥ 1.3 — `penguins_full` with its 2 numeric-NaN rows kept fits untouched, CV 0.9535). Honest limit:
  string categoricals still need numeric encoding in sklearn (the implementation, not CART theory).
- **Feature importance** — Gini importance introduced (one figure) with the **bias caveat** (continuous
  / high-cardinality features look more important); **permutation importance named** as the honest
  alternative (shown in NB 5, not here).
- **Honest tuning:** `GridSearchCV` over `max_depth` / `min_samples_leaf` / `criterion` / `ccp_alpha`
  on TRAIN; one sealed test — the module-00 NB 10 discipline.
- **Your turn:** tune a knob and read the boundary; verify raw == standardized → an identical tree;
  raise `min_samples_leaf` and watch the leaf count fall and the boundary smooth out.

## Notebook 5 — the demanding practical case: **breast cancer (interpretability vs accuracy; where a single tree fails)**

A single decision tree on **breast_cancer** (569 × 30) — the full honest workflow, and the place where
the chapter's payoff (readable rules) and its honest limit (accuracy & variance) **both** bite:
- **Look → fit → tune → sealed test:** class balance; **no standardization** (a tree point, stated);
  fit a tree; CV-tune depth / `ccp_alpha` on train; one held-out evaluation. *"The dataset KNN felt
  the curse on and logistic regression read calibrated probabilities from — now read it as rules."*
- **Interpretability — the payoff:** the **depth-3 tree as a flowchart** (test 0.918, 7 leaves) — a
  short, clinically sensible rule set (`mean concave points`, `worst area`, concavity thresholds) a
  clinician could read. The model *is* its explanation.
- **Accuracy — the honest cost:** tree **CV-on-train 0.940 / tuned test 0.906** vs ch 03 **LogReg
  0.985 / 0.953**. The cross-method **test** spine, all on the same pinned 70/30 seed-0 split (KNN
  re-measured here, CV-best k = 5): **KNN 0.942 → LogReg 0.953 → single tree 0.906** — the most
  interpretable, the least accurate here.
- **Where a single tree fails — variance:** the **root split feature changes across bootstraps**
  (`mean concave points` 15× / `worst perimeter` 6× / …), test std 0.021. A single tree is
  **high-variance** — the weakness the next five chapters exist to fix.
- **Feature importance, honestly:** the Gini-importance bar (`mean concave points` dominates) **with**
  the high-cardinality-bias caveat and a permutation-importance cross-check.
- **Bridge forward — ensembles (ch 06+):** averaging many trees on many resamples cancels the
  variance (bagging → random forests); adding trees to fix mistakes (boosting, ch 07–10). The
  "harder" exercise is a **hand-built preview**: bootstrap a handful of trees, majority-vote, and
  check that test variance drops — random forests in miniature.
- **Your turn:** read the tree's rule path for a given patient; compare tree vs LogReg and state the
  interpretability/accuracy trade-off; (harder) hand-bag K bootstrap trees + majority vote and measure
  whether the variance falls.

**Five notebooks** — the standard per-method arc (fundamentals one-concept-each across NB 1–3, the
estimator NB 4, the demanding case NB 5). No 6th is needed: trees do not have a load-bearing concept
under-budgeted the way logistic regression's optimizer was. *(If Rémy wants an optional advanced 6th —
e.g. regression trees, or a fuller bagging-by-hand bridge — it is a clean add, but not proposed by
default.)*

## Resolved decision

**NB 5's dataset = `breast_cancer`** — Rémy chose it (over `penguins_full` and a new Titanic loader).
It matches course_map §04's NB-5 theme exactly ("interpretability vs accuracy; where a single tree
fails"), carries the strongest measured failure story (tree CV-on-train 0.940 < LogReg 0.985;
root-feature instability; the ensemble bridge), and keeps the cross-method demanding-case spine (KNN →
LogReg → single tree). Trees' native strengths (multiclass / missing values / scale-invariance) are
demonstrated separately in **NB 4**, so they are not lost.

## Library additions (decided per-notebook, with tests; none forced now)

- **Likely none in `src/`.** `datasets.load_breast_cancer` already exists (ch 03); penguins and
  `make_moons` are in place; the impurity-decrease curve, variance panels, importance bar and tree
  flowchart are one-off in-notebook figures (charter colours) or sklearn's `plot_tree`.
- **Possible (NB 2, reused NB 5): `viz.plot_decision_tree(model, *, feature_names, class_names)`** — a
  thin charter-aware wrapper over `sklearn.tree.plot_tree` (the flowchart appears in NB 2 *and* NB 5,
  so it clears the "reused ≥ 2×" bar). **Charter watch:** `plot_tree`'s default node fills are its own
  blue/orange, not the palette — the wrapper either renders `filled=False` (charter-neutral, rule text
  + class only) or fills leaves by class via `CLASS_CYCLE`. Decided at NB 2 build; if promoted, add a
  smoke test (`pytest` 17 → 18).
- **Reused as-is from `ml_course.viz`:** `use_course_style`, `plot_decision_boundary` (the axis-aligned
  boxes — NB 2/3/4), `plot_train_test_curve` (the depth U-curve — NB 3/4), `plot_confusion_matrix`,
  `plot_class_balance`, `plot_feature_histograms`.

## Honest limits stated in the notebooks

- **A single tree overfits** (unpruned → 0 training error by memorizing) — shown on moons, fixed by
  pruning / CV, never hidden.
- **A single tree is high-variance** (resample → different tree, root feature flips) — measured in
  NB 4 and NB 5; named as the precise motivation for ensembles (ch 06+), not waved away.
- **Greedy ≠ optimal** — the tree takes the locally best split each step; the globally optimal tree is
  NP-hard (Hyafil & Rivest 1976). Works well in practice; stated, not oversold.
- **Axis-aligned only** — splits are single-feature thresholds, so a diagonal boundary becomes a
  staircase of boxes (visible on moons); rotations / oblique splits are out of scope.
- **Gini importance is biased** toward continuous / high-cardinality features (Strobl 2007) —
  permutation importance named as the honest cross-check.
- **Interpretability has a cost here** — on breast_cancer the readable tree is *less* accurate than
  the linear model; interpretability vs accuracy is a real trade, stated with the numbers.
- **Scale-invariance is a genuine strength** (no standardization, unlike KNN / LogReg) — and string
  categoricals still need encoding in sklearn (the implementation, not the theory).

## References (with DOI where they resolve; **dereferenced & pinned by the ml-expert reviewer at build**)

- Breiman L, Friedman J, Olshen R, Stone C (1984). *Classification and Regression Trees (CART).*
  Wadsworth (book; edition pinned at build) — the foundational method, Gini & cost-complexity pruning.
- Quinlan JR (1986). *Induction of decision trees (ID3).* Machine Learning 1:81–106.
  DOI: 10.1007/BF00116251 — entropy / information gain.
- Hyafil L, Rivest RL (1976). *Constructing optimal binary decision trees is NP-complete.* Information
  Processing Letters 5(1):15–17. DOI: 10.1016/0020-0190(76)90095-8 — why growth is greedy.
- Hastie T, Tibshirani R, Friedman J (2009). *The Elements of Statistical Learning*, §9.2 (trees) /
  §9.2.3 (cost-complexity pruning). DOI: 10.1007/978-0-387-84858-7.
- James G, Witten D, Hastie T, Tibshirani R (2021). *An Introduction to Statistical Learning*, §8.1
  (decision trees). DOI: 10.1007/978-1-0716-1418-1.
- Strobl C, Boulesteix A-L, Zeileis A, Hothorn T (2007). *Bias in random forest variable importance
  measures.* BMC Bioinformatics 8:25. DOI: 10.1186/1471-2105-8-25 — the Gini-importance caveat.
- Breiman L (1996). *Bagging predictors.* Machine Learning 24:123–140. DOI: 10.1007/BF00058655 — the
  variance fix (the bridge to ch 06).

## Verification (per notebook, at its commit)

Every number re-run and reconciled into prose at build on sklearn **1.9.0** (NB 1: root Gini 0.4948 /
entropy 0.9925, best split flipper ≤ 206 dec 0.4732 vs bill ≤ 43.25 dec 0.4044, stump 0.9891; NB 2:
depth-2 train 0.9964 / 4 leaves / CV 0.9855 vs full 1.0 / CV 0.9818, by-hand == sklearn; NB 3: depth
sweep test 0.744 → 0.889 (CV-best 6) → 0.900 (depth 7) → 0.878, ccp 0.01 → 8 leaves / test 0.900;
NB 4: criterion default-depth near-tie 0.910 / 0.914 / 0.914 (flips to 0.871 / 0.867 at depth 4),
min_samples_leaf 0.933 at 5, bootstrap variance under the pinned recipe (`default_rng(0)`, 20
resamples, `random_state=0`, 150×150 grid) full std 0.032 / disagree 6.3 % & depth-3 0.022 / 5.6 %,
scale-invariance raw == std, penguins_full 2 numeric-NaN rows CV 0.9535; NB 5: **CV-on-train** tree
0.940 vs LogReg 0.985 (matching shipped ch 03 NB 6), tuned test 0.906 vs 0.953, depth-3 test 0.918,
the **test spine** KNN 0.942 → LogReg 0.953 → tree 0.906 re-measured on the pinned split, root-feature
flip across bootstraps, importance concave points 0.74). Runs top-to-bottom (nbconvert to scratchpad;
tracked file **output-free**, `--clear-output --inplace`); **banned-word scan over the JSON real text**
(`just / simply / obviously / trivially` + FR) = 0; `check_no_hardcoded_hex` passes; `gen_llms_txt`
re-run; `pytest` green (17, → 18 only if a `viz.plot_decision_tree` helper lands); `ruff` / `black`
clean; `course_map.md` §04 aligned to the final titles; `common_errors.md` extended per new intuition
trap. Both reviewers pass (no BLOCK) on each notebook; Rémy validates visually; commit per notebook;
**chapter close via PR into `main`** (protected).

## Per-notebook status (update as the chapter is built)

| NB | Branch | Status |
|----|--------|--------|
| 1 | `notebook/04_DecisionTree__01_impurity_and_splits` | **done** — built (23 cells, 4 figures), both reviewers folded (pedagogy PASS; ml-expert REVISE → 1 MAJOR fixed: the ch-03 `bill`-vs-`flipper` callback reframed to "the criterion *measures* the best feature"), Rémy validated visually, merged to `chapter/04_DecisionTree` |
| 2 | `notebook/04_DecisionTree__02_growing_and_reading` | **done** — built (20 cells, 2 figures: a custom charter flowchart + depth-1-vs-depth-2 regions), both reviewers **PASS** (no BLOCK; MINORs folded — box-counting back-ref reworded, CV qualifier/gloss/prereq, flowchart 47.20), Rémy validated visually, merged to `chapter/04_DecisionTree` |
| 3 | `notebook/04_DecisionTree__03_overfitting_and_pruning` | **done** — built (21 cells, 4 figures: 3 boundaries, train/test U-curve, ccp pruning path, unpruned-vs-pruned), both reviewers **PASS** (no BLOCK; MINORs folded — dangling figure refs, training-impurity, CV-coincidence guard), Rémy validated visually (asked to re-check the thin bands → verified they are real tree fences/overfitting, not a render bug; both "Read the figure" cells tightened to name them), merged to `chapter/04_DecisionTree` |
| 4 | `notebook/04_DecisionTree__04_estimator_and_parameters` | not started |
| 5 | `notebook/04_DecisionTree__05_breast_cancer_interpretability` | not started |

## Reviewer notes (chapter-plan gate — both reviewers REVISE → all folded; re-verified at plan time)

- **ML expert — REVISE (no BLOCK; ~18/22 anchors reproduced exactly, the rest re-measured & folded):**
  MAJOR — NB 5's cross-method CV used **full-data** CV while the prose promised CV-on-train and the
  shipped ch 03 NB 6 reports **0.985** → re-measured **CV-on-train tree 0.940 / LogReg 0.985**, tuned
  test 0.906 / 0.953 kept. MAJOR — the `criterion` triple (0.871 / 0.867) was an undocumented
  `max_depth=4` artefact where Gini wins → re-measured at the **default depth (0.910 / 0.914 / 0.914**,
  entropy edges Gini), flagged the ranking as noise-level / depth-dependent, and rested "Gini = default"
  on the **no-logarithm cost** argument. MINOR — "depth-6 = the peak" → depth 6 is **CV-best**, the test
  optimum is depth 7 (0.900); reworded as a CV-selection-≠-test-max teaching moment. MINOR — "11 NaN
  rows" → only **2** rows miss the numeric features the tree splits on. MINOR — depth-3 variance unpinned
  → **pinned the recipe** (`default_rng(0)`, 20 resamples, `rs=0`, 150×150 grid → full 0.032 / 6.3 %,
  depth-3 0.022 / 5.6 %). All re-verified by Claude at plan time. Praised: mechanism correctness
  throughout, the sklearn-1.9 API verified to the letter, the complete honest-limits section, immaculate
  fundamentals numbers, all six DOIs resolve.
- **Pedagogy — REVISE (no BLOCK):** MAJOR — **banned words in the plan's seed prose** ("trivially"
  l.29, "simply" l.43 / l.172) → reworded clean. MAJOR — **NB 4 overloaded** for ~20 cells →
  restructured: **four core knobs shown** (`max_depth` / `min_samples_leaf` / `ccp_alpha` / `criterion`),
  `max_features` / `class_weight` **named not shown**, native-strengths **compacted to one section**,
  permutation importance **moved to NB 5**, soft ceiling **~24 cells** stated. MINOR — "more depth ⇒
  more boxes" never an explicit beat → added to NB 2's figure read. MINOR — KNN spine point (0.947) came
  from a different ch-01 protocol → re-measured on **this chapter's pinned split (KNN 0.942, CV-best
  k = 5)**, added to Verification. MINOR — charter close not named → added ("Your turn → What you built
  → … "; NB 5 frames variance as the **door to ch 06**). Praised: exemplary first-contact fencing, the
  NB 1→2→3 carving, the by-hand parity discipline, the measured honesty spine, and the "hand-bag K
  trees" capstone bridge.
