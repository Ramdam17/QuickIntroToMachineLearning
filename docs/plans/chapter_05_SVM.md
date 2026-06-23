# Chapter plan — 05_SVM (Support Vector Machines)

> Status: **APPROVED** (2026-06-22, by Rémy; reviewer-gated — `@pedagogy-reviewer` **PASS**,
> `@ml-expert-reviewer` **REVISE → 1 MAJOR + MINORs all folded**; 21/22 headline anchors reproduced
> exactly and the SVM API facts **verified on the live sklearn 1.9.0 install**, not quoted). Drives the
> notebook loop in `docs/WORKFLOW.md`. Numbers re-measured at each notebook's build and reconciled
> into prose.
>
> **Five notebooks** (the standard per-method arc — the fundamentals split cleanly into three
> one-concept notebooks: the hard margin, the soft margin, the kernel trick). The course's **fifth
> method**: the first built on the **maximum-margin** principle, and the home of the **kernel trick** —
> the single idea that turns a linear classifier into a non-linear one without changing the machinery.

## Context

Logistic regression (ch 03) drew **one straight line** by minimizing log-loss — but once the classes
are separable, *many* lines separate them perfectly and all score about the same loss. Logistic
regression never asks **which separating line is best placed**. Decision trees (ch 04) went non-linear
with axis-aligned boxes, but at the cost of high variance. Support vector machines answer the question
logistic regression left open, with a new and geometric principle:

1. **The maximum margin — the first margin-based model.** Among all boundaries that separate the
   classes, the SVM picks the one with the **widest margin**: the boundary that keeps the largest
   empty "street" between the two classes. The few points that touch the edges of the street — the
   **support vectors** — are the only ones that matter; move any other point and the boundary does not
   change. Widening the margin is a robustness/generalization argument (Vapnik's structural risk
   minimization), and it gives a *unique* boundary where logistic regression had a whole family.
2. **The soft margin — honesty about overlap.** Real classes overlap, so a perfectly separating street
   rarely exists. The soft margin lets points sit inside the street or on the wrong side (**slack**)
   and charges a cost **C** for each violation. `C` is the chapter's first dial: a small `C` buys a
   wide, forgiving street; a large `C` insists on a narrow, strict one. (The penalty it charges *is*
   the **hinge loss** — the close cousin of ch 03's log-loss.)
3. **The kernel trick — the chapter's showpiece.** A straight margin cannot separate curved data
   (think concentric rings). Lift the points into a higher-dimensional space where they *are* linearly
   separable, find the max-margin plane there, and the boundary back in the original space is curved.
   The trick: the whole optimization only ever needs **inner products between points**, so we can
   compute the inner product in the lifted space directly — the **kernel** — and never build the space.
   One substitution (dot product → kernel) turns the linear SVM into a non-linear one. This idea
   (kernel methods) reaches far beyond SVMs.

Two properties tie the chapter to what came before and after. SVMs are **distance-based** (the RBF
kernel is a function of `‖x − x′‖`), so they are **scale-sensitive** — the ch 01 KNN scale trap
returns, and standardization is mandatory (the NB 5 headline). And the chapter's honest spine is the
**cost of the kernel**: a kernel SVM does not scale to large `n` (training is super-linear, measured
here), and it returns a **signed distance**, not a calibrated probability.

## Prerequisites (re-established briefly; **first-contacts flagged, not mislabelled as recaps**)

- **Module 00 (genuine recaps):** the train/test split & leakage (NB 04); accuracy + a baseline
  (NB 06); the confusion matrix, precision/recall (NB 07) — NB 5 watches **recall on malignant** under
  imbalance; **the signed-distance score & sliding a threshold** (NB 08) — the SVM's
  `decision_function` is exactly such a score; **over-/under-fitting and the generalization gap**
  (NB 09) — the bias–variance picture returns in NB 4, controlled by **`C` and `gamma`**;
  **cross-validation** (NB 10) — chooses `C`/`gamma`/kernel; **preprocessing, the `Pipeline`,
  fit-on-train-only** (NB 11) — load-bearing in NB 5 (the scaling headline), unlike trees (ch 04).
- **Chapter 01 (KNN):** the **scale trap** (NB 2) — SVMs are distance-based like KNN, so the trap
  applies again (NB 5); **`make_moons`** is reused as the non-linear playground (NB 3–4); the
  **breast_cancer** demanding-case spine (NB 5).
- **Chapter 02 / 03 — calibration (named before, *built* here):** the reliability diagram, the Brier
  score, and **Platt (sigmoid) scaling** were *named* in ch 02 NB 4–5 and ch 03 NB 6 (in prose);
  NB 4 **builds** the `CalibratedClassifierCV` idiom for the first time, now on an SVM score.
- **Chapter 03 (Logistic Regression) — genuine recaps:** the **linear decision boundary**
  `z = w·x + b` and that **`w` ⟂ the boundary** with `‖w‖` setting its steepness (NB 1 reuses both —
  the SVM boundary is the *same* `w·x + b = 0`, now chosen by the margin); **log-loss** (NB 3) — the
  foil for the hinge loss (NB 2); the linear model **underfits curved truth** (the motivation for
  kernels, NB 3); **softmax / OvR multiclass** (NB 5) — the contrast for SVM's one-vs-one; **CV-on-train
  + one sealed test**; the breast_cancer **LogReg baseline** (test 0.953) that NB 5 measures against.
- **Chapter 04 (Decision Trees):** the **breast_cancer cross-method spine** (KNN → LogReg → tree →
  **now SVM**); the **complexity dial / bias–variance** language, now realized as `C` and `gamma`;
  trees as the deliberate **scale-invariant counter-example** to the scale trap SVMs fall into.
- **Re-established from scratch (assumed from nobody) — the FIRST CONTACTS, each budgeted as a
  taught-from-scratch concept, never called a recap:**
  - **the margin** — the width of the empty "street" between the classes; **maximizing** it — **NB 1**;
  - **support vectors** — the few boundary-touching points that alone determine the solution — **NB 1**;
  - **the margin width = `2 / ‖w‖`** (so maximizing the margin = minimizing `‖w‖`) — **NB 1**;
  - **slack variables / the soft margin** and the **cost `C`** (penalty per violation) — **NB 2**;
  - **the hinge loss** `max(0, 1 − y·f(x))` and its tie to ch 03's log-loss — **NB 2**;
  - **the kernel trick** — a feature map `φ`, and a **kernel** `K(x,x′) = φ(x)·φ(x′)` computed without
    building `φ`; the optimization needs only inner products (the dual intuition, kept light) — **NB 3**;
  - **the RBF (Gaussian) kernel** `exp(−γ‖x−x′‖²)` and **`gamma`** (its reach); the **polynomial**
    kernel — **NB 3 / NB 4**;
  - **`decision_function`** (the signed-distance score) vs a calibrated probability, and how to get a
    probability (**`CalibratedClassifierCV`**, not the deprecated `probability=True`) — **NB 4 / NB 5**;
  - **multiclass one-vs-one** (`n(n−1)/2` classifiers; contrasted with ch 03's softmax/OvR) — **NB 4**.

## Verified sklearn 1.9.0 API facts (measured on the live install — pinned, not quoted)

- **`SVC(probability=True)` is DEPRECATED in 1.9, removed in 1.11.** The exact `FutureWarning`:
  *"Use `CalibratedClassifierCV(SVC(), ensemble=False)` instead of `SVC(probability=True)`."* The
  course teaches the **`CalibratedClassifierCV` / `FrozenEstimator` Platt-scaling pattern from ch 03
  NB 6** (verified to work and return `predict_proba` on 1.9.0) — never `probability=True`.
- **`gamma='scale'` (the default) = `1 / (n_features · X.var())`** — verified exactly. A concrete reason
  to **standardize**: the RBF reach is tied to the features' spread.
- **Multiclass `SVC` is one-vs-one**: 3 classes → `n(n−1)/2 = 3` pairwise classifiers;
  `decision_function_shape='ovr'` (default) aggregates to shape `(n, 3)`. Contrast ch 03's softmax/OvR.

## Datasets (measured at plan time, sklearn **1.9.0**; seeds fixed; re-measured at build)

- **NB 1 — a separable synthetic 2-blob set** (`make_blobs(n_samples=40, centers=[[-2.2,-2.2],
  [2.2,2.2]], cluster_std=0.7, random_state=0)`, standardized). The hard margin **only exists when the
  data is separable**; the binary penguins subset is measurably **not** perfectly separable (a
  hard-margin `C=1e6` leaves 1 error), so the first notebook uses a cleanly separable set small enough
  to reason about by hand (the course already uses synthetic `make_moons`/`make_circles` for its
  non-linear notebooks).
  - hard-margin `SVC(kernel="linear", C=1e6)`: **train acc 1.000**, **‖w‖ ≈ 1.16**, **margin = 2/‖w‖ ≈
    1.72**, exactly **2 support vectors** (one per class) — the street is pinned by two points.
  - **Real-data reference (the bridge to NB 2):** penguins Adélie vs Gentoo is **near-separable** — a
    hard margin still leaves **1 stubborn misclassified point** (train 0.9964). Hard margin alone cannot
    cope → the soft margin (NB 2).
- **NB 2 — Palmer penguins**, the binary 2-feature subset (`load_penguins`: Adélie / Gentoo,
  `bill_length_mm`, `flipper_length_mm`, standardized), the course fil-rouge, now real and
  **near-separable** (so a soft margin is *needed*, not optional). Linear `SVC`, sweep `C`:
  | `C` | margin `2/‖w‖` | # support vectors | train acc | CV |
  |----|----|----|----|----|
  | 0.01 | 2.28 | 124 | 0.996 | 0.989 |
  | 0.1 | 1.25 | 44 | 0.993 | 0.993 |
  | 1 | 0.65 | 17 | 0.993 | 0.993 |
  | 10 | 0.46 | 8 | 0.993 | 0.993 |
  | 100–1000 | 0.35 | 6 | 0.996 | 0.996 |
  As `C` grows the **street narrows** (2.28 → 0.35) and the **support-vector count collapses**
  (124 → 6). **Honest framing:** on near-separable data the *accuracy* barely moves — `C` chooses the
  **geometry** (margin width, which points are support vectors, how many violations are tolerated), not
  accuracy. The accuracy consequence of `C` bites hardest with a kernel and noisier data (NB 4). *(An
  overlapping synthetic blob set is held in reserve if a sharper accuracy-vs-`C` contrast is wanted; it
  needs `cluster_std ≈ 2` to actually swing accuracy (`1.35` stays ~flat) — so the penguins geometry
  story is the default.)*
- **NB 3 — `make_circles(n_samples=300, factor=0.4, noise=0.10, random_state=0)`** (the hero) and
  **`make_moons(n_samples=300, noise=0.20, random_state=0)`** (the second example), standardized.
  *(`make_circles` is **new course vocabulary** — used inline like `make_moons`, seed pinned; flagged
  as a first-contact loader, not silently introduced.)*
  - circles: **linear `SVC` CV 0.557** (chance — a line is hopeless on concentric rings) vs **RBF
    `SVC` CV 0.997**.
  - the explicit lift: add the feature **`r² = x₁² + x₂²`** → the rings become linearly separable in
    3-D (`LinearSVC` train **1.000**); a flat plane in the lifted space is a circle back in 2-D.
  - the RBF kernel reaches the same curved boundary **in 2-D** (CV 0.997) without ever forming `r²`;
    a **polynomial** kernel at `degree=2` (CV 1.000) is shown as a second family, with the honest
    degree-matters catch (default `degree=3` → CV 0.613, `degree=4` → 0.997: the degree must match the
    radial geometry — a real kernel-choice lesson, not a simplification).
  - moons: linear CV **0.840** vs RBF **0.970** (a curved-but-not-radial second case).
- **NB 4 — `make_moons(n_samples=300, noise=0.30, random_state=0)`** (the over/underfitting set from
  ch 00 NB 09 / ch 04 NB 3), standardized inside a `Pipeline`; **3-species `load_penguins_full`** (2
  numeric features, standardized) for the OvO demo. The `C × gamma` CV grid (RBF):
  | `C` \ `gamma` | 0.01 | 0.1 | 1 | 10 |
  |----|----|----|----|----|
  | 0.1 | 0.830 | 0.827 | 0.903 | 0.890 |
  | 1 | 0.827 | 0.833 | 0.930 | 0.937 |
  | 10 | 0.823 | 0.877 | 0.933 | 0.930 |
  | 100 | 0.827 | 0.927 | 0.937 | 0.910 |
  Low `gamma` underfits (a flat ~0.83 wall); the best region is `C` 1–100 with `gamma` ≈ 1; very high
  `gamma`+`C` overfits (0.937 → 0.910). **`gamma` as the bias–variance dial**, by support-vector count
  at `C=1`: `gamma` 0.01 → **167 SVs** (smooth/underfit), 1 → **88** (good), 10 → **163 SVs**
  (wiggly/overfit) — a U in complexity, train acc climbing 0.833 → 0.950. OvO 3-class RBF **CV 0.956**.
- **NB 5 — breast cancer** (`datasets.load_breast_cancer`, 569 × 30, **malignant = 1 = positive**,
  37.3 %), reused from ch 01 / 03 / 04 — the cross-method spine, now met by an SVM. Pinned like the
  earlier capstones: 70/30 stratified split (seed 0), `StratifiedKFold(5, shuffle=True,
  random_state=0)` on train.
  - **Scaling headline (course_map §05's "scaling matters"):** RBF `SVC` **raw CV 0.910 →
    standardized CV 0.965** — a distance-based method, so the ch 01 scale trap is back; standardization
    is not optional.
  - **CV-tune (GridSearchCV over `C`/`gamma`/kernel on train) → one sealed test:** best
    `{C=100, gamma=0.001, kernel="rbf"}`, **CV 0.982**, **test 0.965**, **42 support vectors of 398**.
  - **Cross-method TEST spine (same pinned split):** **KNN(k=5) 0.942 → DecisionTree 0.906 → LogReg
    0.953 → SVM 0.965** — the margin method is the strongest *here* (stated honestly: on this dataset
    and split, not universally).
  - **Confusion (tuned SVM, test):** `[[104, 3], [3, 61]]` — **3 of 64 cancers missed** (recall
    **0.953**), 3 false alarms; fewer of both than the ch 04 tree (`[[95,12],[4,60]]`).
  - **Honest limit on large data (course_map §05), MEASURED not quoted:** kernel `SVC` `.fit` time vs
    `n` on synthetic 20-feature data — `n` 500 → 16 000 grows the fit from **0.003 s → 0.62 s** (about
    **n^1.6** empirically here; worst-case `O(n³)`), while `LinearSVC` stays ~flat (0.001 → 0.008 s) —
    the kernel-vs-linear time ratio explodes **4.5× → 81×**. Kernel SVMs do not scale; `LinearSVC` /
    `SGDClassifier(loss="hinge")` are the large-`n` alternatives.
- All datasets offline (sklearn-bundled / generated); colours from `ml_course.colors`; "Read the
  figure" after every figure; a **"Your turn" (2–3 tiered)** in every notebook; every notebook closes
  on the charter arc (**Your turn → What you built → optional Going further → References**); a running
  SVM vocabulary box; **NB 5 closes by framing the kernel SVM's scaling / probability limits as a
  sign-post toward the ensembles (ch 06+) for large tabular data**, never as a wall.

## Primordial concepts → notebooks 1–3 (one concept each; by hand before any library)

| NB | Title | The one concept | Done by hand | Key figure(s) → "Read the figure" (one per figure) | Your turn (tiered sketch) |
|----|-------|-----------------|--------------|----------------------------------------------------|---------------------------|
| 1 | The widest street: **the maximum margin** | **Max-margin & support vectors.** Among all lines that separate two classes, the SVM picks the one with the **widest margin** (the empty "street"). The few points touching the street — the **support vectors** — alone define the boundary; the margin width is **`2/‖w‖`**, so widest margin = smallest `‖w‖` | on a small **separable** set: draw several separating lines and **measure each one's margin** (distance to the nearest point) → the widest wins; identify the **2 support vectors** by eye; compute `margin = 2/‖w‖`; fit `SVC(kernel="linear", C=1e6)` and confirm it finds the **same** street (‖w‖≈1.16, margin≈1.72, 2 SVs) | (A) several candidate separating lines, each with its margin shaded — the widest is the SVM's; (B) the max-margin boundary with the **street** (the `±1` contours) and the **support vectors ringed**; (C) **delete a far point → boundary unchanged; move a support vector → boundary shifts** — *"the boundary sits as far as possible from both classes, and listens only to the ringed support vectors"* | easy — given two lines, say which has the wider margin; medium — from a plot, name the support vectors; harder — from figure C, explain why moving a non-touching point leaves the street where it is (argue from the shown invariance, not the dual) |
| 2 | When the data overlaps: **the soft margin & the cost `C`** | **Slack, the cost `C`, and the hinge loss.** Real classes overlap, so allow violations (**slack**: points inside the street or on the wrong side) and pay a cost **`C`** per violation — a penalty that *is* the **hinge loss** `max(0, 1 − y·f(x))`. `C` is the dial: **small `C`** → wide, forgiving street (many support vectors, more bias); **large `C`** → narrow, strict street (few SVs, toward the hard margin) | on **penguins** (real, near-separable — a hard margin is infeasible, one stubborn point): sweep `C` and **measure** the margin width (2.28 → 0.35) and the support-vector count (124 → 6); compute the **hinge loss** `max(0, 1 − y·f(x))` for a few points by hand, using the **`y ∈ {−1,+1}`** convention the formula needs (clear → 0, in-street → ~0.80, wrong-side → ~2.40) | (A) **hinge loss vs `y·f(x)`, overlaid with ch 03's log-loss** — the same "confident-and-wrong is expensive" shape, drawn two ways (hinge has a flat safe zone past the margin, log-loss never reaches zero); (B) the same data at **small `C` vs large `C`** — wide vs narrow street, violations marked; (C) **margin width and #support-vectors vs `C`** — *"raising `C` narrows the street and leans on fewer points; on near-separable data the accuracy hardly moves — `C` is choosing the shape"* | easy — predict whether small or large `C` gives the wider street; medium — count the violations at a given `C` from a plot; harder — compute the hinge loss for three given points and relate it to their position vs the street |
| 3 | Curved boundaries: **the kernel trick** | **Lift, then kernelize.** A straight margin cannot separate curved data (concentric rings). **Lift** the points to a space where they *are* linearly separable (add `r² = x₁²+x₂²`); the max-margin plane there is a **curved** boundary back in 2-D. The **kernel** computes the inner product in the lifted space **without building it** (RBF `K=exp(−γ‖x−x′‖²)`, `gamma='scale'`=`1/(n_feat·var)`) — one substitution turns the linear SVM non-linear | on **circles**: show a linear SVM fails (CV 0.557); **add `r²` by hand** → linearly separable in 3-D (a plane; train 1.000); then an **RBF `SVC`** draws the same circular boundary in 2-D (CV 0.997) **without** forming `r²`; show a **polynomial** kernel at **`degree=2`** (CV 1.000 — the rings are a degree-2 form) as a compact second family, with the honest catch that the kernel must **match the geometry** (the default `degree=3` nearly fails, CV 0.613; `degree=4` recovers, 0.997); note the optimization needs only **dot products** (the dual, kept light) | (A) circles in 2-D (no line works) → the **3-D lift** with a flat separating plane; (B) the **RBF and degree-2 polynomial** decision boundaries in 2-D, with the failing degree-3 poly as the honest contrast — *"in the lifted space a flat plane separates the rings; that plane, seen back in 2-D, is the closed curve the kernel SVM draws — and the kernel got us there without computing the third coordinate; but the poly's degree must match the rings, or it fails"* | easy — explain why no straight line separates the rings; medium — compute `r²` for three given points and check they become separable; harder — fit the polynomial kernel at degree 2 vs degree 3 and explain why the degree must match the rings' geometry |

## Notebook 4 — the estimator & its parameters (~24 cells; the integrative notebook)

`sklearn.svm.SVC` for real (sklearn **1.9.0**). **Parity first:** the by-hand max-margin line (NB 1) ==
`SVC(kernel="linear", C=1e6)` (same `w`, `b`, support vectors within tolerance). Then walk the knobs —
**four things shown, two named** — keeping a **soft ~24-cell ceiling** so the headline (the `C × gamma`
bias–variance map) gets room (the ch-04 NB-4 discipline):
- **Shown:**
  - **`C` and `gamma` jointly — the `C × gamma` CV heatmap** (the classic SVM picture, the bias–variance
    map) + a row of boundaries from **underfit** (low `gamma`) → **good** → **overfit** (high `gamma`),
    with the **support-vector count** as the complexity read-out (167 → 88 → 163). `gamma` is the new
    dial (large = local = wiggly = variance; small = smooth = bias); `gamma='scale'`=`1/(n_feat·X.var())`
    is the default, and *why it depends on feature spread* is another reason to standardize.
  - **`kernel`** (`linear` / `rbf` / `poly`) — the boundary family (poly's `degree`/`coef0` named).
  - **Multiclass = one-vs-one** — on 3-species `load_penguins_full`, `SVC` trains `n(n−1)/2 = 3`
    pairwise classifiers (CV 0.956); the explicit contrast with ch 03's softmax/OvR and ch 04's native
    multiclass.
  - **`decision_function` → calibrated probability** — `SVC` natively outputs a **signed-distance
    score** (`predict` is its sign), **not** a probability. To get one, wrap it with
    **`CalibratedClassifierCV` (Platt / sigmoid scaling)** — the calibration approach *named* in
    ch 02 NB 4 / ch 03 NB 6, now **built for the first time** on an SVM score (a single raw-vs-calibrated
    Brier figure via `plot_calibration_curve`). **Pin the deprecation:** `SVC(probability=True)` is
    deprecated in 1.9 / removed in 1.11; the message itself prescribes `CalibratedClassifierCV(SVC(),
    ensemble=False)`. (The probability is *used* to set a threshold in NB 5.)
- **Named, one line each (no figure):** **`LinearSVC` / `SGDClassifier(loss="hinge")`** — the linear
  SVM that scales far better than `SVC(kernel="linear")` (the sign-post to NB 5's large-`n` limit) —
  and **`class_weight`** (imbalance, used lightly in NB 5).
- **Honest tuning:** `GridSearchCV` over `C` / `gamma` / `kernel` on TRAIN; one sealed test — the
  module-00 NB 10 discipline.
- **Build-note (the ~24-cell ceiling):** protect the `C × gamma` map + boundary row as the headline;
  keep `kernel` and OvO compact (one figure / short section each); render calibration as a *single*
  raw-vs-calibrated figure (`plot_calibration_curve`), not a second reliability study; OvO (one
  measured contrast, CV 0.956) is the most compressible beat if the count runs over.
- **Your turn:** turn `gamma` and read the boundary (smooth ↔ wiggly); find the under/overfit corners
  of the `C × gamma` grid; confirm the linear `SVC` == NB 1's by-hand line.

## Notebook 5 — the demanding practical case: **breast cancer (scaling, model selection, honest limits)**

A kernel SVM on **breast_cancer** (569 × 30) — the full honest workflow, **visualization-first**
(capstone; ~20 cells is a floor not a ceiling), where the chapter's strength (a strong, well-tuned
non-linear classifier) and its honest limits (scaling, no probabilities, no importances, no scaling to
large `n`) both bite:
- **Look → scale → tune → sealed test:** class balance; the **scaling headline** first — RBF `SVC`
  **raw CV 0.910 vs standardized CV 0.965** inside a `Pipeline` (fit-on-train-only) — the ch 01 scale
  trap, now for SVMs; then `GridSearchCV` over `C`/`gamma`/kernel on train → best
  `{C=100, gamma=0.001, rbf}` (CV 0.982) → **one held-out test 0.965**. *"The dataset KNN felt the
  curse on, logistic regression read probabilities from, and a tree turned into rules — now meet it
  with the widest margin."*
- **Cross-method spine — the payoff, stated honestly:** on the same pinned split, **KNN 0.942 →
  tree 0.906 → LogReg 0.953 → SVM 0.965**. The margin method is the strongest *here* (not a universal
  claim); 42 support vectors carry the boundary.
- **Error analysis:** the confusion matrix `[[104,3],[3,61]]` — **3 of 64 cancers missed** (recall
  0.953), 3 false alarms — fewer of both than the ch 04 tree. The `decision_function` score is what a
  threshold slides (ties to ch 00 NB 08 / ch 03 NB 6); since it is **not** a probability, a
  **calibrated** probability (NB 4's `CalibratedClassifierCV`) is what to threshold for the
  malignant-recall trade — shown lightly, not a second full calibration study (the §05 mandate is
  scaling / selection / limits).
- **Honest limits on large data — MEASURED:** time `SVC.fit` vs `n` (the kernel SVM's training is
  ~`n^1.6` here, worst-case `O(n³)`) against `LinearSVC` (~linear) — the ratio explodes 4.5× → 81× as
  `n` grows. Kernel SVMs do not scale to large `n`; `LinearSVC` / `SGDClassifier` are the alternatives.
  Plus: **no native feature importance** (unlike trees) and **no probability by default** — the
  interpretability / probability costs, stated.
- **Bridge forward:** SVMs are a strong off-the-shelf choice for **small-to-medium, well-scaled** data
  with a clear margin; for **large tabular data, feature importance, and built-in probabilities**, the
  **ensembles** (ch 06+) often win — the next chapters.
- **Figures (visualization-first):** (A) class balance; (B) **raw-vs-standardized accuracy bar** (the
  scaling headline); (C) the **`C × gamma` CV heatmap** on breast_cancer (model selection); (D) the
  **cross-method test spine** bar (KNN / tree / LogReg / SVM); (E) the **confusion matrix**; (F) the
  **measured fit-time-vs-`n`** curve (kernel `SVC` vs `LinearSVC`, log–log). Each + "Read the figure".
- **Your turn:** standardize vs not and measure the gap; read the `C × gamma` heatmap and pick a cell;
  (harder) subsample and time `SVC` vs `LinearSVC` as `n` grows, and say where you'd switch.

**Five notebooks** — the standard per-method arc. No 6th is needed: the deep math (the Lagrangian
**dual**, why the optimum depends only on dot products) is taught as **intuition** in NB 3 and offered
as an optional **"Going further"**, not a separate notebook — the right altitude for this audience.
*(An optional advanced 6th — the dual & KKT properly, `NuSVC`/ν-SVM, or SVR for regression — is a
clean add if Rémy wants it, but not proposed by default.)*

## Library additions (decided per-notebook, with tests)

- **Proposed (NB 1; reused NB 2, NB 3, NB 4): `viz.plot_svm_decision(model, X, y, *, ax=None,
  resolution=300)`** — the SVM signature picture: fill the decision regions (via `predict`, like
  `plot_decision_boundary`), overlay the **street** as `decision_function` contours at **−1 / 0 / +1**
  (0 solid = boundary, ±1 dashed = margins), and **ring the support vectors** (`model.support_`).
  Appears in **four** notebooks → clears the "reused ≥ 2×" bar. **Charter watch:** all colours from
  `ml_course.colors` (no hardcoded hex) — region fill via `CLASS_CYCLE`, contours via
  `COLORS["text"]/["muted"]`, SV rings as an open marker edged in `COLORS["highlight"]`. Add a **smoke
  test** (`pytest` 17 → 18). Decided at NB 1 build.
- **Reused as-is from `ml_course.viz`:** `use_course_style`, `plot_decision_boundary` (where the street
  is not needed), `plot_class_balance` (NB 5), `plot_confusion_matrix` (NB 5),
  `plot_calibration_curve` (NB 4 / NB 5), `plot_feature_histograms`. The `C × gamma` heatmap, the
  hinge-vs-log-loss curve, the cross-method bar, and the fit-time curve are one-off in-notebook figures
  in charter colours (`CMAP_PROBA` for the heatmap).
- **No new dataset** — `load_penguins`, `load_penguins_full`, `load_breast_cancer` exist;
  `make_blobs`/`make_circles`/`make_moons` come from sklearn (used inline, seeds pinned).

## Honest limits stated in the notebooks

- **SVMs are scale-sensitive** (distance-based; the RBF kernel is a function of `‖x−x′‖`, and
  `gamma='scale'` depends on feature variance) — standardization is mandatory (the NB 5 headline; the
  ch 01 scale trap returns). A genuine contrast with trees (ch 04), which needed none.
- **Kernel SVMs do not scale to large `n`** — training is super-linear (worst-case `O(n²–n³)`, LIBSVM;
  ~`n^1.6` *on this data/C*, with the kernel-vs-`LinearSVC` time ratio exploding 4.6× → 82×);
  `LinearSVC` / `SGDClassifier` are the large-`n` alternatives. Measured, not quoted.
- **`SVC` returns a signed distance, not a probability** — `decision_function` ranks; a probability
  needs **`CalibratedClassifierCV` (Platt / sigmoid)**, the ch 03 NB 6 pattern. **`probability=True`
  is deprecated** (1.9; removed 1.11). Contrast ch 02 / ch 03 calibration.
- **No native feature importance** (unlike trees) — a kernel SVM is comparatively opaque; the
  interpretability cost is stated.
- **`C` and `gamma` must be tuned jointly by CV** — defaults are not always good; the wrong `gamma`
  overfits badly (NB 4 shows it). Never tune on test.
- **The hard margin needs separable data** — hence the soft margin for real data (NB 2); the dual /
  hinge-loss derivation is kept at intuition level (Going further), not oversold.

## Resolved decisions / open for Rémy

- **NB 1 dataset = a separable synthetic 2-blob set** (not penguins) — *resolved by measurement*: the
  binary penguins subset is **not** perfectly linearly separable (hard margin leaves 1 error), and the
  hard margin is only coherent on separable data. Penguins returns in NB 2 as the near-separable real
  set that *motivates* the soft margin. (The course already uses synthetic sets for its non-linear
  notebooks, so this is consistent.) **Open for Rémy** if he prefers an alternative separable set.
- **`make_circles` is new course vocabulary** — used **inline** like `make_moons` (seed pinned), not as
  a new `datasets` wrapper. Flagged here so it is a deliberate first-contact, not silent.
- **NB 5 = breast_cancer + a synthetic fit-time ramp** — breast_cancer (n=569) keeps the cross-method
  spine intact, but the `O(n²–n³)` wall cannot be *felt* at that size, so the scaling figure uses a
  synthetic `make_classification` ramp (measured). The same spine-vs-scale split Rémy approved for the
  ch 04 capstone.
- **Five notebooks, no 6th** (the arc fits; an optional advanced 6th is offered above, not default).
- **Calibration realized in NB 4, used in NB 5** — to honour `decision_function`-vs-probability
  (c20) without overloading NB 5's scaling/selection mandate.

## References (with DOI where they resolve; **dereferenced & pinned by the ml-expert reviewer at build**)

- Cortes C, Vapnik V (1995). *Support-vector networks.* Machine Learning 20:273–297.
  DOI: 10.1007/BF00994018 — the soft-margin SVM.
- Boser BE, Guyon IM, Vapnik VN (1992). *A training algorithm for optimal margin classifiers.* COLT.
  DOI: 10.1145/130385.130401 — the kernel trick.
- Vapnik V (1995). *The Nature of Statistical Learning Theory.* Springer.
  DOI: 10.1007/978-1-4757-2440-0 — structural risk minimization / max-margin generalization.
- Platt J (1999). *Probabilistic outputs for support vector machines.* Adv. in Large Margin
  Classifiers, MIT Press — Platt (sigmoid) scaling (`CalibratedClassifierCV(method="sigmoid")`).
- Schölkopf B, Smola A (2002). *Learning with Kernels.* MIT Press — kernels, RBF, Mercer's condition
  (the NB 3 "Going further").
- Chang C-C, Lin C-J (2011). *LIBSVM: a library for support vector machines.* ACM TIST 2(3):27.
  DOI: 10.1145/1961189.1961199 — the solver behind sklearn `SVC`; the OvO scheme and the
  `O(n²–n³)` training-complexity claim.
- Fan R-E, Chang K-W, Hsieh C-J, Wang X-R, Lin C-J (2008). *LIBLINEAR: a library for large linear
  classification.* JMLR 9:1871–1874 — the solver behind `LinearSVC` (the large-`n` alternative).
- Hastie T, Tibshirani R, Friedman J (2009). *The Elements of Statistical Learning*, §12 (SVMs).
  DOI: 10.1007/978-0-387-84858-7.
- James G, Witten D, Hastie T, Tibshirani R (2021). *An Introduction to Statistical Learning*, §9
  (SVMs). DOI: 10.1007/978-1-0716-1418-1.

## Verification (per notebook, at its commit)

Every number re-run and reconciled into prose at build on sklearn **1.9.0** (NB 1: separable blobs
hard-margin train 1.0 / ‖w‖≈1.16 / margin≈1.72 / 2 SVs, penguins near-separable hard-margin 0.9964 =
1 error; NB 2: penguins `C`-sweep margin 2.28→0.35, SVs 124→6, accuracy ~flat, hinge-vs-log-loss; NB 3:
circles linear CV 0.557 vs RBF 0.997, `r²` lift train 1.000, moons linear 0.840 vs RBF 0.970, poly
shown; NB 4: `C×gamma` CV grid as tabulated, n_SV 167→88→163, OvO 3-class CV 0.956, `gamma='scale'`
formula, the `CalibratedClassifierCV` calibration + the `probability=True` deprecation pin; NB 5:
scaling raw 0.910 vs std 0.965, GridSearch best `{C100,γ0.001,rbf}` CV 0.982 / test 0.965, spine KNN
0.942 / tree 0.906 / LogReg 0.953 / SVM 0.965, confusion `[[104,3],[3,61]]` recall 0.953, fit-time
~`n^1.6` vs LinearSVC). Runs top-to-bottom (nbconvert to scratchpad; tracked file **output-free**,
`--clear-output --inplace`); **banned-word scan over the JSON real text** (`just / simply / obviously /
trivially` + FR) = 0; `check_no_hardcoded_hex` passes; `gen_llms_txt` re-run; `pytest` green (17, → 18
when `viz.plot_svm_decision` lands with its test); `ruff` / `black` clean; `course_map.md` §05 aligned
to the final titles; **`common_errors.md` extended** per new SVM trap ("SVMs don't need scaling" —
false; "the `decision_function` value is a probability" — false; "larger `gamma` is always better" —
false; "`SVC(probability=True)` gives calibrated probabilities" — deprecated + not free; "any
polynomial kernel fits a curved boundary" — false, the `degree` must match the geometry; "hinge loss
with `{0,1}` labels" — wrong, hinge needs `y ∈ {−1,+1}`). Both
reviewers pass (no BLOCK) on each notebook; Rémy validates visually; commit per notebook; **chapter
close via PR into `main`** (protected).

## Per-notebook status (update as the chapter is built)

| NB | Branch | Status |
|----|--------|--------|
| 1 | `notebook/05_SVM__01_maximum_margin` | **done** — built (22 cells, 4 figures: candidate-lines/margins, the max-margin street via the new `viz.plot_svm_decision`, support-vector invariance, the LogReg contrast), both reviewers folded (pedagogy PASS; ml-expert REVISE → 1 MAJOR fixed: the closest-pair/perpendicular-bisector recipe reframed as a special case with the **convex-hull** scope caveat + MINORs), Rémy validated visually, merged to `chapter/05_SVM`. `src/` added `viz.plot_svm_decision` + 2 tests (pytest 19). |
| 2 | `notebook/05_SVM__02_soft_margin_C` | **done** — built (22 cells, 3 figures: hinge-vs-log-loss, small-`C` vs large-`C` street, margin & #SV vs `C`), both reviewers folded (pedagogy PASS; ml-expert REVISE → 1 MAJOR fixed: "support vector = pays slack" corrected to SV = `m ≤ 1`, on-edge SVs pay zero slack, 17 SVs vs 15 slack-payers + reconnected to NB 1 + MINORs), Rémy validated visually, merged to `chapter/05_SVM`. No `src/` change (pytest 19). |
| 3 | `notebook/05_SVM__03_kernel_trick` | **done** — built (21 cells, 4 figures: 2-D→3-D `r²` lift with a separating plane, the RBF circular boundary, poly degree-2 vs degree-3, RBF on moons), **both reviewers PASS** (no BLOCK/MAJOR; 3 MINOR polish folded — poly default `coef0=0` named as why odd degrees miss the radial form, RBF default `gamma='scale'` noted, `make_circles` flagged as new vocab), Rémy validated visually, merged to `chapter/05_SVM`. No `src/` change (pytest 19). |
| 4 | `notebook/05_SVM__04_estimator_and_parameters` | **done** — built (21 cells ≤24 ceiling, 4 figures: `C × gamma` heatmap, gamma boundary grid, OvO 3-class regions, calibration reliability), both reviewers folded (pedagogy PASS, cell budget exemplary; ml-expert REVISE → 1 MAJOR fixed: calibration switched from in-sample `FrozenEstimator` to leak-free `CalibratedClassifierCV(SVC(), ensemble=False)` matching the printed deprecation idiom, Brier 0.106→0.072 + MINOR decision_function `(5,3)`), Rémy validated visually, merged to `chapter/05_SVM`. No `src/` change (pytest 19). |
| 5 | `notebook/05_SVM__05_breast_cancer_scaling_limits` | planned |

## Reviewer notes (chapter-plan gate — both reviewers passed, no BLOCK; all folded)

- **ML expert — REVISE (no BLOCK; 21/22 headline anchors reproduced exactly on sklearn 1.9.0):**
  MAJOR — the NB 3 polynomial kernel was framed as a working "second family" but the **default
  `degree=3` poly nearly fails on the rings** (CV 0.613, barely above linear's 0.557) — only `degree=2`
  separates them (CV 1.000), `degree=4` recovers (0.997). **Folded:** NB 3 pins **`degree=2`** and turns
  the default-degree failure into the deliberate teaching beat *"the kernel's degree must match the
  geometry"* (poly kept subordinate to the kernel-trick concept; movable to NB 4 if the cell budget is
  tight). MINORs folded: (1) calibration "reused verbatim from ch 03 NB 6" overstated provenance
  (ch 03 only *named* `CalibratedClassifierCV` in prose) → reworded "named before, **built** here";
  (2) the NB 2 by-hand hinge needs the **`y ∈ {−1,+1}`** convention stated (the course encodes
  `{0,1}`) → added, with a `common_errors` row; (3) keep the fit-time **worst-case `O(n²–n³)`** primary,
  the measured `n^1.6` flagged *this-data/C* and tied to the LinearSVC ratio; (4) the reserve
  overlap-blobs need `cluster_std ≈ 2` to swing accuracy (`1.35` is flat). Praised: the max-margin
  `2/‖w‖` re-derived and closing by hand; the honest C-geometry framing; the fair, no-leakage NB 5
  spine; the complete honest-limits section; load-bearing DOIs.
- **Pedagogy — PASS (no BLOCK, no MAJOR):** 3 MINOR build-time watch-flags, all folded into the plan:
  (1) keep the NB 3 polynomial kernel **subordinate** to the kernel-trick concept (one figure + one
  read) — converges with the ML MAJOR; (2) the NB 4 **~24-cell ceiling** is tight (the `C × gamma` map
  and the calibration beat are both headline-weight) → added a build-note (protect the map; compact
  `kernel`/OvO; calibration as a single figure; OvO most compressible); (3) the NB 1 "harder" exercise
  should argue from the **shown figure-C invariance**, not the dual met later → reworded. Praised:
  exemplary first-contact-vs-recap fencing (every recap tied to its prior NB id); the clean NB 1→2→3
  carving (each opening from the prior's stated limit); the measured NB-1 dataset decision; the genuine
  honest-limits spine; charter compliance (no banned words in learner prose, no hardcoded hex, the
  `viz.plot_svm_decision` helper names only real palette keys); the visual-first capstone.
