# Chapter plan — 03_LogisticRegression (Logistic Regression)

> Status: **APPROVED** (2026-06-18, by Rémy; reviewer-gated — both reviewers REVISE → all
> BLOCK/MAJOR/MINOR folded, every number re-measured on the pinned sklearn **1.9.0**). **SIX
> notebooks**: Rémy approved splitting NB 3 (log-loss) from NB 4 (gradient descent) — a deliberate
> exception to the 5-ceiling, exactly as KNN earned its 6th. Drives the notebook loop in
> `docs/WORKFLOW.md`.
> The course's **third method** — the first **discriminative** one and the first that **learns by
> iterative optimization**.

## Context

Naive Bayes (ch 02) was **generative**: it modelled how each class *generates* the features
(P(x∣y)), then inverted with Bayes' rule — and we measured the cost of that indirection (over-confident
probabilities). Logistic regression takes the opposite, **discriminative** route promised at the close
of ch 02 (Ng & Jordan 2001): model **P(y∣x) directly**, with a straight line (a linear decision
boundary) squashed through an S-curve into a probability. It is the natural next method on the
progression — KNN voted by distance, NB multiplied probabilities, and logistic regression draws **one
weighted line** and reads a calibrated probability off it.

Two firsts make this chapter pivotal for everything after:
1. **The first model trained by *iterative optimization*.** Nearest-centroid, KNN and NB all had
   closed-form or lazy fits — nothing was *optimized*. Logistic regression has **no closed form**: we
   define a loss (log-loss / cross-entropy — NB 3) and walk **gradient descent** downhill to the weights
   (NB 4). This is the optimizer the whole back half of the course (MLP, neural networks) is built on; it
   earns its own notebook, taught from scratch, by hand, and *watched* as it converges. (Rémy-approved
   6th notebook; the pedagogy reviewer argued — and the ML reviewer agreed it is defensible — that a
   load-bearing first optimizer should not be the back half of the log-loss notebook.)
2. **The first model whose weights are directly *interpretable* — and whose probabilities are genuinely
   *calibrated***. Each weight is a change in **log-odds** per unit of a feature; and where NB was
   over-confident (ch 02 NB 5), logistic regression's probabilities can be read as probabilities
   (measured: Brier **0.027** vs GaussianNB **0.088** on breast cancer). This closes the
   generative-vs-discriminative loop ch 02 opened.

## Prerequisites (re-established briefly; **first-contacts flagged, not mislabelled as recaps**)

- **Module 00 (genuine recaps):** train/test split & leakage (NB 04); accuracy + baseline (NB 06);
  confusion matrix, precision/recall/F1 + the **PR curve** (NB 07–08) and the **score→threshold**
  picture (NB 08) — all load-bearing in NB 6's threshold choice; cross-validation (NB 10);
  **standardization / `Pipeline` / fit-on-train-only** (NB 11) — weights are comparable only on
  standardized features, and gradient descent converges nicely only on them.
- **Chapter 01 (KNN):** the **scale trap** (NB 2) — same lesson, now for reading weight *magnitudes*;
  **breast_cancer** (NB 5) is reused in NB 6 here, a deliberate "revisit with new eyes" (KNN felt the
  curse; logistic regression reads interpretable weights and calibrated probabilities).
- **Chapter 02 (Naive Bayes) — now genuine recaps, no longer first contacts:** **generative vs
  discriminative** (the NB 5 bridge is the entry point here); **calibration, the reliability diagram,
  the Brier score** (taught in full in ch 02 NB 5) — reused, not re-taught; NB's **over-confidence** is
  the foil logistic regression is measured against; the **likelihood** (ch 02 maximized it by counting)
  — NB 3's log-loss is its negative log, the bridge made explicit.
- **Re-established from scratch (assumed from nobody) — the FIRST CONTACTS, each budgeted as a
  taught-from-scratch concept, never called a recap:**
  - the exponential **e** and the **sigmoid / logistic function** σ(z)=1/(1+e⁻ᶻ) — **NB 1** (one line
    on e; the curve built and plotted from scratch);
  - **odds and log-odds** as arithmetic objects: from a probability the learner already trusts, form
    odds p/(1−p), then its log — a small p→odds→log-odds table — *then* identify the score z as the
    **log-odds** — **NB 1** (a genuine first-contact; the learner has only ever *counted* probabilities);
  - the **linear decision boundary** w·x+b=0 and **weights as log-odds contributions** — **NB 2**;
  - **log-loss / cross-entropy** as a *training objective* (= the negative log-likelihood of the
    Bernoulli model — the bridge from ch 02's likelihood) and **why not squared error** (non-convex on
    the sigmoid) — **NB 3**;
  - **the gradient as the direction of steepest increase** (gradient-as-slope, built on a 1-D bowl
    before the 2-parameter surface — the calculus kept light but *introduced*, never assumed) and
    **gradient descent** with a **learning rate** — **NB 4** (the course's first optimizer);
  - **L1 / L2 regularization** as penalties on the weights (ridge shrinks, lasso zeroes) — **NB 5**;
  - **multinomial / softmax** — the sigmoid generalized to >2 classes (its own section + figure, not a
    parameter bullet) — **NB 5**.

## Datasets (measured at plan time, sklearn **1.9.0**; seeds fixed; re-measured at build & reconciled into prose)

- **NB 1–5: Palmer penguins** — the binary 2-feature subset (`load_penguins`/`penguins_xy`: Adélie /
  Gentoo, `bill_length_mm`, `flipper_length_mm`), the course fil-rouge. Measured facts the fundamentals
  rest on:
  - **The sigmoid's sweet spot is `bill_length` (Adélie vs Gentoo), used in RAW mm for an interpretable
    axis.** 1-D logistic regression: **acc 0.947**, the P=½ crossing at **≈43 mm**, with **~16 % of
    points in the transition band** P∈[0.1, 0.9] (raw mm; the ~30 % figure was a standardized-feature
    artefact and is dropped). Read qualitatively: a **genuine S-curve with a real overlap band, not a
    step** — `flipper` is too separable (only ~16 % even standardized, steeper), Adélie/Chinstrap on
    bill_depth/body_mass is ≈ chance (no signal). NB 1's illustrative weights are **chosen by hand / read
    off the ≈43 mm crossing — never fitted** (fitting is NB 3–4; the "we'll learn to find these" promise
    stays honest).
  - **The 2-feature data is near-, but NOT perfectly, separable**, so the MLE is **finite** and gradient
    descent **converges** (no weight blow-up to wreck NB 4): train acc **0.9962** (≈1 misclassified
    point), |coef| **plateaus** (identical at C=1e4 and 1e8). The separation→divergence pathology is
    therefore **taught deliberately in NB 5** (motivating regularization), on a constructed
    perfectly-separable slice — it never accidentally breaks NB 1–4.
  - **NB 2–4 standardize** the two features (recap of ch 01 NB 2 / module-00 NB 11): weight *magnitudes*
    are comparable only when standardized, and gradient descent is well-conditioned only then.
  - **Regularization (NB 5), current sklearn-1.9 API — `C` + `l1_ratio`, NOT `penalty=`:** **L2**
    (`l1_ratio=0`, default, lbfgs) shrinks weights smoothly: **‖w‖₂ over the four standardized features**
    = **0.83 / 1.89 / 3.25 / 6.78** for C = 0.01 / 0.1 / 1 / 100. **L1** (`l1_ratio=1`, solver `saga`) on
    the 4 penguin features keeps **all 4 nonzero for any reasonable C** (every feature is informative; it
    only zeroes *real* features under very strong L1 — 1/4 nonzero at C=0.01). So penguins **cannot**
    cleanly show sparsity → NB 5's L1 demo **injects pure-noise feature columns** into penguins
    (controlled: L1 drives the noise weights to exactly 0; L2 only shrinks them).
  - **Multi-class (NB 5):** on the 3 species (2 features, pinned CV), **multinomial/softmax** (the sklearn
    default, `coef_` shape (3, 2)) CV **0.955** and **one-vs-rest** (`OneVsRestClassifier`) CV **0.952**,
    **disagreeing on 0.0 % of predictions** — data this separable can't separate them on predictions, so
    the difference is taught on **how the probabilities are formed** (softmax normalizes jointly; OvR fits
    3 independent sigmoids then renormalizes) and on the three boundaries — **stated as measured**, never
    "they're the same model".
  - **NB 4 parity:** by-hand gradient descent on **standardized 1-D `bill_length`** converges to
    `LogisticRegression(C=np.inf)` (w≈6.18, b≈−0.51 — exact to the unregularized MLE), **NOT** the default
    `LogisticRegression()` (which is regularized, C=1) — the parity cell must use `C=np.inf` or the
    learner will see a spurious mismatch.
- **NB 6: breast cancer** (`sklearn.datasets.load_breast_cancer`, 569×30, malignant 212 / benign 357 ≈
  37 % positive) — reused from ch 01 NB 5, a new lens. Numbers (pinned: `StratifiedKFold(5,
  shuffle=True, random_state=0)` for CV; 30 % held-out, seed 0; **a single standardized `Pipeline` for
  both estimators** — the fair comparison the ch 02 gate also enforced):
  - LogReg **CV acc 0.979** (> GaussianNB **0.930**) — a ~5-point gap (larger than an earlier mis-measure
    suggested; the lesson is stronger, not weaker).
  - **Calibration — the ch 02 bridge, both under the standardized pipeline:** LogReg **Brier 0.0265** vs
    GaussianNB **Brier 0.0880** (~3× better); pile-up past 0.99/0.01 **123/171** vs **167/171**. Honest
    nuance to state: near-separable data still pushes LogReg fairly confident (123/171), so it is
    *better*-calibrated, **not perfectly**.
  - **Threshold choice — medical asymmetry (malignant = the costly miss):** at the default 0.5, malignant
    recall **0.953** (3 of 64 cancers missed, confusion `[[103,4],[3,61]]`); the threshold to 0.3 lifts
    recall to **0.969** (2 missed, confusion `[[101,6],[2,62]]`) at the cost of more false alarms (4→6).
  - **L1 feature selection** (`l1_ratio=1`, saga, **train-only standardized**, high `max_iter`): keeps
    **3 / 8 / 14** of 30 features at C = 0.02 / 0.2 / 1.0 — lasso as automatic selection, and a chance to
    *read which* features drive malignant. (Counts are `saga`-tolerance-sensitive near the soft-threshold;
    pinned to the NB-6 setup with a fixed `max_iter`.)
- All datasets offline (sklearn-bundled); colours from `ml_course.colors`; "Read the figure" after every
  figure; a **"Your turn" (2–3 tiered)** in every notebook; a running logistic-regression vocabulary box.

## Primordial concepts → notebooks 1–4 (one concept each; by hand before any library)

| NB | Title | The one concept | Done by hand | Key figure(s) → "Read the figure" (one per figure) | Your turn (tiered sketch) |
|----|-------|-----------------|--------------|----------------------------------------------------|---------------------------|
| 1 | From a linear score to a probability | **The sigmoid & log-odds.** A linear score z = w·x + b is any real number; the **logistic function** σ(z)=1/(1+e⁻ᶻ) squashes it into a probability in (0,1); inverted, z **is the log-odds** of the class. Predict by thresholding P at ½ (⇔ z at 0) | one line on **e**; code σ(z) from scratch and plot it; build a **p → odds → log-odds** table from a probability the learner already trusts, then name z = log-odds; apply σ to `bill_length` (raw mm) with **hand-chosen** weights (a labelled **preview** — "NB 3–4 *find* these") and mark the ≈43 mm ½-crossing | **the sigmoid over `bill_length`** with the two species at P=0/1 and the ½-crossing — *"the S-curve turns 'how long is the bill' into 'how probable is Gentoo'; steep where classes overlap, flat where they don't"* | easy — read P off the curve at a given bill length; medium — flip a weight's sign and predict the effect on the curve; harder — from w, b solve for the bill length where P=½ |
| 2 | The decision boundary & reading the weights | **A weighted line, and what its weights mean.** In 2-D, z = w₁x₁+w₂x₂+b; the **decision boundary** is the line z=0 (P=½); **w is perpendicular** to it and sets the sigmoid's steepness; each **wⱼ = change in log-odds per (standardized) unit of feature j** — sign = direction, magnitude = strength | on **standardized** bill+flipper, *set* w, b by hand and watch the boundary rotate (with w) and shift (with b); read the two weights as log-odds contributions; contrast with module-00 nearest-centroid's *unweighted* bisector | **decision boundary** over the cloud with the w arrow, beside a **weight bar** — *"the line is where the model is 50/50; w points toward Gentoo and its length is confidence; the taller bar is the feature that moves the odds more"* | easy — predict which side a given penguin falls; medium — double w₁ and describe the boundary's change; harder — set b so the boundary passes through a named point |
| 3 | Fitting I — what we optimize: **log-loss** | **The objective.** Fitting needs a number that says how wrong the weights are. **Log-loss = cross-entropy = the negative log-likelihood of the Bernoulli model** (the bridge from ch 02's likelihood): it punishes confident-and-wrong hardest. **Why not squared error?** on the sigmoid it is **non-convex** (bumpy — a descent could stall); log-loss is **convex** (one bottom) | for a 1-D fit, write log-loss as a function of the weight; **plot log-loss vs a weight (a convex bowl)** beside **squared-error-on-sigmoid (bumpy)**; compute the loss of a few hand-set weights to feel "less surprised = lower loss" | **log-loss bowl vs squared-error bumps** — *"log-loss has a single bottom, so there is one best weight and a downhill path always reaches it; squared error on the sigmoid has bumps a search could get stuck in — that is why we use log-loss"* | easy — rank three weight settings by loss; medium — explain why a confident wrong prediction costs more than an unsure one; harder — write the loss of one example by hand for y=1 and y=0 |
| 4 | Fitting II — how we find them: **gradient descent** | **The optimizer (the course's first).** The **gradient** is the direction the loss rises fastest; step the **opposite** way by a **learning rate**, repeat — the weights roll to the bottom of NB 3's bowl. The gradient here is **∝ (P−y)·x** (stated and used; full derivation pointed to, not ground through). Convergence is *shown*, not proved | build gradient-as-slope intuition on a **1-D bowl** first; implement the (P−y)·x update; run gradient descent by hand on **standardized 1-D `bill_length`** (2 params → a visualizable loss surface); confirm the by-hand weights ≈ `LogisticRegression(C=np.inf)` | (a) **gradient-as-slope on a 1-D bowl**; (b) **loss surface + descent path** walking to the minimum; (c) **loss vs iteration** falling to a floor — each with its own read: *"training = rolling downhill; each step nudges w to be less surprised by the labels; it stops where sklearn's solver stops"*; and the **learning-rate panel** (too small crawls, good converges, too big oscillates/diverges — constructed deliberately, e.g. ill-conditioned/raw features, since standardized 1-D is forgiving) | easy — change the learning rate and describe the path; medium — do one update step by hand from given w, P, y, x; harder — start from a bad initial w and predict whether it still converges (convex ⇒ yes) |

## Notebook 5 — the estimator & its parameters

`sklearn.linear_model.LogisticRegression` for real (sklearn **1.9.0** — the API verified at plan time).
**Parity first:** by-hand gradient-descent weights ≈ `LogisticRegression(C=np.inf)` on penguins (same
boundary). Then walk the knobs, each *shown*, on the **current** API:
- **`C`** — inverse regularization strength: small C = strong penalty = small, smooth weights; large C →
  the near-separable penguins push ‖w‖₂ up to a plateau. The **regularization path** (‖w‖₂, and each
  weight, vs C — figure) and the **separation→divergence** demo (one figure, on a constructed separable
  slice: weights run away ⇒ *why regularization exists*).
- **`l1_ratio` (NOT `penalty=`)** — the chapter's correctness pivot: **`penalty` was deprecated in
  sklearn 1.8 and is removed in 1.10**; regularization *type* is now set by **`l1_ratio`** (0 = L2 ridge,
  default; 1 = L1 lasso; in between = elastic-net) with **`C`** for strength, `C=np.inf` for none.
  **L2 shrinks** all weights toward 0; **L1 zeroes** some (feature selection) — shown on **penguins +
  injected noise columns** (L1 kills the noise, L2 only shrinks it). L1/elastic-net need **solver
  `saga`** (lbfgs is L2-only) — stated as the practical constraint, with the solver table named.
- **Multi-class — `softmax`, in its own section with its own figure + "Read the figure"** (a conceptual
  extension, not a parameter): `LogisticRegression` now minimizes the **full multinomial (softmax) loss**
  by default for ≥3 classes (no `multi_class=` argument — removed); **one-vs-rest** is the explicit
  `OneVsRestClassifier` wrapper. On the 3 species: the **three boundaries**; the two routes agree on
  predictions here (measured 0 % disagreement) but differ in how probabilities are formed (softmax joint
  vs renormalized independent sigmoids).
- **Honest tuning:** choose C (and l1_ratio) by `cross_val_score` / `GridSearchCV` on TRAIN; one sealed
  test eval — the module-00 NB 10 discipline.
- **Your turn:** trace the regularization path yourself; turn up L1 until a feature that genuinely carries
  signal (not only the injected noise) gets zeroed; compare softmax vs OvR probabilities on a borderline
  penguin.

## Notebook 6 — the demanding practical case: **breast cancer (calibration, threshold, error analysis)**

Logistic regression on **breast_cancer** (569×30) — the full honest workflow, mobilizing the whole arc,
and the place where the chapter's two payoffs (calibrated probabilities, interpretable weights) bite:
- look (class balance, standardization in a `Pipeline`) → fit `LogisticRegression` → CV (**0.979**) → one
  held-out evaluation. The same dataset KNN met in ch 01 NB 5 — *"you felt the curse with distances;
  watch a linear model read it instead."*
- **Calibration — the ch 02 loop closes, both under the standardized pipeline:** reliability diagram +
  Brier via `viz.plot_calibration_curve`, LogReg vs GaussianNB on the same split (**Brier 0.027 vs
  0.088**; pile-up 123 vs 167 / 171). Read honestly: discriminative logistic regression gives
  probabilities you can *trust as probabilities* far more than generative NB — *and* the stated nuance
  that near-separable data still leaves it somewhat confident (better-calibrated, not perfectly).
- **Threshold choice — the asymmetric-cost decision (module-00 NB 08 made real):** malignant is the
  costly miss; slide the threshold (`viz.plot_score_threshold` + a precision/recall-vs-threshold read),
  default 0.5 → recall **0.953** (3 cancers missed) vs 0.3 → **0.969** (2 missed, more false alarms). The
  probability is calibrated, so the threshold is a *defensible policy choice*, not a guess.
- **Error analysis & reading the model:** confusion matrix; *which* test cases it misses (borderline,
  near P≈½); and the **coefficients as a story** — read the largest |weights| (which measurements push
  toward malignant), then **L1** (`l1_ratio=1`) to **3/8/14** of 30 features as automatic selection.
  Interpretable in a way KNN/NB were not.
- **Bridge forward:** logistic regression is a **linear** boundary; when the truth is non-linear it
  underfits — pointing to **decision trees** (ch 04, non-linear partitions) and, far later, to neural
  networks (the sigmoid neuron *is* a logistic unit; the gradient descent of NB 4 *is* how they train).
- **Your turn:** pick a threshold for a stated cost ratio and justify it from the PR curve; read the top
  malignant-driving coefficients; (harder) `CalibratedClassifierCV` — does it improve the Brier?

**Six notebooks** — a Rémy-approved exception to the 5-ceiling (the per-method arc preserved:
fundamentals one-concept-each across NB 1–4, the estimator NB 5, the demanding case NB 6). The
generative-vs-discriminative capstone ch 02 promised is paid off across NB 1 (direct P(y∣x)), NB 4
(discriminative fitting) and NB 6 (calibration), not as a separate notebook.

## Library additions (decided per-notebook, with tests; none forced now)

- **Likely (NB 6):** `datasets.load_breast_cancer()` — a thin, **pandas-first** wrapper over
  `sklearn.datasets.load_breast_cancer` returning a tidy `DataFrame` with the **30 named feature
  columns + target** (NB 6 needs the names to read coefficients), mirroring the `load_penguins_full`
  shape, with INFO logging. Schema test (`pytest` 16 → 17).
- **Possible (NB 1, reused NB 4/NB 6):** `viz.plot_logistic_curve(x, y, *, proba=…/model=…)` — a 1-D
  fitted-sigmoid plot (points at 0/1, the S-curve, the ½ line) in charter colours, **only if** genuinely
  reused ≥2×; otherwise a one-off in-notebook figure. Smoke test if promoted.
- **One-off in-notebook figures (charter colours, not helpers):** NB 1's sigmoid + p/odds/log-odds table;
  NB 2's boundary+weight bar; NB 3's **log-loss bowl vs squared-error bumps**; NB 4's **gradient-on-bowl**,
  **loss surface + descent path**, **loss-vs-iteration**, **learning-rate panel**; NB 5's **regularization
  path**, **separation→divergence**, **noise-vs-real weight bars**, **3-class softmax boundaries**; NB 6's
  **coefficient-story bar**.
- **Reused as-is:** `use_course_style`, `plot_decision_boundary` (NB 2/NB 5 boundaries, incl. 3-class),
  `plot_confusion_matrix`, `plot_class_balance`, `plot_feature_histograms`, `plot_roc_curve`,
  `plot_score_threshold`, `plot_calibration_curve` (all in `ml_course.viz` today).

## Honest limits stated in the notebooks

- Logistic regression draws a **linear** boundary; on non-linearly-separable data it **underfits** (named,
  fix pointed to — features/kernels/trees/nets — not taught). make_moons is *not* used for the
  fundamentals precisely because LogReg can't fit it; penguins is near-linear.
- **Perfect separation → the MLE diverges** (weights →∞): shown deliberately in NB 5 as *why
  regularization exists*, on a constructed separable slice — not hidden.
- **Weights are interpretable only after standardization** (raw-scale weights confound importance with
  units — the ch 01 scale trap); stated wherever weights are read.
- **Calibration is better than NB but not automatic:** Brier 0.027 vs 0.088, yet near-separable data still
  leaves LogReg somewhat over-confident (123/171 piled up) — "better-calibrated, not perfect; recalibrate
  (`CalibratedClassifierCV`) if the number must be exact."
- **Gradient descent is taught as a picture and a working loop, not a convergence proof**; the learning
  rate is shown to matter (too big → oscillates/diverges — constructed deliberately, since standardized
  1-D is forgiving), the heavy calculus kept light (the gradient stated and used; full derivation →
  ESL/Bishop).
- **multinomial vs OvR agree on penguins** because the data is near-separable — a measured property of
  *this* data, not a general identity; on overlapping multiclass data they differ.
- Logistic regression is **eager and parametric** (fits w, b then discards the data — unlike lazy KNN),
  **fast** to train (small data) and predict, a **strong, interpretable, well-calibrated linear
  baseline**. **When to use:** a linear-ish boundary, interpretable weights, trustworthy probabilities, a
  baseline before anything fancier. **When not:** strongly non-linear structure (→ trees, SVM-kernel,
  nets), or when the boundary is visibly curved.

## References (with DOI where they resolve; **dereferenced & pinned by the ml-expert reviewer at build**)

- Cox DR (1958). *The regression analysis of binary sequences.* J. R. Stat. Soc. B 20(2):215–242.
  DOI: 10.1111/j.2517-6161.1958.tb00292.x — logistic regression's origin.
- Berkson J (1944). *Application of the logistic function to bio-assay.* JASA 39(227):357–365.
  DOI: 10.1080/01621459.1944.10500699 — the logistic function for classification.
- Ng AY, Jordan MI (2001). *On discriminative vs. generative classifiers: a comparison of logistic
  regression and naive Bayes.* NeurIPS 14:841–848 — the generative↔discriminative bridge from ch 02.
- Hastie T, Tibshirani R, Friedman J (2009). *The Elements of Statistical Learning*, §4.4 (logistic
  regression), §3.4 / §4.4.4 (ridge/lasso, regularized logistic). DOI: 10.1007/978-0-387-84858-7.
- James G, Witten D, Hastie T, Tibshirani R (2021). *An Introduction to Statistical Learning*, §4.3
  (logistic regression). DOI: 10.1007/978-1-0716-1418-1.
- Tibshirani R (1996). *Regression shrinkage and selection via the lasso.* J. R. Stat. Soc. B
  58(1):267–288. DOI: 10.1111/j.2517-6161.1996.tb02080.x — L1 / sparsity.
- Hoerl AE, Kennard RW (1970). *Ridge regression: biased estimation for nonorthogonal problems.*
  Technometrics 12(1):55–67. DOI: 10.1080/00401706.1970.10488634 — L2 / shrinkage.
- Bishop CM (2006). *Pattern Recognition and Machine Learning*, §4.3.2–4.3.4 (logistic regression,
  cross-entropy, the gradient). Springer (no DOI; page/section pinned at build) — the log-loss/gradient
  grounding.
- Niculescu-Mizil A, Caruana R (2005). *Predicting good probabilities with supervised learning.*
  ICML-2005. DOI: 10.1145/1102351.1102430 — calibration (LogReg well-calibrated; reused from ch 02).
- Platt J (1999). *Probabilistic outputs for SVMs…* Adv. Large Margin Classifiers (no DOI; volume pinned
  at build) — calibration / recalibration (NB 6 going-further).

## Verification (per notebook, at its commit)

Every number re-run and reconciled into prose at build on the pinned sklearn **1.9.0** (sigmoid raw-mm:
acc 0.947, crossing ≈43 mm, ~16 % in [0.1,0.9]; 2-D near-separable: train 0.9962, |coef| plateau;
‖w‖₂ L2 path 0.83→6.78 over 4 std features; L1 penguins 4/4 for reasonable C (1/4 at C=0.01) → noise-
injection demo; multinomial 0.955 / OvR 0.952 / 0 % disagreement; GD parity vs `C=np.inf`; breast_cancer
CV 0.979 vs 0.930 (StratifiedKFold5 shuffle seed0), Brier 0.027 vs 0.088 (both std-pipeline), pile-up
123 vs 167, threshold recall 0.953→0.969, L1 train-std 3/8/14). **API correctness pinned:** `C` +
`l1_ratio` (not deprecated `penalty=`), `OneVsRestClassifier` for OvR (no `multi_class=`), `saga` for L1,
GD parity against `C=np.inf`. **Banned-word grep** ("just/simply/obviously/trivially" + FR) over every
notebook before commit. Runs top-to-bottom (nbconvert to /tmp; **output-free**, `--clear-output
--inplace`); `check_no_hardcoded_hex` passes; `gen_llms_txt` re-run; `pytest` green (grows from 16 only as
`src/` helpers land); `course_map.md` §03 aligned to the six final titles; `common_errors.md` extended per
new intuition trap. Both reviewers pass (no BLOCK) on each notebook; Rémy validates visually; commit per
notebook; **chapter close via PR into `main`** (protected).

## Per-notebook status (update as the chapter is built)

| NB | Branch | Status |
|----|--------|--------|
| 1 | `notebook/03_LogisticRegression__01_score_to_probability` | **done** — built (19 cells), both reviewers PASS (2 MINORs folded: log=natural-log; "a fifth" tied to the P∈[0.1,0.9] band), Rémy validated visually, merged to `chapter/03_LogisticRegression` |
| 2 | `notebook/03_LogisticRegression__02_boundary_and_weights` | pending |
| 3 | `notebook/03_LogisticRegression__03_logloss_objective` | pending |
| 4 | `notebook/03_LogisticRegression__04_gradient_descent` | pending |
| 5 | `notebook/03_LogisticRegression__05_estimator_and_parameters` | pending |
| 6 | `notebook/03_LogisticRegression__06_breast_cancer_calibration_threshold` | pending |

## Reviewer notes (chapter-plan gate — both reviewers REVISE → folded; split decided by Rémy)

- **ML expert — REVISE (2 BLOCK, 2 MAJOR, MINORs) → all folded; numbers independently re-measured:**
  BLOCK1 — NB 1's "30 % in transition / ≈46 mm" was self-contradictory; raw-mm truth is **acc 0.947,
  crossing ≈43 mm, ~16 % in [0.1,0.9]** (the 30 % was standardized) → corrected, NB 1 uses raw mm.
  BLOCK2 — breast_cancer CV (0.9724/0.9497) didn't reproduce → pinned **StratifiedKFold(5, shuffle, seed
  0): LogReg 0.979, GaussianNB 0.930** (gap larger). MAJOR — GaussianNB calibration was measured raw vs
  LogReg-standardized → both now under **one standardized pipeline** (GaussianNB Brier **0.088**, pile-up
  **167/171**; the ch 02 fairness fix); 3× gap survives. MAJOR — L1 counts setup-sensitive → penguins
  qualified ("4/4 for reasonable C, 1/4 at C=0.01"), breast_cancer 3/8/14 pinned to the train-std setup
  with high `max_iter`. MINORs — GD parity stated against **`C=np.inf`** (not C=1); **‖w‖₂ over 4 std
  features** named; OvR **0.952** (not 0.955); DOIs to dereference at build. **Split: ml-expert prefers 5
  (objective+optimizer cohere; the parity binds them) but agrees 6 is defensible.** Praised: the sklearn-
  1.9 API pivot (all four claims verified), gradient ∝(P−y)·x verified to machine precision, the
  separation→divergence placement, the honest calibration framing, NB 6's mobilization of the method.
- **Pedagogy — REVISE (1 BLOCK, 3 MAJOR, MINORs) → all folded:** BLOCK — 2 banned words ("just",
  "obviously") reworded; banned-word grep added to Verification. MAJOR — **gradient descent under-budgeted
  → split to 6** (its own NB 4): the course's first optimizer, load-bearing for ch 11/12, deserves its own
  intuition/loop/learning-rate/convergence — **Rémy approved the split.** MAJOR — **odds/log-odds** and
  **the gradient-as-slope** were used but not budgeted → both added as explicit NB 1 / NB 4 first-contacts
  (+ a line on e). MAJOR — NB 5 softmax was a parameter bullet → promoted to **its own section + figure +
  "Read the figure"**, after C and L1/L2. MINORs — per-NB "Your turn" sketched for NB 1–4; NB 4's three
  figures each get their own read; NB 1 weights stated hand-chosen, never fitted. Praised: the
  generative/discriminative payoff (not re-opened), the breast_cancer revisit-with-new-eyes, first-contacts
  fenced from recaps, the honesty spine, measured-and-defended dataset choices.
