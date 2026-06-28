# Course map — twelve methods, the per-method plan

> The canonical plan of what each module teaches. Each method = 3–5 notebooks of ~20 cells, on the
> arc **fundamentals (1–3) → the method & its parameters (4) → a demanding practical case (5)**.
> This map is the contract the `@pedagogy-reviewer` checks each notebook against. Notebook *titles*
> below are a proposed skeleton — refine per method during authoring; the **arc and the concept
> coverage are fixed**.

## Progression rationale

Ordered so each method leans on intuition already built: a distance/vote method first (KNN), then
probability (Naive Bayes), then a linear decision boundary (Logistic Regression), then non-linear
partitions (Decision Tree), then margins (SVM), then ensembles of trees (Random Forest → the
boosting family AdaBoost → Gradient Boosting → XGBoost → LightGBM), then learned representations
(MLP → Neural Networks).

## 00_GettingStarted

The foundational module: the vocabulary and skills every ML student needs before the twelve methods.
Exempt from the per-method arc and the 3–5 ceiling — a one-concept-per-notebook progression of **11
notebooks**. Through-line: Palmer penguins (binary, 2 features), nearest-centroid as the first
classifier, a polynomial-degree complexity dial for the over/underfitting and cross-validation
notebooks. Full plan: `docs/plans/chapter_00_GettingStarted.md`.

1. What is machine learning? — learn a rule from examples; supervised; classification vs regression; the toolkit; fetching and caching the data.
2. Features, labels, and the feature space — `X`/`y`; the mean of a point cloud and Euclidean distance.
3. Look before you model (EDA) — distributions, class balance, feature scales.
4. Generalize, don't memorize — the stratified train/test split; the cardinal sin; leakage; i.i.d. as an assumption.
5. Your first classifier: nearest centroid — fit→predict by hand; the bisector boundary, its bias and scale-sensitivity.
6. Is it any good? accuracy + a baseline — accuracy, the majority baseline, and accuracy's limit under imbalance.
7. The confusion matrix, precision & recall — TP/FP/FN/TN, precision/recall/F1, asymmetric error costs.
8. Scores, thresholds, ROC & AUC — the signed-distance score, sliding the threshold, ROC/PR, AUC.
9. Over-/under-fitting and the generalization gap — complexity, the train/test U, bias–variance, the learning curve.
10. Validating honestly: cross-validation — hyperparameters vs parameters, stratified k-fold, model selection.
11. Preprocessing & leakage — scaling, encoding, fit-on-train-only, the `Pipeline`; data leakage (the wrong way to cross-validate).

## 01_KNN — k-Nearest Neighbours
1. Predict by the neighbourhood vote; k = neighbourhood size; the lazy learner (by hand, on make_moons).
2. Distance & the scale trap (Euclidean vs Manhattan; k-NN is pure distance → standardize).
3. The k dial: under- vs over-fitting; choose k by cross-validation.
4. The estimator & its parameters (KNeighborsClassifier: k, weights, metric); decision boundary vs k.
5. Demanding case: breast_cancer (30-D) — full honest workflow; the curse of dimensionality, felt.
6. Advanced (optional): distances & choosing k — Minkowski p (L1/L2/L∞), Mahalanobis, cosine; the metric matters in high dimensions, not low (penguins vs noisy breast_cancer); nested CV.

## 02_NaiveBayes  *(complete — 5 notebooks, merged to `main`)*
1. Bayes' rule, from counts — prior, likelihood, posterior, evidence; predict by argmax (by hand on one binned feature).
2. The "naive" assumption (conditional independence) — what it buys, where it breaks, and why classification survives (Domingos & Pazzani).
3. The Gaussian likelihood, computed safely — continuous density; by-hand Gaussian NB == `GaussianNB`; log-space vs underflow.
4. The estimators & their parameters — `GaussianNB`/`MultinomialNB`/`BernoulliNB`; var_smoothing, alpha (the zero-frequency cure), priors; honest tuning; calibration named.
5. Text classification (the demanding case) — bag-of-words + `MultinomialNB`; honest evaluation under imbalance; calibration; generative vs discriminative.

## 03_LogisticRegression  *(plan approved — six notebooks; gradient descent earns its own, like KNN's 6th)*
1. From a linear score to a probability: the sigmoid & log-odds, by hand.
2. The decision boundary; reading the weights (set by hand, not fitted).
3. Fitting I — what we optimize: log-loss / cross-entropy (= negative log-likelihood of Bernoulli; why not squared error).
4. Fitting II — how we find them: gradient descent by hand (the course's first optimizer; learning rate, convergence).
5. The estimator & its parameters (`LogisticRegression`: C, `l1_ratio` for L1/L2, multinomial/softmax; honest tuning).
6. Demanding case: breast_cancer — calibrated probabilities, threshold choice, error analysis, reading coefficients.

## 04_DecisionTree
1. A question splits the data: impurity (Gini/entropy) by hand.
2. Growing a tree greedily; reading a tree.
3. Overfitting and pruning; depth as the complexity dial.
4. Parameters (max_depth, min_samples, criterion); instability and variance.
5. Demanding case: interpretability vs accuracy; where a single tree fails.

## 05_SVM  *(complete — five notebooks, merged to `main`)*
1. The widest-margin idea, by hand on separable 2D data.
2. Soft margin: the cost of mistakes (C).
3. The kernel trick: non-linear boundaries without leaving the notebook's intuition.
4. Parameters (C, kernel, gamma); the bias/variance picture they control.
5. Demanding case: scaling matters; model selection by CV; honest limits on large data.

## 06_RandomForest  *(complete — five notebooks, merged to `main`)*
1. The wisdom of trees — averaging cuts variance (bagging), by hand on `make_moons`.
2. The "random" in the forest — bootstrap + feature subsampling decorrelate the trees (the Var = ρσ² + (1−ρ)σ²/B law, derived).
3. Out-of-bag estimation — the bootstrap's free validation set (the ~1/e left out per tree).
4. The estimator `RandomForestClassifier` & its parameters (n_estimators, max_features, depth); diminishing returns; feature importance introduced.
5. Demanding case: a strong tabular baseline on forest cover type (covtype); honest evaluation under imbalance; reading importances honestly.

## 07_AdaBoost  *(complete — five notebooks, merged to `main`)*
1. Boosting intuition: focus on the mistakes; reweighting, by hand (SAMME α; by-hand == sklearn).
2. Weak learners and the additive model; the exponential-loss / forward-stagewise view, from scratch.
3. Learning rate vs number of rounds; overfitting behaviour — resistant on clean data, not immune under noise.
4. The estimator `AdaBoostClassifier` & its parameters (`estimator`, `n_estimators`, `learning_rate`; `algorithm` removed — SAMME only; the base learner must stay weak).
5. Demanding case — spambase (ESL ch 10): AdaBoost shines; where noise hurts it, framed honestly (exponential-loss non-robustness, not an RF horse-race).

## 08_GradientBoosting  *(COMPLETE — merged to `main` via PR #8; six notebooks, regression-first + a classification notebook)*
1. Boosting as fitting residuals, by hand (regression): F₀ = mean → fit a regression tree to the residuals → add a shrunken slice → repeat; exact by-hand == `GradientBoostingRegressor` (1e-16). *Does not yet name "gradient".*
2. The residual was the gradient: gradient descent in function space (regression); the ensemble is a point in function space, each tree a downhill step, ν the step size.
3. Gradient boosting for classification (the added notebook): a different loss (log-loss) → a different residual (y − p); the honest Newton leaf-step (regression leaves = mean, exact; classification leaves need a Newton correction); `loss='exponential'` = AdaBoost's objective (the unifying reveal).
4. Shrinkage and the trees: ν × n_estimators, depth as interaction order, and why GB overfits with too many trees at large ν (the structural contrast with RF, which does not).
5. The estimator `GradientBoostingRegressor`/`Classifier` & its parameters: `loss` (`'deviance'` removed), `subsample` (stochastic GB + OOB), early stopping, importances; `HistGradientBoosting*` named as the fast modern default + the ch 09–10 bridge.
6. Demanding case (visualization-first capstone): tuning a competitive model honestly — California housing (regression); R²/MAE in dollars, residual error analysis, RF/linear foil, HistGB teaser.

## 09_XGBoost  *(in progress — five notebooks; **NB 1 built**; the regularized, second-order refinement of ch 08's engine)*
1. The second-order view: gradients **and** curvature, by hand — `w* = −G/H` minimizes the 2nd-order loss; it unifies ch 08's two leaf rules (squared-error leaf=mean with h=1; log-loss Newton leaf with h=p(1−p)). Its own λ=0 XGBoost parity.
2. The regularized objective: λ, γ, and the gain that decides splits, by hand — `Ω=γT+½λΣw²` → leaf `w*=−G/(H+λ)`, the structure-score gain (Chen & Guestrin eq. 6–7); the measured 2×/½ parity detail (`Cover`=ΣH).
3. Sparsity-aware splits: a learned default direction for missing values, by hand (C&G §3.4) — try both directions, keep the higher gain; GB rejects NaN, HistGBR & XGBoost accept it.
4. The estimator `XGBClassifier`/`XGBRegressor` & its parameters — owns the histogram method (`tree_method='hist'`, `max_bin`, speed measured); `reg_lambda`/`reg_alpha`/`gamma`, `max_depth`/`min_child_weight`/`grow_policy`, `subsample`/`colsample_*`, eta×n_estimators; honest tuning (the aggressive defaults overfit).
5. Demanding case (visualization-first capstone): Adult/Census Income (informative missing + imbalance + native categoricals; Ames fallback) — early stopping, honest cross-method comparison naming native-NaN-vs-imputed as a deliberate axis, gain MDI vs permutation, the LightGBM teaser.

## 10_LightGBM
1. Leaf-wise growth vs level-wise; why it is fast (intuition).
2. Histogram binning and `num_leaves` as the central dial.
3. Categorical handling; key parameters and their traps (overfitting with deep leaves).
4. Tuning and early stopping; comparison points with XGBoost.
5. Demanding case: larger tabular data; speed/accuracy trade-offs, measured.

## 11_MLP
1. A neuron: weighted sum + activation, by hand.
2. Layers and non-linearity; what depth buys.
3. Training intuition: loss, backpropagation (picture, not heavy derivation), learning rate.
4. Parameters (layers, units, activation, regularization, optimizer); over/underfitting.
5. Demanding case: a small end-to-end MLP; scaling, seeds, honest evaluation.

## 12_NeuralNetworks
1. From MLP to networks: representations learned layer by layer.
2. Key building blocks and where they matter (intuition over a chosen framework).
3. Training dynamics: batches, epochs, regularization (dropout, early stopping).
4. Parameters and diagnostics (learning curves, over/underfitting signatures).
5. Demanding case: a complete, honestly evaluated network on a realistic problem; stated limits.
