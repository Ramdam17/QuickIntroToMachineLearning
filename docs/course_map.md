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
11. Preprocessing & leakage — scaling, encoding, fit-on-train-only, the `Pipeline`.

## 01_KNN — k-Nearest Neighbours
1. Prediction = vote of the neighbourhood; k = size of the neighbourhood (by hand, 2D).
2. Distance, and the scale trap (Euclidean vs Manhattan; why normalize).
3. The k dial: under- vs over-fitting; choosing k on held-out points.
4. The estimator & its parameters (k, weights, metric); decision boundary vs k.
5. Demanding case: full workflow on a small realistic set; curse of dimensionality, felt.

## 02_NaiveBayes
1. From counts to probabilities; Bayes' rule by hand.
2. The "naive" conditional-independence assumption — what it buys, where it breaks.
3. Likelihoods: Gaussian vs multinomial/Bernoulli; log-probabilities and underflow.
4. The estimators & parameters (var smoothing, priors, alpha); calibration.
5. Demanding case: text or tabular classification; honest evaluation under imbalance.

## 03_LogisticRegression
1. From a linear score to a probability: the sigmoid, by hand.
2. The decision boundary; reading weights.
3. Fitting: log-loss and what "training" optimizes (gradient intuition, no heavy calculus).
4. Parameters: regularization (C, L1/L2), multi-class; coefficients under regularization.
5. Demanding case: calibrated probabilities, threshold choice, error analysis.

## 04_DecisionTree
1. A question splits the data: impurity (Gini/entropy) by hand.
2. Growing a tree greedily; reading a tree.
3. Overfitting and pruning; depth as the complexity dial.
4. Parameters (max_depth, min_samples, criterion); instability and variance.
5. Demanding case: interpretability vs accuracy; where a single tree fails.

## 05_SVM
1. The widest-margin idea, by hand on separable 2D data.
2. Soft margin: the cost of mistakes (C).
3. The kernel trick: non-linear boundaries without leaving the notebook's intuition.
4. Parameters (C, kernel, gamma); the bias/variance picture they control.
5. Demanding case: scaling matters; model selection by CV; honest limits on large data.

## 06_RandomForest
1. Why average many trees: variance reduction (bagging), by hand.
2. Bootstrap samples and feature subsampling; decorrelating trees.
3. Out-of-bag estimation; feature importance (and its caveats).
4. Parameters (n_estimators, max_features, depth); diminishing returns.
5. Demanding case: a strong tabular baseline; reading importances honestly.

## 07_AdaBoost
1. Boosting intuition: focus on the mistakes; reweighting, by hand.
2. Weak learners and the additive model.
3. Learning rate vs number of rounds; overfitting behaviour.
4. Parameters (n_estimators, learning_rate, base estimator).
5. Demanding case: where AdaBoost shines and where noise hurts it.

## 08_GradientBoosting
1. Boosting as fitting residuals: gradient descent in function space, by hand.
2. The loss and the role of shrinkage.
3. Trees as the base learner; depth, learning rate, n_estimators interplay.
4. Parameters & early stopping; the bias/variance trade-off.
5. Demanding case: tuning a competitive tabular model honestly.

## 09_XGBoost
1. What XGBoost adds to gradient boosting: regularized objective, second-order view (intuition).
2. Handling missing values and sparsity; the histogram split.
3. Key knobs (eta, max_depth, subsample, colsample, lambda/alpha).
4. Early stopping, evaluation sets, and overfitting control.
5. Demanding case: a realistic dataset; tuning + honest comparison to the simpler boosters.

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
