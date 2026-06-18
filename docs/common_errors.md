# Common errors & intuition traps

A curated, growing list of the narrow, recurring confusions in this course, so an AI tutor diagnoses
fast and correctly instead of reasoning from scratch. Add a row whenever a learner hits a new one;
point to the notebook that addresses it.

## Methodology traps (any method)

| Symptom / confusion | What's really going on | Where it's addressed |
|---|---|---|
| "My accuracy is 100% on the data I trained on." | You measured memorization, not generalization. Evaluate on held-out data. | `00_GettingStarted` |
| Scaling/feature selection done on the whole dataset before the split | Leakage — test info bleeds into training. Fit preprocessing on train only. | `00_GettingStarted`, `01_KNN/02` |
| "I changed the seed and the result changed — which is right?" | Single-seed results are noisy; fix and document the seed, and judge over CV folds. | per method, notebook 4 |
| "I picked the k (or any hyperparameter) that scored best on the test set." | That tunes the model to the test set, so the reported score is optimistic. Choose hyperparameters by cross-validation on the training data; touch the test set once, at the end. | `00_GettingStarted` (NB 10), `01_KNN/03` |
| "I'll report the best cross-validation score from my grid/sweep." | Optimistic too — you kept the luckiest setting (winner's curse). Estimate with nested CV (inner tunes, outer estimates), or a final sealed-test score. | `01_KNN/06` |
| Comparing models on different splits/preprocessing | Not a fair comparison. Same splits, same pipeline. | per method, notebook 5 |
| Accuracy looks great on imbalanced data | A constant predictor can score high. Use a metric that respects the minority class. | `02_NaiveBayes/05` |

## Method-specific (seed entries — extend during authoring)

| Method | Trap | Note |
|---|---|---|
| KNN | "Distances don't need scaling." | A large-range feature dominates the distance; normalize. |
| KNN | "The single nearest neighbour (k=1) is the most reliable guide." | One nearest point can be a noise stray; the majority vote over k≥3 neighbours is more robust near class overlap. The vote: `01_KNN/01`; choosing k: `01_KNN/03`. |
| KNN | "Odd k avoids ties, so I'm safe." | True only for **two** classes. With 3+ classes an odd k can still tie (e.g. 1-1-1 at k=3); sklearn then breaks it by the lowest class label. | `01_KNN/04` |
| KNN | "More features can only help." | For distance-based k-NN, irrelevant/noise features make all points look equally far (the curse of dimensionality; near/far distance ratio → 1), so accuracy *falls*. Select/reduce features. | `01_KNN/05` |
| KNN | "A fancier distance (Mahalanobis, fractional p) will rescue my k-NN." | The metric matters mainly in high dimensions; in low-d, well-scaled data it is a wash. Fix scale (NB 2) and cut dimensions (NB 5) first. | `01_KNN/06` |
| Naive Bayes | "Features must really be independent." | The assumption is *conditional* independence, and the method is often useful even when it's violated. |
| Naive Bayes | "A likelihood/posterior of exactly 0 or 1 is a hard fact." | A category never seen in a class gives likelihood 0, which zeroes the whole product (the zero-frequency problem) — it means *unobserved*, not impossible. Smoothing (add a little to every count) cures the overconfidence. Surfaced in `02_NaiveBayes/01`; fixed in `02_NaiveBayes/04`. |
| Naive Bayes | "The features are correlated overall (r≈0.87), so naive Bayes must fail." | The naive assumption is about *within-class* (conditional) correlation; the overall number is mostly the gap *between* classes. Measure the within-class value (here 0.33/0.66) — and even when it's nonzero, the *decision* often survives (the probabilities suffer more). `02_NaiveBayes/02`. |
| Naive Bayes | "The product of the per-feature likelihoods is 0 for every class." | With many features the product underflows to 0.0 in floating point — it's *unrepresentable*, not actually zero. Work in **log-space**: sum log-likelihoods instead of multiplying; the argmax is unchanged. `02_NaiveBayes/03`. |
| Naive Bayes / densities | "A probability density above 1 means something is wrong." | For a continuous feature it's the **area** under the density, not its height, that is a probability; a narrow distribution can have a peak above 1 while still integrating to 1. `02_NaiveBayes/03`. |
| Naive Bayes | "Turn `var_smoothing` (or `alpha`) up to be safe." | Smoothing is a floor, not a free lunch: too much inflates every variance (or flattens every count) until the classes stop differing and accuracy collapses toward the majority baseline. It's a safety net to tune, not a knob to max out. `02_NaiveBayes/04`. |
| Naive Bayes | "It reported 99%, so it's almost certainly right." | NB is **over-confident**: treating correlated features as independent double-counts evidence and piles probabilities at 0/1. Trust its *ranking*, not the raw number; recalibrate (`CalibratedClassifierCV`) if you need a probability. `02_NaiveBayes/05`. |
| Logistic Regression | "It's regression." | It's a classifier; the output is a probability via the sigmoid. |
| Decision Tree | "Deeper is better." | Depth is the complexity dial; too deep memorizes. |
| SVM | "Forgot to scale." | SVMs are scale-sensitive; standardize features. |
| Random Forest | "Feature importance = causal importance." | Impurity importance is biased toward high-cardinality features; read it with care. |
| Gradient Boosting / XGBoost / LightGBM | "More estimators always help." | Without shrinkage/early stopping, more rounds overfit. |
| LightGBM | "num_leaves like max_depth." | Leaf-wise growth means large `num_leaves` overfits fast. |
| MLP / Neural Networks | "It won't converge." | Check scaling, learning rate, and seed before concluding anything. |
