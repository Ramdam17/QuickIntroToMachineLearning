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
| Comparing models on different splits/preprocessing | Not a fair comparison. Same splits, same pipeline. | per method, notebook 5 |
| Accuracy looks great on imbalanced data | A constant predictor can score high. Use a metric that respects the minority class. | `02_NaiveBayes/05` |

## Method-specific (seed entries — extend during authoring)

| Method | Trap | Note |
|---|---|---|
| KNN | "Distances don't need scaling." | A large-range feature dominates the distance; normalize. |
| Naive Bayes | "Features must really be independent." | The assumption is *conditional* independence, and the method is often useful even when it's violated. |
| Logistic Regression | "It's regression." | It's a classifier; the output is a probability via the sigmoid. |
| Decision Tree | "Deeper is better." | Depth is the complexity dial; too deep memorizes. |
| SVM | "Forgot to scale." | SVMs are scale-sensitive; standardize features. |
| Random Forest | "Feature importance = causal importance." | Impurity importance is biased toward high-cardinality features; read it with care. |
| Gradient Boosting / XGBoost / LightGBM | "More estimators always help." | Without shrinkage/early stopping, more rounds overfit. |
| LightGBM | "num_leaves like max_depth." | Leaf-wise growth means large `num_leaves` overfits fast. |
| MLP / Neural Networks | "It won't converge." | Check scaling, learning rate, and seed before concluding anything. |
