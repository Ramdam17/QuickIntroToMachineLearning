# STATE — where the course build is right now

> The compaction lifeline. Update this at **every** phase transition. On any session start / after a
> compaction, read this first, then `docs/WORKFLOW.md`, then the active plan. See the phase
> vocabulary in `docs/WORKFLOW.md`.

| Field | Value |
|---|---|
| Current chapter | `00_GettingStarted` (8 of 11 notebooks done) |
| Current notebook | — (08 merged; next is 09) |
| Phase | `chapter-plan-approved` (ready to open the next notebook) |
| Active branch | `chapter/00_GettingStarted` |
| Active plan | `docs/plans/chapter_00_GettingStarted.md` (APPROVED) |
| Next concrete action | Open notebook 09: `git switch -c notebook/00_GettingStarted__09_overfitting_generalization_gap` off the chapter branch; phase `notebook-plan`; enter plan mode; draft from the chapter plan's NB 09 row — **over-/under-fitting** and the **generalization gap** (named precisely, ≠ variance), **bias–variance** as the conceptual explanation, the **learning curve**. **Complexity dial:** a polynomial-feature decision boundary whose degree is the knob (method-agnostic — NOT one of the 12; per chapter plan). Likely a new `viz` helper (`plot_complexity_curve` / `plot_learning_curve`) + tests. Rémy approves; then build. |

## Notes / blockers

- NB 01 shipped: `ml_course.datasets.load_penguins()` + vendored `penguins.csv` + tests are in place
  and reusable by later notebooks. pandas-first convention recorded in CLAUDE.md/AGENTS.md.
- Through-line reminder: Palmer penguins (binary, 2 features), nearest-centroid first classifier (NB
  05), polynomial-degree complexity dial (NB 09–10).
- Confirmed: notebooks all-English; maths re-established not presupposed; git history = preserve
  per-notebook, mark chapters (`--no-ff` chapter → main).

## Progress log (most recent first)

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
