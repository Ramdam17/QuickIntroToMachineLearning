# STATE — where the course build is right now

> The compaction lifeline. Update this at **every** phase transition. On any session start / after a
> compaction, read this first, then `docs/WORKFLOW.md`, then the active plan. See the phase
> vocabulary in `docs/WORKFLOW.md`.

| Field | Value |
|---|---|
| Current chapter | `00_GettingStarted` (10 of 11 notebooks done) |
| Current notebook | — (10 merged; next is 11) |
| Phase | `chapter-plan-approved` (ready to open the next notebook) |
| Active branch | `chapter/00_GettingStarted` |
| Active plan | `docs/plans/chapter_00_GettingStarted.md` (APPROVED) |
| Next concrete action | Open notebook 11 (the chapter's last): `git switch -c notebook/00_GettingStarted__11_preprocessing_leakage` off the chapter branch; phase `notebook-plan`; enter plan mode; draft from the chapter plan's NB 11 row — **standardization** (and that NB 05's Euclidean nearest-centroid was already trusting raw scales → the scale-sensitivity payoff promised in NB 02/05); **encoding**; **fit-on-train-only**; the **`Pipeline`**; demonstrate **scale-before-split leakage** vs the correct estimate; build a `Pipeline` evaluated under CV (reuse NB 10's cross-validation). **Returns to penguins** (binary, 2 features). Prereqs 04, 05, 10. Rémy approves the plan; then build. After NB 11 ships: **chapter 00 close** — `git switch main && git merge --no-ff chapter/00_GettingStarted`. |

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
