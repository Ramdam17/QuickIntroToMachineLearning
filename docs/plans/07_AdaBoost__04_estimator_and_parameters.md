# Notebook plan — 07_AdaBoost / 04_estimator_and_parameters

> Status: **APPROVED by Rémy & persisted** (no reviewer gate at plan stage). Building now → guards →
> two-reviewer gate (no BLOCK) → Rémy visual → commit → ff-merge to `chapter/07_AdaBoost`.

## Context

NB **4 of 5**, the **integrative** notebook (the per-method arc's "the estimator & its parameters").
De-overloaded: one job — meet `AdaBoostClassifier` and its real dials, knowing from NB 1–3 exactly
what each does. Headline dial: **the base learner must stay weak** (a stronger base overfits faster —
the opposite of a random forest). Then `n_estimators × learning_rate` (CV heatmap), the removed
`algorithm` param, multiclass SAMME, `feature_importances_` (MDI, ch 06 caveat), and honest
`GridSearchCV`. On the moons-0.20 through-line. ~22 cells (9 code / 13 md), 3 figures.

## Anchors (re-measured at plan time, sklearn 1.9.0, seed 0; moons-0.20, n_train 280)

- **API:** `AdaBoostClassifier(estimator=None, *, n_estimators=50, learning_rate=1.0,
  random_state=None)`; default `estimator_` = `DecisionTreeClassifier(max_depth=1)` (stump); the
  `algorithm` kwarg is **absent** (SAMME only; SAMME.R removed). Parity recap: AdaBoost(50) test
  **0.9417**, `estimator_weights_[:3]` = [1.6796, 1.1338, 1.3854] (== by-hand, NB 1/2).
- **Base-learner strength** (n_estimators=200, lr=1.0): every depth memorises the train set
  (train **1.0000**), but TEST drops as the base deepens — depth-1 **0.9417**, depth-2 **0.9333**,
  depth-3 **0.9333**, depth-5 **0.9167**. A stronger base overfits faster; the stump default is right.
  (Contrast ch 06: a random forest wants *deep* trees — boosting wants *weak* ones.)
- **n_estimators × learning_rate, 5-fold CV on train** (mean accuracy; rows = learning_rate, cols =
  n_estimators — built via a pandas pivot so the orientation can't drift; GridSearchCV expands params
  by *alphabetically-sorted* keys, so a naive `reshape` transposes the grid):
  | lr \ n | 50 | 100 | 200 | 400 |
  |---|---|---|---|---|
  | 0.1 | 0.911 | 0.911 | 0.943 | 0.954 |
  | 0.5 | 0.946 | 0.957 | 0.957 | 0.961 |
  | 1.0 | 0.954 | 0.957 | 0.957 | 0.957 |
  Bottom-left (few rounds, small steps: n=50/lr=0.1) underfits ~0.911; rises to a broad ~0.95–0.96
  **plateau**; small lr needs many rounds (lr=0.1 reaches 0.954 only at n=400; lr=1.0 is 0.957 by
  n=100). Best CV: **lr=0.5, n=400 (0.9607)**.
- **Honest tuning:** default (n=50, lr=1.0) CV **0.9536**, sealed test **0.9417**; tuned (lr=0.5, n=400)
  CV **0.9607**, sealed test **0.9417** — **identical on the test set**. The CV gain (+0.007) did NOT
  transfer; a fraction-of-a-point CV improvement on a small problem is within noise. (Echoes ch 06's
  "tuning barely beats the default.")
- **`feature_importances_`** (AdaBoost(200) on moons): ~**[0.61, 0.39]** (x1, x2; MDI, sums to 1).

## Cell-by-cell (~22 cells; intuition → implementation → "Read the figure")

1. (md) **Header** — `# 04 — The estimator AdaBoostClassifier and its parameters`; *Chapter 07 · NB 4
   of 5*. Open: NB 1–3 built AdaBoost by hand and explored its behaviour; now meet the library
   estimator and turn its dials *knowing what each one does*. **Prerequisites:** NB 1–3 (the by-hand
   loop & parity, the additive model, learning_rate & overfitting); module 00 (cross-validation &
   `GridSearchCV`, NB 10); ch 06 (feature importance / MDI & its bias). **What you'll be able to do:**
   read `AdaBoostClassifier`'s constructor; set the **base learner** and explain why it must stay weak;
   tune `n_estimators × learning_rate` honestly by CV; read `feature_importances_` with the right
   caveat; state the current-API facts (SAMME-only).
2. (md) **Where we are.** Three notebooks of mechanism and behaviour; this one is short and practical —
   the estimator and its knobs. Nothing here is new magic: every dial maps to something you already
   built.
3. (code) **Setup + the constructor** — moons-0.20 split; print
   `inspect.signature(AdaBoostClassifier.__init__)`, the default `estimator_` (stump), and that
   `algorithm` is absent.
4. (md) **Read it.** The four parameters: **`estimator`** (the weak learner; default a depth-1 stump),
   **`n_estimators`** (rounds), **`learning_rate`** (shrinkage, NB 3), **`random_state`**. The old
   `algorithm` parameter (`SAMME` / `SAMME.R`) is **gone** in 1.9 — only SAMME remains, the algorithm
   we derived in NB 1–2. So there is no SAMME.R speed/accuracy choice to make any more.
5. (code) **Parity recap** — `AdaBoost(50)`: print test 0.9417 and `estimator_weights_[:3]`; one line
   confirming they equal our by-hand α (NB 1/2).
6. (md) **Read it.** The library estimator *is* the loop you wrote — same learner weights, same
   predictions. From here we explore its dials, not its internals.
7. (md) **The dial that matters most: the base learner.** Boosting's bet (NB 1) is that *weak*
   learners combine into a strong one. So the base should stay weak. What happens if we make it
   stronger — a deeper tree?
8. (code) **Base-strength sweep** — `AdaBoost(estimator=DecisionTreeClassifier(max_depth=d),
   n_estimators=200)` for d ∈ {1, 2, 3, 5}; print train/test (all train 1.000; test 0.9417 → 0.9167).
9. (code) **Figure A — boundary: stump base vs depth-3 base** (both AdaBoost(200)), side by side.
10. (md) **Read the figure (A).** Every base depth drives **training** accuracy to 1.000 — they all
    memorise. But **test** accuracy *falls* as the base deepens (0.9417 with a stump → 0.9167 at
    depth 5): a stronger base overfits faster, and the boundary (right panel) is visibly more contorted
    than the stump's clean staircase. This is the mirror image of chapter 06: a **random forest** wants
    *deep*, low-bias trees and averages away their variance; **boosting** wants *weak*, high-bias
    learners and reduces bias by adding many. Deep base learners defeat boosting — keep the default
    stump (or near it).
11. (md) **Rounds and step size, together.** `n_estimators` and `learning_rate` interact (NB 3): a
    smaller step needs more rounds. The honest way to choose both is cross-validation on the training
    set — never the test set.
12. (code) **`GridSearchCV`** on TRAIN over `n_estimators ∈ {50,100,200,400} × learning_rate ∈
    {0.1,0.5,1.0}`, cv=5; print the mean-CV grid, best params, best CV.
13. (code) **Figure B — the `n_estimators × learning_rate` CV heatmap** (mean CV accuracy; course
    colormap, cells annotated).
14. (md) **Read the figure (B).** The bottom-left (few rounds, small step) **underfits** (~0.91);
    accuracy climbs with more rounds and/or a larger step to a broad **~0.95–0.96 plateau**. A small
    learning rate needs many rounds to catch up (n=50/lr=0.1 → 0.91 vs n=400/lr=0.1 → 0.957). Best CV
    is lr=0.5, n=400 (0.961) — but notice how *flat* the plateau is: like the random forest (ch 06),
    AdaBoost is **forgiving** once it has enough rounds.
15. (code) **Tuned vs default — one sealed test.** Print default (n=50,lr=1.0) CV 0.943 / test 0.9417,
    and tuned (lr=0.5,n=400) CV 0.961 / test 0.9417.
16. (code) **Figure C — CV vs sealed test, default vs tuned** (grouped bars: each model's CV accuracy
    and its test accuracy).
17. (md) **Read the figure (C).** Cross-validation preferred the tuned model (CV 0.943 → 0.961), but on
    the **sealed test both score exactly 0.9417** — the CV gain did **not** transfer. On a problem this
    size a fraction-of-a-point CV improvement is within the noise; do not over-claim tuning wins.
    (Chapter 06 found the same for the forest.) Touch the test set **once**, at the end — which is what
    we did.
18. (code) **`feature_importances_`** — `AdaBoost(200).feature_importances_` on moons (≈ [0.61, 0.39]).
19. (md) **Read it.** Like a tree or a forest, AdaBoost exposes `feature_importances_`: the MDI
    (impurity-decrease) of each feature, averaged over the stumps and weighted by their α. Here both of
    the two features carry real signal (the boundary needs both), x1 a little more. **The ch 06 caveat
    carries over**: MDI is biased toward continuous / high-cardinality features and is **not causal** —
    so read it at the group level and cross-check with permutation importance. We do the honest reading
    on real data (54 features) in NB 5. Multiclass, for the record, is automatic: SAMME's `+ ln(K−1)`
    term (NB 2) lets the same estimator handle K > 2 classes.
20. (md) **Honest scoping.** (a) The base learner must stay **weak** — a deeper base overfits faster
    (measured); the stump default is right (the opposite of a random forest). (b) No `algorithm` /
    SAMME.R choice remains (SAMME only). (c) Tuning gains can be **within noise** — here CV rose but the
    sealed test did not move. (d) `feature_importances_` is **MDI**, biased and non-causal (NB 5 reads
    honestly). (e) One sealed test, touched once.
21. (md) **Your turn** (tiered) — *easy:* from Figure A / the sweep, which base depth generalises best,
    and state in one sentence why boosting prefers it (vs a random forest's preference). *medium:* add
    `learning_rate = 0.05` and `n_estimators = 800` to the grid and re-run `GridSearchCV` — does the
    plateau extend, and does the best CV move? *harder:* set `estimator=DecisionTreeClassifier(max_depth=3)`
    and tune `n_estimators` alone — does a stronger base need **fewer** rounds, and does it beat the
    stump on the sealed test? Explain from the bias/variance picture.
22. (md) **What you built** + vocabulary (`estimator` / base learner & its strength · `n_estimators` ·
    `learning_rate` · SAMME-only / `algorithm` removed · `GridSearchCV` (CV-on-train, one sealed test) ·
    `feature_importances_` / MDI) + **References** — Freund & Schapire 1997 (DOI 10.1006/jcss.1997.1504);
    Zhu et al. 2009 SAMME (DOI 10.4310/SII.2009.v2.n3.a8); Pedregosa et al. 2011 — scikit-learn
    (JMLR 12:2825–2830); ESL §10 (DOI 10.1007/978-0-387-84858-7). `Previous: 03 — Learning rate, rounds,
    and overfitting behaviour. Next: 05 — A demanding case (spambase).`

## `src/` & guards

No `src/` change (reuse `use_course_style`, `plot_decision_boundary`; the CV heatmap and the CV-vs-test
bars are notebook-local matplotlib with a course colormap from `ml_course.colors`). **pytest stays 20.**
Build via `uv run python - < build`; banned-word JSON scan = 0; ruff/black clean; no hardcoded hex;
output-free; nbconvert from project cwd on a scratchpad copy; **rebuild from the script right before
`git add`** (editor kernel-drift habit); `gen_llms_txt`; both reviewers (no BLOCK) + Rémy visual.

## Note on anchors vs the chapter plan

The chapter plan §NB 4 sketched the base-strength effect on **moons-0.30** (depth-1 0.892 / depth-3
0.875); NB 4 uses the **moons-0.20 through-line** (depth-1 0.9417 / depth-3 0.9333 / depth-5 0.9167) for
consistency with NB 1–3. The qualitative result (a deeper base overfits faster) is identical; only the
dataset noise level (and so the absolute numbers) differs.
