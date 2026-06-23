# Notebook plan — 06_RandomForest / 03_out_of_bag

> Status: **APPROVED** (2026-06-23, by Rémy). Built next; both reviewers (`@ml-expert-reviewer` +
> `@pedagogy-reviewer`) gate the **built** notebook (no gate at the plan stage). Anchors re-measured
> at build on sklearn **1.9.0**, every random forest `random_state`-pinned.

## Context

NB **3 of 5** — the last fundamentals notebook. One concept: **out-of-bag (OOB) estimation** — the
bootstrap's free validation set. NB 1 noted that each bootstrap leaves ~37 % of points unseen ("they
will earn their keep in notebook 3"); NB 2 finished the forest. Now we cash in: for each training
point, the trees that did **not** see it form a held-out mini-forest that can predict it honestly, so
the forest **scores itself for free** — no separate validation split. We build the OOB vote by hand,
match it to sklearn's `oob_score_`, see it track the sealed test (a touch optimistically), and learn
when it is trustworthy (enough trees) and when it is not (sklearn warns).

## Anchors (measured at plan time, sklearn **1.9.0**; `random_state` pinned; re-measured at build)

breast_cancer (569 × 30, **malignant = 1**), 70/30 stratified split (seed 0); n_train = 398.

- **The ~1/e left-out fraction:** `(1 − 1/n)^n` = **0.3674** (n = 398) → `1/e` = 0.3679; empirical mean
  left-out fraction over 3000 draws **0.3676**. (At n = 398 the formula already sits on its limit —
  unlike NB 1's n = 10 demo at 0.349.)
- **OOB by hand (B = 200 bagged `'sqrt'` trees):** track each tree's in-bag indices; for each training
  point, majority-vote the trees that missed it → **hand OOB accuracy 0.962**; every point is graded
  (398/398 have ≥ 1 OOB tree), by a mean of **73.4 trees** each (≈ 0.37 × 200). 
- **Parity with sklearn:** `RandomForestClassifier(n_estimators=200, oob_score=True).oob_score_` =
  **0.955** ≈ the hand OOB (0.962); the small gap is the hand **hard vote** vs sklearn's
  **probability-averaged** OOB vote (decided at build — may switch the hand demo to soft vote for a
  tighter match). **Sealed test = 0.942.**
- **OOB ≈ test, mildly optimistic:** OOB ≈ 0.955 vs test ≈ 0.942 — close, but OOB runs ~1–2 points
  high here (it is still an in-training estimate). Stated, not hidden.
- **OOB needs enough trees (the honest caveat):** OOB-vs-`n_estimators`: B = 3 → oob **0.842** (and
  sklearn **warns**: P(a point is never OOB) ≈ 0.632³ = 0.25), B = 5 → 0.910 (warns, 0.10), B = 10 →
  0.937 (warns, 0.01), **B ≥ 25 → ~0.955–0.960 and stable** (no warning); test hovers ~0.94 throughout.
  OOB settles into a reliable estimate by a few dozen trees.

## Library / figures

- **No `src/` change** (`pytest` stays 19). Reuse `viz.use_course_style`, `viz.plot_train_test_curve`
  (the OOB-error vs test-error curve — two lines, ylabel "error"); `ml_course.colors`. One one-off
  charter figure: the **in-bag / OOB schematic** (a trees × points grid, in-bag vs OOB shaded).
  Sklearn: `RandomForestClassifier(oob_score=True)`, `train_test_split`, `accuracy_score`;
  `DecisionTreeClassifier` + numpy for the by-hand OOB.
- **Two figures:** **A** the in-bag/OOB schematic (which points each of a few trees saw vs missed);
  **B** OOB error vs test error vs number of trees. Each + "Read the figure".

## Cell-by-cell (~21 cells)

1. (md) **Header** — `# 03 — Out-of-bag estimation`; *notebook 3 of 5*; warm welcome. **Prerequisites:**
   NB 1 (bootstrap, and the ~37 % left out); NB 2 (the full random forest); module 00 — the train/test
   split (NB 04) and cross-validation (NB 10). **What you'll be able to do:** derive the ~37 % out-of-bag
   fraction; build the OOB vote by hand; match it to sklearn's `oob_score_`; use OOB as a free estimate
   of generalization; say when OOB is trustworthy and when it is not.
2. (code) **Imports + seed + style + data** — breast_cancer, malignant = 1, 70/30 split (seed 0); print
   shapes.
3. (md) **Recap / footing** — NB 1: a bootstrap samples n with replacement and leaves ~**37 %** of
   points unseen (we parked them, "they will earn their keep in notebook 3"). NB 2: those resampled
   trees, decorrelated, are now a random forest. Time to collect on the parked points.
4. (md) **Intuition — the bootstrap's leftover is a free test set.** Each tree trains on ~63 % of the
   points; the ~37 % it never saw are, *for that tree*, unseen data — a personal held-out set. Pool
   across the forest: each point sits out ~37 % of the trees, so it has a whole **mini-forest of trees
   that never saw it** and can judge it honestly. The forest can grade itself, with no data set aside.
5. (code) **Derive & measure the ~37 %.** `(1 − 1/n)^n` for n = 398 → **0.367**, which is `1/e`;
   empirical mean left-out fraction over many draws → **0.368**.
6. (md) **Read the result.** The left-out fraction is a robust ~37 %. So with B trees, each point is
   out-of-bag for about 0.37·B of them — enough, once B is reasonable, to form a real committee that
   never saw it.
7. (md) **Intuition — the OOB prediction.** For one training point: gather the trees that did *not*
   include it in their bootstrap, take their **majority vote**, and call that the point's **OOB
   prediction**. Do this for every training point; the error of these predictions is the **OOB error**
   — an estimate of generalization that touched no held-out split.
8. (code) **Build OOB by hand.** B = 200 bagged `'sqrt'` trees; track each tree's in-bag indices; for
   each point, vote the trees that missed it → **hand OOB accuracy 0.962**; mean **73 OOB trees per
   point** (≈ 0.37 × 200); all 398 points covered.
9. (md) **Read the result.** Every training point was graded by ~73 trees that never saw it, and the
   pooled accuracy (0.962) is an honest out-of-sample estimate — computed for free, during the same fit
   that built the forest.
10. (code) **Figure A — the in-bag / OOB schematic.** A small grid (a handful of trees × ~20 points),
    each cell shaded for in-bag vs OOB, so the ~37 %-OOB pattern is visible per row and per column.
11. (md) **Read the figure (A).** Each row is one tree: the shaded ~37 % are the points it never saw.
    Read down a column (one point): the shaded cells are exactly the trees that grade it. No point
    grades a tree that trained on it — that is what makes the estimate honest.
12. (code) **Parity with sklearn.** `RandomForestClassifier(oob_score=True).oob_score_` = **0.955** ≈
    our hand OOB (0.962); **sealed test = 0.942**.
13. (md) **Read the parity — and the honest caveat.** The library is doing exactly what we did:
    `oob_score_` (0.955) lands next to our hand OOB (0.962) — the small gap is our hard vote vs
    sklearn's probability-averaged vote. And OOB (0.955) sits close to the **sealed test (0.942)**, a
    touch **optimistic** (~1–2 points high): OOB is a strong free estimate, but it is still measured on
    training points, so report the sealed test as the final number.
14. (md) **Intuition — when is OOB trustworthy?** A point is graded only by the trees that missed it
    (~37 % of them). With **few** trees, some points are caught in-bag by *every* tree and have **no**
    OOB grader at all — the estimate is then unreliable, and sklearn says so.
15. (code) **Figure B — OOB error vs test error vs number of trees.** Sweep B = 3, 5, 10, 25, 50, 100,
    200, 500 (sklearn **warns** at the small B — shown, not silenced); plot OOB error and test error.
16. (md) **Read the figure (B).** At a handful of trees the OOB error is wild and sklearn warns (some
    points never sit out — P ≈ 0.632ᴮ, a quarter of them at B = 3); by ~**25 trees** it settles and
    then **tracks the test error**, sitting a hair below it (the mild optimism). Past a few dozen trees,
    OOB is a reliable, free read on generalization.
17. (md) **OOB vs cross-validation.** Both estimate generalization without spending the test set. OOB
    is **nearly free** — computed during the single forest fit, no refitting — where k-fold CV refits k
    times. OOB is the forest's built-in shortcut (handy for quick `n_estimators`/`max_features` checks);
    CV is the general gold standard and the right tool when you compare across model *families*. For
    the headline number you ship, still use the sealed test.
18. (md) **Your turn** (3 tiered) — *easy:* given a 4-tree × 6-point in-bag table, name a point's OOB
    graders and compute its OOB vote; *medium:* for n = 1000 and B = 100, estimate how many trees grade
    each point, and the chance a point has no OOB grader; *harder:* sweep `n_estimators`, plot OOB error
    and test error, and find the smallest forest whose OOB you would trust.
19. (md) **What you built** — the **~37 % OOB fraction** (derived `(1−1/n)^n → 1/e` and measured); the
    **OOB vote by hand**; **parity** with sklearn's `oob_score_`; **OOB ≈ test**, free but mildly
    optimistic; and **when OOB is reliable** (enough trees). **Vocabulary:** out-of-bag (OOB) · OOB
    error · `oob_score_` · free validation estimate.
20. (md) **Going further (optional)** — OOB is handy for cheap model selection on large forests, and
    OOB samples also power **permutation importance computed on out-of-bag data** (notebooks 4–5).
    Caveat to carry: OOB is slightly optimistic and unreliable for tiny forests; for a final, defensible
    number, a sealed test (or nested CV) still wins. **References:** Breiman 1996 (bagging — introduces
    OOB, DOI 10.1007/BF00058655); Breiman 2001 (Random Forests — OOB error & importance, DOI
    10.1023/A:1010933404324); ESL §15.3.1 (out-of-bag samples, DOI 10.1007/978-0-387-84858-7).
    `Previous: 02 — decorrelating the trees` · `Next: 04 — the estimator & its parameters`.

## Honest scoping (stated in the notebook)

- **OOB is mildly optimistic** — it tracks the sealed test but sits ~1–2 points high here; the final
  reported number stays the sealed test.
- **OOB needs enough trees** — at a handful of trees some points have no OOB grader and the estimate is
  unreliable; sklearn warns, and we surface the warning rather than silence it.
- **OOB ≠ a substitute for the test set for final reporting** — it is a cheap in-training estimate, like
  CV; the sealed test is still the honest headline.
- **The by-hand OOB is the mechanism, not a bit-exact clone** — it lands next to `oob_score_`; the gap
  is hard vote vs sklearn's averaged-probability vote.

## Verification

Build via `uv run python - < <scratchpad>/build_ch06_nb3.py` (stdin). Re-measure at build: `(1−1/n)^n`
0.367 / empirical 0.368; hand OOB 0.962 / ~73 graders/point / 398 covered; sklearn `oob_score_` 0.955
≈ hand, sealed test 0.942; OOB-vs-n_estimators (B=3 oob 0.842 + warning → B≥25 ~0.955 stable, test
~0.94). Runs top-to-bottom (nbconvert from project cwd to scratchpad; tracked file **output-free**);
**banned-word scan over the JSON real cell text** = 0; `check_no_hardcoded_hex` passes; `gen_llms_txt`
re-run; `pytest` 19 (no `src/` change); `ruff` clean. Both reviewers PASS (no BLOCK); Rémy validates
visually; commit `feat(06_random_forest): notebook 03 — out-of-bag estimation`; merge
`notebook → chapter`.
