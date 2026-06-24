# Notebook plan — 06_RandomForest / 01_averaging_cuts_variance

> Status: **APPROVED** (2026-06-23, by Rémy). Built next; both reviewers (`@ml-expert-reviewer` +
> `@pedagogy-reviewer`) gate the **built** notebook (no gate at the plan stage). Anchors re-measured
> at build on sklearn **1.9.0**, every random forest `random_state`-pinned.

## Context

NB **1 of 5** — the first fundamentals notebook. One concept: **averaging many high-variance trees
(each on a bootstrap resample), by majority vote, gives one low-variance model** — *bagging*, built
entirely by hand. Feature subsampling is deliberately **OFF** here (it is NB 2). The chapter's
through-line picks up ch 04's closing weakness — a single tree is high-variance — and turns it into
the method's first idea. We work on `make_moons` because it is 2-D: the variance, and its cancellation
by averaging, can be **drawn**.

## Anchors (measured at plan time, sklearn **1.9.0**; re-measured at build; every RF `random_state`-pinned)

`make_moons(n_samples=300, noise=0.30, random_state=0)`, 210/90 **stratified** split (seed 0) — the
same set ch 04 NB 3/4 carved, so the single-tree number is a continuity anchor.

- **The shaky expert:** a single unlimited `DecisionTreeClassifier(random_state=0)` → train **1.000**,
  test **0.8778** (= ch 04's 0.878). Refit on 50 bootstrap resamples of train → test mean 0.885,
  **std 0.0314** (≈ ch 04's 0.032). High variance.
- **Bootstrap fact:** drawing n indices with replacement leaves out ≈ **37 %** of points (the seed of
  NB 3's OOB); shown by hand on a 10-point set.
- **Hand-built bagging (majority vote of B unlimited trees, each on its own bootstrap):** test
  **0.900** (B=1) → 0.922 (B=5) → **0.9333** (B=10, plateau through B=200).
- **Variance reduction (the headline):** run-to-run std of test accuracy across 20 bagging seeds:
  **0.0465 (B=1) → 0.0310 (B=5) → 0.0089 (B=25) → 0.0053 (B=100)** — ~**9×** smaller; mean accuracy
  0.894 → 0.909 → 0.927 → 0.929.
- **Honest parity:** hand-bag(200) = **0.9333** = `RandomForestClassifier(n_estimators=200,
  max_features=None, random_state=0)` exactly — *bagging is a random forest with feature subsampling
  off*. The default `RandomForestClassifier()` (`max_features='sqrt'` → 1 of 2 feats here) scores
  **0.900** — lower on this 2-feature toy, a deliberate hook into NB 2 (subsampling pays with many
  features, not two).

## Library / figures

- **No `src/` change** (`pytest` stays 19). Reuse `viz.use_course_style`, `viz.plot_decision_boundary`
  (single-tree and bagged boundaries); `ml_course.colors`. Fig B (acc + run-to-run std vs B) is a
  one-off twin-axis chart in charter colours. Sklearn: `make_moons`, `train_test_split`,
  `DecisionTreeClassifier`, `RandomForestClassifier`, `accuracy_score`; numpy for the bootstrap.
- **Two figures:** **A** several single bootstrap-tree boundaries (jagged, each different) vs the
  smooth averaged/bagged boundary; **B** test accuracy and run-to-run std vs number of trees. Each +
  "Read the figure".

## Cell-by-cell (~22 cells)

1. (md) **Header** — `# 01 — The wisdom of trees: averaging cuts variance`; *notebook 1 of 5*; warm
   welcome. **Prerequisites:** ch 04 (decision trees — a deep tree memorizes and is high-variance,
   NB 3/4; the hand-bagged-25 taste, NB 5); module 00 — train/test split (NB 04), accuracy (NB 06).
   **What you'll be able to do:** bootstrap-resample by hand; aggregate trees by majority vote; measure
   how averaging cuts variance; explain the σ²/B effect and its diminishing returns; recognize that
   hand-bagging == a random forest with feature subsampling off.
2. (code) **Imports + seed + style + data** — `make_moons(300, noise=0.30, random_state=0)`; 210/90
   stratified split (seed 0); `viz.use_course_style()`; print shapes.
3. (md) **Recap / footing** — from ch 04: a deep tree drives training error to 0 (memorizes) and is
   **high-variance** — resample the data, get a different tree (the root feature flipped; on moons the
   boundary wobbles, std 0.032). Ch 04 ended by hand-bagging 25 trees and seeing accuracy steady. Here
   we explain *why*, on moons, where we can **draw** it.
4. (md) **Intuition — one tree is a shaky expert.** A deep tree fits its sample exactly, noise and all;
   a slightly different sample → a different boundary. Confident, but shaky. The plan: ask **many**
   shaky experts, each on a slightly different sample, and **vote** — idiosyncratic errors cancel.
5. (code) **The shaky expert, measured.** Single unlimited tree → train 1.000 / test 0.878; refit on
   50 bootstrap resamples → test mean 0.885, **std 0.031**.
6. (md) **Read the result.** 0.878 is real but *fragile*: resample and it swings (std 0.031, ~0.85–0.92
   run to run). Can we keep the flexibility but tame the wobble?
7. (md) **Intuition — bootstrap resampling** (first contact). Draw n points **with replacement** from
   the n training points: some repeat, ~**37 %** never appear. Each bootstrap is a plausible "alternate
   training set" — the data we might have collected.
8. (code) **Bootstrap by hand.** On a 10-point set, `rng.integers(0, n, n)` → print drawn indices,
   repeats, and omitted points; show ~37 % left out.
9. (md) **Read it.** This resample kept some points twice, dropped others — a slightly different view of
   the same problem. Grow a tree on each of many views → a committee.
10. (md) **Intuition — majority vote (aggregation).** The **ensemble** predicts the class most trees
    vote for (probabilities = the average). If each tree is right more often than not and their errors
    differ, the vote is right far more often — the σ²/B intuition, made precise in NB 2.
11. (code) **Hand-built bagging.** `bag_predict(Xtr, ytr, Xte, B)`: B bootstraps, B unlimited trees,
    majority vote. B = 1/5/25/100 → test 0.900 / 0.922 / 0.933 / 0.933.
12. (md) **Read the result.** From one tree's 0.878 to **0.933** with 25 trees — the committee beats
    its members, then plateaus. But the real prize is *stability* — the next figure shows it.
13. (code) **Figure A — the variance, drawn.** Several single bootstrap-tree boundaries (jagged, each
    different) vs the smooth bagged boundary (e.g. a small-multiples grid of single trees + the bag).
14. (md) **Read the figure (A).** Each pale boundary is one tree on one bootstrap — confident, jagged,
    and *different* (that difference IS the variance). The bold boundary is the vote: it drops each
    tree's idiosyncratic spikes and keeps what they agree on — smooth and stable. Averaging made no
    single tree better; it made the **committee** steadier.
15. (code) **Figure B — variance shrinks with B.** Test accuracy and run-to-run std (across 20 bagging
    seeds) vs number of trees: std 0.047 (B=1) → 0.009 (B=25) → 0.005 (B=100); accuracy 0.894 → 0.929.
16. (md) **Read the figure (B).** Two stories: accuracy climbs then **plateaus** (diminishing returns —
    past ~25 trees, more barely move the mean), while the std falls **~9×** — the answer stops depending
    on the luck of the resample. **This is variance reduction:** averaging B noisy predictions shrinks
    the spread roughly like σ²/B. More trees never hurt accuracy — they only cost compute.
17. (md) **Intuition — this is (almost) a random forest.** Bootstrap + many trees + vote = **bagging**
    (Breiman 1996). A random forest adds one twist (NB 2). Let's confirm the library matches us.
18. (code) **Parity.** `RandomForestClassifier(n_estimators=200, max_features=None, random_state=0)` →
    test **0.9333** = our hand-bag(200). The default `RandomForestClassifier()` (`max_features='sqrt'`)
    → **0.900** here.
19. (md) **Read the parity.** The library is not magic: `max_features=None` *is* the bagging we built.
    The default forest adds a second idea — each split sees only some features — which we take up in
    **NB 2**; on this 2-feature toy it changes little (even dips), but on real many-feature data it is
    what makes a forest a *forest*.
20. (md) **Your turn** (3 tiered) — *easy:* given three trees' votes on a point, give the ensemble
    prediction; *medium:* bootstrap a small set by hand, list repeated and omitted points, estimate the
    omitted fraction; *harder:* sweep B = 1, 2, 5, 10, 25, 50, 100, plot test accuracy and its
    run-to-run std (over several seeds), and find where diminishing returns set in.
21. (md) **What you built** — bootstrap resampling by hand; majority-vote aggregation; measured variance
    reduction (single-tree std 0.031 → bagged std 0.005, ~9×); accuracy 0.878 → 0.933; the parity
    hand-bag == `RF(max_features=None)`. **Vocabulary:** ensemble · bootstrap (sampling with
    replacement) · bagging (bootstrap aggregating) · majority vote · variance reduction · diminishing
    returns.
22. (md) **Going further (optional)** — the σ²/B sketch: for B *independent* estimators each of
    variance σ², the mean has variance σ²/B; real trees are correlated, so the gain is less than 1/B —
    the exact correction (the correlation ρ) is **NB 2**. **References:** Breiman 1996 (bagging,
    DOI 10.1007/BF00058655); Breiman 2001 (random forests, DOI 10.1023/A:1010933404324); ESL §8.7 /
    §15; ISLR §8.2. `Previous: Chapter 05 — Support Vector Machines (NB 5)` · `Next: 02 — The "random"
    in the forest: decorrelating the trees`.

## Honest scoping (stated in the notebook)

- **Averaging cancels variance, not bias** — bagging steadies a flexible (low-bias) learner; it cannot
  fix a wrong model class. Stated in "What you built" / Going further.
- **Feature subsampling is OFF here** — this is bagging; the "random" twist is NB 2. The default-RF
  0.900 < hand-bag 0.933 on moons is flagged as a hook, not a contradiction.
- **Diminishing returns** — more trees steady the estimate but never raise accuracy without bound; the
  cost is compute/memory.
- **The single-tree 0.878 is a continuity anchor** with ch 04 (same split), not a re-derivation.

## Verification

Build via `uv run python - < <scratchpad>/build_ch06_nb1.py` (stdin). Re-measure at build: single tree
train 1.000 / test 0.878 / bootstrap std 0.031; hand-bag B-sweep 0.900→0.933; run-to-run std
0.0465→0.0053; parity hand-bag(200) == RF(max_features=None) == 0.9333, RF(sqrt) 0.900. Runs
top-to-bottom (nbconvert to scratchpad; tracked file **output-free**); **banned-word scan over the JSON
real cell text** = 0; `check_no_hardcoded_hex` passes; `gen_llms_txt` re-run; `pytest` 19 (no `src/`
change); `ruff` clean. Both reviewers PASS (no BLOCK); Rémy validates visually; commit
`feat(06_random_forest): notebook 01 — the wisdom of trees: averaging cuts variance`; merge
`notebook → chapter`.
