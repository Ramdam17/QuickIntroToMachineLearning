# Notebook plan — 04_DecisionTree / 01_impurity_and_splits

> Status: **APPROVED** (2026-06-20, by Rémy; notebook plan validated by Rémy alone — the two reviewers
> gate the *built* notebook). Numbers re-measured at build. Drives the NB-1 build in `docs/WORKFLOW.md`.

## Context

NB **1 of 5** — the chapter's first fundamentals notebook, **one concept: impurity & the best split**
— the single move a decision tree makes, done **by hand before any library**. We do not grow a tree
yet (that is NB 2's recursion); here we learn to **score one yes/no question** and find the best one.
On the binary penguins subset (Adélie / Gentoo, `bill_length_mm` + `flipper_length_mm`), in **raw
units** — a tree's split is a threshold, so standardization changes nothing (a first taste of
scale-invariance, the contrast with ch 01 / ch 03). This is the floor the chapter stands on: NB 2
recurses this move into a tree, NB 3 over-does it (overfitting), NB 4 is the real estimator.

## Anchors (measured at plan time, sklearn **1.9.0**; re-measured at build)

- **Root node** (n = 274; Adélie 151 / Gentoo 123; p(Gentoo) = 0.4489): **Gini = 1 − Σpₖ² = 0.4948**,
  **entropy = −Σpₖlog₂pₖ = 0.9925 bits**. By-hand == sklearn root impurity (0.4948).
- **Impurity shape:** Gini peaks at **0.5** (p = ½), entropy at **1 bit** (p = ½); both **0** at a pure
  node (p = 0 or 1).
- **Best single split** (scan thresholds, maximize impurity decrease = parent − sample-weighted
  children): **`flipper_length_mm` ≤ 206.0 → Gini decrease 0.4732** beats **`bill_length_mm` ≤ 43.25 →
  0.4044**. **Gini and entropy pick the *same* thresholds** (entropy info-gain: flipper 0.9069, bill
  0.7174) — the choice of impurity measure does not change the best question here.
- **Decrease-vs-threshold curve** is a clean single hump (flipper peaks ≈ 0.45 near 201–206; bill peaks
  ≈ 0.39 near 43).
- **The winning split's children** (`flipper ≤ 206`): **left** n = 150 → 149 Adélie / 1 Gentoo, Gini
  **0.0132**; **right** n = 124 → 2 Adélie / 122 Gentoo, Gini **0.0317**; weighted child Gini **0.0216**
  → decrease **0.4948 − 0.0216 = 0.4732**. One cut nearly separates the two species.
- **Parity:** `DecisionTreeClassifier(max_depth=1)` → root split `flipper ≤ 206`, root impurity 0.4948,
  **train acc 0.9891** — exactly the by-hand best split (the stump *is* one split).
- **Scale-invariance:** the split is the test "is flipper ≤ 206?"; rescaling the feature only rescales
  the threshold — the partition (and the tree) is identical. Stated here; measured in full in NB 4.

## Library / figures

- **No `src/` change** (pytest stays 17). Reuse `viz.use_course_style`, **`viz.plot_feature_histograms`**
  (Fig A). Three **one-off in-notebook figures** in charter colours (`ml_course.colors`,
  `CLASS_CYCLE`): impurity-vs-p curves (B), decrease-vs-threshold (C), the chosen split on the cloud +
  child class-mix bars (D). `DecisionTreeClassifier(max_depth=1)` is imported for the **parity** cell
  only (the by-hand mechanism comes first). Classes sort Adélie → class_a, Gentoo → class_b.

## Cell-by-cell (~22 cells; by hand before the library; "Read the figure" after every figure)

1. (md) **Header** — `# 01 — A question that splits the data: impurity`; *notebook 1 of 5*; warm
   one-line welcome. **Prerequisites:** module 00 — features & the feature space (NB 02), the
   train/test split (NB 04), accuracy (NB 06); chapter 03 NB 6 (the linear boundary that *underfits*
   curved data — the door to trees). **What you'll be able to do:** measure how class-mixed a group is
   (Gini / entropy); score a yes/no split by how much it *reduces* impurity; find the best split by
   scanning thresholds; recognise that this single move is what `DecisionTreeClassifier` does first.
2. (code) **Imports + seed + style + data** — `numpy`, `pandas`, `from sklearn.tree import
   DecisionTreeClassifier` (parity only), `from ml_course import datasets, viz`, `from
   ml_course.colors import COLORS, CLASS_CYCLE`; `np.random.seed(0)`; `viz.use_course_style()`;
   `X, y = datasets.penguins_xy()` (raw mm); print class counts (Adélie 151 / Gentoo 123).
3. (md) **Recap & footing** — the course has drawn *lines* (nearest centroid, logistic regression); a
   tree asks **yes/no questions** instead. Re-establish: a classifier carves the feature space; today
   we learn the **one move** a tree makes — a single split — and how it *chooses* it. Raw units on
   purpose (a tree is scale-invariant; we will see why). Nothing is fitted with a library until the
   very end (parity check).
4. (md) **Intuition — what makes a split good?** A node is a group of points; **mixed = bad, pure =
   good**. To choose a split we need a *number* for "how mixed" — that number is **impurity**.
5. (code) **Fig A — `plot_feature_histograms(df, ["bill_length_mm", "flipper_length_mm"], by="species")`**
   — how each feature separates the two species.
6. (md) **Read the figure (A)** — both features separate the species, but `flipper_length` splits into
   two cleaner humps with less overlap — a hint it will make the better single cut (we will *measure*
   this, not eyeball it).
7. (md) **Intuition — impurity: Gini & entropy.** Define both as functions of the class proportion p:
   **Gini = 1 − Σpₖ²** (expected mis-label rate), **entropy = −Σpₖlog₂pₖ** (bits of surprise). Both are
   0 at a pure node and largest at 50/50.
8. (code) **Gini & entropy by hand** (`gini(p)`, `entropy(p)`) + **Fig B — impurity vs p** (both curves
   over p ∈ [0, 1]).
9. (md) **Read the figure (B)** — Gini tops out at **0.5**, entropy at **1 bit**, both at p = ½; both
   fall to **0** at a pure node. Two measures, same shape — "how far from pure."
10. (code) **Root impurity on penguins** — apply by hand: **Gini 0.4948**, **entropy 0.9925 bits**;
    confirm `== DecisionTreeClassifier(max_depth=1).fit(X, y).tree_.impurity[0]` (0.4948). The library
    computes the same number.
11. (md) **Read the result** — the root is **close to maximally mixed** (0.4948, near the 0.5 ceiling):
    a 45/55 split of two species. A good question should drop this sharply.
12. (md) **Intuition — the best split = the biggest impurity drop.** A threshold split sends points to
    a **left** child (feature ≤ t) and a **right** child (> t). Score it by the **impurity decrease** =
    parent − the **sample-weighted** average of the two children's impurity (= *information gain* for
    entropy). Bigger drop = purer children = better question.
13. (code) **Scan `flipper_length` thresholds** (midpoints between sorted values), compute the decrease
    at each, and **Fig C — decrease vs threshold** (flipper, with `bill_length`'s curve overlaid); mark
    each peak.
14. (md) **Read the figure (C)** — each curve is a single hump; its **peak is the best split on that
    feature**. Flipper peaks higher (≈ 0.47 at **206**) than bill (≈ 0.40 at **43.25**).
15. (code) **Pick the winner across features** — print flipper best (≤ 206, decrease **0.4732**) vs
    bill best (≤ 43.25, **0.4044**); show the **entropy** info-gain picks the **same thresholds**
    (0.9069 / 0.7174).
16. (md) **Read** — **`flipper ≤ 206` is the best single question** (0.4732 > 0.4044), and Gini and
    entropy *agree* on it. A callback: ch 03's sigmoid favoured `bill_length`; the impurity criterion
    favours `flipper` — **the "best feature" depends on the question you ask.**
17. (code) **The winning split's children** — `flipper ≤ 206`: **left** 149 Adélie / 1 Gentoo (Gini
    **0.0132**), **right** 2 Adélie / 122 Gentoo (Gini **0.0317**); weighted **0.0216** → decrease
    **0.4948 − 0.0216 = 0.4732**. **Fig D — the cloud with the split line at flipper = 206 + the two
    children's class-mix bars.**
18. (md) **Read the figure (D)** — one horizontal cut nearly separates the two species: each child is
    **almost pure** (Gini ≈ 0.01–0.03). That near-purity *is* the large impurity decrease, made
    visible.
19. (code) **Parity + scale-invariance** — `DecisionTreeClassifier(max_depth=1)`: root split
    `flipper ≤ 206`, root impurity 0.4948, **train acc 0.9891** — exactly the split we found by hand
    ("the library's first move is the split we found by hand"). Note: the split is the test *is flipper
    ≤ 206?*; standardizing `flipper` only rescales the threshold — same partition. (Measured in full in
    NB 4.)
20. (md) **Your turn** (3 tiered) — *easy*: compute a node's Gini from given class counts; *medium*:
    scan `bill_length` yourself and confirm its best split (≤ 43.25), then split a child node further;
    *harder*: argue why a **pure** node has impurity 0 and a **50/50** node is maximal (and why entropy's
    maximum is 1 bit for two classes).
21. (md) **What you built** — impurity (Gini / entropy) as "how class-mixed"; the **impurity decrease**
    that scores a split; finding the **best split** by scanning; and the match to sklearn's stump.
    **Vocabulary box:** node · impurity · Gini · entropy · split · threshold · child node · impurity
    decrease / information gain · stump · scale-invariance.
22. (md) **Going further (optional) + References** — Gini vs entropy (rarely disagree; Gini is cheaper —
    no logarithm); CART makes **binary** splits; the optimal tree is NP-hard so growth is **greedy**
    (NB 2). **References:** Breiman et al. 1984 (CART); Quinlan 1986 (ID3 / information gain, DOI
    10.1007/BF00116251); ESL §9.2 (DOI 10.1007/978-0-387-84858-7); ISLR §8.1 (DOI
    10.1007/978-1-0716-1418-1). `Previous: Module 03 — Logistic Regression (NB 6)` · `Next: 02 —
    Growing a tree, and reading it`.

## Honest scoping (stated in the notebook)

- **No tree is grown** — this is one split, scored by hand; recursion is NB 2 (flagged forward).
- **Greedy is named, not yet exercised** — the "best split" is locally best; the globally optimal tree
  is NP-hard (NB 2 / references).
- **Scale-invariance is asserted with a one-line argument** here, **measured** (raw == standardized
  tree) in NB 4 — not over-claimed.
- **Gini & entropy agree on this data** — stated as a measured property here, not a general identity
  (NB 4 returns to criterion choice).
- Raw units throughout; no standardization; no `Pipeline`; nothing fitted until the parity cell.

## Verification

Build via `uv run python - < <scratchpad>/build_nb1.py` (stdin, to dodge the `/tmp/struct.py`
stdlib shadow). Re-measure at build: root Gini 0.4948 / entropy 0.9925; flipper ≤ 206 decrease 0.4732
vs bill ≤ 43.25 0.4044 (entropy picks the same thresholds); children Gini 0.0132 / 0.0317 → weighted
0.0216; stump root `flipper ≤ 206`, acc 0.9891. Runs top-to-bottom (nbconvert to scratchpad; tracked
file **output-free**, `--clear-output --inplace`); **banned-word scan over the JSON real text** = 0;
`check_no_hardcoded_hex` passes; `gen_llms_txt` re-run; `pytest` 17 (no `src/` change); `ruff` /
`black` clean. Both reviewers PASS (no BLOCK); Rémy validates visually; commit `feat(04_decision_tree):
notebook 01 — a question that splits the data: impurity`; merge `notebook → chapter`.
