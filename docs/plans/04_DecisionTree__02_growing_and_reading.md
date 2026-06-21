# Notebook plan — 04_DecisionTree / 02_growing_and_reading

> Status: **APPROVED** (2026-06-20, by Rémy; notebook plan validated by Rémy alone — the two reviewers
> gate the *built* notebook). Numbers re-measured at build. Drives the NB-2 build in `docs/WORKFLOW.md`.

## Context

NB **2 of 5** — one concept: **recursive greedy growth, and reading the tree**. NB 1 found a single
best split (`flipper ≤ 206`); a tree is nothing more than that move **repeated** on each child. We
grow a depth-2 tree **by hand** (re-split each child of NB 1's root), read it as a **flowchart**,
trace a penguin down to its leaf, watch it carve the plane into **axis-aligned boxes**, and confirm
the by-hand tree **is** `DecisionTreeClassifier(max_depth=2)`. We stop at depth 2 on purpose; *when*
to stop (overfitting, pruning, the depth dial) is NB 3. Penguins binary subset, raw units.

## Anchors (measured at plan time, sklearn **1.9.0**; re-measured at build)

- **Root** (from NB 1): `flipper ≤ 206`, Gini decrease 0.4732. Its two children:
  - **left** (`flipper ≤ 206`): n = 150, **149 Adélie / 1 Gentoo**, Gini 0.0132 → best 2nd split
    **`bill ≤ 47.20`** (decrease 0.0132, isolates the lone Gentoo);
  - **right** (`flipper > 206`): n = 124, **2 Adélie / 122 Gentoo**, Gini 0.0317 → best 2nd split
    **`bill ≤ 40.85`** (decrease 0.0157, isolates one Adélie).
- **The 4 leaves** (depth-2): `flipper≤206 & bill≤47.20` → **149 A / 0 G** (Adélie, pure); `flipper≤206
  & bill>47.20` → **0 A / 1 G** (Gentoo, pure); `flipper>206 & bill≤40.85` → **1 A / 0 G** (Adélie,
  pure); `flipper>206 & bill>40.85` → **1 A / 122 G** (Gentoo, Gini 0.0161 — the one remaining error).
- **Train accuracy 0.9964** (273/274); **by-hand depth-2 predictions are bit-identical to
  `DecisionTreeClassifier(max_depth=2)`** (parity).
- **The one error: row 128** — an Adélie with **bill 44.1 mm, flipper 210 mm** (a long-flippered
  Adélie) falls in the `flipper>206 & bill>40.85` Gentoo leaf. An honest overlap case, not a bug.
- **CV depth-2 0.9855 > full tree 0.9818** — the unpruned tree already generalizes slightly *worse*
  (the first whiff of overfitting → NB 3).
- **Depth 3** → 5 leaves but **train accuracy unchanged at 0.9964**: the extra split (right branch,
  `bill ≤ 44.25`) separates points that are *both* still predicted Gentoo. More depth ≠ more accuracy
  here — the dial NB 3 tunes.
- **Greedy ≠ optimal**: each split is the locally best one; the globally optimal tree is NP-hard
  (Hyafil & Rivest 1976), so trees are grown greedily, top-down.

## Library / figures

- **No `src/` change** (pytest stays 17). Reuse `viz.use_course_style`, **`viz.plot_decision_boundary`**
  (Fig B — the axis-aligned boxes), `ml_course.colors` (`CLASS_CYCLE`, `COLORS`). Re-define the small
  `gini` + best-split-on-a-subset helpers in-notebook (standalone; same as NB 1's mechanism).
  `DecisionTreeClassifier(max_depth=2)` for the parity + the boundary figures.
- **Fig A — a custom charter-coloured flowchart** drawn in-notebook (matplotlib boxes + arrows): root
  `flipper ≤ 206?` → two internal nodes (`bill ≤ 47.2?`, `bill ≤ 40.85?`) → four leaf boxes **filled
  by majority class** (`CLASS_CYCLE[0]` Adélie / `CLASS_CYCLE[1]` Gentoo), each labelled with its class
  and counts; internal/root boxes neutral (charter `grid`/`text`); arrows annotated *yes* / *no*. This
  keeps the tree's class colours **consistent** with the scatter/boundary figures (sklearn's
  `plot_tree` fills with its own blue/orange, which would clash; mentioned in *Going further* as the
  built-in alternative). *(Fallback if the custom layout proves fiddly: `plot_tree(filled=False)`,
  structure-only. The `viz.plot_decision_tree` helper question from the chapter plan is **deferred** —
  NB 2 uses a one-off; revisit at NB 5 if its depth-3 tree wants the same.)*

## Cell-by-cell (~20 cells; by hand before the library; "Read the figure" after every figure)

1. (md) **Header** — `# 02 — Growing a tree, and reading it`; *notebook 2 of 5*; warm welcome.
   **Prerequisites:** NB 1 (impurity & the best single split — the move we now repeat); module 00 —
   the train/test split (NB 04), accuracy (NB 06). **What you'll be able to do:** grow a small tree by
   repeating the best-split rule; read a fitted tree as a flowchart of questions; trace a sample to its
   leaf and predict; see the axis-aligned boxes a tree carves; explain why growth is greedy.
2. (code) **Imports + seed + style + data** — `numpy`, `DecisionTreeClassifier`, `from ml_course import
   datasets, viz`, `from ml_course.colors import CLASS_CYCLE, COLORS`; `np.random.seed(0)`;
   `use_course_style()`; load penguins (`X, y`), `y_bin = (y=="Gentoo")`. Re-define `gini(labels)` and a
   `best_split(mask)` helper (scan both features within a subset) — the NB-1 mechanism, compact.
3. (md) **Recap — one split, from NB 1.** We measured that `flipper ≤ 206` is the single best question
   (Gini 0.4948 → 0.0216, a 0.4732 drop). A decision tree adds only one new idea: it asks the same
   kind of question **again**, of each group the first split made.
4. (md) **Intuition — recursion.** Each child is itself a node with its own class mix, so we can ask
   *its* best question, and split again. Repeat, and a tree grows. We stop at **depth 2** here (two
   questions deep); choosing where to stop is NB 3.
5. (code) **Grow depth-2 by hand** — call `best_split` on the left child (`flipper ≤ 206`) and the
   right child separately; print each child's best 2nd split (left `bill ≤ 47.20`, right `bill ≤
   40.85`) and the **four leaves'** compositions (counts + majority class + Gini).
6. (md) **Read the result** — the left child is almost all Adélie; its split peels off the lone Gentoo
   (`bill > 47.20`). The right child is almost all Gentoo; its split peels off one Adélie (`bill ≤
   40.85`). Four leaves: three perfectly pure, one (122 G / 1 A) nearly so.
7. (md) **Intuition — a leaf predicts its majority class; the tree is a flowchart.** Reading a tree is
   following yes/no questions from the top until you reach a leaf, whose majority class is the answer.
8. (code) **Fig A — the depth-2 tree as a flowchart** (custom charter drawing: root + two internal
   questions + four class-coloured leaves with counts).
9. (md) **Read the figure (A)** — start at the top, answer each question, move down the matching
   branch; the leaf you land in gives the prediction. Three leaves are single-colour (pure); one is
   Gentoo with a single Adélie mixed in.
10. (code) **Trace two penguins** — a clear Gentoo (long flipper, long bill) and the borderline Adélie
    of **row 128** (bill 44.1, flipper 210); print each one's path root → leaf → predicted class.
11. (md) **Read the trace** — the clear Gentoo drops into a pure Gentoo leaf. The borderline Adélie has
    an unusually long flipper (210 mm > 206) and a mid bill (> 40.85), so it lands in the Gentoo leaf —
    the tree's **one** training error. It sits in the species overlap, not a flaw in the method.
12. (md) **Intuition — the tree carves the plane into boxes.** Every leaf is an axis-aligned rectangle;
    a horizontal cut (`flipper`) then vertical cuts (`bill`) tile the feature space.
13. (code) **Fig B — decision regions, depth 1 vs depth 2** (`plot_decision_boundary` ×2, trees fit on
    the species labels so the legend reads Adélie/Gentoo): one cut → **2 boxes**, two levels → **4
    boxes**.
14. (md) **Read the figure (B)** — depth 1 splits the plane in two along `flipper = 206`; depth 2 cuts
    each half again along `bill`, giving four boxes whose colours are the leaf predictions. More depth →
    more, smaller boxes (the seed of NB 1's box-count note, and of NB 3's dial). The lone misclassified
    Adélie is the stray point inside the upper Gentoo box.
15. (code) **Parity + a first CV peek** — by-hand depth-2 predictions `==`
    `DecisionTreeClassifier(max_depth=2)` (identical), train accuracy 0.9964; then `cross_val_score`
    depth-2 **0.9855** vs the unpruned full tree **0.9818**.
16. (md) **Read the result** — the library grows the very same tree by the very same greedy rule (the
    mechanism, not magic). And notice: the **unpruned** tree already scores a touch *worse* under
    cross-validation (0.9818 < 0.9855) — the first sign that more tree is not always better. NB 3 makes
    that precise.
17. (md) **Greedy, and "what one more level adds."** Each split is chosen as the locally best one;
    finding the globally best tree is NP-hard (Hyafil & Rivest 1976), so we grow greedily, top-down.
    Growing to depth 3 here adds a fifth leaf but leaves training accuracy at 0.9964 — the extra split
    separates points that were already both called Gentoo. Depth is a dial, and turning it up does not
    always buy accuracy (NB 3).
18. (md) **Your turn** (3 tiered) — *easy*: from the flowchart, predict the class of a penguin with
    flipper 195 mm, bill 50 mm; *medium*: take the right child (`flipper > 206`) and rerun `best_split`
    on it — confirm `bill ≤ 40.85`, and say which leaf a (flipper 215, bill 39) penguin reaches;
    *harder*: fit `DecisionTreeClassifier(max_depth=3)`, count the leaves, and explain why training
    accuracy did **not** improve over depth 2.
19. (md) **What you built** — a tree grown by **recursively** repeating the best-split move; a **leaf**
    that predicts its majority class; how to **read** a tree as a flowchart and **trace** a sample to
    its leaf; the **axis-aligned boxes** it carves; the parity with `DecisionTreeClassifier`; and the
    first hint that an unpruned tree can generalize worse. **Vocabulary:** recursion · greedy growth ·
    internal node · branch · leaf · majority-class prediction · flowchart · axis-aligned box · parity.
20. (md) **Going further (optional) + References** — `sklearn.tree.plot_tree` is the built-in tree
    renderer (it shades nodes by class/purity in its own palette); CART grows **binary**, top-down;
    greedy growth is standard because the optimal tree is NP-hard. **References:** Breiman et al. 1984
    (CART); Hyafil & Rivest 1976 (NP-completeness, DOI 10.1016/0020-0190(76)90095-8); ESL §9.2 (DOI
    10.1007/978-0-387-84858-7); ISLR §8.1 (DOI 10.1007/978-1-0716-1418-1). `Previous: 01 — A question
    that splits the data: impurity` · `Next: 03 — Overfitting & pruning: depth is the complexity dial`.

## Honest scoping (stated in the notebook)

- **We stop at depth 2 by choice** — *when* to stop (overfitting, pruning, the depth dial) is NB 3,
  flagged forward; the CV 0.9855 > 0.9818 peek is the hook, not the lesson.
- **Greedy ≠ optimal** is stated (NP-hard), and "more depth ≠ more accuracy" is shown (depth-3 adds a
  leaf, not accuracy) — honest, not oversold.
- **The one error is a real overlap case** (row 128), named, not hidden.
- **Parity is exact** (by-hand predictions identical to sklearn's), reaffirming the mechanism.
- Raw units; no standardization (scale-invariance, established NB 1); nothing tuned (NB 3/4).

## Verification

Build via `uv run python - < <scratchpad>/build_ch04_nb2.py` (stdin). Re-measure at build: left child
best split `bill ≤ 47.20`, right `bill ≤ 40.85`; 4 leaves 149/0, 0/1, 1/0, 1/122; train 0.9964; by-hand
== `DecisionTreeClassifier(max_depth=2)` (identical predictions); the error is row 128 (bill 44.1,
flipper 210, Adélie); CV depth-2 0.9855 vs full 0.9818; depth-3 = 5 leaves, train 0.9964. Runs
top-to-bottom (nbconvert to scratchpad; tracked file **output-free**, `--clear-output --inplace`);
**banned-word scan over the JSON real text** = 0; `check_no_hardcoded_hex` passes; `gen_llms_txt`
re-run; `pytest` 17 (no `src/` change); `ruff` clean. Both reviewers PASS (no BLOCK); Rémy validates
visually; commit `feat(04_decision_tree): notebook 02 — growing a tree, and reading it`; merge
`notebook → chapter`.
