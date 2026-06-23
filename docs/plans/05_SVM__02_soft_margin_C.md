# Notebook plan — 05_SVM / 02_soft_margin_C

> Status: **APPROVED** (2026-06-22, by Rémy; NB plans are validated by Rémy alone — the two reviewers
> gate the *built* notebook, not this plan). Drives the build. Numbers re-measured at build and
> reconciled into prose.

## Context

NB **2 of 5** — one concept: **the soft margin (slack) and the cost `C`.** NB 1 found the widest street
on *separable* data. Real data is rarely that clean: on the penguins (Adélie vs Gentoo) the two clouds
overlap by one stubborn point, so a **hard** margin has no solution. The fix is to **allow
violations** — let points sit inside the street, or even on the wrong side — and charge a cost for
each. That cost is governed by one dial, **`C`**: small `C` buys a wide, forgiving street; large `C`
insists on a narrow, strict one. The penalty a violation pays *is* the **hinge loss**, the close cousin
of Chapter 03's log-loss — which lets us line the two methods up. The learner computes the slack and
the hinge **by hand**, then sweeps `C` and watches the geometry move.

## Anchors (measured at plan time, sklearn **1.9.0**; re-measured at build)

Dataset: penguins binary subset (`load_penguins`: Adélie / Gentoo, `bill_length_mm`,
`flipper_length_mm`), **standardized** — real, and **near-separable** (so the soft margin is *needed*,
not optional). Labels mapped to `y ∈ {−1, +1}` for the hinge formula.

- **Hard margin is infeasible here:** `SVC(kernel="linear", C=1e6)` scores train **0.9964** — it cannot
  reach 1.000, one point stays misclassified. A hard margin (no violations allowed) has no solution →
  we must allow slack.
- **`C`-sweep (linear), measured:**
  | `C` | margin `2/‖w‖` | # support vectors | # slack > 0 | # misclassified | train | CV |
  |----|----|----|----|----|----|----|
  | 0.01 | 2.280 | 124 | 122 | 1 | 0.996 | 0.989 |
  | 0.1 | 1.248 | 44 | 43 | 2 | 0.993 | 0.993 |
  | 1 | 0.651 | 17 | 15 | 2 | 0.993 | 0.993 |
  | 10 | 0.462 | 8 | 8 | 2 | 0.993 | 0.993 |
  | 100 | 0.351 | 6 | 4 | 1 | 0.996 | 0.996 |
  | 1000 | 0.351 | 6 | 3 | 1 | 0.996 | 0.996 |
  As `C` grows the **street narrows** (2.28 → 0.35) and the support vectors **collapse** (124 → 6);
  accuracy stays ~flat (0.989–0.996) — on near-separable data **`C` sets the geometry, not the score**.
- **Hinge loss by hand at `C=1`** (`hinge = max(0, 1 − y·f(x))`, `y ∈ {−1,+1}`): a point well outside
  the street (`m = y·f = 8.57`) → **hinge 0**; a point inside the street on the right side
  (`m = 0.60`) → **hinge 0.40**; a point on the wrong side (`m = −0.31`) → **hinge 1.31**. At `C=1`
  there are 15 points with slack > 0 (the support vectors) and 2 misclassified; total hinge 6.55.
- **Hinge vs log-loss (functions of the margin `m = y·f`):** hinge `max(0, 1−m)` is **zero** once
  `m ≥ 1` (a flat "safe zone") and rises linearly for `m < 1`; log-loss `log(1 + e^{−m})` is **never**
  exactly zero (it always wants more confidence) but is smooth. Both punish confident-and-wrong; the
  flat zero of the hinge is what *creates* a margin. (ESL §12.3.)

## Library / figures

- **No `src/` change** — reuse `viz.plot_svm_decision` (from NB 1) for the two streets, and
  `viz.use_course_style`; the hinge-vs-log-loss curve and the margin/#SV-vs-`C` plot are one-off
  in-notebook figures in charter colours. Sklearn: `SVC`, `StandardScaler`, `StratifiedKFold`,
  `cross_val_score`. (`pytest` stays 19.)
- **Three figures** (each + "Read the figure"): **A** the **hinge loss vs log-loss** as functions of
  the margin `m = y·f` (the unifying-loss beat, tying to ch 03); **B** the street at **small `C` (0.1)**
  vs **large `C` (100)** side by side (`plot_svm_decision`), violators visible; **C** the **margin
  width and #support-vectors vs `C`** (log-`C` axis) — the two moving together.

## Cell-by-cell (~22 cells; one concept; "Read the figure" after every figure)

1. (md) **Header** — `# 02 — The soft margin and the cost C`; *notebook 2 of 5*; warm welcome.
   **Prerequisites:** NB 1 (the maximum margin & support vectors, `2/‖w‖`); ch 03 NB 3 (log-loss —
   the foil for the hinge loss); ch 01 NB 2 (the scale trap — named, paid off in NB 5). **What you'll
   be able to do:** explain why real data needs a *soft* margin; define **slack** and the **hinge
   loss**; say what the cost **`C`** trades and predict the effect of turning it; read the margin /
   support-vector count off a `C`-sweep; connect the hinge loss to ch 03's log-loss.
2. (code) **Imports + seed + style + data** — penguins Adélie/Gentoo standardized; `y ∈ {−1,+1}`;
   print counts; fit `SVC(kernel="linear", C=1e6)` and show train **0.9964** (hard margin leaves 1
   error).
3. (md) **Recap / footing** — NB 1's hard margin assumed the classes were *separable*. Here they are
   not: one Adélie sits among the Gentoos, so no street separates them cleanly. Demanding zero
   violations (a hard margin) is now impossible — we need a gentler rule.
4. (md) **Intuition — allow violations, at a cost** — let a point sit **inside** the street, or even on
   the **wrong side**, but **pay** for it. The amount it pays grows with how far it intrudes — that
   "how far" is the **slack**.
5. (code) **The offending point** — identify the misclassified point under the (near) hard margin and
   show it on the cloud — the one a hard margin cannot place.
6. (md) **Intuition — slack and the hinge loss** — define the **functional margin** `m = y·(w·x+b)`
   (positive & ≥ 1 = safely outside the street; between 0 and 1 = inside it; negative = wrong side).
   The price a point pays is the **hinge loss** `max(0, 1 − m)`: zero when it is safely outside, rising
   as it intrudes.
7. (code) **Hinge by hand** — at `C=1`, compute `m` and `hinge = max(0, 1−m)` for three points: one
   well outside (`m=8.57` → **0**), one inside the street (`m=0.60` → **0.40**), one on the wrong side
   (`m=−0.31` → **1.31**).
8. (md) **Read the result** — the hinge charges nothing to points that are already safely classified,
   and a linearly growing fee to those that intrude — exactly the "slack" each violator contributes.
9. (md) **Intuition — the same shape as log-loss** — Chapter 03 trained by log-loss, which also
   punished confident-and-wrong. Lining the two losses up against the margin makes the SVM's choice
   visible.
10. (code) **Fig A — hinge vs log-loss** vs the margin `m`: `max(0,1−m)` and `log(1+e^{−m})`.
11. (md) **Read the figure (A)** — both fall as the margin grows and both punish confident mistakes,
    but the hinge hits **exactly zero** at `m=1` (a flat safe zone — a point past the street costs
    nothing) while log-loss never quite reaches zero (it always asks for more confidence). That flat
    zero is what gives the SVM a *margin* rather than a single line.
12. (md) **Intuition — the cost `C` is the dial** — the soft-margin objective is
    `½‖w‖² + C·Σ hinge`: the first term widens the street, the second pays for violations. **`C`** sets
    the exchange rate — small `C` makes violations cheap (a wide, forgiving street, more bias); large
    `C` makes them expensive (a narrow, strict street, toward the hard margin, more variance).
13. (code) **The `C`-sweep** — table of `margin = 2/‖w‖`, #support vectors, #slack>0, #misclassified,
    train, CV for `C ∈ {0.01, 0.1, 1, 10, 100, 1000}` (margin 2.28→0.35, SVs 124→6).
14. (md) **Read the result** — raising `C` narrows the street and leans on fewer points; the accuracy
    hardly moves (0.989–0.996). On this near-separable data **`C` is choosing the geometry** — the
    margin width and which points are support vectors — not rescuing the score. (Its accuracy bite is
    sharper with a kernel and noisier data — NB 4.)
15. (code) **Fig B — small `C` vs large `C`** — `viz.plot_svm_decision` for `C=0.1` and `C=100` side by
    side; the wide forgiving street (many points inside) vs the narrow strict one (few).
16. (md) **Read the figure (B)** — small `C` keeps a broad street and tolerates the points inside it;
    large `C` squeezes the street until almost nothing violates it. Same data, two very different
    boundaries — chosen by one number.
17. (code) **Fig C — margin & #support-vectors vs `C`** — `2/‖w‖` and the support-vector count against
    `C` on a log axis.
18. (md) **Read the figure (C)** — the two move together: a wider margin needs **more** support vectors
    to hold it, a narrow margin needs few. This is the bias–variance dial of NB 1's complexity language,
    now in support-vector-machine form.
19. (md) **Intuition — honest limits** — on near-separable data `C` mostly moves the geometry; where it
    *also* moves accuracy is on harder, kernelized problems (NB 4). And the whole picture assumes the
    features are on a comparable scale — *named here, demonstrated in NB 5*.
20. (md) **Your turn** (3 tiered) — *easy*: predict whether small or large `C` gives the wider street,
    and why; *medium*: from the `C`-sweep, say at which `C` the model leans on the fewest points, and
    what that costs; *harder*: compute the hinge loss for three points given their `m = y·f` values, and
    say which are support vectors.
21. (md) **What you built** — saw that real data needs a **soft margin**; defined **slack** and the
    **hinge loss** and computed them by hand; understood the objective `½‖w‖² + C·Σ hinge` and what
    **`C`** trades; read the margin and support-vector count off a `C`-sweep; connected the hinge to ch
    03's log-loss. **Vocabulary:** slack · violation · the hinge loss · the cost `C` · soft margin ·
    the `C` dial.
22. (md) **Going further** — the soft-margin program
    `min ½‖w‖² + C Σ ξᵢ` s.t. `yᵢ(w·xᵢ+b) ≥ 1 − ξᵢ, ξᵢ ≥ 0`, with `ξᵢ = hinge`; still depends on the
    data only through dot products — the door to bending the boundary (the **kernel trick**, NB 3).
    **References:** Cortes & Vapnik 1995 (DOI 10.1007/BF00994018); Hastie/Tibshirani/Friedman ESL §12.3
    (the hinge loss as a regularized loss, DOI 10.1007/978-0-387-84858-7); James et al. ISLR §9.2 (DOI
    10.1007/978-1-0716-1418-1). `Previous: 01 — The maximum margin` · `Next: 03 — The kernel trick`.

## Honest scoping (stated in the notebook)

- **Real data needs slack** — the hard margin is shown to be infeasible (1 error) before the soft
  margin is introduced; the soft margin is the fix, not an optional extra.
- **On near-separable data, `C` sets the geometry, not the accuracy** — stated with the flat CV column;
  the accuracy consequence of `C` is deferred to the kernelized, noisier case (NB 4), not over-claimed.
- **The objective is stated, not derived** — `½‖w‖² + C·Σ hinge` and the slack constraints are given
  geometrically; the QP/dual stays a "Going further" teaser.
- **Scale** — named here (the features are standardized), the *why* paid off in NB 5; not over-claimed.

## Verification

Build via `uv run python - < <scratchpad>/build_ch05_nb2.py` (stdin). Re-measure at build: hard margin
train 0.9964 (1 error); the `C`-sweep table (margin 2.28→0.35, SVs 124→6, slack 122→3, accuracy
~flat); hinge at C=1 (0 / 0.40 / 1.31); hinge & log-loss curve shapes. Runs top-to-bottom (nbconvert
to scratchpad; tracked file **output-free**, `--clear-output --inplace`); **banned-word scan over the
JSON real text** = 0; `check_no_hardcoded_hex` passes; `gen_llms_txt` re-run; `ruff` / `black` clean;
`pytest` 19 (no `src/` change). Both reviewers pass (no BLOCK); Rémy validates visually; commit
`feat(05_svm): notebook 02 — the soft margin and the cost C`; merge `notebook → chapter`.
