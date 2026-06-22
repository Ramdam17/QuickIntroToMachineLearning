# Notebook plan — 05_SVM / 01_maximum_margin

> Status: **APPROVED** (2026-06-22, by Rémy; NB plans are validated by Rémy alone — the two reviewers
> gate the *built* notebook, not this plan). Drives the build. Numbers re-measured at build and
> reconciled into prose.

## Context

NB **1 of 5** — the chapter's first fundamentals notebook, one concept: **the maximum margin and its
support vectors.** Logistic regression (ch 03) drew *one* separating line by minimizing log-loss, but
once the classes are separable, *many* lines separate them and it never asks **which line is best
placed**. The SVM's answer: pick the line with the **widest empty street** between the classes. The
learner builds this **by hand** — measure several separating lines' margins, find the widest, see that
the **two closest opposite-class points** (the support vectors) pin it, and that the boundary is their
**perpendicular bisector** — before `SVC(kernel="linear")` is called once to confirm the same street.

## Anchors (measured at plan time, sklearn **1.9.0**; re-measured at build)

Dataset: `make_blobs(n_samples=40, centers=[[-2.2,-2.2],[2.2,2.2]], cluster_std=0.7, random_state=0)`,
**standardized** (small `n` so the street and its support vectors are visible and reasoned about by
hand; cleanly separable so the **hard** margin is honest). Why blobs not penguins: the binary penguins
subset is **not** perfectly separable (a hard margin leaves 1 error), and the hard margin only coheres
on separable data — penguins returns in NB 2 as the near-separable real set that motivates the soft
margin.

- **Hard-margin `SVC(kernel="linear", C=1e6)`:** train acc **1.000**, `w = [0.9017, 0.7315]`,
  `b = −0.0997`, **‖w‖ = 1.1612** → **half-margin `1/‖w‖` = 0.8612**, **street `2/‖w‖` = 1.7224**.
- **Exactly 2 support vectors** (`n_support_ = [1, 1]`, indices 23 & 26): class-0 SV `[−0.5782, −0.518]`,
  class-1 SV `[0.7595, 0.5671]`.
- **The by-hand mechanic, verified exactly:** the distance between the 2 SVs `‖SV⁺ − SV⁻‖ = 1.7224`
  **equals** the street `2/‖w‖`; their midpoint lies **on** the boundary (`w·m + b ≈ 0`); the boundary
  is **perpendicular** to the SV segment (`cos(w, SV⁺−SV⁻) = 1.0000`) — i.e. the max-margin boundary
  **is the perpendicular bisector of the closest opposite-class pair**; functional margins at the SVs
  are exactly `±1`.
- **Contrast (a separator that does *not* maximize the street):** `LogisticRegression` also separates
  perfectly, but its nearest-point distance (its "margin") is **0.7736 < 0.8612** — it draws *a* line,
  not the *widest* street.

## Library / figures

- **`src/` addition (this notebook): `viz.plot_svm_decision(model, X, y, *, ax=None, resolution=300)`**
  — the SVM signature picture, reused in NB 1–4 (clears the "≥ 2×" bar). Fills the decision regions
  (via `predict`, like `plot_decision_boundary`), overlays the **street** as `decision_function`
  contours at **−1 / 0 / +1** (0 solid = boundary, ±1 dashed = margins), and **rings the support
  vectors** (`model.support_vectors_`). **Charter:** colours only from `ml_course.colors` (regions
  `CLASS_CYCLE`; contours `COLORS["text"]`/`COLORS["muted"]`; SV rings an open marker edged in
  `COLORS["highlight"]`). Add a **smoke test** in `tests/` (fit a tiny linear `SVC`, assert a Figure is
  returned and the support-vector count drawn matches `model.support_.size`) → **pytest 17 → 18**.
- **Reused:** `viz.use_course_style`; for the candidate-lines and SV-invariance figures, small in-notebook
  matplotlib in charter colours (no hardcoded hex). Sklearn: `make_blobs`, `StandardScaler`, `SVC`,
  `LogisticRegression`.
- **Three figures** (each + "Read the figure"): **A** several candidate separating lines with their
  margins shaded → the widest is the SVM's; **B** the max-margin street (`plot_svm_decision`) with the
  support vectors ringed; **C** support-vector invariance (delete a far point → boundary unchanged;
  move a support vector → boundary shifts).

## Cell-by-cell (~22 cells; one concept; "Read the figure" after every figure)

1. (md) **Header** — `# 01 — The widest street: the maximum margin`; *notebook 1 of 5*; warm welcome.
   **Prerequisites:** ch 03 NB 2 (the linear boundary `w·x+b=0`; `w` ⟂ the boundary); module 00 NB 04
   (the train/test split — used lightly), NB 05 (nearest-centroid, a *single* line, the visual foil).
   **What you'll be able to do:** explain what the margin is and why widest = most defensible; find the
   support vectors and the max-margin boundary **by hand** on separable data; state `margin = 2/‖w‖`;
   confirm a linear `SVC` finds the same street; explain why only the support vectors matter.
2. (code) **Imports + seed + style + data** — `np.random.seed(0)`; `from ml_course import viz, colors`;
   `viz.use_course_style()`; `make_blobs(...)` standardized; `X, y`; print shapes & that it is separable.
3. (md) **Recap / footing** — re-establish (briefly, genuinely): a linear classifier is a line
   `w·x + b = 0`; its **sign** gives the side/class; **`w` is perpendicular** to the line (ch 03). On
   *separable* data, **many** such lines classify every point correctly — logistic regression picked
   one by probability; here we ask a different question.
4. (md) **Intuition — which separating line?** — picture several lines that all separate the two
   clouds. They are not equally good: a line that grazes one cloud is fragile; a line with lots of room
   on both sides is robust. The SVM picks the line with the **widest empty street**.
5. (code) **Fig A — candidate lines + their margins** — plot the two clouds; draw 3 separating lines (a
   tilted one, one shifted toward a cloud, and the soon-to-be max-margin one); for each, compute its
   **margin** = the distance to the nearest point, and shade the street.
6. (md) **Read the figure (A)** — all three lines separate every point, but their streets differ; the
   widest street sits farthest from both clouds. *That* width is what the SVM maximizes.
7. (md) **Intuition — the margin, precisely** — the **margin** is the distance from the boundary to the
   nearest point on each side; the **street** is the band of width = twice that. Maximizing it is a
   robustness argument (a wider gap tolerates more noise) — Vapnik's structural-risk idea, stated, not
   derived.
8. (md) **Intuition — support vectors & the by-hand recipe** — the widest street is pinned by the
   **closest pair of opposite-class points**: the boundary is the **perpendicular bisector** of that
   pair, and the street's width is the distance between them. Those pinning points are the **support
   vectors**.
9. (code) **By hand** — find the closest opposite-class pair (the 2 SVs: indices 23 & 26); compute the
   **midpoint** (on the boundary), the **perpendicular bisector** (the boundary), and the **street
   width** `‖SV⁺ − SV⁻‖ = 1.7224`; the half-margin = 0.8612.
10. (md) **Read the result** — two points out of forty fix the entire boundary; the midpoint sits on
    the line, and the line is perpendicular to the segment joining them. Everything else is slack.
11. (md) **Intuition — `2/‖w‖`** — writing the boundary as `w·x + b = 0` with the canonical scaling
    `y(w·x+b) ≥ 1` (equality at the support vectors), the street width is exactly **`2/‖w‖`**, so
    **maximizing the margin = minimizing `‖w‖`** (the SVM's objective — stated; the optimizer is the
    library's job, as gradient descent was in ch 03).
12. (code) **The library finds the same street** — `SVC(kernel="linear", C=1e6).fit(Xs, y)`; print
    `w`, `‖w‖ = 1.1612`, `margin = 2/‖w‖ = 1.7224`, `support_` (== the by-hand pair 23 & 26); confirm
    parity: `cos(w, SV⁺−SV⁻)=1.0`, midpoint on the line, functional margins `±1`.
13. (code) **Fig B — the max-margin street** — `viz.plot_svm_decision(svc, Xs, y)`: regions, the solid
    boundary, the dashed `±1` margins, the support vectors ringed.
14. (md) **Read the figure (B)** — the `SVC` drew the same widest street we found by hand; the ringed
    points are the support vectors, and the dashed lines are the street's edges they sit on.
15. (md) **Intuition — only the support vectors matter** — because the street is pinned by those few
    points, moving or deleting a *non*-support point changes nothing; moving a *support* vector moves
    the whole boundary.
16. (code) **Fig C — support-vector invariance** — two panels: (left) delete a far point and re-fit →
    identical boundary; (right) nudge a support vector → the boundary shifts. (Charter colours.)
17. (md) **Read the figure (C)** — the boundary listens only to the support vectors; the crowd of
    interior points has no vote. This is what makes the SVM's solution *sparse* in the data.
18. (code) **Contrast — a line that separates but doesn't maximize the street** — fit
    `LogisticRegression`; compute its nearest-point distance (**0.7736**) vs the SVM's (**0.8612**);
    overlay both boundaries.
19. (md) **Read the result** — logistic regression separates the clouds perfectly too, but its line
    sits closer to the data (a narrower street). Same data, a different *criterion*: probability vs the
    widest margin.
20. (md) **Your turn** (3 tiered) — *easy*: given two drawn lines, say which has the wider margin and
    why; *medium*: from a plot, name the support vectors and explain how you know; *harder*: from
    figure C, explain why moving a non-touching point leaves the street where it is (argue from the
    shown invariance, not machinery from later notebooks).
21. (md) **What you built** — found the **maximum-margin** boundary by hand, identified its **support
    vectors**, derived **`margin = 2/‖w‖`**, confirmed a linear `SVC` finds the same street, and saw
    that only the support vectors hold it up. **Vocabulary:** margin · the street · support vectors ·
    maximum-margin boundary · `2/‖w‖`.
22. (md, optional) **Going further** — the hard-margin program `min ½‖w‖²` s.t. `yᵢ(w·xᵢ+b) ≥ 1`, and
    a one-line teaser that its solution depends on the data **only through dot products** — the door to
    the soft margin (NB 2) and the kernel trick (NB 3). **References:** Cortes & Vapnik 1995 (DOI
    10.1007/BF00994018); Boser/Guyon/Vapnik 1992 (DOI 10.1145/130385.130401); ESL §12 (DOI
    10.1007/978-0-387-84858-7); ISLR §9 (DOI 10.1007/978-1-0716-1418-1). `Previous: Module 04 — Decision
    Trees` · `Next: 02 — The soft margin & the cost C`.

## Honest scoping (stated in the notebook)

- **Hard margin needs separable data** — that is why NB 1 uses a separable synthetic set; real data
  (penguins) is near-separable and motivates the **soft margin** in NB 2 (named at the close).
- **The objective is stated, not derived** — `min ½‖w‖²` and `margin = 2/‖w‖` are presented
  geometrically; the QP/dual is a "Going further" teaser, not a derivation (the right altitude here).
- **`C=1e6` ≈ a hard margin** — flagged as "make violations very expensive"; the real `C` dial is NB 2.
- **Scale** — the data is standardized; *why* SVMs need it is named here and paid off in NB 5 (the
  scale trap), so this notebook does not overclaim it.

## Verification

Build via `uv run python - < <scratchpad>/build_ch05_nb1.py` (stdin). Re-measure at build: hard-margin
‖w‖ 1.1612 / margin 1.7224 / 2 SVs (idx 23, 26); the by-hand bisector parity (cos 1.0, midpoint on the
line, ‖SV⁺−SV⁻‖ = 1.7224); LogReg margin 0.7736 < 0.8612. Add `viz.plot_svm_decision` + a smoke test
(**pytest 17 → 18**). Runs top-to-bottom (nbconvert to scratchpad; tracked file **output-free**,
`--clear-output --inplace`); **banned-word scan over the JSON real text** (`just / simply / obviously /
trivially` + FR) = 0; `check_no_hardcoded_hex` passes; `gen_llms_txt` re-run; `ruff` / `black` clean.
Both reviewers pass (no BLOCK); Rémy validates visually; commit `feat(05_svm): notebook 01 — the
maximum margin`; merge `notebook → chapter`.
