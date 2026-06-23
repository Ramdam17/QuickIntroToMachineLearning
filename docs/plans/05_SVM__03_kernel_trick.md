# Notebook plan — 05_SVM / 03_kernel_trick

> Status: **APPROVED** (2026-06-22, by Rémy; NB plans are validated by Rémy alone — the two reviewers
> gate the *built* notebook, not this plan). Drives the build. Numbers re-measured at build and
> reconciled into prose.

## Context

NB **3 of 5** — the chapter's showpiece, one concept: **the kernel trick.** NB 1–2 drew the widest
*straight* street. A straight street cannot separate **curved** data — two concentric rings are the
textbook case. The fix is to **lift** the points into a higher-dimensional space where they *do* become
linearly separable (add a coordinate `r² = x₁² + x₂²`), find the flat separating plane there, and read
the boundary back in the original plane as a **curve**. The trick: the whole optimization needs the
data **only through dot products**, so we can compute the dot product *in the lifted space directly* —
the **kernel** — and never build the lift. One substitution (dot product → kernel) turns the linear SVM
into a non-linear one. The learner builds the lift **by hand** on the rings, then watches an RBF kernel
reach the same curved boundary without ever forming the extra coordinate.

## Anchors (measured at plan time, sklearn **1.9.0**; re-measured at build)

Datasets: **`make_circles(n_samples=300, factor=0.4, noise=0.10, random_state=0)`** (the hero) and
**`make_moons(n_samples=300, noise=0.20, random_state=0)`** (the generality check), both standardized.
*(`make_circles` is **new course vocabulary** — used inline like `make_moons`, seed pinned; flagged as
a first-contact loader.)*

- **A line is hopeless on the rings:** linear `SVC` **CV 0.557** (≈ chance).
- **The lift `r² = x₁² + x₂²` separates them outright:** outer class r² mean **3.39** (range
  [1.96, 5.26]), inner class r² mean **0.61** (range [0.05, 1.48]) — the ranges do not overlap, so a
  **single threshold on r² alone** scores **acc 1.000** (at t ≈ 1.48). Lifting to `[x₁, x₂, r²]` and
  fitting a `LinearSVC` gives **train 1.000**, with the plane weights dominated by the r² term
  (`w ≈ [0.06, −0.08, −1.52]`) — i.e. the separating plane is **~horizontal in r²** (a height cut).
- **The RBF kernel reaches the same boundary without building r²:** RBF `SVC` **CV 0.997** (38 support
  vectors); its 2-D decision boundary is a **closed curve** (a circle) around the inner ring.
- **The kernel is a choice, and its shape must match the data:** polynomial `SVC` **degree 2 → CV
  1.000** (the rings are a degree-2 form `x₁²+x₂²`), **default degree 3 → CV 0.613** (barely above
  the linear chance line), **degree 4 → 0.997**. Odd degrees miss the radial structure — the degree
  must match the geometry (a real kernel-choice lesson, not a simplification).
- **Generality (not just circles):** on `make_moons`, linear `SVC` **CV 0.840** vs RBF **CV 0.970** —
  the set ch 03's straight line could not fit, carved by the kernel.

## Library / figures

- **No `src/` change** — reuse `viz.plot_svm_decision` (from NB 1) for the RBF and polynomial
  boundaries, and `viz.use_course_style`; the 2-D→3-D lift is a one-off `matplotlib` 3-D figure in
  charter colours (`mpl_toolkits.mplot3d`). Sklearn: `SVC`, `LinearSVC`, `StandardScaler`,
  `StratifiedKFold`, `cross_val_score`, `make_circles`, `make_moons`. (`pytest` stays 19.)
- **Four figures** (each + "Read the figure"): **A** the rings in 2-D (no line works) → the **3-D lift**
  `(x₁, x₂, r²)` with a near-horizontal **separating plane**; **B** the **RBF** decision boundary in
  2-D (a closed curve, via `plot_svm_decision`); **C** the **polynomial degree 2 vs degree 3**
  boundaries on the rings (degree 2 succeeds, degree 3 nearly fails — the degree-matters beat); **D**
  the **RBF on `make_moons`** (the trick is general).

## Cell-by-cell (~22 cells; one concept; "Read the figure" after every figure)

1. (md) **Header** — `# 03 — The kernel trick`; *notebook 3 of 5*; warm welcome. **Prerequisites:**
   NB 1 (the maximum margin), NB 2 (the soft margin & `C`); ch 03 NB 6 (a straight line **underfits
   curved** truth — the door this opens). `make_circles` is introduced here. **What you'll be able to
   do:** explain why a straight margin cannot separate curved data; **lift** data with a feature map so
   it becomes linearly separable; state the **kernel trick** (the optimization needs only dot products,
   so replace the dot product with a kernel); use the **RBF** and **polynomial** kernels and say why the
   kernel (and its degree) must fit the data.
2. (code) **Imports + seed + style + data** — `make_circles` standardized; print the linear `SVC` CV
   **0.557** ("a straight line is no better than a coin flip on the rings").
3. (md) **Recap / footing** — NB 1–2 found the widest *straight* street, and ch 03's straight line
   could not fit curved data. Concentric rings are the sharpest example: **no** line separates an inner
   disk from an outer ring.
4. (md) **Intuition — lift it** — give each point a **new coordinate** that encodes what a line is
   missing. For rings, the telling quantity is the **distance from the centre**, `r² = x₁² + x₂²`:
   inner points have small `r²`, outer points large `r²`.
5. (code) **By hand — the lift separates them** — compute `r² = x₁² + x₂²`; show the two classes' r²
   ranges do not overlap (inner [0.05, 1.48], outer [1.96, 5.26]); a **single threshold** on r²
   (t ≈ 1.48) classifies every point correctly (acc 1.000).
6. (md) **Read the result** — one extra coordinate did what no line in the original plane could: the
   classes, hopelessly nested in 2-D, fall on two sides of a single cut in `r²`. A straight rule in the
   *lifted* space is a curved rule back in the original one.
7. (code) **Fig A — the 2-D→3-D lift** — left: the rings in 2-D (no line works); right: the points at
   `(x₁, x₂, r²)` in 3-D with the near-horizontal **separating plane** (the `LinearSVC` on `[x₁,x₂,r²]`,
   train 1.000).
8. (md) **Read the figure (A)** — lifted into three dimensions the two classes sit at different
   *heights* (`r²`), so a **flat plane** slices cleanly between them. That plane, projected back to the
   2-D plane, is the **circle** that separates the rings — a line up there, a curve down here.
9. (md) **Intuition — the kernel trick** — building the lift by hand worked, but it does not scale (the
   useful lifts are huge, sometimes infinite-dimensional). The escape: the SVM's solution depends on
   the data **only through dot products** between points. So we never build the lift — we compute the
   dot product *in the lifted space* directly, with a **kernel** `K(x, x′)`. The **RBF kernel**
   `K(x, x′) = exp(−γ‖x − x′‖²)` is the workhorse (`gamma` = its reach; tuned in NB 4).
10. (code) **The RBF reaches the same boundary** — `SVC(kernel="rbf")` on the rings (CV **0.997**),
    *without* forming `r²`; `viz.plot_svm_decision` shows the closed circular boundary.
11. (md) **Read the figure (B)** — the RBF drew the **same circular street** we built by hand, but in
    the original 2-D plane and without ever computing `r²`. The kernel did the lift implicitly — that is
    the whole trick: a curved boundary from the very same maximum-margin machinery.
12. (md) **Intuition — the kernel is a choice (and must fit the data)** — RBF is one kernel; the
    **polynomial** kernel `(γ x·x′ + c)^d` is another, with `degree` setting the curviness. The kernel's
    shape has to match the data's.
13. (code) **Fig C — polynomial degree 2 vs degree 3** on the rings — degree 2 (CV **1.000**) draws the
    circle; default degree 3 (CV **0.613**) nearly fails; print both, plot both boundaries.
14. (md) **Read the figure (C)** — degree 2 nails the rings because they are a **degree-2 form**
    (`x₁²+x₂²`); the default degree 3 misses the radial structure and barely beats the linear line.
    Choosing a kernel — and its degree — is a real modelling decision, not a default to accept blindly
    (we tune it honestly in NB 4).
15. (md) **Intuition — not just circles** — the trick is general. The two-moons set ch 03's line could
    not fit is carved just as cleanly.
16. (code) **Fig D — RBF on `make_moons`** — linear CV **0.840** vs RBF **0.970**; `plot_svm_decision`
    shows the curved boundary following the two crescents.
17. (md) **Read the figure (D)** — the same RBF kernel bends the boundary around the interlocking moons.
    Whatever the curve, the kernel lets the linear max-margin machinery follow it.
18. (md) **Honest limits** — the kernel is **implicit** (for the RBF there is no finite feature map to
    write down — the lift is infinite-dimensional); the kernel and its knobs (`degree`, `gamma`) **must
    fit the data**, and `gamma` controls under/over-fitting (tuned in NB 4); and, as always, the
    features must be on a comparable **scale** (named here, the headline of NB 5).
19. (md) **Your turn** (3 tiered) — *easy*: explain in one sentence why no straight line separates the
    rings; *medium*: compute `r² = x₁²+x₂²` for three given points and say which side of the threshold
    `t = 1.5` each lands on; *harder*: fit the polynomial kernel at degree 2 vs degree 3 on the rings
    and explain, from the geometry, why the degree matters.
20. (md) **What you built** — saw a straight margin fail on curved data; **lifted** the rings with
    `r² = x₁²+x₂²` so a flat plane separates them; learned the **kernel trick** (the optimization needs
    only dot products → replace them with a kernel); used the **RBF** to reach the curved boundary
    *without* building the lift, and the **polynomial** kernel where the degree must match the geometry.
    **Vocabulary:** feature map · the lift · linearly separable in a higher space · the kernel · the RBF
    kernel · `gamma` · the polynomial kernel and its `degree`.
21. (md) **Going further** — the dot-product-only structure (NB 1–2's "Going further") is what licenses
    the swap: any function that is a valid dot product in *some* space is a kernel (**Mercer's
    condition** — a positive-semidefinite kernel matrix). The RBF corresponds to an
    infinite-dimensional lift, which is why we never form it. The parameters `C`, `kernel`, `gamma`, and
    `degree` are the dials of the next notebook.
    **References:** Boser/Guyon/Vapnik 1992 (the kernel trick, DOI 10.1145/130385.130401); Cortes &
    Vapnik 1995 (DOI 10.1007/BF00994018); Schölkopf & Smola 2002 (*Learning with Kernels* — Mercer /
    valid kernels); ESL §12.3 (DOI 10.1007/978-0-387-84858-7); ISLR §9.3 (DOI 10.1007/978-1-0716-1418-1).
    `Previous: 02 — The soft margin and the cost C` · `Next: 04 — The estimator and its parameters`.

## Honest scoping (stated in the notebook)

- **A straight margin cannot do curves** — shown (linear CV 0.557 on the rings) before the lift is
  introduced; the kernel is the fix, motivated, not asserted.
- **The kernel is implicit** — for the RBF there is no finite feature map to build (infinite-dimensional);
  we lift by hand only with the toy `r²` to *show* the idea, then let the kernel do it.
- **The kernel/degree must fit the data** — degree 3 nearly failing on the rings is shown as the honest
  contrast, not hidden; `gamma`'s tuning is named and deferred to NB 4.
- **Scale** — named here (features standardized); the *why* is the headline of NB 5; not over-claimed.
- **The dual is intuition-level** — "needs only dot products" is stated and used; the Lagrangian /
  Mercer formalism stays a "Going further" pointer.

## Verification

Build via `uv run python - < <scratchpad>/build_ch05_nb3.py` (stdin). Re-measure at build: circles
linear CV 0.557 vs RBF 0.997 (n_SV 38); r² class ranges (inner [0.05,1.48], outer [1.96,5.26]) and the
1-D threshold acc 1.000 at t≈1.48; lifted LinearSVC train 1.000 (plane ~horizontal in r²); poly deg 2
1.000 / deg 3 0.613 / deg 4 0.997; moons linear 0.840 vs RBF 0.970. Runs top-to-bottom (nbconvert to
scratchpad; tracked file **output-free**, `--clear-output --inplace`); **banned-word scan over the JSON
real text** = 0; `check_no_hardcoded_hex` passes; `gen_llms_txt` re-run; `ruff` / `black` clean;
`pytest` 19 (no `src/` change). Both reviewers pass (no BLOCK); Rémy validates visually; commit
`feat(05_svm): notebook 03 — the kernel trick`; merge `notebook → chapter`.
