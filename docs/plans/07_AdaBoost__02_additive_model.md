# Notebook plan — 07_AdaBoost / 02_additive_model

> Status: **APPROVED by Rémy & persisted** (no reviewer gate at plan stage). Building now → guards →
> two-reviewer gate (no BLOCK) → Rémy visual → commit → ff-merge to `chapter/07_AdaBoost`.

## Context

NB **2 of 5**, the second fundamentals notebook and **the chapter's hardest maths** (budgeted
intuition-first). **One concept: the additive model `F(x)=sign(Σ αₜ hₜ(x))` and *where α comes from*.**
NB 1 ran the reweighting loop and *used* α=ln((1−ε)/ε) as a given; here we (a) combine the weak
learners into one prediction — the weighted additive vote — and (b) *earn* α by deriving it as the
minimiser of the **exponential loss** (forward stagewise additive modelling). ~21 cells (6 code / 15
md), 3 figures.

## Anchors (re-measured at plan time, sklearn 1.9.0, seed 0; moons-0.20, n_train 280)

- **Boundary sharpens with T** (`AdaBoostClassifier`, test acc): T=1 **0.8667** (one cut = the stump),
  T=5 0.9250, T=10 **0.9417**, T=50 0.9417 (train 0.8429 → 0.9857). The weighted sum of stumps grows a
  curved boundary.
- **The additive model fits train perfectly at T=114** (staged train error: T=1 0.157 → T=50 0.014 →
  **T=114 0.000**). An ever-more-expressive sum (bridge to NB 3: does it overfit?).
- **Where α comes from** (round 1, uniform weights, ε₁=**0.1571**): the per-round exponential loss
  `L(α) = Σ wᵢ e^(−α yᵢ hᵢ) = (1−ε)e^(−α) + ε e^(α)` is minimised at **α\* = ½ ln((1−ε)/ε) = 0.8398**
  (grid argmin over [−0.5, 2.5] = **0.8400**; closed-form vs sum-form agree to **1.3e-15**). `L(0)=1.000`,
  `L(α\*)=0.728` (the bottom).
- **SAMME vs classic**: SAMME's α = ln((1−ε)/ε) = **1.6796 = 2α\***. It is *not* the per-round
  exp-loss minimiser, yet gives the **same classifier**: the reweighting uses only the renormalised
  *ratio* and the vote uses only the *sign* — both invariant to a global positive scale on α. (Neat
  check, optional: `L(2α\*) = L(0) = 1`.)
- **Multiclass SAMME** (K=3, `make_classification`): by-hand round-1 α = ln((1−ε)/ε) + **ln(K−1)** =
  **1.0788** == sklearn `estimator_weights_[0]` (|diff| **2e-16**). For K=2, ln(K−1)=0 → binary AdaBoost.

## Cell-by-cell (~21 cells; intuition → implementation → "Read the figure")

1. (md) **Header** — `# 02 — Weak learners and the additive model`; *Chapter 07 · Notebook 2 of 5*.
   Open: NB 1 built the reweighting loop and *used* α; two questions were left open — how do the stumps
   **combine** into one prediction, and **where does α=ln((1−ε)/ε) come from**? Both answered here.
   **Prerequisites:** NB 1 (reweighting, ε, α, the by-hand↔sklearn parity); ch 04 (the stump); **ch 03
   NB 3 (log-loss / cross-entropy — a *loss* is a number we drive down, and it punishes confident-wrong
   without bound)**; module 00 (decision boundary, accuracy). **What you'll be able to do:** write the
   weighted additive vote `F=sign(Σ αₜ hₜ)`; see the boundary sharpen as rounds add; **derive** α as
   the exponential-loss minimiser; reconcile sklearn's SAMME α with the classic ½ form; state the
   multiclass rule.
2. (md) **Where we are — from a loop to a model.** NB 1 gave a *procedure* (reweight, fit, repeat) and
   a number (α) we took on faith. A method needs a *model*: a function that turns a point into a
   prediction. AdaBoost's is the **weighted additive vote**. And α is not arbitrary — it falls out of a
   loss. This notebook earns both.
3. (code) **Setup** — imports, `use_course_style()`, moons-0.20 split (same as NB 1). Print shapes.
   Brief: fit `AdaBoostClassifier` at a few `n_estimators` for the boundary figure; keep one by-hand
   round-1 stump for the derivation.
4. (md) **The weighted additive vote.** Define `F(x) = Σₜ αₜ hₜ(x)`, predict **sign(F)**. The reveal:
   the **same α** that reweighted points in NB 1 is each learner's **vote weight** — a confident weak
   learner (small ε, large α) speaks louder. Contrast: bagging takes an *equal* vote; boosting takes a
   *weighted* one. (One honest line: we *use* this vote here; its α is *derived* below.)
5. (code) **Figure A — boundary sharpens with T** — `plot_decision_boundary` for
   `AdaBoostClassifier(n_estimators=1, 10, 50)`, triptych; titles carry test acc 0.867 / 0.942 / 0.942.
6. (md) **Read the figure (A).** T=1 is the single straight cut (the stump). By T=10 the weighted sum
   of cuts bends into a rough curve; by T=50 it tracks the two crescents smoothly. No stump moved — the
   **sum** grew expressive. A complex boundary, built from simple weighted pieces.
7. (md) **Weak learner, precisely.** A weak learner only needs ε < ½ (better than chance) ⇒ α > 0.
   Why summing them works: each `αₜ hₜ` is a small correction; the additive model is a flexible
   function assembled from simple terms (like adding terms to a series). The art is *choosing* each term.
8. (md) **Where does α come from? Re-lay the idea of a loss (ch 03).** A **loss** is a single number
   we drive down; in ch 03, log-loss punished a confident-wrong prediction without bound (unlike
   squared error, which caps out). AdaBoost minimises a close cousin: the **exponential loss**.
9. (code) **Figure B — exponential loss as a picture** — plot `exp(−margin)` against the 0/1 step loss
   over margin = y·F ∈ [−2, 2].
10. (md) **Read the figure (B).** Margin = y·F: positive when right, negative when wrong, larger when
    more confident. `exp(−margin)` is ~0 for confident-correct and **blows up exponentially** for
    confident-wrong — a smooth, differentiable **surrogate** for the jagged 0/1 step. (Flag: that
    exponential blow-up on wrong points is exactly why *mislabeled* points will dominate — NB 3/5.)
11. (md) **Forward stagewise: add one term at a time.** We build F greedily: freeze `F_{t−1}`, add
    `α·h`, and choose (h, α) to most reduce the total exponential loss `Σ exp(−yᵢ(F_{t−1}+α hᵢ))`. For
    a *fixed* weak learner h with weighted error ε, which α is best?
12. (code) **Derive α (round 1).** With uniform weights, `L(α) = (1−ε)e^(−α) + ε e^(α)`; sweep α on a
    grid, find the argmin, compare to **½ ln((1−ε)/ε)**. Print ε=0.157, α\*=0.8398, grid argmin 0.840.
13. (code) **Figure C — the exp-loss bowl `L(α)`** — plot L(α) vs α; mark the minimiser α\*=0.84;
    annotate SAMME's α=1.68 (=2α\*).
14. (md) **Read the figure (C).** L(α) is a convex bowl; its bottom is **α\* = ½ ln((1−ε)/ε) = 0.84** —
    *this* is where α comes from, the loss-minimising step, not a guess. It has NB 1's behaviour built
    in: ε→0 ⇒ α→∞ (trust a great learner), ε=½ ⇒ α=0 (a coin flip earns no vote).
15. (md) **SAMME vs the classic ½.** sklearn's SAMME uses α = ln((1−ε)/ε) = **2α\*** (what NB 1 used).
    Why is doubling fine? The reweighting depends only on the **renormalised ratio**, and the final
    prediction only on the **sign** of F — both unchanged by a global positive factor. So the classic
    (½ ln) and SAMME (ln) build the **same stumps** and the **same classifier**; the 2 is bookkeeping.
16. (code) **Multiclass SAMME (K=3).** On a 3-class set, by-hand round-1 α = ln((1−ε)/ε) + **ln(K−1)**;
    print it (1.0788) against sklearn `estimator_weights_[0]` (match to 2e-16).
17. (md) **Read the result.** SAMME generalises to K classes by adding **ln(K−1)**; for K=2 that term
    is 0, recovering binary AdaBoost. The "better than chance" bar rises to ε < 1 − 1/K (else α < 0).
18. (md) **Honest scoping.** (a) Exponential loss is a **surrogate** for the 0/1 error — chosen for its
    clean minimiser, not handed down; its exponential penalty on negative margins is the seed of
    AdaBoost's **noise-sensitivity** (NB 3/5). (b) **Greedy** forward-stagewise ≠ global optimum: we
    never go back and refit earlier stumps. (c) The classic/SAMME factor is convention, not a different
    algorithm. (d) The additive model drove **train** error to 0 by T≈114 — expressive, but whether
    that generalises is NB 3.
19. (md) **Your turn** (tiered) — *easy:* from Figure A, at what number of rounds does the boundary
    first look *curved* rather than a single straight cut? *medium:* compute α\* = ½ ln((1−ε)/ε) and the
    SAMME α for ε = 0.1 and ε = 0.3; confirm SAMME = 2·classic each time. *harder:* show that the SAMME
    step α = 2α\* gives the **same** exponential loss as taking no step (`L(2α\*) = L(0)`), yet yields
    the **same classifier** as the minimiser α\* — resolve the apparent paradox (hint: the renormalised
    reweighting ratio and the sign of the vote are both scale-invariant).
20. (md) **What you built** + vocabulary — additive model · weighted vote · weak learner (ε<½) ·
    exponential loss · surrogate loss · forward stagewise additive modelling · the α derivation ·
    SAMME vs classic ½ · multiclass +ln(K−1).
21. (md) **References** — Freund & Schapire 1997 (DOI 10.1006/jcss.1997.1504); Friedman, Hastie &
    Tibshirani 2000 — the statistical/exp-loss view (DOI 10.1214/aos/1016218223); Zhu et al. 2009 SAMME
    (DOI 10.4310/SII.2009.v2.n3.a8); ESL §10.1–10.4 (DOI 10.1007/978-0-387-84858-7). `Previous: 01 —
    Boosting intuition · Next: 03 — Learning rate, rounds, and overfitting behaviour.`

## `src/` & guards

No `src/` change (reuse `use_course_style`, `plot_decision_boundary`; the exp-loss picture and the
L(α) bowl are notebook-local matplotlib). **pytest stays 20.** Build via `uv run python - < build`;
banned-word JSON scan = 0; ruff/black clean; no hardcoded hex; output-free; nbconvert from project cwd
on a scratchpad copy; `gen_llms_txt`; both reviewers (no BLOCK) + Rémy visual before commit.
