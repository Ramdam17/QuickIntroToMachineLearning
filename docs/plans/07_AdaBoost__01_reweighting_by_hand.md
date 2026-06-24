# Notebook plan — 07_AdaBoost / 01_reweighting_by_hand

> Status: **APPROVED by Rémy & persisted** (no reviewer gate at plan stage). Building now → guards →
> two-reviewer gate (no BLOCK) → Rémy visual → commit → ff-merge to `chapter/07_AdaBoost`.

## Context

NB **1 of 5**, the first **fundamentals** notebook. **One concept: the AdaBoost reweighting loop, by
hand** — boosting trains weak learners *sequentially*, each re-focused on the previous ensemble's
mistakes via sample weights. The whole notebook is by hand; the only library AdaBoost call is the
parity check at the end. ~24 cells (9 code / 15 md), 2 figures.

## Anchors (re-measured at plan time, sklearn 1.9.0, seed 0; through-line set)

`make_moons(n_samples=400, noise=0.20, random_state=0)`, 70/30 stratified (seed 0) → n_train **280**
(140/140), n_test **120**.
- Single stump: train **0.8429**, **test 0.8667** (one cut, feature 1 @ 0.136) — weak.
- Round 1: ε₁ **0.1571**, α₁ = ln((1−ε)/ε) **1.6796**; the 44/280 (16 %) misclassified points hold
  0.157 of the weight → **0.500** after one reweight.
- Round 2: ε₂ **0.2435** (> ε₁ — the reweighted problem is deliberately harder), α₂ **1.1338**.
- Running-ensemble **train error**: T=1 0.157 → T=3 0.082 → T=10 0.043 → T=50 **0.0143**.
- **Parity:** by-hand α == sklearn `estimator_weights_` (max |diff| over 50 rounds **1e-15**); by-hand
  staged test acc == sklearn (**0.9417** @ T=50; T=1/2/3 = 0.8667/0.8667/0.925 both).
- Misclassified-point weight grows by **×exp(α₁) ≈ 5.36** relative to a correct point (before renorm).

## Cell-by-cell (~24 cells; intuition → implementation → "Read the figure")

1. (md) **Header** — `# 01 — Boosting intuition: focus on the mistakes`; *Chapter 07 · Notebook 1 of
   5*. Warm open: ch 06 built a forest by averaging *independent* trees; now the opposite idea — weak
   learners built one after another, each fixing the last's mistakes. **Prerequisites:** the decision
   stump / one split (ch 04 NB 1–2); ensembles & why averaging helps (ch 06 NB 1); train/test split &
   accuracy (module 00 NB 4, 6). **What you'll be able to do:** run AdaBoost's reweighting loop by
   hand; explain sample weights, the weighted error ε, the learner weight α; watch weak stumps compound
   into a strong classifier; recognise the by-hand loop *is* `AdaBoostClassifier`.
2. (md) **Where we are — from bagging to boosting.** 3-sentence ch 06 recap (parallel, independent
   bootstrap samples, *equal* votes → variance↓). Boosting = the opposite: sequential weak learners,
   each focused on what's still wrong, combined with *unequal* votes. "Bagging asks many independent
   opinions and averages; boosting builds one, sees where it errs, and corrects." Small contrast table
   (parallel/independent/equal-vote vs sequential/dependent/weighted-vote).
3. (code) **Setup** — imports, `use_course_style()`, `make_moons(400, 0.20, 0)`, 70/30 stratified
   split (seed 0); print shapes (280 = 140/140, test 120); scatter the training set.
4. (md) **Read the data** — two interleaving crescents, balanced; no single straight cut separates
   them — exactly why one weak learner won't be enough.
5. (md) **The weak learner: a decision stump.** Re-lay (ch 04): stump = depth-1 tree = one yes/no
   question on one feature; the canonical *weak* learner (barely better than chance on a hard problem).
6. (code) **A single stump** — `DecisionTreeClassifier(max_depth=1)`; print train/test acc
   (0.843/0.867); plot its boundary (`plot_decision_boundary`).
7. (md) **Read the figure** — one cut at feature-1 ≈ 0.14; ~87 % right but a whole crescent missed. A
   stump alone is weak. Question: can many stumps, each focused on the previous misses, add up to strong?
8. (md) **Intuition — the reweighting loop (in words).** Give each point a weight (start equal); fit a
   weak learner that minimises the *weighted* error; *increase* the weight on what it got wrong so the
   next learner attends to it; repeat. Each learner also earns a vote weight α (better learner → louder
   vote). Foreshadow: the same α does double duty (NB 2).
9. (code) **Round 1 by hand** — uniform wᵢ=1/n; fit stump with `sample_weight=w`; ε = Σwᵢ𝟙[miss]/Σwᵢ;
   **α = ln((1−ε)/ε)**. Print ε₁ 0.157, α₁ 1.680.
10. (md) **Read the result** — ε₁≈0.16 (>chance ⇒ α₁>0); α grows as ε shrinks (near-perfect → big vote;
    coin-flip ε=0.5 → α=0). One honest sentence: this is sklearn's **SAMME** α; an older form carries a
    **½** and gives the same classifier — why, in NB 2.
11. (code) **The reweighting step** — wᵢ ← wᵢ·exp(α·𝟙[miss]); renormalise. Print: the 44 misses held
    0.157 of the weight → **0.500** after reweighting.
12. (md) **Read the result** — half the total weight now sits on round 1's mistakes; the next stump is
    *paid* to fix them. "Focus on the mistakes," made literal.
13. (code) **Round 2 by hand** — fit a stump on the reweighted data; ε₂ 0.244, α₂ 1.134.
14. (md) **Read the result** — ε₂ (0.24) > ε₁ (0.16): not failure — we deliberately made the data
    harder (easy points downweighted), so even a good stump scores worse; higher ε ⇒ smaller α (less
    vote), which is correct.
15. (code) **Figure A — the reweighting story** — 3 panels (after rounds 1/2/3): training points,
    **size ∝ weight**, coloured by class, each round's stump split drawn (notebook-local matplotlib +
    `ml_course.colors`).
16. (md) **Read the figure (A)** — the heavy dots migrate to the boundary's hard cases; each new stump
    cuts where the weight piled up. The committee spreads its attention where it's needed.
17. (code) **The full loop** — wrap into `boost(Xtr, ytr, T)` → stumps, αs, running vote
    `F(x)=Σ αₜ hₜ(x)`; run T=50; track running-ensemble **train error** each round.
18. (code) **Figure B — train error vs rounds** — line plot, running-ensemble train error T=1..50
    (0.157 → 0.014). (notebook-local.)
19. (md) **Read the figure (B)** — each stump alone ~84 % on train; chained, the ensemble's train error
    falls 16 %→~1 %. Weak learners, combined the boosting way, become strong. Caveat sentence: driving
    *training* error down isn't the whole story — generalisation & the noise question are NB 3.
20. (code) **Parity — this is AdaBoost** — `AdaBoostClassifier(n_estimators=50, random_state=0)`; print
    `estimator_weights_[:5]` == our α[:5] (max diff 1e-15) and `staged_score` test acc == ours (0.9417
    @ T=50).
21. (md) **Read the result** — the by-hand loop reproduces `AdaBoostClassifier` to machine precision:
    same learner weights, same predictions. sklearn just runs this loop fast — you know what's inside.
22. (md) **Honest scoping** — (a) α is sklearn's SAMME form; classic Freund–Schapire ½·ln is the same
    classifier (NB 2 derives it). (b) Boosting attacks **both bias and variance** by chasing hard cases
    — which is *also* why it's sensitive to mislabeled points (a flipped label looks eternally hard and
    hoards weight); met head-on in NB 3 & NB 5. (c) We drove *training* error down; generalisation is NB 3.
23. (md) **Your turn** (tiered) — *easy:* set `n_estimators` to 1 and 5 in the parity cell; at how many
    stumps does boosted test acc first beat the single stump's 0.867? *medium:* compute α for ε = 0.01,
    0.25, 0.5 by hand; explain what α=0 (ε=0.5) says about a coin-flip learner. *harder:* show a
    misclassified point's weight grows by ×exp(α₁) ≈ 5.36 relative to a correct one (derive from
    `exp(α·𝟙[miss])`, check against the printed weights).
24. (md) **What you built** + **References.** Recap: sample weights, weighted error ε, learner weight α,
    reweight-and-repeat, the weighted vote — matched `AdaBoostClassifier` exactly. **Vocabulary:** weak
    learner · sample weights · weighted error ε · learner weight α · reweighting · sequential ensemble ·
    boosting vs bagging. **References:** Freund & Schapire 1997 (DOI 10.1006/jcss.1997.1504); Zhu et al.
    2009 SAMME (DOI 10.4310/SII.2009.v2.n3.a8); ESL §10.1 (DOI 10.1007/978-0-387-84858-7). `Previous:
    Chapter 06 — Random Forests · Next: 02 — Weak learners and the additive model`.

## `src/` & guards

No `src/` change (reuse `use_course_style`, `plot_decision_boundary`; the reweighting figure and the
train-error line are notebook-local). **pytest stays 20.** Build via `uv run python - < build`; banned-word
JSON scan = 0; ruff/black clean; no hardcoded hex; output-free; nbconvert from project cwd on a
scratchpad copy; `gen_llms_txt`; both reviewers (no BLOCK) + Rémy visual before commit.
