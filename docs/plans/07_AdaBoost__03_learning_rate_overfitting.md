# Notebook plan — 07_AdaBoost / 03_learning_rate_overfitting

> Status: **APPROVED by Rémy & persisted** (no reviewer gate at plan stage). Building now → guards →
> two-reviewer gate (no BLOCK) → Rémy visual → commit → ff-merge to `chapter/07_AdaBoost`.

## Context

NB **3 of 5**, the last fundamentals notebook, **richer scope (chapter Decision A)**. **One declared
concept: how boosting controls its own complexity — the rounds × `learning_rate` trade-off, and what
it does to generalization.** We establish the learning-rate knob, then meet AdaBoost's two-faced
overfitting behaviour: **resistant** on clean data, **not immune** under label noise. Entirely on the
moons-0.20 through-line (2-D lets us *see* the boundary contort around noise). ~20 cells (5 code / 15
md), 3 figures.

## Anchors (re-measured at plan time, sklearn 1.9.0, seed 0; moons-0.20, n_train 280)

- **`learning_rate` scales α directly:** `estimator_weights_[0] = lr · ln((1−ε)/ε)` — measured
  **1.6796 / 0.8398 / 0.1680** for lr = 1.0 / 0.5 / 0.1 (ε₁ = 0.1571). So F = Σ (ν·αₜ) hₜ and the
  reweight uses ν·αₜ: smaller ν = smaller steps = more rounds needed.
- **Clean resistance** (lr = 1.0): staged **train** error → **0 at round 114** (and stays 0); staged
  **test** error bottoms **0.0417 @ round 35**, then holds in a **0.04–0.06 band** out to 400 (a mild
  **+0.017** drift from min to final — *no runaway climb*, even long after train hit 0).
- **lr × rounds trade-off:** test error vs rounds — lr = 1.0 reaches its ~0.058 plateau by **~10
  rounds**; lr = 0.5 by ~25–100; lr = 0.1 climbs slowly and only reaches ~0.05 by **~400 rounds**
  (gentler steps, more rounds, a touch lower in the end).
- **Noise overfit** (flip **25 %** of TRAIN labels, test clean, lr = 1.0): staged **train** error still
  marches to ~0 (AdaBoost memorises the noise), but staged **test** error bottoms **0.067 @ round 18**
  then **climbs to 0.150 @ 400** (a **+0.083 runaway** — more rounds make it worse). At flip = 15 % the
  rise is mild (+0.008); at 25 % it is unmistakable. (breast_cancer 20 % flip gives the same shape,
  +0.088, but no 2-D picture — deferred to NB 5's real-data noise study.)

## Cell-by-cell (~20 cells; intuition → implementation → "Read the figure")

1. (md) **Header** — `# 03 — Learning rate, rounds, and overfitting behaviour`; *Chapter 07 · Notebook
   3 of 5*. Open: NB 2 ended on a worry — the additive model drove **training** error to 0; doesn't
   that overfit? AdaBoost's answer is famously two-faced, and there is a knob — `learning_rate` — that
   shapes it. **Prerequisites:** NB 1–2 (reweighting, α, the additive model, exp-loss); **module 00**
   (over/under-fitting and the train/test U-curve; the generalization gap); ch 06 (a forest's
   diminishing returns). **What you'll be able to do:** use `learning_rate` (shrinkage) and read how it
   trades off against the number of rounds; explain AdaBoost's **resistance** to overfitting on clean
   data (and why — margins); recognise and diagnose its **overfitting under label noise**; reach for
   early stopping / smaller ν.
2. (md) **Where we are — one concept: controlling complexity.** NB 2 left the question "train → 0,
   does it overfit?" Boosting's complexity grows with *rounds*; `learning_rate` shrinks each round's
   step. Together they are the dials that control how hard the ensemble fits. This notebook's single
   thread: turn those dials and watch generalization — first on clean data (a happy surprise), then on
   noisy data (an honest limit).
3. (code) **Setup** — imports, `use_course_style()`, moons-0.20 split (same as NB 1–2). Print shapes.
4. (md) **The learning rate ν (shrinkage).** Re-lay: AdaBoost can multiply every learner's contribution
   by a factor ν ∈ (0, 1], so F = Σ **ν·αₜ** hₜ and the reweighting uses ν·αₜ. It is a brake on each
   step — like the step size in chapter 03's gradient descent. Smaller ν = smaller steps = more rounds
   to get anywhere.
5. (code) **ν scales α — by hand vs sklearn.** Round-1 α = ln((1−ε)/ε) = 1.68 (ν = 1). Show
   `AdaBoostClassifier(learning_rate=ν).estimator_weights_[0]` equals ν·α for ν ∈ {1, 0.5, 0.1}
   (1.680 / 0.840 / 0.168).
6. (md) **Read the result.** `learning_rate` does exactly one thing: scale every vote weight (and so
   every reweighting step) by ν. ν = 0.1 means each round counts a tenth as much — you will need many
   more rounds to reach the same model.
7. (md) **Does fitting train to zero overfit? (the clean surprise).** Module 00's lesson: past some
   complexity, test error turns back up (the U-curve). NB 2 drove **training** error to 0 by ~114
   rounds — so we should brace for a rising test curve. Let us actually measure it.
8. (code) **Figure A — clean staged train/test error vs rounds** (lr = 1.0, 400 rounds): train and test
   error each round; mark the round where train hits 0 (114).
9. (md) **Read the figure (A).** Training error reaches **0 at round 114** and stays there — the model
   memorises the training set. Yet the **test** error does **not** turn up: it bottoms near **0.042**
   around round 35 and then holds in a narrow **0.04–0.06** band all the way to 400 (a mild ~0.017
   drift, not a climb). This is AdaBoost's celebrated **resistance to overfitting**. The explanation
   (Schapire et al., 1998): once every point is on the right side, extra rounds keep pushing them
   *further* from the boundary — widening the **margin** — and larger margins tend to generalise
   better. Honest word: **resistance, not immunity** — the next experiment shows the other face.
10. (md) **The lr × rounds trade-off.** If ν shrinks each step, a smaller ν needs more rounds. Watch
    three learning rates race.
11. (code) **Figure B — test error vs rounds for lr ∈ {1.0, 0.5, 0.1}** (3 curves).
12. (md) **Read the figure (B).** lr = 1.0 reaches its plateau (~0.058) within ~10 rounds; lr = 0.1
    crawls — it needs ~400 rounds to get there, and edges a touch lower (~0.05). Smaller ν = gentler
    steps = more rounds, often slightly better generalization. The two knobs trade off; you tune them
    **together** (NB 4's `GridSearchCV`). More rounds is not free — and, as we are about to see, not
    always safe.
13. (md) **The other face: noise.** Resistance held on clean data. But AdaBoost *chases the hardest
    points* (NB 1), and a **mislabeled** point is eternally hard — NB 2's exponential loss punishes it
    without bound. So on noisy data, more rounds should *hurt*. We flip 25 % of the **training** labels
    (the test set stays clean and honest) and watch.
14. (code) **Figure C — noisy data (25 % train-label flip), two panels.** Left: staged train/test error
    vs rounds (train → 0, test bottoms ~round 18 then climbs to ~0.150). Right: the decision boundary
    at 400 rounds, with the flipped training points marked — the boundary carving islands around them.
15. (md) **Read the figure (C).** The story **reverses**. Training error still marches toward 0 —
    AdaBoost memorises the flipped labels too. But **test** error bottoms early (~round 18, ~0.067) and
    then **climbs to ~0.150**: past that point, every extra round makes the model *worse*. The right
    panel shows why — the boundary contorts into little islands chasing points whose labels are wrong.
    This is AdaBoost's known weak spot (Dietterich, 2000): the exponential loss feeds ever more weight
    to the impossible points, and the ensemble overfits the noise. **Resistance was never immunity.**
16. (md) **What to do about it.** Three honest levers: **early stopping** (use the round where held-out
    error bottoms, not the last), a **smaller learning rate** (gentler steps overfit noise more slowly),
    and a **simpler base learner** (NB 4). On genuinely noisy problems, a more robust loss
    (gradient boosting, ch 08) or bagging's averaging (ch 06) can beat AdaBoost outright — the
    capstone (NB 5) measures this on real data.
17. (md) **Honest scoping.** (a) "Resistance" is an **empirical**, margin-explained behaviour (Schapire
    1998) that holds at low noise — not a guarantee. (b) The noise overfit is real and measured
    (Dietterich 2000). (c) These exact rounds are **this dataset, this seed** (moons-0.20, seed 0); the
    *qualitative* behaviour generalises, the precise numbers do not. (d) No leakage: one clean sealed
    test throughout; noise was injected into **train only**.
18. (md) **Your turn** (tiered) — *easy:* from Figure C, name the round where test error bottoms on the
    noisy run, and say what happens to test error after it. *medium:* re-run the noise experiment at
    flip = 0.10 and flip = 0.40; does the overfit get worse as noise rises, and does the *bottom* move
    earlier? *harder:* on the noisy run, pick the early-stopping round (lowest test error) and compare
    its accuracy to 400 rounds; then explain why early stopping rescues the noisy model but barely
    changes the clean one (Figure A).
19. (md) **What you built** + vocabulary — `learning_rate` / shrinkage · the rounds × learning-rate
    trade-off · resistance to overfitting (margins) · overfitting under label noise · early stopping.
20. (md) **References** — Schapire, Freund, Bartlett & Lee 1998 — boosting the margin
    (DOI 10.1214/aos/1024691352); Dietterich 2000 — noise sensitivity (DOI 10.1023/A:1007607513941);
    Freund & Schapire 1997 (DOI 10.1006/jcss.1997.1504); ESL §10.4–10.6 (shrinkage; DOI
    10.1007/978-0-387-84858-7). `Previous: 02 — Weak learners and the additive model. Next: 04 — The
    estimator AdaBoostClassifier & its parameters.`

## `src/` & guards

No `src/` change (reuse `use_course_style`, `plot_decision_boundary`; the staged-error and lr-sweep
line plots are notebook-local matplotlib). **pytest stays 20.** Build via `uv run python - < build`;
banned-word JSON scan = 0; ruff/black clean; no hardcoded hex; output-free; nbconvert from project cwd
on a scratchpad copy; `gen_llms_txt`; both reviewers (no BLOCK) + Rémy visual before commit.

## Note on scope (for the reviewer gate)

NB 3 is **richer scope by design** (chapter Decision A, approved): its single declared concept is
"what controls boosting's complexity (rounds × ν) and how that shows up in generalization." The
learning-rate knob is established **by hand first** (cell 5) before any sweep, and the clean-resistance
and noise-overfit are the two faces of that one concept (not two concepts). The noise *demonstration*
lives here at the toy-2-D level (mechanism + a visible contorted boundary); the **real-data** noise
study is the capstone (NB 5, spambase). "By hand before library" was satisfied by NB 1–2 (the engine);
NB 3 is a **behaviour** notebook and observes the trained estimator (legitimate for a generalization
study), with the one by-hand check being ν-scales-α.
