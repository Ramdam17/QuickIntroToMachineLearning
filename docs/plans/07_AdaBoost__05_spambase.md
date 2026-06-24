# Notebook plan — 07_AdaBoost / 05_spambase  (chapter capstone)

> Status: **APPROVED by Rémy & persisted** (no reviewer gate at plan stage — reviewers return on the
> built notebook). Building now → guards → two-reviewer gate (no BLOCK) → Rémy visual → commit →
> ff-merge to `chapter/07_AdaBoost`. **Last NB of chapter 07 — after it ships, close the chapter via PR
> into `main` (`--no-ff`).**

## Context

NB **5 of 5** — the chapter's **demanding practical case and capstone**, **visualization-first**.
AdaBoost on **spambase** (UCI / ESL ch 10's canonical boosting set): predict spam / not-spam for 4601
emails from 57 hand-crafted features (word- and char-frequencies + capital-run lengths). Two questions
drive it: does AdaBoost **shine** here, and can we tell its **honest noise story** — the one the whole
chapter has been building toward? Capstone rule: ~24–26 cells a **floor**, **7 figures**.

**The honest finding (re-measured, aligned with the approved chapter plan).** AdaBoost shines (test acc
**0.949**) — competitive with RF (**0.959**, a hair ahead), far above a single stump (0.782). Its
clean-data **resistance** holds on real data (test bottoms ≈round 280, then flat). And the noise story
is **nuanced, not a slogan**: the *mechanism* is real and measurable (the exponential loss piles weight
onto mislabeled points — by hand, ~21% of points come to hold ~40% of the weight), but the *consequence*
on spam is mild until high noise (test error rises only ~0.005 at 20% flip, clearly only at 40%), and
the folk claim **"RF beats AdaBoost under noise" REVERSES on spam** — from 20% noise up, AdaBoost is the
*more* robust of the two (40%: 0.823 vs RF 0.704). So we show the mechanism, show the comparison, and
**refuse to ship "RF beats AdaBoost" as a law** — it is dataset-dependent. *Figure captions and "Read
the figure" stay measured ("competitive with", never triumphal.)*

## Anchors (re-measured at plan time, **sklearn 1.9.0**, seed 0; every estimator `random_state`-pinned)

`fetch_openml("spambase", version=1, as_frame=True)` → **4601 × 57**, target `class` (`'0'` not-spam
2788 / `'1'` spam 1813 → **spam 39.4%**, mild imbalance). 70/30 **stratified** split (seed 0) → **train
3220 / test 1381**. Columns descriptively named by openml (`word_freq_*`, `char_freq_*`,
`capital_run_length_*`). **No scaling for AdaBoost / RF / trees**; KNN / LogReg / SVM get a
`StandardScaler` pipeline.

- **AdaBoost shines (cross-method, sealed test):** stump **0.782** · deep tree 0.915 · KNN(scaled)
  0.899 · LogReg(scaled) 0.926 · SVM-rbf(scaled) 0.936 · **AdaBoost(400) 0.949** · **RF(300) 0.959**.
  The two ensembles lead; AdaBoost is competitive with RF (RF a hair ahead — no universal best). Depth
  recap: AdaBoost(50) 0.931 → (200) 0.946 → (400) 0.949 (more rounds keep helping). *Contrast bc, where
  SVM led 0.965 and the ensembles trailed — the right tool tracks the problem's shape.*
- **Resistance on real data (staged, clean, n=400):** test err 0.218@1 → **bottoms 0.0485 @ round 279**
  → 0.0507 @400 (flat, +0.0022, no upward drift — the ESL resistance curve). **Train err floors at
  0.045 and NEVER reaches 0** — real data has irreducible overlap; "train→0" was a low-noise-2D
  (moons) phenomenon, corrected here.
- **Confusion (AdaBoost 400):** `[[810, 27], [43, 501]]` → spam **recall 0.921** (43 missed), **precision
  0.949** (27 legit mails flagged — the costlier false alarm). The asymmetry is the lesson.
- **Importance honestly (AdaBoost 200; MDI nonzero 36/57):** MDI top — `george` .096, `char_!` .093,
  `meeting` .077, `hp` .068, `edu` .060, `char_$` .056, `1999` .038, `remove` .037, `cs` .037,
  `project` .036. Permutation top — `george` .033, `hp` .033, `char_!` .021,
  `capital_run_length_average` .020, `char_$` .019, `remove` .018, `our` .014, `free` .010, `your`
  .009, `edu` .008. They **broadly agree on the leaders but reorder**. The honest read: a mix of (i)
  **spam markers** (`!`, `$`, `remove`, `free`, `your`) and (ii) **corpus artifacts** — `george`, `hp`,
  `edu`, `meeting`, `cs`, `1999` (the set was collected at HP Labs by someone named George; those tokens
  mark *ham*). Importance reflects **this corpus**, is **not causal** (`george` is no general spam
  signal), and is read at the semantic/group level.
- **Noise mechanism (by-hand SAMME, 20% train flip → 668 pts = 20.7%):** weight held by the flipped
  points climbs 0.207 → 0.288 @round 1 → **~0.40 by rounds 100–200** — roughly **double their share**:
  the exponential loss (NB 2) chases the points it can't get right, including the mislabeled ones.
- **Noise consequence (staged test on clean test):** 20% flip — bottoms 0.081 @286, +**0.005** to 400
  (mild — spam is well separated); **40% flip — bottoms 0.142 @ round 39, climbs to 0.177 @400 (+0.035)**:
  at high noise more rounds memorize the noise → early stopping / smaller `learning_rate` / a robust loss
  (ch 08) matter. Resistance is real, **not immunity**.
- **"RF beats AdaBoost under noise" is NOT a law (degradation, clean test):** AdaBoost(400) vs RF(300) at
  noise {0, .1, .2, .3, .4} = AdaBoost {.949, .933, **.914, .897, .823**} vs RF {.959, .939, **.902,
  .839, .704**}. From **20% up AdaBoost is the MORE robust of the two** (the reversal). Why: AdaBoost's
  base is a **weak stump** (NB 4 — the base must stay weak), too simple to memorize individual noisy
  points, while RF's **deep** trees have the capacity to fit the noise. On breast_cancer the ranking goes
  the other way — **dataset-dependent**, cited (25% flip: AdaBoost ≈0.82 vs RF ≈0.91), not re-run.
- **Cross-method spine (referenced for continuity, not re-run here):** breast_cancer — KNN 0.942 · tree
  0.906 · LogReg 0.953 · SVM **0.965** · RF ≈0.95.

## Library / figures

- **No `src/` change.** `fetch_openml(as_frame=True)` returns a named DataFrame + Series (no loader
  wrapper, cf. ch 06 NB 5 with `fetch_covtype`); **INFO logging shown** (`logging.basicConfig(INFO)`) so
  the one-time openml fetch is visible, never silenced. `pytest` stays **20**.
- **Reused `viz` helpers:** `use_course_style`; `plot_class_balance` (Fig A); `plot_train_test_curve`
  (Fig C, staged train/test pair); `plot_confusion_matrix` (Fig D); `plot_feature_importances` (Fig E,
  MDI vs permutation, 2 panels). The **cross-method bar** (B), the **2-panel noise figure** (F:
  by-hand weight concentration + clean-vs-noisy staged test) and the **degradation curve** (G) are
  **notebook-local matplotlib** with a course colormap from `ml_course.colors`.
- Sklearn: `fetch_openml`, `AdaBoostClassifier`, `RandomForestClassifier`, `DecisionTreeClassifier`,
  `KNeighborsClassifier`, `LogisticRegression`, `SVC`, `StandardScaler`, `make_pipeline`,
  `permutation_importance`, `train_test_split`, `confusion_matrix`.
- **Seven figures** (each + "Read the figure"): **A** class balance · **B** cross-method accuracy ·
  **C** staged resistance (clean, train+test) · **D** confusion matrix · **E** MDI vs permutation
  importance · **F** noise (2 panels: by-hand weight-on-flipped + clean-vs-40%-noise staged test) ·
  **G** AdaBoost-vs-RF degradation vs noise level.

## Cell-by-cell (~27 cells; visualization-first; "Read the figure" after every figure)

1. (md) **Header** — `# 05 — A demanding case: spam (where AdaBoost shines, and where label noise
   bites)`; *Chapter 07 · Notebook 5 of 5*; warm welcome (the capstone — the whole chapter converges).
   **Prerequisites:** NB 1–4 (reweighting by hand & parity; the additive model & exponential loss; rounds
   × learning_rate & overfitting; the estimator & its dials, the base must stay weak); the cross-method
   spine (ch 01 KNN, 03 LogReg, 04 tree, 05 SVM, 06 RF); module 00 — confusion / precision / recall (NB
   07), the split (NB 04). **What you'll be able to do:** run an honest AdaBoost workflow on a real,
   high-dimensional problem; see it shine; read its staged resistance curve; read importances honestly;
   understand its noise behaviour **from the inside** (weight concentration) and refuse to over-generalize.
2. (md) **Where we are & the stakes** — spambase: 4601 emails, 57 features (word/char frequencies +
   capital-run lengths), spam/not-spam. ESL ch 10's canonical boosting dataset, fresh to the course. Two
   questions: does AdaBoost **shine** here, and can we tell its **honest noise story**?
3. (code) **Setup + INFO logging + fetch + split** — `logging.basicConfig(INFO)` (the one-time openml
   fetch visible); `fetch_openml("spambase", version=1, as_frame=True)`; map target to int; 70/30
   stratified (seed 0); print shapes, the three feature groups, class counts. **No scaling for AdaBoost.**
4. (code) **Fig A — class balance** (`plot_class_balance`, 2 bars).
5. (md) **Read the figure (A)** — spam 39.4% / not-spam 60.6%: **mild** imbalance (unlike covtype's
   severe one). Accuracy is a reasonable headline here, but we still read the confusion matrix because the
   two errors cost differently (a flagged real mail hurts more than a missed spam).
6. (md) **Intuition — the honest workflow & the cross-method question.** Fit AdaBoost (no scaling —
   tree-based, NB 4) and a spine of baselines (stump, deep tree, KNN/LogReg/SVM scaled, RF); compare on
   the sealed test. Which model *family* fits spam's shape?
7. (code) **Fig B — cross-method accuracy** — fit all seven; bar of test accuracy.
8. (md) **Read the figure (B)** — AdaBoost **0.949**, RF **0.959** lead; the ensembles beat the single
   tree (0.915), the linear/margin/distance methods (LogReg 0.926, SVM 0.936, KNN 0.899), and far outrun a
   lone stump (0.782). AdaBoost is **competitive with RF** (RF a hair ahead — no universal best). *Contrast
   bc, where SVM led (0.965) and the ensembles trailed:* on spam the boosted/bagged ensembles win — the
   right tool tracks the problem's shape. AdaBoost turned a 0.782 stump into a 0.949 classifier by stacking
   400 of them — **the chapter's promise, realized.**
9. (md) **Intuition — resistance on real data (re-lay NB 3).** On clean data more rounds keep helping even
   after they could stop — margin maximisation (Schapire 1998), not overfitting. Watch the staged test error.
10. (code) **Fig C — staged resistance curve (clean)** — `AdaBoost(400)` staged **train + test** error vs
    rounds (`plot_train_test_curve`).
11. (md) **Read the figure (C)** — test error 0.218 → **bottoms 0.0485 @ round 279**, then holds flat to
    0.0507 @400 (no upward drift): the ESL resistance curve, on **real** data. Note **train error floors at
    0.045 and never reaches 0** — real data has irreducible overlap; "train→0" was a low-noise-2D (moons)
    phenomenon (NB 2/3). Resistance is real — and we are about to see it is **not immunity**.
12. (code) **Fig D — confusion matrix** (`plot_confusion_matrix`, AdaBoost 400).
13. (md) **Read the figure (D)** — `[[810,27],[43,501]]`: spam **recall 0.921** (43 spam slipped through),
    **precision 0.949** (27 legit mails wrongly flagged). The **asymmetry** is the point — a filter's false
    alarms (lost real mail) hurt more than its misses; threshold-moving (ch 03) is the lever, here we report
    the default 0.5.
14. (md) **Intuition — reading importance honestly (MDI + permutation; ch 04/06 caveat).** AdaBoost exposes
    `feature_importances_` (MDI, α-weighted over the stumps); permutation is the honest cross-check — NB 4's
    promise, now on real, 57-feature data.
15. (code) **Fig E — MDI vs permutation** (`plot_feature_importances`, two panels, top-12); print MDI
    nonzero = 36/57.
16. (md) **Read the figure (E)** — both rank a **mix**: (i) classic **spam markers** — char `!`, char `$`,
    `remove`, `free`, `your`; and (ii) **corpus artifacts** — `george`, `hp`, `edu`, `meeting`, `cs`,
    `1999` (this set was collected at HP Labs by someone named George; those tokens mark *ham*). MDI and
    permutation **agree on the leaders but reorder**, and only 36/57 features carry any MDI. The honest
    lesson: importance reflects **this corpus** and is **not causal** — `george` is not a general spam
    signal. Read at the semantic/group level; never deploy on importance alone.
17. (md) **Intuition — the noise story, from the inside (the honest core).** NB 3 showed AdaBoost can
    overfit label noise. *Why?* Its **exponential loss** (NB 2) punishes confidently-wrong predictions ever
    harder, so reweighting keeps piling weight onto the points it cannot get right — **including mislabeled
    ones.** We measure that pile-up by hand, watch the consequence, then refuse to over-generalize.
18. (code) **Fig F — noise (2 panels)** — (left) by-hand SAMME: **weight held by the flipped points vs
    rounds** at 20% flip (a 0.21 share reference line + the curve rising to ~0.40); (right) **staged test
    error vs rounds, clean vs 40%-noise** (clean flat, bottoms @279; 40% bottoms early @≈39 then climbs to
    0.177).
19. (md) **Read the figure (F)** — (left) the ~21% mislabeled points come to **hoard ~40% of the total
    weight** — roughly double their share — exactly because the exponential loss chases them (NB 2's
    surrogate, lived). (right) at modest noise spam still largely **resists** (20%: +0.005 over 400 rounds),
    but at **40%** the test error bottoms early (round ≈40) then **climbs** (+0.035): more rounds now
    memorize the noise. The levers: **early stopping**, a smaller `learning_rate` (NB 3), or a robust loss
    (ch 08). Resistance is real but **not immunity**.
20. (md) **Intuition — is "RF beats AdaBoost under noise" a law?** The folk claim says boosting is fragile
    to noise and bagging is robust. Let's test it **honestly** on spam.
21. (code) **Fig G — degradation** — AdaBoost(400) vs RF(300) test accuracy vs noise {0, .1, .2, .3, .4}
    (clean test), two lines.
22. (md) **Read the figure (G)** — on clean spam RF edges AdaBoost (0.959 vs 0.949), but **from 20% noise
    up AdaBoost is the MORE robust** (0.3: 0.897 vs 0.839; 0.4: 0.823 vs 0.704). The folk claim **reverses**
    on spam. *Why:* AdaBoost's base is a **weak stump** (NB 4 — keep the base weak), too simple to memorize
    individual noisy points, while RF's **deep** trees have the capacity to fit the noise. So "RF beats
    AdaBoost under noise" is **dataset-dependent, NOT a law** — the chapter refuses to ship it (on
    breast_cancer the ranking reverses again; cited, not re-run). Know the **mechanism**, not a slogan.
23. (md) **Error analysis & honest limits + the boosting bridge.** AdaBoost here is **strong** (0.949),
    competitive with RF, resistant on clean data, and — on this problem — *more* noise-robust than RF. But:
    (1) it did **not win** (RF a hair ahead) — no universal best; (2) resistance is **not immunity** (high
    noise → use early stopping); (3) importance reflects the **corpus**, not cause; (4) it inherits whatever
    the stump base cannot separate (train floor 0.045). **When to push further:** **ch 08 Gradient
    Boosting** generalizes AdaBoost (the exponential-loss special case) to **any differentiable loss** —
    including losses far less noise-sensitive than exponential — via gradient descent in function space.
24. (md) **Your turn** (tiered) — *easy:* from Fig C, name the round where the clean test error bottoms and
    say in one sentence why running to 400 doesn't hurt. *medium:* raise the injected noise to 0.5 and
    re-measure where AdaBoost's test error turns upward (Fig F right) — relate it to the weight-concentration
    panel. *harder:* swap the stump base for `DecisionTreeClassifier(max_depth=3)` and re-run the
    degradation (Fig G) — does the stronger base overfit the noise faster (NB 3/4), and does AdaBoost still
    beat RF at 40%? Explain from bias/variance.
25. (md) **What you built** — an honest AdaBoost workflow on a real, high-dimensional problem: it **shines**
    (0.949, competitive with RF, far above a stump); reads its **staged resistance** honestly (bottoms ≈280,
    flat, train floor never 0); reads **importance** honestly (spam markers + corpus artifacts, MDI vs
    permutation, not causal); understands the **noise behaviour from the inside** (weight piles on mislabeled
    points) and **refuses to over-generalize** the RF comparison (dataset-dependent, not a law).
    **Vocabulary:** staged resistance · irreducible error / train floor · false-alarm asymmetry · MDI vs
    permutation · corpus artifact · exponential-loss non-robustness · weight concentration · no universal law.
26. (md) **Chapter wrap — AdaBoost, end to end** + **References.** From reweighting by hand (NB 1) → the
    additive model & where α comes from / exponential loss (NB 2) → rounds × learning rate & overfitting
    behaviour (NB 3) → the estimator & its dials, the base stays weak (NB 4) → a demanding case where it
    shines and its honest noise signature (NB 5). The first **boosting** method: sequential
    error-correction, the opposite of ch 06's parallel bagging, built on ch 04's stump. Its **exponential
    loss** is both its engine and its noise-sensitivity; **ch 08 Gradient Boosting** generalizes it to any
    differentiable loss. **Going further (optional):** early stopping via a validation curve; SAMME with a
    slightly stronger base; robust losses (ch 08). **References (DOIs):** Freund & Schapire 1997
    (10.1006/jcss.1997.1504); Schapire et al. 1998 — margins (10.1214/aos/1024691352); Friedman, Hastie &
    Tibshirani 2000 — statistical view / exp loss (10.1214/aos/1016218223); Zhu et al. 2009 — SAMME
    (10.4310/SII.2009.v2.n3.a8); Dietterich 2000 — noise (10.1023/A:1007607513941); Hopkins et al. —
    spambase, UCI (10.24432/C53G6X); ESL §10 (10.1007/978-0-387-84858-7); ISLR §8.2
    (10.1007/978-1-0716-1418-1). `Previous: 04 — The estimator & its parameters.` `Next: Module 08 —
    Gradient Boosting.`

## Honest scoping (stated in the notebook)

- **AdaBoost shines but does not "win"** — competitive with RF (0.959), which edges it on clean spam; no
  method is universally best (bc's SVM led there). "Competitive with", never triumphal.
- **Resistance is real but not immunity** — clean test flat to 400 rounds (train floor 0.045, never 0,
  unlike moons); under heavy label noise more rounds *hurt* (early stopping the lever).
- **The noise weakness is framed internally** — the *mechanism* (exp-loss piles weight on mislabeled
  points, by-hand ~0.21→~0.40) is the robust, dataset-independent statement; the *consequence* magnitude
  is dataset-dependent.
- **"RF beats AdaBoost under noise" is NOT shipped as a law** — it **reverses** on spam (AdaBoost more
  robust from 20% up); the RF comparison is shown, the bc counter-case cited, no law claimed.
- **Importance is not causal and reflects this corpus** — `george`/`hp` are HP-Labs artifacts; read MDI
  with permutation, at the semantic level.
- **One sealed test**; the cross-method comparison and the degradation sweep use fixed defaults (no
  test-set tuning); noise is injected into **train only**, the test set stays clean.

## `src/` & guards

No `src/` change (reuse `use_course_style`, `plot_class_balance`, `plot_train_test_curve`,
`plot_confusion_matrix`, `plot_feature_importances`; the cross-method bar, the 2-panel noise figure, and
the degradation curve are notebook-local matplotlib with a course colormap from `ml_course.colors`).
**pytest stays 20.** Build via `uv run python - < <scratchpad>/build_ch07_nb5.py` (stdin); re-measure every
anchor at build; nbconvert top-to-bottom **from project cwd** on a scratchpad copy (tracked file
**output-free**); **banned-word scan over JSON real cell text** = 0; `check_no_hardcoded_hex` passes;
`ruff`/`black` clean; `gen_llms_txt` re-run; **rebuild from the build script right before `git add`**
(editor kernel-drift habit). Both reviewers PASS (no BLOCK) + Rémy validates visually; commit
`feat(07_adaboost): notebook 05 — a demanding case: spam (shines, and the honest noise signature)`;
ff-merge `notebook → chapter`; **then close CHAPTER 07 via PR into `main` (`--no-ff`)** on Rémy's go.

## Note on anchors vs the chapter plan

The chapter plan §NB 5 cited the **breast_cancer** by-hand noise figures (20% → "0.198 → 0.449" weight,
test 0.082 → 0.170). NB 5 is **spambase**, so the by-hand weight concentration is re-measured **on spam**
(0.207 → ~0.40) and the staged-noise rise is the **spam** curve (mild at 20%, clear at 40%). The
qualitative claims (weight piles on mislabeled points; resistance is not immunity; the RF horse-race is
not a law) are unchanged and, if anything, **sharper** — the spam reversal makes the "not a law" point
vivid. spam clean numbers match the chapter plan (AdaBoost ≈0.949, RF ≈0.956–0.959, stump 0.782).
