# Notebook plan — 03_LogisticRegression / 02_boundary_and_weights

> Status: **APPROVED** (2026-06-20, by Rémy — validated alone; reviewers gate the *built* notebook).
> Build via `uv run python - < <scratchpad>/build_nb2.py` (stdin — avoids the `/tmp/struct.py` shadow).

## Context

NB **2 of 6** — the chapter's second fundamental, **one concept: a weighted line and what its weights
mean.** NB 1 turned *one* feature's score into a probability through σ and showed the score **is the
log-odds**. NB 2 adds the second feature and reads the geometry: z = w₁x₁ + w₂x₂ + b, the **decision
boundary** is the line z = 0 (P = ½), **w is perpendicular** to it and **‖w‖ sets its steepness**, and
each **wⱼ is a change in log-odds per standardized unit** (sign = direction, magnitude = strength).
Built **by hand**, **maximally visual**, on the fil-rouge (penguins, both features). **No fitting** — the
weights are **hand-chosen** (rotate the line with w, shift it with b), so the learner meets the geometry
cleanly *before* optimization (NB 3 writes the loss, NB 4 finds the weights). Closes with the contrast to
module 00's **unweighted nearest-centroid bisector** — the tilt between the two boundaries *is* the
weighting.

## Dataset & measured anchors (penguins; sklearn 1.9.0; re-measured at build)

- `datasets.load_penguins()` / `penguins_xy()` — Adélie **151** / Gentoo **123** (n = 274); features
  `bill_length_mm`, `flipper_length_mm`. **Gentoo = positive** (P = P(Gentoo)), as in NB 1.
- **Standardized** (the comparison only makes sense on a shared scale — ch 01 NB 2 scale trap; module 00
  NB 11): `bill` μ = **42.70**, σ = **5.19**; `flipper` μ = **202.18**, σ = **15.02**.
- **Hand-chosen weights (standardized; NOT fitted):** **w = (w_bill, w_flipper) = (1.0, 2.0), b = 0.0** —
  flipper weighted **exactly twice** bill (a clean weight bar). Re-measured: **acc 0.9891** (271/274),
  transition band P ∈ [0.1, 0.9] ≈ **37 %**, **‖w‖ = 2.24**, w-direction **63.4°**, boundary-line angle
  **−26.6°**. The band is **wide on purpose** (gentle ‖w‖) so the shading and the perpendicular arrow are
  visible — taught as *"unsure ≠ wrong"* (still 98.9 % correct; NB 3–4 find steeper, more decisive
  weights). *Reviewer-steepenable with identical geometry:* (1.5, 3.0) keeps the 1:2 ratio / direction /
  tilt but tightens the band to ≈17.5 % — I default to (1.0, 2.0) for the cleanest bars and the most
  visible band.
- **Reading a weight:** +1 standardized unit (≈ 5.19 mm bill / 15.02 mm flipper) **adds wⱼ to the
  log-odds** ⇔ **multiplies the odds by e^{wⱼ}**: bill ×e¹ ≈ **2.72**, flipper ×e² ≈ **7.39**. Sign =
  direction (toward Gentoo); magnitude = strength; fair only because standardized.
- **Nearest-centroid contrast (module 00 NB 05, same standardized space):** the unweighted boundary is the
  **perpendicular bisector** of the class centroids — its normal = (Gentoo − Adélie means) = **(1.68,
  1.81)**, weighting the features **≈ 0.93 : 1 (nearly equal)**, direction **47.18°**, line angle
  **−42.82°**, **NC acc 0.9927**. The hand-logistic boundary is **tilted ≈ 16°** from it — the tilt *is*
  the weighting. **Doubling w_bill → (2, 2)** rotates the line to **−45°**, almost onto the NC bisector
  (equal weighting).
- **Honest aside (not over-claimed):** the hand direction (63.4°) happens to sit almost on the
  unregularized fit (≈ 64°) — mentioned in one line as a teaser, **never presented as fitted** (NB 3–4
  fit).
- **Example penguins under the hand weights** (pinned exactly at build): clear Adélie bill 33.1 / flipper
  178 → P ≈ 0.00; clear Gentoo bill 59.6 / flipper 230 → P ≈ 1.00; **borderline** bill ≈ 41 / flipper ≈
  205 → P ≈ 0.5.

## Library / figures

Reuse `viz.use_course_style`, `ml_course.colors` (`CLASS_CYCLE` Adélie/Gentoo; `model` for the boundary &
arrow; `highlight`; **`CMAP_PROBA`** for the P-shading). All three figures are **one-off in-notebook**
(charter colours, **numpy under the hood** for the meshgrid / linear algebra). `viz.plot_decision_boundary`
is built for a *fitted* estimator's `predict` (it returns next in NB 5); NB 2's boundary is a hand score,
so a small custom `contourf`(σ over a meshgrid) + line + `annotate` arrow keeps the mechanism visible. **No
`src/` change → `pytest` stays 16.**

- **Figure A (2 subplots) — the headline figure:** *left* — standardized cloud + light **P-shading**
  (CMAP_PROBA) + the **boundary line** z = 0 + the **w arrow** (⟂ the line, toward Gentoo, length ∝ ‖w‖);
  *right* — a **weight bar** |w_bill| vs |w_flipper| (flipper twice as tall).
- **Figure B — the contrast:** the hand-logistic boundary overlaid with the **nearest-centroid bisector**
  (both class centroids marked) — the ≈16° tilt.
- **Figure C — the knobs:** **rotate & shift** — (1, 2, 0); doubling w_bill → (2, 2, 0) rotating toward the
  NC direction; and b = −2, 0, +2 as parallel shifts, on a faint cloud.

## Cell-by-cell (~21 cells; one concept; "Read the figure" after every figure)

1. (md) **Header** — `# 02 — The decision boundary & reading the weights`; *notebook 2 of 6 — Logistic
   Regression*; one-line purpose; warm welcome. **Prerequisites:** NB 1 (σ, odds/log-odds, the score *is*
   the log-odds, predict at ½); module 00 (the feature space NB 02; the **nearest-centroid** first
   classifier NB 05; **standardization / fit-on-train** NB 11); chapter 01 (the **scale trap** NB 02).
   **What you'll be able to do:** write the 2-feature score z = w₁x₁+w₂x₂+b; find and draw the **decision
   boundary** (z=0); explain why **w ⟂ the boundary** and why **‖w‖ is its steepness**; read each **weight
   as a change in log-odds per standardized unit** (sign & magnitude); contrast a weighted boundary with
   the **unweighted nearest-centroid bisector**.
2. (code) **Imports + seed + style** — `matplotlib.pyplot`, `numpy`, `pandas`; `from ml_course import
   colors, datasets, viz`; `viz.use_course_style()`; `np.random.seed(0)`. Load `df =
   datasets.load_penguins()`; `X_df, y = datasets.penguins_xy(df)`; show `df.head()` + class counts
   (Adélie 151 / Gentoo 123).
3. (md) **Recap & footing** — NB 1 took ONE feature (`bill_length`): score → σ → P(Gentoo), threshold ½.
   Re-establish from module 00: each penguin is a **point in the 2-D feature space** (NB 02), and the
   **nearest-centroid** classifier (NB 05) already drew a boundary — the perpendicular bisector between the
   two class means, treating both features **equally**. Here: give each feature **its own weight**, draw
   the **weighted** line, and learn to **read** what the weights mean. Still **no fitting** (NB 3–4).
4. (md) **Why standardize first** — bill ~32–60 mm, flipper ~170–230 mm: different ranges. To compare
   weight *magnitudes* ("which feature moves the odds more") the features must share a scale — the **scale
   trap** (ch 01 NB 2), now for weights. Standardize to z-scores (subtract mean, divide by std); no
   train/test split yet (NB 6 does fit-on-train-only). State μ/σ (42.70/5.19; 202.18/15.02).
5. (code) **Standardize** — `StandardScaler().fit_transform(X)` (numpy under the hood); confirm mean ≈ 0,
   std ≈ 1 (print); keep a small standardized DataFrame for display; print the scaler means/stds.
6. (md) **Intuition — the 2-feature score** — z = w₁·(std bill) + w₂·(std flipper) + b: a **weighted sum**,
   still ONE number per penguin, still the **log-odds** (NB 1), still squashed by σ to a probability. Two
   weights (one per feature) + a bias b. Predict Gentoo when P ≥ ½ ⇔ z ≥ 0.
7. (code) **Set the weights BY HAND & compute P** — `w = np.array([1.0, 2.0])` (bill, flipper), `b = 0.0`
   (comment: hand-chosen, **NOT fitted** — NB 3–4 find these; flipper weighted twice bill, to see the
   geometry). `def sigmoid(z)`; `z = Xs @ w + b`; `P = sigmoid(z)`. Show a small table — a clear Adélie, a
   clear Gentoo, a **borderline** penguin (raw mm, z, P).
8. (md) **Intuition — the decision boundary is the line z = 0** — the points where the model is exactly
   50/50 (P=½ ⇔ z=0) form a **straight line** w₁x₁+w₂x₂+b=0; z>0 on the Gentoo side, z<0 on the Adélie
   side. Two facts to SEE next: **w ⟂ the line**, and **‖w‖ sets how steeply P swings** across it.
9. (code) **Figure A (2 subplots)** — *left*: standardized cloud (CLASS_CYCLE), **P-shading** (σ over a
   meshgrid, CMAP_PROBA), the **boundary line** z=0 (solve x₂ = −(w₁x₁+b)/w₂), the **w arrow** (from a
   point on the line, direction w/‖w‖ toward Gentoo, length ∝ ‖w‖); *right*: **weight bar** |w_bill| vs
   |w_flipper|. Print the hand-rule **accuracy (0.9891)**.
10. (md) **Read the figure (A)** — *left*: the line splits the plane into a Gentoo side (P>½) and an Adélie
    side; the shaded band hugging the line is where the model is **unsure** (P near ½) — ~37 % of penguins
    here, yet the rule is **98.9 % correct** (unsure, P≈0.3, is **not** wrong; NB 3–4 find steeper, more
    decisive weights). The arrow crosses the line at a **right angle** toward Gentoo; its length is the
    "confidence slope". *right*: the **flipper bar is twice the bill bar**.
11. (md) **Intuition — read each weight as a change in log-odds** — because z **is** the log-odds (NB 1),
    raising a standardized feature by 1 unit (≈5.19 mm bill / 15.02 mm flipper) **adds wⱼ to the log-odds**
    ⇔ **multiplies the odds by e^{wⱼ}**: bill ×e¹≈**2.72**, flipper ×e²≈**7.39**. Sign = direction
    (positive → Gentoo), magnitude = strength; fair only because standardized; a negative weight pushes
    toward Adélie. (Restate: **w ⟂ boundary, ‖w‖ = steepness** — double the weights ⇒ same line, steeper
    transition, narrower band.)
12. (md) **Intuition — contrast with the nearest-centroid bisector (module 00 NB 05)** — nearest-centroid's
    boundary normal = (Gentoo − Adélie means), weighting the two features **≈ equally (0.93 : 1)**;
    logistic gives flipper ~2× the weight, so its boundary is **tilted** from the bisector. The tilt is the
    weighting made visible.
13. (code) **Figure B** — overlay on the standardized cloud: the **hand-logistic boundary** (≈ −26.6°) and
    the **nearest-centroid bisector** (≈ −42.82°, from the centroids), both class-centroid markers. Print
    **NC acc 0.9927** vs **hand-logistic 0.9891**, and the **angle between them (≈16°)**.
14. (md) **Read the figure (B)** — same data, two boundaries: nearest-centroid weights features by their
    centroid gap (≈ equal here), logistic weights flipper more, so the line rotates ~16°. They score about
    the same on THIS near-separable data (0.9927 vs 0.9891) — but logistic adds what nearest-centroid
    cannot: a **probability** (the shading) and **interpretable weights** (which feature, how much). Where
    features deserve unequal weight, that tilt is the difference between a good and a poor boundary.
15. (md) **Intuition — w rotates the line, b shifts it** — changing the **ratio** of the weights **rotates**
    the boundary; changing **b slides it parallel** (no rotation; distance = −Δb/‖w‖). These are exactly
    the knobs we stop **guessing** and start **optimizing** in NB 3–4.
16. (code) **Figure C — rotate & shift** — on a faint cloud: the (1, 2, 0) line; doubling w_bill →
    (2, 2, 0) rotating to ≈ −45° (toward the equal-weighting / NC direction); and b = −2, 0, +2 as three
    parallel lines.
17. (md) **Read the figure (C)** — doubling bill's weight gives bill an equal say and swings the line
    toward the nearest-centroid bisector; raising b slides the whole line toward the Adélie corner (more
    Gentoo predictions) **without turning it**. **Weights turn the line; the bias moves it.**
18. (md) **Your turn** (3 tiered) — *easy*: for a penguin at a stated standardized point, compute z and say
    which side it falls on; *medium*: double w_bill (→ 2.0) and describe how the boundary turns and which
    feature now matters more; *harder*: keep w = (1, 2) and choose **b so the boundary passes through a
    named standardized point** (set z = 0, solve for b).
19. (md) **What you built** — bullets: the 2-feature score & the **decision boundary** (z=0); **w ⟂** it
    and **‖w‖ = steepness**; each **weight = a change in log-odds (×e^w to the odds) per standardized
    unit**; the **weighted line vs the unweighted nearest-centroid bisector**. **Vocabulary box:** decision
    boundary, weight vector w, normal/perpendicular, ‖w‖ (steepness), bias b, weight = log-odds
    contribution, standardized units. Restate the new abilities.
20. (md) **Going further (optional)** — in >2 features the boundary is a flat **hyperplane** w·x+b=0 (same
    idea, more axes); a **straight** boundary **underfits curved data** (named; fix pointed to —
    features/kernels (SVM ch 05), trees (ch 04)). Forward pointer: **NB 3** stops guessing the weights and
    writes a number for how wrong they are — the **log-loss**.
21. (md) **References** — Cox DR (1958) DOI 10.1111/j.2517-6161.1958.tb00292.x; ISLR §4.3 DOI
    10.1007/978-1-0716-1418-1; ESL §4.4 DOI 10.1007/978-0-387-84858-7. `Previous: 01 — From a linear score
    to a probability` · `Next: 03 — Fitting I: what we optimize (log-loss)`.

## Honest scoping (stated in the notebook)

- **Nothing is fitted in NB 2.** Weights are hand-chosen to expose the geometry; the **0.9891** is the
  hand rule's accuracy, **not** an optimum — NB 3–4 *find* the weights. Stated plainly.
- **Weights are interpretable only after standardization** (raw-scale weights confound importance with
  units — the ch 01 scale trap); stated wherever weights are read.
- The **wide band is gentle-by-choice** for visibility; **"unsure ≠ wrong"**; it motivates NB 3–4 (find
  steeper, decisive weights).
- Nearest-centroid is **comparable here** (0.9927 vs 0.9891) **only because the data is near-separable**;
  what logistic adds is the **weighting**, the **probability**, and the **interpretable weights**.
- The boundary is **linear** → it **underfits curved data** (named; fix pointed to, not taught).

## Verification

Build via `uv run python - < <scratchpad>/build_nb2.py` (stdin — avoids the `/tmp/struct.py` stdlib
shadow). Re-measure at build: scaler μ/σ; hand-rule **acc 0.9891** / band ≈37 %; **NC acc 0.9927**; line
angles **−26.6° / −42.82°** and the **≈16°** tilt; the three example penguins. Runs top-to-bottom
(nbconvert to /tmp; **output-free**, `--clear-output --inplace`); **banned-word grep**
("just/simply/obviously/trivially" + FR) = 0; `check_no_hardcoded_hex` passes; **no `src/` change**
(`pytest` stays 16); `gen_llms_txt` re-run; `ruff`/`black` clean. Both reviewers PASS (no BLOCK); Rémy
validates visually; commit `feat(03_logistic_regression): notebook 02 — the decision boundary & reading
the weights`; merge `notebook → chapter`.
