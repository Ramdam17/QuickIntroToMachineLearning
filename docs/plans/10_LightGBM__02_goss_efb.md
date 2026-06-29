# NB plan — 10_LightGBM / 02_goss_efb — GOSS (built) + EFB (named)

> Status: **APPROVED by Rémy (via ExitPlanMode, 2026-06-28)**. One built concept (GOSS) + one named
> companion (EFB). No reviewer gate at the NB-plan stage (reviewers return on the built notebook).
> **Reframed vs the chapter-plan wording after live measurement** (see Context) — Rémy signed off.

## Context

Chapter 10, NB 2 of 5 (a fundamental). One concept built by hand — **GOSS** (Gradient-based
One-Side Sampling, Ke et al. 2017 §3.2) — plus one named companion — **EFB** (Exclusive Feature
Bundling, §4). It answers "how does LightGBM get *light*?": train each tree on fewer rows / fewer
features without losing the signal. Builds on ch 08 (per-row gradients) and ch 09 NB 2 (the
split-gain is a **sum over rows** of `(g, h)`).

### Material reframing vs the approved chapter-plan wording (measured, Rémy-approved)

The chapter plan said NB 2 should "show **GOSS ≈ full-data quality & beats a uniform subsample at
matched fraction**." Measured live before drafting (3 scratchpad scripts): the **"beats uniform"
half is regime-dependent**, exactly as Ke et al.'s Theorem 3.2 predicts. GOSS's edge over a uniform
subsample is governed by **gradient concentration** (share of `Σ|g|` in the top rows):

| concentration (top-20 % share of `Σ|g|`) | GOSS std / uniform std |
|---|---|
| 0.20 (flat) | 1.64 — uniform much tighter |
| 0.36 | 1.24 |
| 0.49 | 0.99 — crossover |
| 0.67 | 0.64 — GOSS wins |
| 0.86 | 0.27 — GOSS ~4× tighter |

On **moderate dense tabular data** concentration stays modest — measured **0.21 → 0.47** even after
200 boosting rounds (below the ~0.5 crossover). So there GOSS **≈ full quality on fewer rows**
(ensemble: GOSS 0.9345 ≈ full 0.9313, on 30 % of rows) but **~ties** a uniform subsample; its real
win (statistical + wall-clock) is the **large / wide / sparse, concentrated-gradient** regime it was
designed for (Ke et al.'s benchmarks) — which NB 5 dials `n` up to find. Both estimators are
**unbiased** (`H` exact: `a·n + (1−a)/b·b·n = n`; `G` unbiased); GOSS's lever is **variance**, only
when gradients are imbalanced. The NB teaches this honest, paper-faithful version (a strict honesty
upgrade, consistent with the chapter's existing "GOSS = statistical efficiency, wall-clock
regime-dependent" bar).

## The one concept

**GOSS:** rank rows by `|gradient|`; keep the top `a` fraction (kept *exactly* — they still carry the
most signal); randomly sample a `b` fraction of the rest; **up-weight the sampled rest by `(1−a)/b`**
so the row-sums `G, H` (hence the split-gain) stay ~unbiased. Fewer rows per tree, same gain in
expectation — lower variance than uniform sampling *when the gradients are concentrated*.

## Live anchors (measured, lightgbm 4.6.0, SEED=0 — scripts in scratchpad)

- **Unbiased reweight (round-1 regression toy, `g=F0−y`, `h=1`, n=2000, a=.2/b=.1, reweight ×8):**
  `H` recovered **exactly** (2000.000, std 0); `G` unbiased (truth 0.000; GOSS mean +2.45, uniform
  −10.9); gradients near-Gaussian (|g| median 1.47, max 8.10) → concentration only 0.20.
- **Concentration sweep (estimating the node `G` on 30 % rows):** GOSS/uniform std ratio
  1.64 / 1.24 / 0.99 / 0.64 / 0.27 at concentration 0.20 / 0.36 / 0.49 / 0.67 / 0.86 — crossover ≈ 0.5;
  both unbiased throughout.
- **Achievable tabular concentration (real boosting gradients, harder data):** 0.21 (1 round) → 0.30
  (10) → 0.41 (50) → 0.47 (200) — stays below the crossover.
- **Ensemble (LightGBM, 45k×30, 300 trees, lr .05):** full 0.9313 / GOSS(.2,.1) 0.9345 / uniform(.3)
  0.9353 — all ≈ equal; GOSS matches full quality on 30 % of rows.
- **Wall-clock (150k×50):** gbdt 2.41 s vs goss 2.45 s — ~flat on dense data (the honesty bar).
- **API:** `LGBMClassifier(data_sample_strategy='goss', top_rate=a, other_rate=b)`; uniform foil =
  `subsample=f, subsample_freq=1`. Keep LightGBM's log visible — never `verbose=-1` in the NB.
- **EFB (one-hot toy, groups 6/5/4 → 15 columns):** same-categorical one-hot columns are mutually
  exclusive (conflict 0.0000); cross-group conflict 0.0354 → EFB bundles 15 columns into ~3
  (tolerating a small conflict rate). Cuts histogram cost `O(n·#feat)→O(n·#bundles)`.

## Cell-by-cell (~22 cells, 4 figures) — intuition → implementation → interpretation

1. **(md) Header** — title; one concept (GOSS); place in the arc (NB 1 grew the trees leaf-wise; NB 2
   = how LightGBM gets light: fewer rows via GOSS, fewer features via EFB named). Warm, rigorous.
2. **(md) Recap** — the split-gain is a **sum over rows** of `(g,h)` (ch 09 NB 2 structure score); the
   per-row gradient `g` (ch 08) = how wrong/uncertain a row still is → a row's *influence* is `|g|`.
3. **(code) Setup** — imports, `viz.use_course_style()`, `SEED=0`; regression toy → `g=F0−y`, `h=1`;
   print `|g|` spread; define "row influence = `|g|`".
4. **(md) Uniform subsampling first** — keep fraction `f`, Horvitz–Thompson reweight `1/f`; blind spot:
   discards a large-`|g|` row as readily as a tiny one.
5. **(md) GOSS in words** — keep top-`a` by `|g|` (exact); sample `b` of the rest; up-weight sampled
   rest by `(1−a)/b`. "One-side" = only the small-gradient side is sampled.
6. **(code) Build GOSS by hand** — rank by `|g|`; `top_idx`; sample rest; weights; print kept fraction
   `a+b`, reweight `(1−a)/b`.
7. **(code) Fig 1 — gradient distribution + GOSS selection** — histogram of `|g|`, top-`a` kept
   highlighted, sampled rest marked.
8. **(md) Read the figure 1** — most rows small `|g|`; GOSS keeps the few influential, samples the crowd.
9. **(md) Why `(1−a)/b`** — small pool `(1−a)n`, keep `bn` → each stands for `(1−a)/b` peers; derive
   `E[G_goss]=G_full`, `H_goss=H_full` exactly.
10. **(code) Measure unbiasedness** — MC draws → `G_goss` ≈ `G_full`, `H_goss`=`n` exact (table;
    uniform unbiased too).
11. **(md) Unbiased isn't enough — variance** — gain `G²/(H+λ)` dominated by large-`|g|` rows; GOSS
    keeps them exactly (zero variance), uniform gambles → GOSS tighter when those rows dominate.
12. **(code) Fig 2 — GOSS vs uniform gain-estimate, concentrated gradients** — MC distribution at a
    fixed split; both centred on truth, GOSS visibly tighter.
13. **(md) Read the figure 2** — same fraction, same centre, far less scatter for GOSS.
14. **(md) When does GOSS win? concentration** — define = top-20 % share of `Σ|g|`; Ke Thm 3.2: GOSS
    beats uniform above a threshold; below it the `(1−a)/b` up-weight *adds* variance.
15. **(code) Fig 3 — concentration crossover (centerpiece)** — synthetic sweep: variance ratio vs
    concentration crossing 1.0 near 0.5; shade "uniform tighter | GOSS tighter"; overlay tabular band
    0.21–0.47.
16. **(md) Read the figure 3** — left of ~0.5 uniform wins; right GOSS wins, dramatically by 0.86;
    dense tabular gradients live near the crossover → GOSS ≈ uniform there.
17. **(md) The honest scope (lead with the limit)** — dense moderate data: GOSS ≈ full on fewer rows
    but ~ties uniform, wall-clock ~flat (measured); GOSS's payoff is the large/wide/sparse,
    concentrated-gradient regime (→ NB 5 dials `n` up).
18. **(code) The LightGBM switch** — `data_sample_strategy='goss'` (+`top_rate`/`other_rate`); ensemble
    row (GOSS ≈ full on 30 %) + ~flat timing; LightGBM log visible.
19. **(md) EFB — second lightness, named** — bundle approximately-exclusive sparse features (tolerates
    a small conflict rate, Ke §4 — more than concatenating one-hot); histogram cost
    `O(n·#feat)→O(n·#bundles)`.
20. **(code+md) Fig 4 — EFB exclusivity on a one-hot toy** — 2–3 mostly-exclusive sparse features → one
    bundle (offset ranges); annotate conflict (same 0.000 / cross 0.035; 15→3). Read-the-figure 4.
21. **(md) Your turn** — (a) vary `top_rate`/`other_rate` → watch the crossover move; (b) compute
    gradient concentration across boosting rounds; (c) where EFB helps (sparse/text) vs hurts (dense).
22. **(md) What you built + References** — GOSS by hand (unbiased reweight, variance gated by
    concentration, honest regime); EFB named. Refs: Ke et al. 2017 NeurIPS (GOSS §3.2 + Thm 3.2; EFB
    §4); ch 09 NB 2 gain; Friedman 2002 (stochastic GB = the uniform foil, DOI
    10.1016/S0167-9473(01)00065-2).

## `src/` & guards
- **No `src/` change** (reuse `viz`; `LGBMClassifier`/`data_sample_strategy`/`subsample`; numpy for the
  by-hand GOSS; pytest 20). Colours only from `ml_course.colors`; seeds fixed.
- Build from `build_ch10_nb2.py` (source of truth; rebuild right before `git add` — kernel-drift guard).
  Cell ids `cell-NN`; kernelspec `ml-course (3.12.12)`.
- **Never silence output** — no `verbose=-1` in the notebook; MC loops print progress.
- Exit guards: nbconvert exit 0 (4 figures), banned-word scan = 0, hex clean, ruff/black clean,
  output-free; **two-reviewer gate** (no BLOCK) → fold → **Rémy visual** → end-of-NB checklist
  (`gen_llms_txt.py`, `common_errors` +rows, `course_map` mark, pytest 20, STATE) → commit
  `feat(10_lightgbm): notebook 02 — GOSS and EFB` → `git merge --ff-only` into `chapter/10_LightGBM`.

## Verification (end-to-end)
1. nbconvert-execute a scratchpad copy → exit 0, 4 figures, anchors reproduce (H exact; crossover ~0.5;
   tabular band 0.21–0.47; ensemble GOSS ≈ full; wall-clock ~flat).
2. hex + banned + ruff + black → clean. 3. pytest → 20 passed. 4. Two-reviewer gate, then Rémy visual.
