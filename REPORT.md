# Ablation study — representations and diffusion for landing-trajectory generation

**Data**: 14,000 Zurich (LSZH) landing approaches, 200 timesteps each, up to the FAF.
Channel superset: `x, y` (UTM m), `altitude`, `timedelta`, `groundspeed`, `track` (unwrapped).
Split 90/10 train/val (fixed seed 42, identical across all experiments).

**Feature sets**: `DYN = [track, groundspeed, altitude, timedelta]` (position only
recoverable by dead-reckoning) vs `XY = [x, y, altitude, timedelta]` (position direct).

**Hypotheses** (stated up front, assessed per experiment):
- **H1** — splines smooth gs/track → dead-reckoned reconstruction degrades badly → drop splines.
- **H2** — fPCA directly on dynamical channels also smooths → still bad.
- **H3** — therefore model x/y directly.
- **H4** — global fPCA+DDPM on x/y works, but not well → motivates clustering.
- **H5** — per-cluster fPCA+DDPM is the best fPCA-based pipeline.
- **H6** — raw-space diffusion baselines on DYN and XY close the loop.

Reproduce: `experiments/e1_smoothing.py`, then the chain
`train/generate/evaluate` with `configs/e2_global.yaml` / `configs/e3_cluster.yaml`,
then `experiments/e4_raw.py --features {dyn,xy}`; shared scores via `experiments/eval_gen.py`.
All artifacts under `results/<exp>/`.

**A note on splines — motivation vs. value here.** Wherever splines appear in
these experiments, keep the framing honest: in the source literature the spline
stage is *load-bearing* — Jarry 2022 needs curves in the Sobolev space 𝕎² for
his FDA/Karhunen-Loève machinery, continuous evaluation at arbitrary t for his
deformation operators (rotate/dilate/cut at "the time the mean crosses 1000 ft"),
registration at non-grid landmarks, and smooth profiles for noise simulation.
**None of those obligations bind our setting** (pre-gridded common 200-tick data,
vector-consuming DDPM, no deformation ops), so for us the spline stage reduces to
a smoothing prior — and whether that prior *adds any value for generation* is an
open question we explicitly assess rather than assume:
- representation side (**answered by E1**): harmful on dynamical channels
  (double-smoothing), the most compact accurate basis on x/y (202 m @ ~20 coeffs);
- generation side (**pending ablation**): latent DDPM on spline-fPCA scores
  (`fpca.basis: bspline`, one config line) vs discrete-fPCA E2/E3 — would test
  whether construction-guaranteed smoothness of decoded samples beats
  `physics_repair`-based cleanup on the shared scorecard. To be queued after the
  raw-space matrix; splines regain first-class status only if/when raw irregular
  ADS-B ingest (spec 1.1-1.3) lands, where a common basis becomes mandatory.

---

## E1 — Representation study (no training): does smoothing kill the dynamics?

**Setup.** Fit each representation on train, encode→decode val, then measure
(a) raw-unit reconstruction RMSE, (b) high-frequency retention of gs/track
(mean |Δ| ratio recon/real; 1 = nothing smoothed), and (c) **position error**:
DYN representations are dead-reckoned forward from the true entry point and
compared to the true x/y path; XY representations decode position directly.
Control row: dead-reckoning with the **real, unsmoothed** dynamics.

**Results** (val split; full table in `results/e1_smoothing/table.md`):

| representation | m | path mean (m) | path final (m) | hf gs | hf track |
|---|---|---|---|---|---|
| **control: dead-reckon real dynamics** | — | **459** | **795** | 1.00 | 1.00 |
| dyn · bspline-20 | 55 | 562 | 999 | 0.88 | 1.10 |
| dyn · bspline-40 | 94 | 486 | 844 | 0.94 | 1.12 |
| dyn · bspline-80 | 140 | 473 | 821 | 0.96 | 1.10 |
| dyn · fPCA-0.95 | 13 | 5,473 | 8,048 | 0.72 | 0.80 |
| dyn · fPCA-0.99 | 25 | 2,169 | 3,236 | 0.83 | 0.94 |
| dyn · fPCA-0.999 | 65 | 570 | 971 | 0.93 | 1.11 |
| xy · bspline-20 | 41 | **202** | **36** | — | — |
| xy · bspline-40 | 52 | 201 | 33 | — | — |
| xy · fPCA-0.99 | 13 | 1,675 | 33 | — | — |
| xy · fPCA-0.999 | 29 | 622 | 33 | — | — |

**Verdicts.**
- **H1 — REFUTED in its mechanism, upheld in its conclusion.** Splines barely
  smooth the dynamics (hf gs 0.88–0.96; track even *rings* slightly, hf > 1), and the
  spline rows sit only ~5–20% above the control. The real problem is the
  **parametrization itself**: even *perfect* dynamics dead-reckon to 459 m mean /
  795 m final error — integration accumulates error and has an irreducible floor.
  So yes, drop the (track, gs) route — but because of integration, not spline smoothing.
- **H2 — CONFIRMED at practical budgets.** fPCA at 0.95/0.99 EV visibly smooths
  (hf gs 0.72/0.83) and path error explodes to 5.5 km / 2.2 km. Only at 0.999
  (m = 65) does it approach the integration floor — at which point the latent is
  large and you *still* can't beat the control.
- **H3 — CONFIRMED, and stronger than hypothesized.** Direct x/y error is
  **bounded, not accumulated**: final-point error is ~33 m for *every* x/y variant —
  vs ≥ 795 m for anything dead-reckoned, control included. The approach *ends at the
  FAF*, exactly where the dynamical route is at its worst and the x/y route at its best.
- **Twist worth reporting**: splines are *excellent* on x/y (bspline-20: 202 m mean,
  36 m final — the best row in the table). "Splines are bad" is only true for the
  dynamical channels; on smooth positional curves the spline basis is the most
  efficient of all. The pipeline still uses discrete fPCA on x/y for simplicity, but
  the honest statement is: **the dynamical parametrization is what's bad; splines
  neither caused it nor can fix it.**

Figures: `dyn_overlays.png` (gs/track smoothing), `path_comparison.png`
(integrated vs direct paths), `path_error_bars.png` (log-scale summary).

---

## E2 — Global fPCA + DDPM on x/y (single model)

**Setup.** One discrete fPCA (EV 0.99 → m = 13) over all 12,600 training flights,
one MLP-denoiser DDPM (1200 epochs, cosine schedule, EMA). `configs/e2_global.yaml`.

**Training.** Converged cleanly: val loss 0.92 → 0.36, flat from epoch ~300 with
no overfitting (12.6k flights for a 13-dim latent is plenty of data).

**Generative scorecard** (`results/e2_global/gen_metrics.json`, 2000 samples):

| metric | value |
|---|---|
| KS marginal x / y / altitude / timedelta | 0.013 / 0.013 / 0.014 / 0.002 |
| sliced-Wasserstein on x/y paths | 0.841 km |
| KS endpoint x / y | **0.490 / 0.476** |
| within training envelope | 100% |

**Reading.** Per-feature marginals are excellent — a single DDPM over the global
latent *can* model the 5-corridor multimodal distribution (the "meaningless global
mean" does not hurt a diffusion model the way it hurts a Gaussian latent).
The failure mode is the **endpoint**: real flights end inside a ~1.5 km FAF ring;
generated endpoints scatter (KS ≈ 0.49). With only 3 x-modes and 3 y-modes, the
endpoint is a delicate joint combination of scores that the global model does not
pin down.

**H4 verdict — PARTIALLY CONFIRMED.** "Works but not the best" is accurate, with
the deficiency localized: marginals fine, path-set decent, FAF convergence poor.

## E3 — Clustered: per-cluster fPCA + per-cluster DDPM

**Setup.** k-means (k = 5) on flattened standardized x/y → 5 corridors
(sizes 1451/3766/2642/3952/2189); an independent fPCA (EV 0.99 → m = 22/21/19/18/19)
+ DDPM per cluster; generation samples clusters at their empirical frequencies.
`configs/e3_cluster.yaml`.

**Training.** All five models show **mild overfitting**: val loss bottoms
(~0.35–0.41) around epoch ~300 and drifts up to ~0.38–0.48 by 1200, while the
global E2 model stayed flat — each per-cluster model sees 5–10× less data.
Practical note: **~300 epochs would be the better budget for per-cluster models**
(EMA weights soften the damage here).

**Generative scorecard** (vs E2):

| metric | E2 global | E3 per-cluster |
|---|---|---|
| KS marginal x / y / alt | 0.013 / 0.013 / 0.014 | **0.010 / 0.008 / 0.010** |
| sliced-Wasserstein paths | 0.841 km | **0.598 km** (−29%) |
| KS endpoint x / y | 0.490 / 0.476 | 0.445 / 0.425 |
| within envelope | 100% | 100% |

**H5 verdict — CONFIRMED for path realism, NOT for the endpoint.** Clustering
buys a −29% path-distribution distance and better marginals, at the cost of
per-cluster overfitting. But the **FAF-endpoint gap persists in both** (KS ~0.43
vs 0.49): E1 showed the basis *can* pin the endpoint to ~33 m given real scores,
so this is a *learned-score-distribution* problem, not a representation problem.
**Future fix**: endpoint-registered basis (model `x(t) − x(T)`, Jarry-style
registration, with the endpoint distribution modelled separately) or
endpoint-conditioned sampling.

## E4 — Raw-space diffusion (no fPCA): DYN and XY

**Setup.** The reference **TCN U-Net** from `diffusion-models-lab` (ported
verbatim to `src/ddpm/tcn_unet.py`: 4 levels 64/128/256/512, dilated res-blocks,
16.7M params, linear β schedule, reference warm-up→decay LR, batch 128) diffusing
the standardized (4, 200) trajectory tensor directly. Budget: **2000 epochs**
(the reference's converged setting). `experiments/e4_raw.py --arch unet`.

**Status: pending — to be run on the CUDA machine** (`bash run_ablation.sh`
with `SKIP_E1=1 SKIP_E2=1 SKIP_E3=1`, or `make e4`). On Apple MPS this model
costs ~41 s/epoch (≈23 h per variant at 2000 epochs), so training was moved
to the GPU box.

**Preliminary data point (archived, `archive/e4_simple_tcn/`)**: a small flat
TCN (200k params, 300 epochs) on DYN converged in-loss (val 0.026) but scored
5–10× worse marginals than E2/E3 (track 0.061, altitude 0.071, timedelta 0.103)
and sliced-W ≈ 9.8 km on centred paths — evidence that raw-space diffusion
needs real capacity (hence the U-Net), and an early sign that the DYN feature
set is as problematic generatively as E1 predicted representationally.
*Caveat: DYN spatial metrics are computed on start-centred dead-reckoned paths
(generated dynamics carry no absolute position — there is no FAF to hit), so
they are not directly comparable with the x/y rows.*

**H6 verdict — OPEN** until the U-Net runs complete.

## Final comparison (state as of this report)

| system | latent | marginal KS (x/y or trk/gs) | sliced-W paths | endpoint KS | notes |
|---|---|---|---|---|---|
| E2 global fPCA+DDPM | 13 | 0.013 / 0.013 | 0.841 km | 0.49 / 0.48 | one model, no FAF pinning |
| **E3 per-cluster fPCA+DDPM** | ~20×5 | **0.010 / 0.008** | **0.598 km** | 0.45 / 0.43 | best overall; overfits past ~300 ep |
| E4 flat-TCN dyn (archived) | — (800 raw dims) | 0.061 / 0.013 | ~9.8 km (centred) | n/a (no absolute pos.) | capacity-starved baseline |
| E4 U-Net dyn / xy | — | *pending (CUDA)* | | | reference architecture, 2000 ep |

**Story so far**: E1 kills the dynamical parametrization (integration error floor,
unbounded endpoint drift) → x/y direct. E2 shows one DDPM handles the multimodal
latent but misses the FAF. E3 (cluster-then-model) is the best system measured,
improving path realism 29%, though the endpoint gap needs a registration/conditioning
fix rather than more clustering. E4 will answer whether 16.7M raw-space parameters
can beat a 13–20-dim fPCA latent at all — at ~1000× the latent models' size.
