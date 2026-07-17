# Physics-aware fPCA + DDPM for landing-trajectory generation

A three-stage generative pipeline for aircraft **landing approaches** (Zurich / LSZH,
14 000 flights up to the Final Approach Fix). Trajectories are compressed with
**functional PCA**, and a **denoising diffusion model (DDPM)** learns the distribution
of the low-dimensional fPCA weights. Generating a trajectory = sample latent → inverse
fPCA → integrate back to lat/lon. Physical plausibility is enforced with a data-driven
repair/rejection step.

```
raw trajectory (200 x 4)  --fPCA-->  weights w (m~13)  --DDPM-->  learn p(w)
        ^                                   |
        |  inverse fPCA + geodesic walk      v  sample
   generated trajectory  <---------------  w ~ p(w)
```

The DDPM **only ever sees latent vectors** — the fPCA basis is frozen after fitting and
never updates during diffusion training.

## Why this design

The four papers in `../` model the fPCA latent with a single Gaussian (Pepper 2024,
Dinh 2025), a GMM / neighbourhood sampler (Jarry 2022), or a normalizing flow
(Hodgkin 2025). This dataset has **5 distinct approach flows** — a genuinely multimodal
latent — so we use a **DDPM** as the latent density model, which captures multimodality
without picking a mixture order or fighting mode collapse.

## Layout

```
src/
  data/      prepare.py  (traffic .pkl -> processed.npz, 6-channel superset)
             dataset.py  (feature subsetting, standardize, split, bounds)
  fpca/      fpca.py     (discrete + Jarry B-spline fPCA, LatentScaler)
  ddpm/      schedule.py  denoiser.py (MLP/TCN/UNet latent + TrajTCN raw-space)
             ddpm.py     (shape-agnostic DDPM + EMA)
  cluster/   cluster.py (exploration)  assigner.py (persistable k-means)
  pipeline/  utils.py  reconstruct.py (physics + direct x/y)  checkpoint.py (single & per-cluster)
experiments/ e1_smoothing.py  e4_raw.py  eval_gen.py  common.py   (ablation study)
configs/     config.yaml + e2_global.yaml + e3_cluster.yaml
train.py  generate.py  evaluate.py  cluster.py  per_cluster_fpca.py
REPORT.md   ablation results & hypothesis verdicts     archive/  old artifacts
```

## Running on another device (e.g. your CUDA machine)

The pipeline is fully portable once `data/processed.npz` exists — **no `traffic`
library and no raw pickle needed** (only Stage-1 prepare requires them, and the
npz ships pre-built). Note `.gitignore` excludes `data/*.npz`, so copy it manually:

```bash
# on the target machine
git clone <this repo> && cd stochastic-modeling
scp <dev-machine>:.../stochastic-modeling/data/processed.npz data/   # 59 MB
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt         # install torch with CUDA per pytorch.org

bash run_ablation.sh                    # full chain E1 -> E2 -> E3 -> E4 (auto-detects CUDA)
# or pieces:
SKIP_E1=1 SKIP_E2=1 SKIP_E3=1 bash run_ablation.sh        # E4 only (U-Net, 2000 epochs)
E4_EPOCHS=500 bash run_ablation.sh                        # custom E4 budget
python experiments/e4_raw.py --features xy --arch unet    # single experiment by hand
tensorboard --logdir runs
```

Device selection is automatic (CUDA > MPS > CPU). E4's default budget is
2000 epochs (the reference model's converged setting); on an RTX-class GPU
expect minutes-per-100-epochs rather than the ~68 min/100 on Apple MPS.

## Ablation study

`REPORT.md` documents the ordered study: **E1** representation fidelity (spline vs
fPCA on dynamical channels vs direct x/y, with dead-reckoning control), **E2** global
fPCA+DDPM on x/y, **E3** per-cluster fPCA+DDPM, **E4** raw-space diffusion baselines.
Each experiment writes to `results/<exp>/`; shared generative metrics come from
`experiments/eval_gen.py` (marginal KS, sliced-Wasserstein on paths, endpoint KS,
envelope plausibility).

## Environment

The source data is a `traffic` library object and that library only lives in the sibling
project venvs, so **run everything with that interpreter** (Stage 1 needs it; the rest is
plain numpy/torch and is cached behind `processed.npz`):

```bash
VPY=/Users/meldor/Desktop/git/deep-traffic-generation-paper/.venv/bin/python
export PYTHONPATH=$(pwd)
```

(TensorBoard was added to that venv via `uv pip install tensorboard`.)

## Run

```bash
# 1. train (Stage 1 prepare runs automatically the first time; ~1 min to build the cache)
$VPY train.py                          # full run (configs/config.yaml: 400 epochs)
$VPY train.py --epochs 1 --tag smoke   # quick smoke test

# 2. generate
$VPY generate.py --n 2000              # -> results/generated.parquet (+ .npz)

# 3. evaluate
$VPY evaluate.py                       # -> results/*.png + metrics.json

# TensorBoard
$VPY -m tensorboard.main --logdir runs
```

A `Makefile` wraps these (`make train`, `make generate`, `make evaluate`, `make tb`,
`make smoke`) — set `VPY` at the top if your path differs.

## The three stages

**1 — Preprocessing** (`src/data`). Load the 200-step flights, keep
`[track, groundspeed, altitude, timedelta]` (the `track` channel uses `track_unwrapped`
so fPCA never sees the 0/360 jump), z-score each feature (train stats), and cache
lat/lon + FAF anchors. The data is already a fixed 200-point time grid, so resampling is
a guard; an arc-length re-parametrization is also implemented (`data.resample: arclength`).

**2 — fPCA** (`src/fpca`). One discrete-fPCA basis per feature (SVD of the centred
profiles); keep enough components for `explained_variance` (default 0.95). Scores are
concatenated into `w in R^m` (here **m = 13**: track 3, gs 6, alt 3, timedelta 1) and
standardized to ~N(0, I) before diffusion.

**3 — DDPM** (`src/ddpm`). A small residual **MLP** denoiser with sinusoidal time
embeddings predicts epsilon on the m-dim latent (cosine schedule, EMA weights). Sampling
clamps the latent each reverse step (~10 sigma) for stability. Generation decodes →
inverse-scales → `physics_repair` (clip groundspeed/altitude to the training envelope,
force monotone timedelta, wrap track) and optionally **rejects** out-of-envelope draws
(a la Hodgkin 2025). Reconstruction integrates `(track, groundspeed, dt)` **backward from
the FAF anchor**, mirroring `deep-traffic-generation-paper/dtg/traffic_builder.py`.

## Outputs

- `checkpoints/pipeline.pt` — frozen fPCA basis, both scalers, physical bounds, EMA DDPM weights (self-contained).
- `results/generated.parquet` — tidy, one row per timestep (features + lat/lon + timestamp).
- `results/*.png`, `results/metrics.json` — explained variance, latent 2D-PCA + per-dim histograms, real-vs-generated feature profiles, spatial tracks, and KS distances.

## Notes / knobs

- `configs/config.yaml` is the single source of truth; nothing is hardcoded in source.
- Fidelity levers: `fpca.explained_variance` (more components = lower reconstruction error, larger m), `ddpm.schedule`, `training.epochs`, `denoiser.*`.
- The included `results/` were produced by a **1-epoch smoke run** — fPCA reconstruction is
  already excellent, but the DDPM samples are noise until a full training run.
- A related differentiable OpenAP flight-envelope penalty lives in
  `../diffusion-models-lab/constraints/physics.py`; the repair step here is its lightweight,
  data-driven analogue on this feature set.
