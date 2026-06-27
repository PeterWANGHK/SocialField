# SocialFieldGraph

Field-theoretic **heterogeneous interaction-graph** model of multi-class road-user
behaviour, learned by supervised trajectory forecasting on the levelX **uniD**
(campus) and **inD** (intersection) bird's-eye-view datasets.

A learned, **class-conditioned interaction potential** `U_{A→B}(Δpos, Δvel)` (a
"social field") supplies the force `f = −∇U` that drives the messages of a
spatio-temporal attention graph (HEIGHT / AttnGraph-style). The learned 4×4
class-pair field is directly interpretable: it recovers the road-user dominance
hierarchy **truck > car > bicycle > pedestrian** purely from naturalistic data.

## Environment
Use the base anaconda interpreter (torch 2.11 + CUDA, torch_geometric):
`C:\Users\ymshu\anaconda3\python.exe`. Run all commands from the repo root
`C:\DREAM_final` so `import tracks_import` resolves.

## Pipeline
```bash
PY="C:/Users/ymshu/anaconda3/python.exe"

# 1. Data — extract multi-class (obs=8 / pred=12 @ 2.5 Hz) windows
$PY -m SocialFieldGraph.data.extract_windows --dataset uniD --recording 0
$PY -m SocialFieldGraph.data.dataset            # PyG graph smoke test

# 2. Baselines (ADE/FDE floor)
$PY -m SocialFieldGraph.baselines --dataset uniD --split val

# 3. Field module sanity
$PY -m SocialFieldGraph.tests.test_field

# 4. Train (full field model) + ablations
$PY -m SocialFieldGraph.train --epochs 50 --tag sfg_field
$PY -m SocialFieldGraph.train --epochs 50 --use_field 0        --tag sfg_nofield
$PY -m SocialFieldGraph.train --epochs 50 --class_conditioned 0 --tag sfg_shared
$PY -m SocialFieldGraph.train --epochs 50 --physics_init 0      --tag sfg_noinit

# 5. Evaluate + interpret
$PY -m SocialFieldGraph.eval --tag sfg_field --dataset uniD --split test
$PY -m SocialFieldGraph.eval --tag sfg_field --dataset inD  --split xdomain   # cross-domain
$PY -m SocialFieldGraph.analysis.field_maps --tag sfg_field   # 4x4 field maps + tables
```

## Results (uniD val, 50 epochs)
| Model | ADE | FDE | car ADE | truck ADE |
|---|---|---|---|---|
| constant velocity | 0.472 | 1.158 | 0.784 | 0.687 |
| analytic field (no learning) | 1.283 | 2.601 | 1.208 | 1.170 |
| vanilla GAT (no field) | 0.305 | 0.829 | 0.412 | 0.836 |
| shared field (no class cond.) | 0.300 | 0.767 | 0.378 | 0.333 |
| **SFG full (field + class-cond + physics-init)** | **0.290** | **0.746** | 0.348 | 0.361 |

Held-out uniD/test: ADE 0.257 / FDE 0.645. Cross-domain inD (zero-shot): ADE 0.946
(≈ CV 0.955; campus→intersection gap motivates fine-tuning).

The field helps most on the high-mass, data-scarce vehicle classes (truck −57% vs
vanilla GAT). The learned strength/asymmetry tables (`field_maps`) expose the
class-conditioned interaction structure — the "how the field explains interaction"
deliverable.

## Layout
```
config.py          canonical classes, radii, window params, splits
data/              extract_windows.py (multi-class windows), dataset.py (PyG graphs)
models/            field_potential.py (learned field + force), predictor.py (encoder+attention+decoder)
baselines.py       constant-velocity, analytic-field
metrics.py         ADE/FDE overall + per class
train.py eval.py   training / evaluation + ablation flags
analysis/          field_maps.py (4x4 learned field), attention_analysis.py
tests/             test_field.py
```
Reuses: `tracks_import.read_from_dataset`, `rl/risk/scene_conditioning.score_agent_relevance`,
DRIFT/APF/ADA field priors, and the PINN/GAT training patterns already in the repo.
