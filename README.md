# LASER

**LASER: A Blind, Label-Free Targeted Attack on U-shaped Split Learning via Post-Training Representation Steering**

Runnable code and experiment pipelines for LASER (IEEE CSCloud 2026).

## Overview

U-shaped Split Learning (USL) splits a model into a client-side bottom network, a server-side middle network, and a client-side top classifier. The server receives only smashed data and is assumed to be isolated from raw inputs, labels, and the classifier.

LASER shows that a malicious-but-protocol-compliant server can still attack after normal training:

1. Cache the intermediate representations received during training.
2. Recover the semantic topology without labels via PCA, silhouette-based K search, and K-means.
3. Select a victim cluster and a proxy cluster from feature geometry.
4. Post-train its own middle network with a Guided Centroid pull and a Feature Anchoring MSE term, collapsing the victim cluster while preserving non-victim features.

The attack requires no labels, raw data, client model, or injected triggers.

## Repository layout

| Path | Contents |
| :--- | :--- |
| `src/` | Core library: USL model definitions, dataloaders, training utilities, clustering/K search, and the LASER attack. |
| `exp_robust/` | Label noise, class imbalance, dynamic client streams, blind victim selection, and attack-overhead experiments. |
| `exp_stealth/` | Client-side detectability and stealth analysis. |
| `exp_scale/` | CIFAR-100 scale experiment. |
| `exp_camera/` | Camera-ready experiments: seed variance, split depth, controls, separability analysis, audit prototype, and case study. |
| `exp/` | Early K-estimation and plotting scripts. |
| `normal_train_*.py`, `Atk_*.py`, `test_*.py` | Original monolithic training/attack/evaluation drivers. They contain hard-coded settings near the top and should be edited before use. |
| `requirements.txt` | Python dependencies. |

## Environment (camera-ready)

- Linux, NVIDIA GPU (RTX 4090D, 24 GB)
- Python 3.10.19, conda environment `cdssc`
- PyTorch 2.8.0 + CUDA 12.8, torchvision 0.23.0
- NumPy 2.0.2, SciPy 1.15.2, scikit-learn 1.7.2, pandas 2.2.3
- matplotlib 3.10.7, seaborn 0.13.2, joblib 1.5.2, torchattacks 3.5.1, tqdm 4.67.1

```bash
conda create -n cdssc python=3.10 -y
conda activate cdssc
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

If your driver requires a different CUDA build, install the matching PyTorch and
torchvision wheels from <https://pytorch.org/get-started/locally/> first.

## Data

CIFAR-10, CIFAR-100, MNIST, and Fashion-MNIST are downloaded automatically by
`torchvision` into `./data/` on first use. The `data/` directory is git-ignored.

## Quick start: VGG-16 / SB=1 / CIFAR-10

The modular `exp_robust/` pipeline is the recommended entry point. It trains a
clean split model, caches features, performs label-free clustering, runs LASER,
and evaluates the attack.

```bash
# Train the clean model (120-epoch controlled protocol, ~25 min on a 4090D)
python -m exp_robust.main train baseline

# Cache embeddings, cluster, run LASER, and evaluate
# (~10 GB cache; freed automatically after the condition finishes)
python -m exp_robust.main run baseline

# Regenerate tables and figures
python -m exp_robust.analyze
```

Results are written to `exp_robust/outputs/`.

The original 400-epoch training driver is `normal_train_cifar10.py`; the
monolithic attack driver `Atk_vgg16.py` contains hard-coded dataset, seed, and
epoch settings at the top of the file.

### Other robustness conditions

```bash
# baseline, lt (rho=100), lt10 (rho=10), noise20, noise40
python -m exp_robust.main train all
python -m exp_robust.main run all
python -m exp_robust.main run dynamic    # reuses the baseline model
python -m exp_robust.main run blind      # attacks every discovered cluster
python -m exp_robust.main run q2         # attack overhead / low-resource sweep
python -m exp_robust.analyze
```

## Other experiment pipelines

### Stealth / client-side detection (`exp_stealth/`)

Reuses the clean model from `exp_robust/outputs/models/baseline`.

```bash
python -m exp_stealth.run
python -m exp_stealth.multiattack
python -m exp_stealth.baselines
python -m exp_stealth.analyze
```

### CIFAR-100 scale experiment (`exp_scale/`)

```bash
python -m exp_scale.main train cifar100
python -m exp_scale.main run cifar100
```

### Camera-ready controlled experiments (`exp_camera/`)

All runs write to `exp_camera/outputs/`.

```bash
# 3 seeds x 3 VGG-16 split points on CIFAR-10
python -m exp_camera.seed_split batch
python -m exp_camera.seed_split attack-batch
python -m exp_camera.report

# CIFAR-100 split-depth study (SB=2/3; SB=1 reuses exp_scale)
python -m exp_camera.cifar100_split batch
python -m exp_camera.report_c100

# Controls, separability analysis, audit prototype, and case study
python -m exp_camera.controls
python -m exp_camera.separability
python -m exp_camera.audit
python -m exp_camera.case_study
```

The analysis/report commands read results produced by earlier runs.

## Outputs and disk usage

Each CIFAR-10 condition produces roughly 10 GB of feature cache plus a ~1.1 GB
model; attack checkpoints add about 1.2 GB each. Plan for at least 30 GB of free
disk space.

Large artifacts are git-ignored:

- `data/`
- `exp_*/outputs/{models,cache,attacks,clusters,attacked_servers,eval}/`
- nested `exp_camera/outputs/cifar100/{models,cache,attacks,clusters}/`
- `*.pt`, `*.pth`, `*.ckpt`, `*.npz`
- logs and LaTeX build files
