
# 🪩 Riemannian Variational Flow Matching 🪩

This is the main repository of the paper **[Riemannian Variational Flow Matching for Material and Protein Design](https://arxiv.org/abs/2502.12981v2)**, accepted at ICLR 2026. 

**Authors**: Olga Zaghen, Floor Eijkelboom*, Alison Pouplin*, Cong Liu, Max Welling, Jan-Willem van de Meent, Erik J. Bekkers

\* equal contribution

___
## Overview 
This repository contains the official implementation of the **Riemannian Gaussian Variational Flow Matching** (RG-VFM) model, together with all the related **synthetic experiments** described in the paper. Since each experiment was implemented independently, we provide separate repositories for clarity:

* Synthetic **checkerboard** experiments are in **this repository**,
* **MOF generation** experiments are in the **[V-MOFFlow repository](https://github.com/olgatticus/rg-vfm-mofflow)**.
* **Protein backbone generation** experiments are in the **[V-ReQFlow repository](https://github.com/olgatticus/rg-vfm-reqflow)**.

The experiments implemented in this repo consist in learning to generate a checkerboard distribution on the hypersphere and hyperbolic space, and the models that can be tested are intrinsic and extrinsic RG-VFM (ours), Riemannian Flow Matching (RFM), Variational Flow Matching (VFM) and standard Conditional Flow Matching (CFM).

<p align="center">
    <img src="misc/sphere_checkerboard_results.png" alt=" Comparison of the spherical checkerboard distribution generated with different methods" width="100%"/>
</p>

___

## Requirements

The main dependencies are:
- Python 3.12
- PyTorch 2.5+
- NumPy, SciPy, Matplotlib
- torchdiffeq
- wandb (optional, for logging)
- tqdm

## Installation

Clone the repository and create the conda environment:

```bash
git clone https://github.com/olgatticus/rg-vfm.git
cd rg-vfm
conda env create -f env_rvfm.yml
conda activate rvfm
pip install -e .
```

## Synthetic Experiments

The repository contains scripts to run checkerboard distribution learning on two different manifolds:
- **Sphere** (`main_sphere.py`): Learning checkerboard on the 2-sphere S²

<p align="center">
    <img src="misc/probability_paths_int_sphere.jpeg" alt="Sphere checkerboard - intrinsic" width="100%"/>
    <br><em>RG-VFM Intrinsic on S²</em>
</p>

<p align="center">
    <img src="misc/probability_paths_ext_sphere.jpeg" alt="Sphere checkerboard - extrinsic" width="100%"/>
    <br><em>RG-VFM Extrinsic on S²</em>
</p>

- **Hyperboloid** (`main_hyperboloid.py`): Learning checkerboard on the hyperboloid H²

<p align="center">
    <img src="misc/probability_paths_hyperboloid_top_with_x1_int.jpeg" alt="Hyperboloid checkerboard - intrinsic" width="100%"/>
    <br><em>RG-VFM Intrinsic on H²</em>
</p>

<p align="center">
    <img src="misc/probability_paths_hyperboloid_top_with_x1_ext.jpeg" alt="Hyperboloid checkerboard - extrinsic" width="100%"/>
    <br><em>RG-VFM Extrinsic on H²</em>
</p>

### Model Options

| Argument | Options | Description |
|----------|---------|-------------|
| `--flow` | `vanilla`, `variational` | Flow matching method. `vanilla` = RFM, `variational` = VFM |
| `--geometry` | `euclidean`, `riemannian` | Geometry type for the loss computation |
| `--support` | `intrinsic`, `extrinsic` | Intrinsic (on-manifold) or extrinsic (ambient space) parametrization of the base distribution p₀ |
| `--p0_distribution` | `uniform`, `gaussian` | Base distribution p₀ |

### Hyperparameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--batch_size` | 1024/2048 | Training batch size |
| `--hidden_dim` | 128 | Hidden dimension of the neural network |
| `--num_epoch` | 3000 | Number of training epochs |
| `--num_samples` | 10000 | Number of training samples |
| `--lr` | 1e-3 | Learning rate |
| `--noise_scale` | 1e-1 | Noise scale for flow matching |
| `--train` | True | Enable training (use `--no-train` to load a saved model) |
| `--wandb` | False | Enable Weights & Biases logging |

### Running the Experiments

All the considered models, i.e. RG-VFM (ours), RFM, VFM and CFM, can be tested through the following combinations of arguments.

#### Sphere Experiments

```bash
# RG-VFM Intrinsic (Ours)
python scripts/main_sphere.py --flow variational --geometry riemannian --support intrinsic --p0_distribution gaussian

# RG-VFM Extrinsic (Ours)
python scripts/main_sphere.py --flow variational --geometry riemannian --support extrinsic --p0_distribution gaussian

# RFM (Riemannian Flow Matching)
python scripts/main_sphere.py --flow vanilla --geometry riemannian --support intrinsic --p0_distribution gaussian

# VFM (Variational Flow Matching - Euclidean)
python scripts/main_sphere.py --flow variational --geometry euclidean --p0_distribution gaussian

# CFM (Conditional Flow Matching)
python scripts/main_sphere.py --flow vanilla --geometry euclidean --p0_distribution gaussian
```

#### Hyperboloid Experiments

```bash
# RG-VFM Intrinsic (Ours)
python scripts/main_hyperboloid.py --flow variational --geometry riemannian --support intrinsic --p0_distribution gaussian

# RG-VFM Extrinsic (Ours)
python scripts/main_hyperboloid.py --flow variational --geometry riemannian --support extrinsic --p0_distribution gaussian

# RFM (Riemannian Flow Matching)
python scripts/main_hyperboloid.py --flow vanilla --geometry riemannian --support intrinsic --p0_distribution gaussian

# VFM (Variational Flow Matching - Euclidean)
python scripts/main_hyperboloid.py --flow variational --geometry euclidean --p0_distribution gaussian

# CFM (Conditional Flow Matching)
python scripts/main_hyperboloid.py --flow vanilla --geometry euclidean --p0_distribution gaussian
```

### Output

Results are saved to `experiments/<manifold>/<flow>/<geometry>/<support>/<p0_distribution>/` and include:
- `config.txt`: Experiment configuration
- `probability_paths.png`: Visualization of the learned flow at different timesteps
- `density_unwrapped.png`: 2D projection of the generated distribution
- `coverage_c2st_metrics.txt`: Evaluation metrics (coverage and C2ST scores)
- `model.pt`: Saved model weights



## BibTeX
If you find this code useful, please consider citing our paper:
```
@article{zaghen2025riemannian,
  title={Riemannian Variational Flow Matching for Material and Protein Design},
  author={Zaghen, Olga and Eijkelboom, Floor and Pouplin, Alison and Liu, Cong and Welling, Max and van de Meent, Jan-Willem and Bekkers, Erik J},
  journal={arXiv preprint arXiv:2502.12981},
  year={2025}
}
```
