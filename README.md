# EdenGNN

**EdenGNN** (Equivariant Density Graph Neural Network) is an E(3)-equivariant Graph Neural Network framework designed for accurate and efficient prediction of charge density and electronic structures directly from atomic configurations.

---

- [Features](#features)
- [Installation](#installation)
- [Usage](#-usage)
  - [Data Preparation](#data-preparation)
  - [Training](#training)
  - [Prediction](#prediction)
- [Troubleshooting](#troubleshooting)
- [Universal Model](#universal-model)
- [Citation](#citation)

---

## Features

- **Bypassing the self-consistent calculations of the Kohn-Sham (KS) equations:** The predicted charge density can be used to construct the KS Hamiltonian and calculate the electronic structures.

- **Support various DFT implementations:** EdenGNN can predict the **augmentation occupancies** in the projector augmented-wave (PAW) formalism, making it capable of predicting the electronic structures with the accuracy of both the plane-wave (PW) basis and the linear combination of atomic orbital (LCAO) basis.


## Installation



### Prerequisites

Ensure you have the following dependencies installed in your python environment:

- `torch`
- `pytorch-geometric`
- `e3nn`
- `nequip`
- `lightning`
- `meson`
- `numpy`

### Install EdenGNN

- Clone the repository and install it via `pip`:

```bash
git clone https://github.com/rubenlee11/EdenGNN.git
cd EdenGNN
pip install .
# for offline installation, use:
# pip install . --no-index --no-build-isolation
```

- Performing non-self-consistent calculations in **OpenMX** with predicted charge density requires some modifications. Please compile **OpenMX** with the patch files provided in `./scripts/openmx/patch_nsc`.

## Usage

* **Important Note:** The following instructions are tailored for the VASP software using PAW pseudopotentials.

* We highly recommend keeping all DFT calculation settings consistent across your dataset. Always use consistent pseudopotentials in the training and predicting stages. 

A complete workflow example can be found in `examples/si`. You can also explore it interactively using the provided Jupyter Notebooks. Detailed parameter descriptions are available in the example `config.yaml` file.

### Data Preparation

#### 1. Calculate Superposition of Atomic Charge Density (VASP only)

This step is necessary for **VASP** but not **OpenMX**. In self-consistent and non-self-consistent calculations, **OpenMX** stores and reads the difference charge density in the restart files.

Run **VASP** to obtain the Superposition of Atomic Charge Density (SACD). This serves as the physical prior for the $\Delta$-Learning strategy and is **required for both training and prediction**.
- **VASP Tags:** Set `ICHARG = 12`, `NELM = 0`, and `ISPIN = 1`.

#### 2. Calculate Self-Consistent Charge Density (Training Only)

Perform normal self-consistent DFT calculations to generate the ground truth data.
- **Consistency is Key:** When using **VASP** to label data, ensure the precision settings match those used in the SACD calculations (except for `KPOINTS` and `ISMEAR`). Inconsistent grids between pseudo charge densities will raise errors.
- **Precision Recommendation:** For an 80 GB GPU, we recommend using the `PREC = Normal` accuracy level for preparing training datasets and making predictions.

#### 3. Create Filelists

Create text files for training, validation, and test sets. 

For **VASP** mode, the filelists should contain the absolute paths to the working directories of the **SACD** calculations.

Example:
```text
/root/.../si/vasp_run_sacd/1_0
/root/.../si/vasp_run_sacd/1_1
...
```

For **OpenMX** mode, the filelists should contain the absolute paths to: the working directories of the DFT calculations in the training stage; or the cif files in the prediction stage.

Example:
```text
/root/.../si/openmx_run/1_0
/root/.../si/openmx_run/1_1
...
```

```text
/root/.../si/openmx_run/1_0.cif
/root/.../si/openmx_run/1_1.cif
...
```


### Training

#### 1. Configure `config.yaml`

Modify your configuration file with the following key settings:

- **Set `run.mode` to `train`.** 

- **Set `run.task`:**
  - `0`: Train the pseudo charge density.
  - `1` Train the augmentation occupancies.
  - `2` Train both. 
  
- **Fine-tuning (Optional):** set `run.checkpoint` to the path of your checkpoint file. 

- **Set `data.dft_software`**.

- **Directories:** 
  - `run.save_dir`: Directory to save logs and checkpoints.
  - `data.dir`: Directory storing your SCF calculations. Directory names here must match those in your filelists. This tag is needed when `data.dft_software` is `vasp`. 
  - `data.path_train` & `data.path_val`: Paths to your training and validation filelists.

- **Optimization:** 
  - Set the initial learning rate via `optimize.lr`.
  - The total loss is a weighted sum: $\mathcal{L} = \sum_{i} w_i \mathcal{L}_i$. Specify the gradient ratios in `optimize.loss_dict.grid_func_out` and `optimize.loss_dict.aug_tensor` if training both tasks (`run.task: 2`). When using **OpenMX**, only the charge density is trained, hence the weight of `grid_func_out` is `1.0`.


- **Model Parameters:** 
  - Set `model.lmix_max` to match the `LMAXMIX` tag used in your VASP dataset (must be consistent across all structures). 
  - Default values for other model parameters are generally robust.

#### 2. Start Training
```bash
python scripts/train.py --config path/to/config.yaml
```

### Prediction

#### 1. Configure `config.yaml`

- **Set `run.mode: predict`**.

- **Set `data.dft_software`**.

- **Set `run.checkpoint`** to the path of the checkpoint file of your trained model. 

- **Set `path_predict`** to the filelist (explained in the **Data Preparation** section).

- **Set `path_template`** to the path of the template files of band structure calculation settings for the DFT softwares.

- **Note:** All model hyperparameters in the `config.yaml` (including `run.task`) must strictly match those of the checkpoint.

#### 2. Run Prediction
```bash
python scripts/train.py --config path/to/config.yaml
```

## Troubleshooting

If you encounter with out of memory (OOM) warnings when training the pseudo charge density, try the following solutions:

1. Use a GPU with larger VRAM.
2. Decrease the maximum number of grids for structures in your training set.
3. Reduce the number of channels: `model.probe.conv.n_channels`.
4. Decrease the `CHUNK_CRITERION` variable in `src/model/model.py`. This parameter controls the number of radial points processed simultaneously during inference.

## Universal Model

**EdenGNN-Uni** is a pre-trained universal charge density model trained on non-magnetic materials from the Materials Project database. 

To use EdenGNN-Uni for predicting electronic structures, please ensure that your input structures use the **exact same PAW pseudopotential versions** as those used as the training set.

## Citation

If you find EdenGNN useful in your research, please cite our paper:
```
X. Li, Z. Xin, H. Yu, Y. Zhong, X. Gong, and H. Xiang, Efficient E(3)-Equivariant Framework for Universal Charge Density Prediction, arXiv:2510.00788.
```