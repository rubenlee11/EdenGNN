# VASP: predict with your model

Use a trained checkpoint to predict PAW charge density, then calculate bands and DOS at fixed density.

## 1. Prepare the SAD prior

Use `scripts/vasp/vasp_sad.py` and run VASP through your configured `scripts/vasp/submit.sh`, as in [training data preparation](vasp-training.md#1-prepare-training-data).

Use the same PAW files, `ENCUT`, `PREC`, FFT-grid conventions, and `LMAXMIX` as training. Generate `KPOINTS_SCF` and `KPOINTS_BAND` for the final calculations.

## 2. Predict charge density

Copy the training configuration and update:

| Parameter | Setting |
| --- | --- |
| `run.mode`, `run.resume` | `predict`, `false` |
| `run.checkpoint` | Saved checkpoint |
| `run.save_dir` | Prediction output directory |
| `data.path_predict` | Filelist of **SAD directories** |
| `data.use_bin` | `false` |

Preserve the checkpoint's task, model parameters, dtype, SAD setting, and PAW configuration. Use `edengnn-predict` with this configuration.

The prediction is a **file** at `run.save_dir/edengnn/<timestamp>/predict/<name>`:

- A joint model (`run.task=2`) outputs a complete CHGCAR; use it directly.
- Separate density and augmentation models require two predictions. Concatenate density first and augmentation second to form CHGCAR.

## 3. Calculate bands and DOS

Prepare `band/` and `nscf/` directories using the generated POSCAR, POTCAR, INCAR, and predicted CHGCAR. Set `ICHARG=11`, positive `NELM`, and `LCHARG=False`.

Use `KPOINTS_BAND` for `band/` and `KPOINTS_SCF` for `nscf/`. Run VASP using `scripts/vasp/submit.sh`.

List all structure result directories in one plotting filelist and use `scripts/vasp/plot_band.py`. Each listed directory contains `band/` and `nscf/`; the resulting figure is `band.png`.

## 4. Compare with DFT

Add matching reference calculations under each structure result directory:

| Directory | Calculation |
| --- | --- |
| `nscf/` | Predicted density, uniform mesh |
| `band/` | Predicted density, high-symmetry path |
| `scf/` | Converged DFT reference, uniform mesh |
| `band_scf/` | Fixed reference density, same high-symmetry path |

Reuse converged training labels when settings match. Preserve CHGCAR, KPOINTS, and `vasprun.xml` in the uniform-grid calculations and KPOINTS and `vasprun.xml` in the band calculations.

Use `scripts/vasp/stat.py` with the structure filelist and a statistics output directory. It produces `band_compare.png` and `stat_mae.json`. A separate reference root can be specified with `--dir-tar`.
