# Train and use a custom model

[User guide](index.md) · [VASP prediction](vasp.md) · [ABACUS prediction](abacus.md)

Training starts from converged DFT labels. Use the ordinary `~/.edengnn/config.yaml` template for a new model; universal-model configurations belong with their pretrained checkpoints.

## Prepare the workspace

Activate the [configured environment](setup.md) and define a fresh working directory outside the checkout:

```bash
set -e
export E="$HOME/.edengnn"
source "$E/cpu.sh"
source "$E/cuda.sh"
export PYTHONDONTWRITEBYTECODE=1
export W=/absolute/path/to/new/training_run
mkdir -p "$W"
```

Create `W/structures.txt` with one absolute structure-file path per line. Split training and validation by structure before constructing their calculation-directory filelists. Use disjoint structures for meaningful validation.

## VASP labels

Use the ordinary PAW map and INCAR. Generate and run SAD for every structure:

```bash
python "$E/scripts/vasp/vasp_sad.py" \
  --filelist "$W/structures.txt" --save-dir "$W/sad" \
  --path-pp "$E/scripts/vasp/pp.json" \
  --path-input "$E/scripts/vasp/INCAR" \
  --for-scf --for-band --nproc 1
for path in "$W/sad/"*; do
  (cd "$path" && bash "$E/scripts/vasp/submit.sh")
done
cp -r "$W/sad" "$W/scf"
python - <<'PY'
import os
import shutil
from pathlib import Path
from pymatgen.io.vasp.inputs import Incar
for path in (Path(os.environ['W']) / 'scf').iterdir():
    shutil.copy2(path / 'KPOINTS_SCF', path / 'KPOINTS')
    incar = Incar.from_file(path / 'INCAR')
    incar.update(ICHARG=1, NELM=100, LCHARG=True)
    incar.write_file(path / 'INCAR')
PY
for path in "$W/scf/"*; do
  (cd "$path" && bash "$E/scripts/vasp/submit.sh")
done
```

Check electronic convergence of every SCF label. SAD and SCF must use the same structure, atom order, PAW files, and density grids. SAD's `NELM=0` output is only a prior.

Create `train.txt` and `val.txt` containing SAD directories, for example:

```text
/absolute/path/to/training_run/sad/structure_001
/absolute/path/to/training_run/sad/structure_002
```

The matching label directories must be `W/scf/structure_001`, `W/scf/structure_002`, etc. `data.vasp.dir` is their parent, not an individual structure directory.

### Configure VASP training

```bash
cp "$E/config.yaml" "$W/train.yaml"
```

Edit these keys in the copied full configuration. Paths below represent their expanded absolute values, not literal `$W` strings in YAML.

| Key | Joint PAW training value |
| --- | --- |
| `run.mode` | `train` |
| `run.task` | `2` |
| `run.save_dir` | `W/train` |
| `run.checkpoint` | `null` for training from scratch |
| `run.resume` | `false` |
| `data.dft_software` | `vasp` |
| `data.vasp.dir` | `W/scf` |
| `data.vasp.path_pp` | Configured ordinary PAW map |
| `data.vasp.lmix_max` | Same as dataset `LMAXMIX`, normally `6` |
| `data.path_train`, `data.path_val` | `W/train.txt`, `W/val.txt` |
| `data.use_bin`, `data.batch_size` | `false`, `1` |
| `model.use_sad` | `true` |
| `optimize.loss_dict.grid_func_out.weight_train`, `.weight_val` | `0.9`, `0.9` |
| `optimize.loss_dict.aug_tensor.weight_train`, `.weight_val` | `0.1`, `0.1` |

Preserve the template interpolations `model.task: ${run.task}`, `optimize.task: ${run.task}`, and `model.lmix_max: ${data.vasp.lmix_max}`.

Separate branches can also be trained:

| Target | `run.task` | `model.use_sad` | Density loss weights (train/val) | Augmentation loss weights (train/val) |
| --- | --- | --- | --- | --- |
| Pseudo density | `0` | `true` | `1 / 1` | `0 / 0` |
| Augmentation | `1` | `false` | `0 / 0` | `1 / 1` |
| Both | `2` | `true` | `0.9 / 0.9` | `0.1 / 0.1` |

With total-density VASP labels, `use_sad=true` enables learning the difference while reconstructing the predicted density using the prior. Do not change this setting when reusing a checkpoint.

## ABACUS labels

Use patched ABACUS, the ordinary pseudopotential/orbital map, and a parameter-only INPUT template with the desired `ecutwfc`:

```bash
python "$E/scripts/abacus/gen_abacus_cal.py" \
  --filelist "$W/structures.txt" --save-dir "$W/scf" \
  --path-pp "$E/scripts/abacus/pp.json" \
  --path-input "$E/scripts/abacus/INPUT" \
  --for-band --nproc 1
for path in "$W/scf/"*; do
  (cd "$path" && bash "$E/scripts/abacus/submit.sh")
done
cp "$E/config.yaml" "$W/train.yaml"
```

The generator sets `calculation scf`, `basis_type lcao`, `init_chg atomic`, and `out_chg 3`. The patched executable writes `scf/<name>/OUT.<suffix>/chgdelta.cube`. Confirm SCF convergence; a total-density cube is not a replacement for this difference-density label.

Create `train.txt` and `val.txt` with **SCF calculation directories**, not structure files or `OUT.<suffix>` paths:

```text
/absolute/path/to/training_run/scf/structure_001
/absolute/path/to/training_run/scf/structure_002
```

Edit the full copied configuration:

| Key | Value |
| --- | --- |
| `run.mode`, `run.task` | `train`, `0` |
| `run.save_dir`, `run.checkpoint`, `run.resume` | `W/train`, `null`, `false` |
| `data.dft_software` | `abacus` |
| `data.abacus.prefix` | Actual training INPUT suffix, e.g. `PBE` |
| `data.abacus.path_pp` | Configured ordinary pseudopotential/orbital map |
| `data.abacus.path_input` | Ordinary INPUT with the label-generation cutoff |
| `data.path_train`, `data.path_val` | `W/train.txt`, `W/val.txt` |
| `data.batch_size` | `1` |
| `model.use_sad` | `false` |
| `optimize.loss_dict.grid_func_out.weight_train`, `.weight_val` | `1`, `1` |
| `optimize.loss_dict.aug_tensor.weight_train`, `.weight_val` | `0`, `0` |

## Start training and select a checkpoint

For either backend:

```bash
edengnn-train --config "$W/train.yaml" > "$W/train.log" 2>&1
```

With the default project name, outputs are:

```text
train/edengnn/<timestamp>/
├── info.log
├── best_step=<step>.ckpt
├── checkpoints/
└── wandb/
```

Monitor `info.log` for losses and validation progress. Configure validation to run during training. Best-checkpoint saving begins from the second epoch; a one-epoch run may only have a regular checkpoint such as `checkpoints/epoch=0-step=1.ckpt`. Use the actual saved filename. For short demonstrations that need a best checkpoint, allow at least two epochs and a validation cycle.

For fine-tuning, set `run.checkpoint` to a compatible model and keep its architecture/task settings. Set `run.resume=true` only when resuming an interrupted training run with its training state.

### Small demonstration datasets

The [EdenGNN skill](https://github.com/rubenlee11/EdenGNN-Skill)'s demonstration convention repeats training entries to reach 50 lines when fewer than 10 structures are supplied. Keep `val.txt` unrepeated. This reduces repeated setup overhead but does not create independent training data.

For the short demonstration schedule set:

```yaml
optimize:
  scheduler:
    type: CosineAnnealingLR
    CosineAnnealingLR:
      T_max: 100
  max_steps: 100
```

These keys are edits to the full template, not a complete configuration. A shared training/validation structure is acceptable only for an explicitly labeled demonstration; it does not measure generalization. Tutorial loss values are illustrative, not universal accuracy thresholds. Validate electronic properties on held-out structures for a real model assessment.

## Predict with your checkpoint

Copy the **training configuration** to a new prediction configuration so the checkpoint's architecture, dtype, task, SAD setting, and DFT conventions remain consistent:

```bash
cp "$W/train.yaml" "$W/self_predict.yaml"
```

Set `run.mode: predict`, `run.resume: false`, `run.checkpoint` to the selected checkpoint's absolute path, `run.save_dir` to a new output location, and `data.path_predict` to the new prediction filelist.

For **VASP**, first generate and run SAD for the new structures using the same PAW/INCAR conventions as training. The filelist contains their SAD directories. Set `data.use_bin: false` and keep `data.vasp.path_pp` and `data.vasp.lmix_max` consistent with training.

For **ABACUS**, the filelist contains new structure files. Keep the same pseudopotentials, orbitals, and `ecutwfc`; set `data.abacus.path_input` and `data.abacus.path_pp` accordingly and `data.plot_band: true` to generate primitive-cell band inputs. Retain `model.use_sad: false`.

```bash
edengnn-predict --config "$W/self_predict.yaml" > "$W/self_predict.log" 2>&1
```

Continue with the matching backend guide, using the actual custom prediction timestamp:

- **VASP, task 2:** the file `T/predict/<name>` already includes density and augmentation. Copy it to `final/<name>/CHGCAR`, skip the universal two-branch prediction and concatenation, then follow [fixed-density calculations](vasp.md#assemble-chgcar-and-calculate-bandsdos).
- **VASP, separate tasks 0 and 1:** predict both checkpoints, concatenate density first and augmentation second, then run the same calculations. A density-only prediction is not the complete PAW workflow.
- **ABACUS:** `T/predict/<name>/` contains the generated inputs and `OUT.EDENGNN/SPIN1_CHG.cube`. The prediction suffix is `EDENGNN`, regardless of the training suffix. Follow [fixed-density bands/DOS](abacus.md#run-fixed-density-bands-and-dos) or [optics](optics.md).

For DFT comparisons, reuse matching converged training labels where possible and follow the backend's four-directory layout. Keep the training split, full YAML, checkpoint, label-generation settings, and prediction timestamp with the results.
