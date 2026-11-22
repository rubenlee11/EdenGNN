# ABACUS: train a model

Train a difference-density model from converged ABACUS calculations using norm-conserving pseudopotentials and localized orbitals.

## 1. Prepare training data

Use the [difference-density-patched ABACUS build](setup.md#dft-backends). Configure `scripts/abacus/pp.json` with the pseudopotential and orbital paths, and `scripts/abacus/INPUT` with the desired cutoff and SCF settings.

Use `scripts/abacus/gen_abacus_cal.py` to generate calculations from a structure filelist. Its INPUT template is parameter-only, without an `INPUT_PARAMETERS` header. Enable `--for-band` to prepare band-path inputs.

Run ABACUS using your configured `scripts/abacus/submit.sh`. The generator sets `calculation=scf`, `basis_type=lcao`, `init_chg=atomic`, and `out_chg=3`.

Confirm SCF convergence. The label is `OUT.<suffix>/chgdelta.cube`; a total-density cube cannot replace this difference-density file.

## 2. Configure training

Copy `~/.edengnn/config.yaml` to your working directory and set:

| Parameter | Setting |
| --- | --- |
| `run.mode`, `run.task` | `train`, `0` |
| `run.save_dir` | Training output directory |
| `run.checkpoint` | Empty for training from scratch; a compatible checkpoint for fine-tuning |
| `data.dft_software` | `abacus` |
| `data.path_train`, `data.path_val` | Filelists of **SCF directories**, split by structure |
| `data.abacus.prefix` | Actual training INPUT suffix, e.g. `PBE` |
| `data.abacus.path_pp`, `data.abacus.path_input` | Configured map and matching INPUT template |
| `model.use_sad` | `false` |

Set the density loss weights to `1` and augmentation loss weights to `0`, for both training and validation. Keep `data.batch_size=1`.

## 3. Train and select a checkpoint

Use `edengnn-train` with the configuration file. Monitor `info.log` under `run.save_dir/edengnn/<timestamp>/` and select a saved `best_step=<step>.ckpt`.

Allow validation and at least two epochs for best-checkpoint saving in a short demonstration. Use distinct training and validation structures to assess generalization.

Continue with [prediction using your model](abacus-prediction.md).
