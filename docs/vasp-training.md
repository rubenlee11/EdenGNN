# VASP: train a model

Train pseudo charge density and PAW augmentation occupancies from SAD priors and converged SCF labels.

## 1. Prepare training data

Use `scripts/vasp/vasp_sad.py` to generate SAD inputs from a structure filelist, with the ordinary `scripts/vasp/INCAR` and `scripts/vasp/pp.json` templates. Enable `--for-scf` and `--for-band` to generate uniform and band-path k points.

Run VASP in the generated directories using your configured `scripts/vasp/submit.sh`. SAD uses `ICHARG=12`, `NELM=0`, and `ISPIN=1` and produces the prior CHGCAR.

Prepare matching SCF calculations with the same structures, atom order, PAW files, and FFT-grid settings. Use `KPOINTS_SCF`, `ICHARG=1`, positive `NELM`, and `LCHARG=True`. Check electronic convergence before using the SCF CHGCAR files as labels.

## 2. Configure training

Copy `~/.edengnn/config.yaml` to your working directory and set:

| Parameter | Setting |
| --- | --- |
| `run.mode` | `train` |
| `run.task` | `2` for joint density and augmentation training |
| `run.save_dir` | Training output directory |
| `run.checkpoint` | Empty for training from scratch; a compatible checkpoint for fine-tuning |
| `data.dft_software` | `vasp` |
| `data.path_train`, `data.path_val` | Filelists of **SAD directories**, split by structure |
| `data.vasp.dir` | SCF parent directory; child names must match the SAD directories |
| `data.vasp.path_pp` | Configured PAW map |
| `data.vasp.lmix_max` | Same as VASP `LMAXMIX` |
| `model.use_sad` | `true` |

Set density and augmentation loss weights in `optimize.loss_dict`; joint training can use `0.9` and `0.1`, respectively, for both training and validation. Keep `data.batch_size=1`.

For separate models, use `run.task=0` for density or `1` for augmentation. Set `model.use_sad=false` for augmentation-only training and give only the selected task a nonzero loss weight.

## 3. Train and select a checkpoint

Use `edengnn-train` with the configuration file. Monitor `info.log` under `run.save_dir/edengnn/<timestamp>/` and select a saved `best_step=<step>.ckpt`.

Allow validation and at least two epochs for best-checkpoint saving in a short demonstration. Reusing one structure for training and validation demonstrates operation but does not measure generalization.

Continue with [prediction using your model](vasp-prediction.md).
