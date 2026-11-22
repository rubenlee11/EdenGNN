# ABACUS: predict with your model

Predict difference density directly from structure files, then calculate bands and DOS using patched ABACUS. No separate SAD calculation is needed.

## 1. Configure prediction

Copy the training configuration and update:

| Parameter | Setting |
| --- | --- |
| `run.mode`, `run.resume` | `predict`, `false` |
| `run.checkpoint` | Saved checkpoint |
| `run.save_dir` | Prediction output directory |
| `data.path_predict` | Filelist of **structure files**, such as CIF or POSCAR |
| `data.abacus.path_pp` | Same pseudopotentials and orbitals as training |
| `data.abacus.path_input` | INPUT template with the matching `ecutwfc` |
| `data.plot_band` | `true` for primitive-cell band inputs |

Preserve the checkpoint's model parameters, dtype, task `0`, and `model.use_sad=false`.

## 2. Predict difference density

Use `edengnn-predict` with the configuration file. Each output directory is `run.save_dir/edengnn/<timestamp>/predict/<name>/` and contains INPUT, STRU, KPT, KPT_BAND, and `OUT.EDENGNN/SPIN1_CHG.cube`.

The cube contains **difference density**. The generated calculation uses `calculation=nscf`, `basis_type=lcao`, `init_chg=drho`, and `out_chg=-1`.

## 3. Calculate bands and DOS

Copy each generated calculation to `band/` and `nscf/` under a structure result directory. Keep the generated primitive STRU, pseudopotentials, orbitals, and predicted cube.

| Directory | K points | Output settings |
| --- | --- | --- |
| `band/` | Replace KPT with KPT_BAND | `out_band=1`, `out_dos=0` |
| `nscf/` | Keep the uniform KPT | `out_dos=2`; also set `out_band=1` for DFT statistics |

Run ABACUS using `scripts/abacus/submit.sh`. List all structure result directories in one filelist and use `scripts/abacus/plot_band.py` to produce `band.png`.

Band data are in `band/OUT.EDENGNN/BANDS_1.dat`; DOS outputs are `nscf/OUT.EDENGNN/TDOS` and `PDOS`.

## 4. Compare with DFT

Prepare `scf/` with a converged DFT reference on the same uniform mesh and `band_scf/` with fixed reference density on the same band path. Keep structures, pseudopotentials, orbitals, cutoff, and spin settings consistent with prediction.

Use `init_chg=file` for total-density reference restarts and `init_chg=drho` for predicted difference density. Set `out_band=1` in all four calculations. Keep the actual `running_scf.log` in `scf/` and `running_nscf.log` in the other three directories.

Use `scripts/abacus/stat.py` with the structure filelist and a statistics output directory. Set `--prefix` and `--prefix-tar` to the actual prediction and reference suffixes. The prediction suffix is `EDENGNN`; training references may use `PBE`. For a separate reference root, use `--dir-tar`.

Outputs include `band_compare.png` and `stat_mae.json`.
