# VASP: use EdenGNN-Uni

Predict non-magnetic electronic structures with the pretrained pseudo-density and PAW augmentation models. Use the exact PAW versions associated with the models.

## 1. Prepare the SAD prior

Use `scripts/vasp/vasp_sad.py` with a structure filelist and the universal templates `edengnn-uni/vasp/INCAR` and `edengnn-uni/vasp/pp.json`. Enable `--for-scf` and `--for-band`.

Run VASP using your configured `scripts/vasp/submit.sh` to produce SAD CHGCAR files. Create the prediction filelist from these **SAD directories**.

## 2. Predict density and augmentation

Use the matching configurations and checkpoints under `~/.edengnn/edengnn-uni/vasp/`:

| Branch | Configuration | Checkpoint |
| --- | --- | --- |
| Pseudo density | `config_vasp_density.yaml` | `EdenGNN-Uni_vasp_pseudo.ckpt` |
| Augmentation | `config_vasp_aug.yaml` | `EdenGNN-Uni_vasp_aug.ckpt` |

In each configuration, set `run.save_dir`, `run.checkpoint`, `data.path_predict`, and `data.vasp.path_pp` to the actual paths. Adjust `model.chunk_size_predict` for available GPU memory; preserve the other model settings.

Use `edengnn-predict` once per configuration. Each branch writes a prediction file at `run.save_dir/edengnn/<timestamp>/predict/<name>`.

## 3. Assemble CHGCAR and calculate bands/DOS

Concatenate the pseudo-density prediction first and the augmentation prediction second. Use the assembled CHGCAR for fixed-density VASP calculations with `ICHARG=11` and positive `NELM`.

Follow [bands and DOS](vasp-prediction.md#3-calculate-bands-and-dos): use the generated band path and uniform mesh, run VASP with `scripts/vasp/submit.sh`, and plot with `scripts/vasp/plot_band.py`.

Outputs include `band/EIGENVAL`, `nscf/DOSCAR`, and `band.png` under each structure result directory.

## 4. Compare with DFT

Follow the same [DFT comparison layout](vasp-prediction.md#4-compare-with-dft) as for a custom model. Use `scripts/vasp/stat.py` after both prediction and reference calculations are complete.
