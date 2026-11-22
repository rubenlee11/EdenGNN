# ABACUS: use EdenGNN-Uni

Predict non-magnetic electronic structures directly from structures using the pretrained ABACUS difference-density model. Use its matching SG15 pseudopotentials, orbital basis, and patched ABACUS build.

## 1. Configure prediction

Use the files under `~/.edengnn/edengnn-uni/abacus/`:

| File | Purpose |
| --- | --- |
| `config_abacus.yaml` | Prediction configuration |
| `EdenGNN-Uni_abacus.ckpt` | Pretrained weights |
| `INPUT` | Matching DFT template |
| `pp.json` | Pseudopotential and orbital map |

Set `run.save_dir`, `run.checkpoint`, `data.path_predict`, `data.abacus.path_pp`, and `data.abacus.path_input` to the actual paths. The prediction filelist contains **structure files**, such as CIF or POSCAR.

Adjust `model.chunk_size_predict` for GPU memory while preserving the pretrained model settings. No separate SAD or SCF calculation is required.

## 2. Predict difference density

Use `edengnn-predict` with the configuration file. The output is `run.save_dir/edengnn/<timestamp>/predict/<name>/`, containing the generated inputs and `OUT.EDENGNN/SPIN1_CHG.cube`.

## 3. Calculate bands and DOS

Follow [bands and DOS](abacus-prediction.md#3-calculate-bands-and-dos): prepare band-path and uniform-mesh calculations, run patched ABACUS with `scripts/abacus/submit.sh`, and plot with `scripts/abacus/plot_band.py`.

Read the predicted difference density with `init_chg=drho`. Outputs include `BANDS_1.dat`, `TDOS`, `PDOS`, and `band.png`.

## 4. Additional calculations

For DFT validation, follow [the reference calculation layout](abacus-prediction.md#4-compare-with-dft) and use `scripts/abacus/stat.py`.

For dielectric functions and optical conductivity, continue directly from prediction to [ABACUS + PyATB optics](optics.md). Bands and DOS are not required for the optical workflow.
