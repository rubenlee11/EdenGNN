# ABACUS: structure to bands and DOS

[User guide](index.md) · [Setup](setup.md) · [Custom models](training.md) · [Optics](optics.md)

ABACUS prediction takes structure files directly. Neither a separate SAD calculation nor a prior SCF is needed. The predicted cube is a **difference density**, read by the patched ABACUS executable.

## Initialize a run

Use the [configured environment and launchers](setup.md). Replace the paths below and run inside your workstation environment or allocated job:

```bash
set -e
export E="$HOME/.edengnn"
source "$E/cpu.sh"
source "$E/cuda.sh"
export PYTHONDONTWRITEBYTECODE=1
export W=/absolute/path/to/new/abacus_run
STRUCTURE=/absolute/path/to/Si.cif
NAME=$(basename "$STRUCTURE")
export NAME="${NAME%.*}"
mkdir -p "$W"
printf '%s\n' "$STRUCTURE" > "$W/structures.txt"
```

## Predict the difference density

Use the universal checkpoint's matching configuration, INPUT template, pseudopotentials, and orbital map. Preserve task/model/sampling defaults, adjusting the chunk size for available GPU memory if necessary.

```bash
python - <<'PY'
import os
from pathlib import Path
from omegaconf import OmegaConf
model_dir = Path(os.environ['E']) / 'edengnn-uni/abacus'
w = Path(os.environ['W'])
cfg = OmegaConf.load(model_dir / 'config_abacus.yaml')
cfg.run.save_dir = str(w / 'results')
cfg.run.checkpoint = str(model_dir / 'EdenGNN-Uni_abacus.ckpt')
cfg.data.path_predict = str(w / 'structures.txt')
cfg.data.abacus.path_pp = str(model_dir / 'pp.json')
cfg.data.abacus.path_input = str(model_dir / 'INPUT')
OmegaConf.save(cfg, w / 'predict.yaml')
PY
edengnn-predict --config "$W/predict.yaml" > "$W/predict.log" 2>&1
runs=("$W"/results/edengnn/*)
T=${runs[0]}
```

For a fresh working directory this selects the single prediction timestamp. Each `T/predict/<name>/` is a complete calculation directory with `INPUT`, `STRU`, `KPT`, `KPT_BAND`, pseudopotentials/orbitals, and `OUT.EDENGNN/SPIN1_CHG.cube`.

For optical properties only, continue directly to [the optics workflow](optics.md); bands and DOS are not prerequisites.

## Run fixed-density bands and DOS

Copy the generated calculation to separate band and uniform-mesh directories:

```bash
mkdir -p "$W/final/$NAME"
cp -r "$T/predict/$NAME" "$W/final/$NAME/band"
cp -r "$T/predict/$NAME" "$W/final/$NAME/nscf"
cp "$W/final/$NAME/band/KPT_BAND" "$W/final/$NAME/band/KPT"
python - <<'PY'
import os
from pathlib import Path
base = Path(os.environ['W']) / 'final' / os.environ['NAME']
for mode in ('band', 'nscf'):
    path = base / mode / 'INPUT'
    fields = dict(line.split(maxsplit=1)
                  for line in path.read_text().splitlines()[1:] if line.strip())
    fields['out_band'] = '1' if mode == 'band' else '0'
    fields['out_dos'] = '0' if mode == 'band' else '2'
    path.write_text('INPUT_PARAMETERS\n' + ''.join(
        f'{key} {value}\n' for key, value in fields.items()))
PY
for mode in band nscf; do
  (cd "$W/final/$NAME/$mode" && bash "$E/scripts/abacus/submit.sh")
done
printf '%s\n' "$W/final/$NAME" > "$W/plot.txt"
python "$E/scripts/abacus/plot_band.py" \
  --filelist "$W/plot.txt" --nproc "${plot_nproc:-1}"
```

Preserve `calculation nscf`, `basis_type lcao`, `init_chg drho`, and `out_chg -1` in the generated inputs. Use `KPT_BAND` only for the band branch and the original uniform `KPT` for DOS. These launcher calls assume synchronous local execution; use [scheduler dependencies](setup.md#local-execution-and-cluster-submission) on a cluster.

For multiple structures, repeat calculation setup for each name, then list every `final/<name>` in the same `plot.txt` and run the plotting CLI once.

## Outputs

```text
abacus_run/
├── results/edengnn/<timestamp>/predict/<name>/
│   ├── INPUT, STRU, KPT, KPT_BAND
│   └── OUT.EDENGNN/SPIN1_CHG.cube
└── final/<name>/
    ├── band/OUT.EDENGNN/BANDS_1.dat
    ├── nscf/OUT.EDENGNN/TDOS
    ├── nscf/OUT.EDENGNN/PDOS
    └── band.png
```

The plotting script reads `TDOS` directly and overlays `PDOS` when present; no renaming to `TDOS.dat` is needed. Keep the actual prediction timestamp with the results.

## Compare with DFT

Organize each comparison under `final/<name>/`:

| Directory | Calculation | Required output under `OUT.<suffix>/` |
| --- | --- | --- |
| `nscf/` | Predicted density, uniform k mesh | `BANDS_1.dat`, `running_nscf.log` |
| `band/` | Predicted density, high-symmetry path | `BANDS_1.dat`, `running_nscf.log` |
| `scf/` | Actual converged reference SCF, uniform mesh | `BANDS_1.dat`, `running_scf.log` |
| `band_scf/` | Reference density, high-symmetry path NSCF | `BANDS_1.dat`, `running_nscf.log` |

Set `out_band 1` in **all four** calculations, including the uniform prediction branch, before running a comparison workflow. The DOS-only configuration above sets that branch to zero and must be changed for statistics.

Keep the same generated primitive STRU, atom order, pseudopotentials, orbitals, functional, cutoff, smearing, and spin settings. Match uniform meshes between `nscf/` and `scf/`, and band paths between `band/` and `band_scf/`.

To create a missing reference:

1. Create a new `scf/` directory with the prediction's `STRU`, matching pseudopotential/orbital files, and uniform `KPT`. Copy its INPUT but change `calculation scf`, `init_chg atomic`, `out_chg 1`, `out_band 1`, and `out_dos 2`. Do not copy the predicted cube or old outputs.
2. Run patched ABACUS and confirm SCF convergence in `running_scf.log`. Reuse a matching converged training calculation instead if available.
3. Copy the converged calculation to `band_scf/`, retaining its density restart files. Replace `KPT` with the prediction's `KPT_BAND`. Set `calculation nscf`, `init_chg file`, `out_chg -1`, `out_band 1`, and `out_dos 0`, then run ABACUS.

`init_chg file` reads the reference **total density**; `init_chg drho` reads the model's **difference density**. If reusing a reference difference-density restart instead, use the patch's `drho` mode and matching restart filename. A fixed-reference-density NSCF must not be relabeled as `scf/`.

Once all four calculations are complete:

```bash
python "$E/scripts/abacus/stat.py" \
  --filelist "$W/plot.txt" --save-dir "$W/statistics" \
  --prefix EDENGNN --prefix-tar EDENGNN
```

Set `--prefix-tar` to the actual reference suffix, such as `PBE` for ordinary training outputs. For references stored separately, add `--dir-tar "$W/target"`, where each reference is `target/<name>/{scf,band_scf}`. The prediction prefix remains `EDENGNN`.

The CLI generates comparison plots and JSON statistics. Check coverage of all requested structures and keep the SCF convergence logs with the comparison.
