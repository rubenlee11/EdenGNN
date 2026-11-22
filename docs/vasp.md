# VASP: structure to bands and DOS

[User guide](index.md) · [Setup](setup.md) · [Custom models](training.md)

The universal PAW workflow needs **two** predictions: pseudo charge density and augmentation occupancies. Assemble them into one CHGCAR before fixed-density diagonalization. For a custom joint model (`run.task: 2`), use its complete prediction directly as described in [Training](training.md#predict-with-your-checkpoint).

## Initialize a run

Activate the configured environment and replace the input and output paths. The commands below run on a workstation or inside an allocated job; see [cluster submission](setup.md#local-execution-and-cluster-submission) for scheduler use.

```bash
set -e
export E="$HOME/.edengnn"
source "$E/cpu.sh"
source "$E/cuda.sh"
export PYTHONDONTWRITEBYTECODE=1
export W=/absolute/path/to/new/vasp_run
STRUCTURE=/absolute/path/to/Si.cif
NAME=$(basename "$STRUCTURE")
export NAME="${NAME%.*}"
mkdir -p "$W"
printf '%s\n' "$STRUCTURE" > "$W/structures.txt"
```

Configure MPI resources in your launchers and set plotting concurrency for the allocation, for example `export plot_nproc=1` for serial plotting.

## Generate and run SAD

```bash
python "$E/scripts/vasp/vasp_sad.py" \
  --filelist "$W/structures.txt" --save-dir "$W/sad" \
  --path-pp "$E/edengnn-uni/vasp/pp.json" \
  --path-input "$E/edengnn-uni/vasp/INCAR" \
  --for-scf --for-band --nproc 1
(cd "$W/sad/$NAME" && bash "$E/scripts/vasp/submit.sh")
printf '%s\n' "$W/sad/$NAME" > "$W/predict.txt"
```

The generator prepares inputs; the following launcher call actually runs VASP. The generator sets `ICHARG=12`, `NELM=0`, and `ISPIN=1`. It also prepares:

| File | Purpose |
| --- | --- |
| `POSCAR` | Generated primitive structure used throughout the workflow |
| `POTCAR`, `INCAR` | PAW and grid settings |
| `CHGCAR` | SAD prior, produced by VASP |
| `KPOINTS_SCF` | Uniform mesh for SCF and DOS |
| `KPOINTS_BAND` | High-symmetry band path |

The prediction filelist contains **SAD directories**, not CIF paths or individual CHGCAR paths.

## Predict density and augmentation

Copy the matched universal configurations, changing output/input paths and expanding resource paths. Preserve their task, architecture, dtype, and SAD settings.

```bash
python - <<'PY'
import os
from pathlib import Path
from omegaconf import OmegaConf
model_dir = Path(os.environ['E']) / 'edengnn-uni/vasp'
w = Path(os.environ['W'])
for kind, weight in [('density', 'pseudo'), ('aug', 'aug')]:
    cfg = OmegaConf.load(model_dir / f'config_vasp_{kind}.yaml')
    cfg.run.save_dir = str(w / kind)
    cfg.run.checkpoint = str(model_dir / f'EdenGNN-Uni_vasp_{weight}.ckpt')
    cfg.data.path_predict = str(w / 'predict.txt')
    cfg.data.vasp.path_pp = str(model_dir / 'pp.json')
    OmegaConf.save(cfg, w / f'{kind}.yaml')
PY
edengnn-predict --config "$W/density.yaml" > "$W/density.log" 2>&1
edengnn-predict --config "$W/aug.yaml" > "$W/aug.log" 2>&1
```

If necessary, adjust `model.chunk_size_predict` to GPU memory before launching. The checkpoint for augmentation is named `EdenGNN-Uni_vasp_aug.ckpt`; some downloaded configurations may still contain a shorter old filename, which the command above replaces.

Each output is a **file** at `<run.save_dir>/edengnn/<timestamp>/predict/<name>`.

## Assemble CHGCAR and calculate bands/DOS

Use a fresh working directory so each glob selects this run's single timestamp:

```bash
density_runs=("$W"/density/edengnn/*)
aug_runs=("$W"/aug/edengnn/*)
T_D=${density_runs[0]}
T_A=${aug_runs[0]}
mkdir -p "$W/final/$NAME"
cat "$T_D/predict/$NAME" "$T_A/predict/$NAME" \
  > "$W/final/$NAME/CHGCAR"
for mode in band nscf; do
  mkdir -p "$W/final/$NAME/$mode"
  cp "$W/sad/$NAME/"{POSCAR,POTCAR,INCAR} "$W/final/$NAME/$mode/"
  cp "$W/final/$NAME/CHGCAR" "$W/final/$NAME/$mode/CHGCAR"
done
cp "$W/sad/$NAME/KPOINTS_BAND" "$W/final/$NAME/band/KPOINTS"
cp "$W/sad/$NAME/KPOINTS_SCF" "$W/final/$NAME/nscf/KPOINTS"
python - <<'PY'
import os
from pathlib import Path
from pymatgen.io.vasp.inputs import Incar
base = Path(os.environ['W']) / 'final' / os.environ['NAME']
for mode in ('band', 'nscf'):
    path = base / mode / 'INCAR'
    incar = Incar.from_file(path)
    incar.update(ICHARG=11, NELM=100, LCHARG=False, NEDOS=2001)
    incar.write_file(path)
PY
for mode in band nscf; do
  (cd "$W/final/$NAME/$mode" && bash "$E/scripts/vasp/submit.sh")
done
printf '%s\n' "$W/final/$NAME" > "$W/plot.txt"
python "$E/scripts/vasp/plot_band.py" \
  --filelist "$W/plot.txt" --nproc "${plot_nproc:-1}"
```

Concatenate density **first**, augmentation **second**, once each. Retaining SAD's `NELM=0` would prevent the required diagonalization. DOS uses the uniform mesh, not the high-symmetry path. To retain wavefunctions, set `LWAVE=True` in the final calculation INCAR files before running VASP.

For multiple structures, repeat the preparation/calculation steps for each name and write all `final/<name>` directories to one `plot.txt`. Invoke the plotting command once after all calculations finish.

## Outputs

```text
vasp_run/
├── sad/<name>/CHGCAR
├── density/edengnn/<timestamp>/predict/<name>
├── aug/edengnn/<timestamp>/predict/<name>
└── final/<name>/
    ├── CHGCAR
    ├── band/       # vasprun.xml, EIGENVAL, ...
    ├── nscf/       # vasprun.xml, DOSCAR, ...
    └── band.png
```

`band.png` is the predicted band/DOS plot. It is not a DFT comparison plot. Keep both model-run timestamps alongside final results for reproducibility.

## Compare with DFT

Only add the reference calculations when a comparison is needed. The statistics CLI requires both uniform-grid and band-path results:

| Directory under `final/<name>/` | Calculation |
| --- | --- |
| `nscf/` | Predicted CHGCAR, `ICHARG=11`, uniform mesh |
| `band/` | Predicted CHGCAR, `ICHARG=11`, high-symmetry path |
| `scf/` | Converged DFT reference, `ICHARG=1`, uniform mesh |
| `band_scf/` | Converged reference CHGCAR, `ICHARG=11`, same band path |

Reuse matching converged training labels or previous reference calculations. Keep structure, atom order, PAW files, `ENCUT`, `PREC`, FFT grids, and uniform k meshes identical across the relevant branches. SAD with `NELM=0` is not a reference SCF.

If a reference is missing, prepare it from the same SAD inputs:

```bash
mkdir -p "$W/final/$NAME/scf"
cp "$W/sad/$NAME/"{POSCAR,POTCAR,INCAR,CHGCAR} "$W/final/$NAME/scf/"
cp "$W/sad/$NAME/KPOINTS_SCF" "$W/final/$NAME/scf/KPOINTS"
python - <<'PY'
import os
from pathlib import Path
from pymatgen.io.vasp.inputs import Incar
path = Path(os.environ['W']) / 'final' / os.environ['NAME'] / 'scf/INCAR'
incar = Incar.from_file(path)
incar.update(ICHARG=1, NELM=100, LCHARG=True, NEDOS=2001)
incar.write_file(path)
PY
(cd "$W/final/$NAME/scf" && bash "$E/scripts/vasp/submit.sh")
```

Confirm electronic convergence before using the reference and generating its band calculation:

```bash
python - <<'PY'
import os
import shutil
from pathlib import Path
from pymatgen.io.vasp.inputs import Incar
from pymatgen.io.vasp.outputs import Vasprun
base = Path(os.environ['W']) / 'final' / os.environ['NAME']
reference = Vasprun(base / 'scf/vasprun.xml', parse_dos=False,
                   parse_eigen=False, parse_projected_eigen=False,
                   parse_potcar_file=False)
if not reference.converged_electronic:
    raise RuntimeError(f'Reference SCF did not converge: {base / "scf"}')
band_dir = base / 'band_scf'
band_dir.mkdir()
for filename in ('POSCAR', 'POTCAR', 'INCAR', 'CHGCAR'):
    shutil.copy2(base / 'scf' / filename, band_dir / filename)
shutil.copy2(Path(os.environ['W']) / 'sad' / os.environ['NAME'] / 'KPOINTS_BAND',
             band_dir / 'KPOINTS')
incar = Incar.from_file(band_dir / 'INCAR')
incar.update(ICHARG=11, NELM=100, LCHARG=False)
incar.write_file(band_dir / 'INCAR')
PY
(cd "$W/final/$NAME/band_scf" && bash "$E/scripts/vasp/submit.sh")
python "$E/scripts/vasp/stat.py" \
  --filelist "$W/plot.txt" --save-dir "$W/statistics" \
  --nproc "${plot_nproc:-1}"
```

`scf/` and `nscf/` must retain `vasprun.xml`, `KPOINTS`, and `CHGCAR`; `band/` and `band_scf/` need `vasprun.xml` and `KPOINTS`. The input predicted CHGCAR remains available with `LCHARG=False`.

For separate reference storage, add `--dir-tar "$W/target"`; references must then be at `target/<name>/{scf,band_scf}`. Run the statistics CLI once for the full filelist.

Results include per-structure `band_compare.png` and `statistics/stat_mae.json`, covering density, augmentation, eigenvalues, energy, gap, and forces. Eigenvalue MAE uses uniform-grid calculations. Comparison plots share the target VBM as energy zero (target Fermi energy for metals). Check that the statistics cover every requested structure.
