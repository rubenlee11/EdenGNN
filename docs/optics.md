# ABACUS and PyATB optical properties

[User guide](index.md) · [ABACUS prediction](abacus.md)

This workflow calculates the independent-particle interband dielectric function and optical conductivity from the predicted ABACUS Hamiltonian. PyATB must be installed in the execution environment, in addition to EdenGNN and patched ABACUS.

## Start from a predicted density

Complete [ABACUS difference-density prediction](abacus.md#predict-the-difference-density), retaining `E`, `W`, and its actual timestamp `T`. Skip bands/DOS if only optics is needed. Existing matching predictions or H/S/rR matrices can be reused.

The input calculation directories are `T/predict/<name>/`. Use their generated primitive `STRU`, not the original input CIF lattice.

## Output Hamiltonian, overlap, and position matrices

Make a matrix-calculation copy for every structure:

```bash
set -e
mkdir -p "$W/calculations"
for source in "$T"/predict/*; do
  name=$(basename "$source")
  calc="$W/calculations/$name"
  cp -r "$source" "$calc"
  sed -i -E '/^(out_mat_hs2|out_mat_r|out_band|out_dos|symmetry)[[:space:]]/d' "$calc/INPUT"
  cat >> "$calc/INPUT" <<'INPUT'
out_mat_hs2 1
out_mat_r 1
out_band 0
out_dos 0
symmetry -1
INPUT
done
printf '%s\n' "$W"/calculations/* > "$W/optics.txt"
while IFS= read -r calc; do
  (cd "$calc" && bash "$E/scripts/abacus/submit.sh")
done < "$W/optics.txt"
```

Retain `calculation nscf`, `basis_type lcao`, `init_chg drho`, `out_chg -1`, and the generated uniform KPT. Do not use the gamma-only localized-orbital algorithm for this matrix workflow. Run the commands inside an allocated job when using a cluster.

Each `calculations/<name>/OUT.EDENGNN/` contains:

| File | Contents |
| --- | --- |
| `data-HR-sparse_SPIN0.csr` | Real-space Hamiltonian, Ry |
| `data-SR-sparse_SPIN0.csr` | Overlap matrix |
| `data-rR-sparse.csr` | Position matrix, Bohr |
| `running_nscf.log` | Calculation log, including Fermi energy in eV |

## Prepare and run PyATB

Choose a Brillouin-zone integration grid for the **actual generated cell** and the desired spectral accuracy. Set the three variables before running, for example `export GRID_X=40 GRID_Y=40 GRID_Z=40` for a trial mesh; this is not a material-independent converged choice.

```bash
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
S="$E/scripts/abacus/abacus_optics.py"
python "$S" prepare --filelist "$W/optics.txt" \
  --grid "$GRID_X" "$GRID_Y" "$GRID_Z" \
  --omega 0.1 10 --domega 0.05 --eta 0.1
while IFS= read -r calc; do
  (cd "$calc/pyatb" && pyatb > pyatb.log 2>&1)
done < "$W/optics.txt"
python "$S" plot --filelist "$W/optics.txt"
```

The helper generates PyATB `Input` files and plots; it does not run ABACUS or PyATB itself. It reads the actual STRU lattice, the calculation's Fermi energy, and absolute matrix paths. Its occupation convention is `occ_band -1`, which determines occupations from the Fermi energy rather than assuming half the electron count for metals. The helper supports `nspin=1` and `4`.

| Parameter | Meaning |
| --- | --- |
| `--grid NX NY NZ` | Full Brillouin-zone optical integration mesh |
| `--omega MIN MAX` | Photon-energy range in eV |
| `--domega STEP` | Photon-energy step in eV |
| `--eta WIDTH` | Broadening in eV |

The example starts at 0.1 eV to avoid the zero-frequency dielectric-function singularity. These spectral parameters demonstrate the workflow; they do not establish convergence.

ABACUS's uniform KPT controls the density/Fermi-energy calculation; PyATB's grid controls optical integration. Refining the PyATB grid reuses existing matrices and does not require another density prediction. It also does not correct an unconverged ABACUS density mesh. One starting estimate is `N_i = max(1, int(|b_i| / 0.05 Å⁻¹))`, using actual reciprocal vectors including `2π`. Narrower broadening generally needs denser sampling. PyATB's `max_kpoint_num 256` controls memory chunking, not the integration grid.

## Outputs

Under each `calculations/<name>/`:

```text
pyatb/
├── Input
├── pyatb.log
└── Out/
    ├── running.log
    └── Optical_Conductivity/
        ├── dielectric_function_real_part.dat
        ├── dielectric_function_imag_part.dat
        ├── optical_conductivity_real_part.dat
        └── optical_conductivity_imag_part.dat
dielectric_function_xx.png
dielectric_function_yy.png
dielectric_function_zz.png
optical_conductivity_xx.png
optical_conductivity_yy.png
optical_conductivity_zz.png
```

Raw spectra have photon energy in the first column (eV), followed by `xx, xy, xz, yx, yy, yz, zx, zy, zz`. The dielectric function is dimensionless; optical conductivity is in S/m.

The six plots show one direction per figure: dielectric real/imaginary parts together, and conductivity real part only. Their displayed energy range is 0–6 eV; raw spectra retain the full calculated range.

The calculation omits excitons and phonon-assisted transitions. For metals, a complete low-frequency response additionally requires the Drude contribution.

## Compare optics with DFT

Reuse the predicted branch. The target must use an actual converged DFT density and the same primitive structure, atom order, pseudopotentials, orbitals, functional, cutoff, KPT, smearing, and spin/SOC settings.

For a new SCF target, copy only inputs into empty target directories. The following assumes the generated pseudopotential and orbital files use `.upf` and `.orb` extensions; if INPUT instead references shared directories, preserve those references and omit the corresponding file-copy arguments.

```bash
mkdir -p "$W/target"
: > "$W/target_optics.txt"
while IFS= read -r prediction; do
  name=$(basename "$prediction")
  target="$W/target/$name"
  mkdir -p "$target"
  cp "$prediction/INPUT" "$prediction/STRU" "$prediction/KPT" \
    "$prediction/"*.upf "$prediction/"*.orb "$target/"
  sed -i -E '/^(calculation|init_chg|out_chg|out_mat_r)[[:space:]]/d' "$target/INPUT"
  cat >> "$target/INPUT" <<'INPUT'
calculation scf
init_chg atomic
out_chg 1
out_mat_r 0
INPUT
  printf '%s\n' "$target" >> "$W/target_optics.txt"
done < "$W/optics.txt"
while IFS= read -r calc; do
  (cd "$calc" && bash "$E/scripts/abacus/submit.sh")
done < "$W/target_optics.txt"
```

Retain `out_mat_hs2 1` to output target H/S. Check SCF convergence before using the matrices. For the same structure, atom order, basis, and spin representation, rR is density-independent and can be reused from prediction:

```bash
while IFS= read -r prediction; do
  name=$(basename "$prediction")
  output="$W/target/$name/OUT.EDENGNN"
  mkdir -p "$output"
  ln -sfn "$(realpath "$prediction/OUT.EDENGNN/data-rR-sparse.csr")" \
    "$output/data-rR-sparse.csr"
done < "$W/optics.txt"
python "$S" prepare --filelist "$W/target_optics.txt" \
  --grid "$GRID_X" "$GRID_Y" "$GRID_Z" \
  --omega 0.1 10 --domega 0.05 --eta 0.1
while IFS= read -r calc; do
  (cd "$calc/pyatb" && pyatb > pyatb.log 2>&1)
done < "$W/target_optics.txt"
python "$S" compare --filelist "$W/optics.txt" \
  --target-filelist "$W/target_optics.txt"
```

Reuse existing converged target H/S when available. If only the converged total density exists, output H/S in a fixed-density NSCF with `init_chg file`; do not read total density using `drho`.

Both branches must use identical PyATB grids, energy range/step, broadening, method, and spin settings, but each branch reads its own Fermi energy. Do not manually shift spectra or add a scissor correction to the comparison. The two filelists must match structure-by-structure in the same line order.

Comparison adds six `*_compare.png` figures and `optical_compare.json` to each prediction calculation directory. The JSON records paths and spectral MAEs for diagonal components and the complete nine-component tensor over the full calculated range. Dielectric-function errors are dimensionless; conductivity errors are in S/m. Retain the actual prediction timestamp, target paths, and spectral parameters with these results.
