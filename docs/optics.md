# ABACUS and PyATB optical properties

Calculate the dielectric function and optical conductivity from ABACUS real-space matrices. Install PyATB in the execution environment before starting.

## 1. Predict difference density

Use the [ABACUS universal model](abacus.md) or [your trained model](abacus-prediction.md). Reuse an existing prediction when available. Bands and DOS are not required.

## 2. Calculate matrices

Copy each generated prediction directory to a matrix-calculation directory. Keep its primitive STRU, pseudopotentials, orbitals, uniform KPT, and difference-density restart.

Retain `calculation=nscf`, `basis_type=lcao`, `init_chg=drho`, and `out_chg=-1`. Set `out_mat_hs2=1`, `out_mat_r=1`, `out_band=0`, `out_dos=0`, and `symmetry=-1`. Use the non-gamma-only LCAO algorithm.

Run patched ABACUS with `scripts/abacus/submit.sh`. The outputs under `OUT.EDENGNN/` are:

| File | Contents |
| --- | --- |
| `data-HR-sparse_SPIN0.csr` | Hamiltonian, Ry |
| `data-SR-sparse_SPIN0.csr` | Overlap matrix |
| `data-rR-sparse.csr` | Position matrix, Bohr |
| `running_nscf.log` | Fermi energy and calculation log |

## 3. Calculate and plot optical spectra

Create a filelist of the matrix-calculation directories. Use `scripts/abacus/abacus_optics.py` in `prepare` mode to generate PyATB inputs. Specify the integration grid, photon-energy range, energy step, and broadening through `--grid`, `--omega`, `--domega`, and `--eta`.

Choose the grid for the actual generated cell and required accuracy. PyATB's optical integration grid is separate from the ABACUS density mesh. Refining it can reuse existing matrices.

Run `pyatb` in each generated `pyatb/` directory, then use `abacus_optics.py` in `plot` mode with the complete filelist. The helper prepares inputs and plots; it does not run ABACUS or PyATB.

## Outputs

Raw spectra are in `pyatb/Out/Optical_Conductivity/`. Each file contains photon energy in eV followed by the nine tensor components. The dielectric function is dimensionless; optical conductivity is in S/m.

Each structure receives separate xx, yy, and zz figures for the dielectric function and real optical conductivity. Raw data retain the full calculated energy range.

## Compare with DFT

Prepare a converged SCF reference using the same structure, pseudopotentials, orbitals, cutoff, k mesh, and spin settings. Output reference H/S with `out_mat_hs2=1`. Reuse prediction-side rR only when the structure, atom order, basis, and spin representation match.

Generate and run reference PyATB calculations with the same integration grid and spectral parameters. Use `abacus_optics.py` in `compare` mode with prediction and reference filelists in matching structure order.

Outputs include six comparison figures and `optical_compare.json`. This workflow calculates independent-particle interband response; excitons and phonon-assisted transitions are excluded, and metals additionally require a Drude contribution for the full low-frequency response.
