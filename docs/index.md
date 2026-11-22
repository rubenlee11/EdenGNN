# EdenGNN user guide

This guide explains how to operate EdenGNN: install and configure the software, predict electronic structures with EdenGNN-Uni, train a material-specific model, compare predictions with DFT, and calculate optical properties with ABACUS and PyATB.

It follows the repository README, the [EdenGNN skill](https://github.com/rubenlee11/EdenGNN-Skill), and the accompanying EdenGNN tutorial. The worked commands focus on VASP and ABACUS. For OpenMX and SIESTA prerequisites, see the [main README](https://github.com/rubenlee11/EdenGNN#installation).

## Choose a workflow

| Goal | Instructions |
| --- | --- |
| Install EdenGNN and configure a workstation or cluster | [Installation and setup](setup.md) |
| Predict VASP band structures and DOS with EdenGNN-Uni | [VASP workflow](vasp.md) |
| Predict ABACUS band structures and DOS with EdenGNN-Uni | [ABACUS workflow](abacus.md) |
| Train and use your own VASP or ABACUS model | [Training and custom checkpoints](training.md) |
| Calculate dielectric functions and optical conductivity | [ABACUS + PyATB optics](optics.md) |
| Compare electronic structures with converged DFT | DFT comparison sections in the [VASP](vasp.md#compare-with-dft) and [ABACUS](abacus.md#compare-with-dft) guides |

## What EdenGNN predicts

EdenGNN predicts charge density from atomic structures. A subsequent DFT calculation constructs and diagonalizes the Kohn–Sham Hamiltonian at the predicted density, avoiding the usual density self-consistency cycle.

- **VASP / PAW:** generate the superposition of atomic charge density (SAD, also called SACD), predict the pseudo charge density and PAW augmentation occupancies, then run fixed-density band/DOS calculations with `ICHARG=11`.
- **ABACUS / norm-conserving pseudopotentials:** predict the difference from the atomic reference density, then use patched ABACUS with `init_chg drho` and `calculation nscf`. A separate SAD calculation is unnecessary.
- **Optics:** use ABACUS to output Hamiltonian, overlap, and position matrices at the predicted density, then integrate the optical response with PyATB.

The universal models target non-magnetic materials. Use the pseudopotential versions, orbital basis, functional, and density-grid settings associated with the selected checkpoint. Changing these changes the prediction problem.

## Command conventions

Examples use Bash and absolute paths on the machine that actually runs the calculations. Keep all generated YAML files, logs, DFT outputs, and checkpoints in a new working directory **outside the source checkout**.

| Variable | Meaning |
| --- | --- |
| `E` | Installed resources, normally `$HOME/.edengnn` |
| `W` | A new working directory for one workflow |
| `STRUCTURE` | Absolute path to an input structure, such as a CIF or POSCAR |
| `NAME` | Structure filename with its final extension removed |
| `T` | Actual timestamp directory created by prediction |
| `ncore` | CPU/MPI resources used by your configured launcher |
| `plot_nproc` | Number of plotting processes within your allocation |

Each prediction example assumes a fresh `W`, so its run directory contains one timestamp. For reruns or existing results, set `T` (or `T_D` and `T_A`) to the intended run explicitly rather than selecting an arbitrary timestamp.

A filelist contains one absolute path per line, without a header. Input structures must have distinct filename stems because these become output names. The expected path type depends on the stage:

| Stage | Filelist entries |
| --- | --- |
| Input generation | Structure files |
| VASP training and prediction | SAD calculation directories |
| ABACUS training | SCF calculation directories |
| ABACUS prediction | Structure files |
| Band plotting / DFT statistics | Structure result directories containing `band/`, `nscf/`, etc. |
| PyATB preparation / plotting | ABACUS matrix-calculation directories |

For multiple structures, prepare and run each calculation, then put **all** result directories in one plotting filelist and call the relevant plotting script once. Choose `--nproc` within the allocated CPU resources.

## Operate through the EdenGNN skill

The [EdenGNN-Skill repository](https://github.com/rubenlee11/EdenGNN-Skill) provides the skill and workflow instructions for operating EdenGNN through a coding agent. After [configuring the job machine and installing the skill](setup.md#install-the-optional-skill), you can describe the full workflow in natural language. Replace the paths and SSH alias in these examples:

```text
$edengnn Use the VASP universal model to predict band structures and DOS for
all structures in /absolute/path/to/materials on my SSH server. Save results
outside the source checkout and compare them with converged DFT.

$edengnn Train a VASP charge-density model for /absolute/path/to/al.
This is a demonstration; use the same structure for training and validation.
Then predict its electronic structure and compare it with DFT.

$edengnn Train an ABACUS difference-density model for
/absolute/path/to/gaas, then predict band structures and DOS with that model.

$edengnn Use the ABACUS universal model to calculate the dielectric function
and optical conductivity of /absolute/path/to/Si.cif with PyATB.
```

Specify whether you want DFT comparisons, only bands, only optics, or additional files such as VASP `WAVECAR` or ABACUS real-space matrices. The standard electronic-structure workflow includes bands and DOS; optics can run directly after density prediction.

## Documentation website

These pages are also the source for the GitHub Pages website. See [Build and publish](publishing.md) for local preview and deployment, and [Troubleshooting](troubleshooting.md) for common workflow issues.

## References

- [Source repository](https://github.com/rubenlee11/EdenGNN)
- [EdenGNN skill repository](https://github.com/rubenlee11/EdenGNN-Skill)
- [Model weights](https://huggingface.co/TrueSavage/EdenGNN)
- [Example density data](https://huggingface.co/datasets/TrueSavage/EdenGNN-Data)
- [Colab tutorial](https://colab.research.google.com/drive/1tSGPZk4XI71GEylYeNDD1218smOEaFKc?usp=sharing)
- Li, X., Xin, Z., Yu, H. et al. *Efficient equivariant framework for universal charge density prediction*. Nature Machine Intelligence (2026). [DOI: 10.1038/s42256-026-01326-5](https://doi.org/10.1038/s42256-026-01326-5).
