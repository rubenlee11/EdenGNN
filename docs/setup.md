# Installation and setup

[User guide](index.md)

## Install the Python package

Use a Python environment with a PyTorch build appropriate for your CPU/GPU. A C compiler and a Fortran compiler such as `gfortran` are needed to build the density-file readers.

```bash
git clone https://github.com/rubenlee11/EdenGNN.git
cd EdenGNN
pip install .
```

For the local workspace described in this project, activate the existing environment with `conda activate eqgnn`. On another machine, activate that machine's EdenGNN environment instead.

Installation copies scripts and the configuration template into `~/.edengnn/` and downloads universal-model resources from Hugging Face. Existing user files are preserved. To initialize missing resources manually from a checkout:

```bash
python scripts/install_user_data.py /absolute/path/to/EdenGNN
```

The README also provides an offline pip invocation:

```bash
pip install . --no-index --no-build-isolation
```

This disables pip's package-index access and isolated build environment. Build/runtime dependencies must already be installed, and the installer still accesses Hugging Face for its model listing and missing model files. It is therefore not a complete air-gapped installation procedure.

## Configure the execution machine

Configure resources on the machine that will run EdenGNN and DFT, including a remote server if applicable. The working layout is:

```text
~/.edengnn/
├── config.yaml
├── cpu.sh
├── cuda.sh
├── edengnn-uni/
│   ├── vasp/
│   │   ├── EdenGNN-Uni_vasp_pseudo.ckpt
│   │   ├── EdenGNN-Uni_vasp_aug.ckpt
│   │   ├── config_vasp_density.yaml
│   │   ├── config_vasp_aug.yaml
│   │   ├── INCAR
│   │   └── pp.json
│   └── abacus/
│       ├── EdenGNN-Uni_abacus.ckpt
│       ├── config_abacus.yaml
│       ├── INPUT
│       └── pp.json
└── scripts/
    ├── vasp/       # SAD generation, plotting, statistics, INCAR, pp.json, launcher
    └── abacus/     # SCF generation, plotting, statistics, optics, patch, launcher
```

Create or customize `cpu.sh` and `cuda.sh` to activate the Python environment and load the required compiler, MPI, CUDA, and library paths. Configure `scripts/vasp/submit.sh` and `scripts/abacus/submit.sh` with the actual DFT executable and launcher. The repository launcher files are placeholders; they must contain a working command before use.

For a local four-process calculation, a launcher could contain `mpirun -n 4 /absolute/path/to/executable`. Use the executable for the selected backend and the resources available on your machine. On a cluster, use the site's environment and submission conventions.

### Pseudopotentials and orbitals

Configure both the universal-model maps and the ordinary training maps:

| Backend | Universal-model map | Custom-training map |
| --- | --- | --- |
| VASP | `~/.edengnn/edengnn-uni/vasp/pp.json` | `~/.edengnn/scripts/vasp/pp.json` |
| ABACUS | `~/.edengnn/edengnn-uni/abacus/pp.json` | `~/.edengnn/scripts/abacus/pp.json` |

A VASP map associates each element with the correct PAW file:

```json
{
  "Si": {
    "path_pp": "/absolute/path/to/matching/Si/POTCAR"
  }
}
```

An ABACUS map also specifies the localized orbital basis:

```json
{
  "Si": {
    "path_pp": "/absolute/path/to/SG15/Si.upf",
    "path_basis": "/absolute/path/to/matching/Si.orb"
  }
}
```

Replace these illustrative filenames with your actual files. Universal VASP models require the exact PAW versions used for training. Universal ABACUS models require the corresponding SG15 pseudopotentials and orbital files; see the [ABACUS resource page](https://abacus.ustc.edu.cn/pseudo/list.htm). Matching element symbols alone is insufficient.

### DFT backends

- **VASP:** keep PAW files, `ENCUT`, `PREC`, FFT-grid settings, and `LMAXMIX` consistent between SAD, labels, and prediction. SAD uses `ISPIN=1`, `ICHARG=12`, `NELM=0`; fixed-density diagonalization uses `ICHARG=11` and positive `NELM`.
- **ABACUS:** use the repository's difference-density patch for ABACUS 3.10.0 (`scripts/abacus/patch_deltarho`). Apply the replacement files to their corresponding source-tree locations and rebuild following the ABACUS build instructions. Dataset generation uses `out_chg 3` to write `chgdelta.cube`; prediction uses `init_chg drho` to read a difference density from `SPIN1_CHG.cube`.
- **OpenMX:** the README describes the OpenMX 3.9 patch under `scripts/openmx/patch_nsc`.
- **SIESTA:** the README requires a build with NetCDF support.

The ABACUS generator and prediction template use parameter-only INPUT templates without an `INPUT_PARAMETERS` header. Generated calculation directories contain the complete INPUT file, including its header. Keep `ecutwfc` in the template so prediction uses the intended real-space grid.

## Install the optional skill

Obtain the `edengnn` skill directory from [EdenGNN-Skill](https://github.com/rubenlee11/EdenGNN-Skill) and place it at `~/.codex/skills/edengnn/` on the machine running the coding agent. The resulting file should be `~/.codex/skills/edengnn/SKILL.md`. Start a new agent session so it can discover the skill.

The agent and job machine can differ: install the skill where the agent runs, and configure `~/.edengnn/` where the jobs run. Invoke it with `$edengnn` and describe your input paths, backend, desired outputs, and remote machine if any. See the [EdenGNN-Skill repository](https://github.com/rubenlee11/EdenGNN-Skill) for the skill's installation instructions and workflow references.

## Local execution and cluster submission

The worked guides show sequential local commands. Their `bash "$E/scripts/<backend>/submit.sh"` calls assume a configured launcher that runs DFT synchronously inside the current directory. Scheduler directives such as `#SBATCH` do not submit a job when the file is executed with Bash.

On a cluster, submit preparation, inference, DFT, and plotting through the scheduler. Do not run Python workflows or DFT on login nodes. Prefer partitions with available resources and minimize requested core-hours. Keep intermediate files on a filesystem shared between job nodes.

For CPU/GPU workflows, submit a dependency chain:

| Backend | Stages |
| --- | --- |
| VASP | CPU preparation/SAD → GPU density and augmentation prediction → CPU bands/DOS and plotting |
| ABACUS | GPU preparation/prediction → CPU bands/DOS or matrices/PyATB and plotting |

Use `set -e` in each stage and success dependencies, such as Slurm `sbatch --dependency=afterok:<job_id> next.sh` or PBS `qsub -W depend=afterok:<job_id> next.sh`. If bands and DOS run as separate jobs, plotting must depend on both. Resolve prediction timestamps in the downstream job after prediction has finished. A CPU-only workflow can be one sequential job.

For this workspace, the remote SSH alias is `lxw`. Long DFT workflows can use a new directory under `lxw:~/.tmp/`, followed by synchronization of the resulting directory back to local storage.

## Memory and runtime settings

- Keep `data.batch_size: 1`.
- For CPU inference or training, set `run.device: cpu` and retain `run.num_gpus: 1`; setting it to zero causes an error.
- Set `model.chunk_size_predict` and `model.chunk_size_train` for available memory. A starting reference is 20,000,000 points for 80 GB GPU memory, scaled proportionally for smaller GPUs. Structure size and other allocations also affect memory use.
- If inference runs out of memory, lower `chunk_size_predict`; for training lower `chunk_size_train`.
- Preserve checkpoint model hyperparameters during inference. Changes to channel counts or architecture require a compatible newly trained model.
- Use absolute paths in YAML and JSON. Expand `~` before writing checkpoint and resource paths.
