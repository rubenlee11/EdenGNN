# Installation and setup

## Install EdenGNN

Download the [EdenGNN repository](https://github.com/rubenlee11/EdenGNN) and install the package with pip in your Python environment. A C compiler, a Fortran compiler, and an appropriate PyTorch build are required. See the [README installation section](https://github.com/rubenlee11/EdenGNN#installation) for commands.

Installation copies scripts and the default configuration to `~/.edengnn/` and downloads universal models. To initialize missing resources, use `scripts/install_user_data.py` from the source checkout. Existing user files are preserved.

## Configure the execution machine

Configure `~/.edengnn/` on the machine that will run the calculations:

| Location | Configure |
| --- | --- |
| `cpu.sh`, `cuda.sh` | Python environment and required MPI/CUDA/library paths |
| `scripts/vasp/submit.sh` | VASP executable and launcher |
| `scripts/abacus/submit.sh` | Patched ABACUS executable and launcher |
| `scripts/vasp/pp.json` | PAW paths for custom training |
| `scripts/abacus/pp.json` | Pseudopotential and orbital paths for custom training |
| `edengnn-uni/vasp/pp.json` | PAW paths matching the universal VASP models |
| `edengnn-uni/abacus/pp.json` | Pseudopotential and orbital paths matching the universal ABACUS model |

VASP map entries use `path_pp`; ABACUS entries use `path_pp` and `path_basis`. Use absolute paths to files available on the execution machine. Universal models require the pseudopotential versions and basis sets used for their training.

## DFT backends

- **VASP:** keep PAW files, cutoff, precision, FFT-grid conventions, and `LMAXMIX` consistent throughout each workflow.
- **ABACUS:** apply `scripts/abacus/patch_deltarho` to ABACUS 3.10.0 and rebuild. The patch supports difference-density labels with `out_chg=3` and prediction restarts with `init_chg=drho`. Use parameter-only INPUT templates without an `INPUT_PARAMETERS` header; include the intended `ecutwfc`.
- **OpenMX / SIESTA:** see the [README](https://github.com/rubenlee11/EdenGNN#installation) for their build requirements.

## Install the optional skill

Install the `edengnn` directory from [EdenGNN-Skill](https://github.com/rubenlee11/EdenGNN-Skill) under `~/.codex/skills/` on the agent machine. Start a new agent session and invoke `$edengnn` with the desired backend, structures, and outputs.

Configure `~/.edengnn/` on the job machine even when the agent runs elsewhere. The [skill repository](https://github.com/rubenlee11/EdenGNN-Skill) provides installation and workflow instructions.

## Run calculations

Keep generated configurations, DFT outputs, logs, and checkpoints outside the source checkout. Filelists contain one absolute path per line. Use distinct structure names for distinct outputs.

On a workstation, use the configured launchers. On a cluster, submit through the scheduler, prefer partitions with available resources, and keep calculations off login nodes. Connect preparation, prediction, DFT, and plotting with success dependencies. CPU-only workflows can run sequentially in one job.

For multiple structures, pass all result directories to one plotting filelist and invoke the plotting script once. Set parallelism within the allocated resources.

## Memory settings

Keep `data.batch_size=1`. For CPU execution, use `run.device=cpu` and retain `run.num_gpus=1`.

Adjust `model.chunk_size_train` and `model.chunk_size_predict` to available memory. A reference starting point is 20,000,000 for an 80 GB GPU, scaled for smaller GPUs. Reduce the relevant chunk size if memory is insufficient; preserve checkpoint architecture settings during prediction.
