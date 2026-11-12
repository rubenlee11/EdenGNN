r"""Generate ABACUS SCF dataset inputs from a list of structures; does not run ABACUS.

Usage:
    python gen_abacus_cal.py \
        --filelist /root/research/ml_charge_density/aips_tutorial/dataset/abacus/dataset.txt \
        --save-dir /root/research/ml_charge_density/aips_tutorial/dataset/abacus/scf \
        --path-pp ./pp.json \
        --path-input ./INPUT \
        --for-band --dk-bz 0.15

Inputs:
    --filelist   Required text file, one CIF/structure path per line.
    --save-dir   Required output directory; each structure uses its file stem.
    --path-pp    Required JSON: element -> {"path_pp": "/path/to/element.upf",
                 "path_basis": "/path/to/element.orb"}. Relative species paths
                 are resolved against the JSON directory.
    --path-input Required INPUT template containing parameters only, without
                 the INPUT_PARAMETERS header.
    --nproc      Parallel input-generation processes (default: 64).
    --dk-bz      Gamma mesh spacing in inverse angstrom (default: 0.15).
                 N_i = max(1, ceil(|b_i| / dk_bz)), including 2*pi in |b_i|.
    --for-band   Use the standard primitive cell and also write KPT_BAND.
                 Without this flag, keep the input cell.
    --dk-band    Band k-point spacing in inverse angstrom (default: 0.037).

Outputs: INPUT, STRU, KPT, copied pseudopotentials/orbitals, optional KPT_BAND.
The template is overridden with calculation=scf, basis_type=lcao,
init_chg=atomic and out_chg=3. With the repository's patched ABACUS, out_chg=3
produces chgdelta.cube for EdenGNN training. Other parameters, including suffix,
ecutwfc, smearing and SCF convergence settings, come from the template.
"""

import argparse
import json
import multiprocessing
from pathlib import Path

from edengnn.data.io.abacus.parse_density import (
    _read_input,
    get_k_grid,
    read_structure,
    write_input_files,
    write_kpath,
)


def generate_inputs(path_structure, args, pseudo_files):
    """Write one SCF directory using the shared ABACUS structure/input writers."""
    structure, band_path = read_structure(path_structure, args.for_band)
    dir_work = args.save_dir / Path(path_structure).stem
    write_input_files(
        dir_work,
        structure,
        get_k_grid(structure, args.dk_bz),
        args.input_parameters,
        pseudo_files,
        args.path_pp,
    )
    if args.for_band:
        write_kpath(
            dir_work / "KPT_BAND",
            structure,
            band_path["point_coords"],
            band_path["path"],
            args.dk_band,
        )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--filelist", required=True, type=Path, help="Text list of structure paths")
    parser.add_argument("--save-dir", required=True, type=Path, help="Output directory")
    parser.add_argument("--path-pp", required=True, type=Path, help="Pseudopotential/orbital JSON mapping")
    parser.add_argument("--path-input", required=True, type=Path, help="Parameter-only INPUT template")
    parser.add_argument("--nproc", type=int, default=64, help="Parallel processes (default: 64)")
    parser.add_argument("--dk-bz", type=float, default=0.15, help="Mesh spacing in inverse angstrom (default: 0.15)")
    parser.add_argument("--for-band", action="store_true", help="Use primitive cell and write KPT_BAND")
    parser.add_argument("--dk-band", type=float, default=0.037, help="Band spacing in inverse angstrom (default: 0.037)")
    args = parser.parse_args()
    args.path_pp = args.path_pp.expanduser().resolve()
    args.input_parameters = _read_input(args.path_input)
    args.input_parameters.update(
        calculation="scf", basis_type="lcao", init_chg="atomic", out_chg=3
    )
    with args.path_pp.open() as f:
        pseudo_files = json.load(f)
    with args.filelist.open() as f:
        paths = [line.strip() for line in f if line.strip()]
    if args.nproc == 1:
        for path in paths:
            generate_inputs(path, args, pseudo_files)
    else:
        with multiprocessing.Pool(min(multiprocessing.cpu_count(), args.nproc)) as pool:
            pool.starmap(generate_inputs, [(path, args, pseudo_files) for path in paths])


if __name__ == "__main__":
    main()
