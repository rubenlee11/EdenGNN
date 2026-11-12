r"""Generate VASP SAD inputs from structure files; does not run VASP.

Usage (activate the eqgnn environment first):
    python vasp_sad.py \
        --filelist ~/research/ml_charge_density/aips_tutorial/dataset/vasp/dataset.txt \
        --save-dir ~/research/ml_charge_density/aips_tutorial/dataset/vasp/sacd \
        --path-pp ~/research/ml_charge_density/EdenGNN-main/scripts/vasp/pp.json \
        --path-input ~/research/ml_charge_density/EdenGNN-main/scripts/vasp/template.incar \
        --for-scf --dk-bz 0.15 --for-band --dk-band 0.037

Inputs:
    --filelist   Required text file, one CIF/POSCAR structure path per line.
    --save-dir   Required output directory; each structure uses its file stem.
    --path-pp    Required JSON: element -> {"path_pp": "/path/to/POTCAR"}.
                 Relative POTCAR paths are relative to the JSON directory.
    --path-input Required INCAR template; all calculation parameters come from it.
    --nproc      Number of parallel input-generation processes (default: 64).
    --for-scf    Also write a Gamma-centered mesh to KPOINTS_SCF.
    --dk-bz      Mesh spacing in inverse angstrom (default: 0.15).
                 N_i = max(1, ceil(|b_i| / dk_bz)), with |b_i| including 2*pi.
    --for-band   Standardize to the primitive cell and write KPOINTS_BAND.
                 Without this flag, keep the input cell.
    --dk-band    Band k-point spacing in inverse angstrom (default: 0.037).

Outputs: POSCAR, INCAR, POTCAR, Gamma-only KPOINTS, and optionally KPOINTS_SCF
and KPOINTS_BAND. All k-points use the cell written to POSCAR.
ICHARG=12 and NELM=0 override the template. Set other SAD parameters, such as
ISPIN=1 and LCHARG=.TRUE., in the template.

For subsequent SCF/NSCF calculations, copy the SAD directory, copy KPOINTS_SCF
to KPOINTS, and set NELM to the desired iteration limit in INCAR. Use ICHARG=1
to start SCF from CHGCAR or ICHARG=11 to keep CHGCAR fixed for NSCF.
"""

import argparse
import json
import multiprocessing
import re
import shutil
from itertools import groupby
from pathlib import Path

import numpy as np
import seekpath
from pymatgen.core import Element, Structure


def write_potcar(path, elements, pseudo_files):
    """Concatenate the selected POTCAR files in POSCAR species-block order."""
    with open(path, "wb") as target:
        for element in elements:
            with open(pseudo_files[element]["path_pp"], "rb") as source:
                shutil.copyfileobj(source, target)


def _poscar_header(z, cell, pos):
    """Return a POSCAR header and species blocks in the original atom order."""
    symbols = [Element.from_Z(int(number)).symbol for number in z]
    species_counts = [(symbol, len(list(sites))) for symbol, sites in groupby(symbols)]
    lines = ["EdenGNN", "1.0"]
    lines.extend(" ".join(f"{value:.16f}" for value in vector) for vector in cell)
    lines.append(" ".join(symbol for symbol, _ in species_counts))
    lines.append(" ".join(str(count) for _, count in species_counts))
    lines.append("Direct")
    lines.extend(
        " ".join(f"{value:.16f}" for value in position)
        for position in pos @ np.linalg.inv(cell)
    )
    return "\n".join(lines) + "\n", [symbol for symbol, _ in species_counts]



def write_kpath(path, cell, k_coords, k_path, dk_band):
    """Write explicit primitive-cell band k-points with spacing in inverse angstrom."""
    reciprocal = 2 * np.pi * np.linalg.inv(cell).T
    segments = []
    points = []
    for start, end in k_path:
        k1 = np.asarray(k_coords[start])
        k2 = np.asarray(k_coords[end])
        distance = np.linalg.norm((k2 - k1) @ reciprocal)
        n_points = max(2, int(np.ceil(distance / dk_band)))
        segments.append([start, end, n_points])
        points.extend(np.linspace(k1, k2, n_points))
    with open(path, "w") as f:
        f.write(f"{json.dumps(segments)}\n{len(points)}\nReciprocal\n")
        for point in points:
            f.write(" ".join(f"{value:.12f}" for value in point) + " 1\n")


def vasp_run(path_structure, args, pseudo_files):
    """Read one structure and write a Gamma-only SAD calculation directory."""
    structure = Structure.from_file(path_structure)
    cell = structure.lattice.matrix
    fractional = structure.frac_coords
    z = np.asarray(structure.atomic_numbers)
    if args.for_band:
        band_path = seekpath.get_path((cell, fractional, z))
        cell = np.asarray(band_path["primitive_lattice"])
        fractional = np.asarray(band_path["primitive_positions"])
        z = np.asarray(band_path["primitive_types"])
        shape_factor = abs(np.linalg.det(cell)) / np.prod(np.linalg.norm(cell, axis=1))
        if shape_factor < 0.15:
            print(f"Skipping {path_structure}: primitive-cell shape factor {shape_factor:.6f}", flush=True)
            return

    path = Path(args.save_dir) / Path(path_structure).stem
    path.mkdir(parents=True, exist_ok=True)
    header, elements = _poscar_header(z, cell, fractional @ cell)
    (path / "POSCAR").write_text(header)
    write_potcar(path / "POTCAR", elements, pseudo_files)
    with (path / "INCAR").open("w") as f:
        for key, value in args.input_parameters.items():
            f.write(f"{key} = {value}\n")
    (path / "KPOINTS").write_text("Gamma mesh\n0\nGamma\n1 1 1\n0 0 0\n")
    if args.for_scf:
        reciprocal = 2 * np.pi * np.linalg.inv(cell).T
        k_grid = np.maximum(
            np.ceil(np.round(np.linalg.norm(reciprocal, axis=1) / args.dk_bz, 5)), 1
        ).astype(int)
        with (path / "KPOINTS_SCF").open("w") as f:
            f.write("Gamma mesh\n0\nGamma\n")
            f.write(" ".join(map(str, k_grid)) + "\n0 0 0\n")
    if args.for_band:
        write_kpath(path / "KPOINTS_BAND", cell, band_path["point_coords"], band_path["path"], args.dk_band)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--filelist", required=True, type=Path, help="Text list of structure paths")
    parser.add_argument("--save-dir", required=True, type=Path, help="Output directory")
    parser.add_argument("--path-pp", required=True, type=Path, help="POTCAR JSON mapping")
    parser.add_argument("--path-input", required=True, type=Path, help="INCAR parameter template")
    parser.add_argument("--nproc", type=int, default=64, help="Parallel input-generation processes (default: 64)")
    parser.add_argument("--for-scf", action="store_true", help="Also write KPOINTS_SCF mesh")
    parser.add_argument("--dk-bz", type=float, default=0.15, help="Mesh spacing in inverse angstrom (default: 0.15)")
    parser.add_argument("--for-band", action="store_true", help="Use primitive cell and write band k-points")
    parser.add_argument("--dk-band", type=float, default=0.037, help="Band spacing in inverse angstrom (default: 0.037)")
    args = parser.parse_args()
    args.input_parameters = {}
    with args.path_input.open() as f:
        for line in f:
            line = re.split(r"#|!", line, maxsplit=1)[0]
            for assignment in line.split(";"):
                if not assignment.strip():
                    continue
                key, value = assignment.split("=", 1)
                args.input_parameters[key.strip().upper()] = value.strip()
    args.input_parameters.update(ICHARG=12, NELM=0)
    path_pp = args.path_pp.expanduser().resolve()
    with path_pp.open() as f:
        pseudo_files = json.load(f)
    for entry in pseudo_files.values():
        source = Path(entry["path_pp"]).expanduser()
        if not source.is_absolute():
            source = path_pp.parent / source
        entry["path_pp"] = str(source)
    with args.filelist.open() as f:
        paths = [line.strip() for line in f if line.strip()]
    if args.nproc == 1:
        for path in paths:
            vasp_run(path, args, pseudo_files)
    else:
        with multiprocessing.Pool(min(multiprocessing.cpu_count(), args.nproc)) as pool:
            pool.starmap(vasp_run, [(path, args, pseudo_files) for path in paths])


if __name__ == "__main__":
    main()
