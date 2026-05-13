"""-----------------------------------------------------------------------------

Generate Superposition of atomic charge from atomic structures

Usage:

    make sure you have set "PMG_VASP_PSP_DIR"

    python vasp_sad.py --config config.yaml

-----------------------------------------------------------------------------"""

import seekpath, json
import numpy as np
import os, pathlib, multiprocessing, argparse
from omegaconf import OmegaConf
from pymatgen.core import Structure
from pymatgen.io.vasp.inputs import Incar, Poscar, Potcar, Kpoints
from pymatgen.core import Structure, Lattice
from edengnn.data.basis_vasp import pseudo_map

CELL_ILL_CONDITION = 0.15
DK_BAND = 0.037


def write_kpath(structure_, dk_band):
    """
    write K points along high-symmetry lines
    """
    # --------------------------------------------------------------------------
    # make sure the input structure is primitive
    # --------------------------------------------------------------------------
    cell = structure_.lattice.matrix
    positions = structure_.frac_coords
    numbers = [site.specie.number for site in structure_]
    sp_res = seekpath.get_path(
        (cell, positions, numbers),
    )

    structure = Structure(
        lattice=Lattice(sp_res["primitive_lattice"]),
        species=sp_res["primitive_types"],
        coords=sp_res["primitive_positions"],
        coords_are_cartesian=False,
    )
    # --------------------------------------------------------------------------
    # Using the Explicit k-point mesh mode of KPOINTS
    # --------------------------------------------------------------------------
    r_cell = structure.lattice.reciprocal_lattice.matrix
    k_path = sp_res["path"]
    k_coords = sp_res["point_coords"]

    nk_total = 0
    lines_kcoords = ""
    header = []
    for start_label, end_label in k_path:
        k1_frac = np.array(k_coords[start_label])
        k2_frac = np.array(k_coords[end_label])

        k1_cart = np.dot(k1_frac, r_cell)
        k2_cart = np.dot(k2_frac, r_cell)

        dist = np.linalg.norm(k2_cart - k1_cart)

        nk = max(2, int(np.round(dist / dk_band)))

        header.append([start_label, end_label, nk])
        nk_total += nk
        kcoords = np.linspace(k1_frac, k2_frac, nk)
        for kcoord in kcoords:
            lines_kcoords += f"{kcoord[0]:6f}  {kcoord[1]:6f}  {kcoord[2]:6f}   1\n"

    header = json.dumps(header)
    lines = f"{header}\n{nk_total}\n" + "Reciprocal\n" + lines_kcoords
    return structure, lines


def _vasp_sad(structure_in, dir_work, cfg):
    """
    If the goal is to perform band structure calculations, the structure is
    transformed to the standardized primitive cell used by seekpath.
    Ill-conditioned primitive cells are excluded for plane-wave band calculations.
    see "Comp. Mat. Sci. 128, 140 (2017). DOI: 10.1016/j.commatsci.2016.10.015"
    """

    if cfg.sad.for_band:
        structure, lines = write_kpath(structure_in, DK_BAND)

        # check if the primitive structure is PW friendly
        cell = structure.lattice.matrix
        volume = structure.volume
        a = np.linalg.norm(cell[0])
        b = np.linalg.norm(cell[1])
        c = np.linalg.norm(cell[2])
        shape_factor = volume / (a * b * c)
        if shape_factor < CELL_ILL_CONDITION:
            print(
                f"bad structure for band calculations: {dir_work}\n shape factor: {shape_factor}",
                flush=True,
            )
            return None
        else:
            print(f"{dir_work} shape factor: {shape_factor}", flush=True)
            os.makedirs(dir_work, exist_ok=True)
            # generate kpoints file and high symmetry points for band structure calculations
            with open(os.path.join(dir_work, "KPOINTS_BAND"), "w") as f:
                f.write(lines)

    else:
        os.makedirs(dir_work, exist_ok=True)
        structure = structure_in

    poscar = Poscar(structure)

    potcar_symbols = list(
        dict.fromkeys([pseudo_map[i] for i in structure.atomic_numbers])
    )
    potcar = Potcar(potcar_symbols, functional="PBE")

    incar_dict = {
        "PREC": "Normal",
        "ISMEAR": 0,
        "SYSTEM": "SAD",
        "ICHARG": 12,
        "IBRION": -1,
        "NSW": 0,
        "NELM": 0,
        "LWAVE": False,
        "LCHARG": True,
        "LREAL": "Auto",
        "NCORE": 4,
        "SYMPREC": 1e-8,
    }
    if cfg.incar.get("encut"):
        incar_dict["ENCUT"] = cfg.incar.get("encut")
    incar = Incar(incar_dict)
    kpoints = Kpoints.gamma_automatic(kpts=[1, 1, 1])

    poscar.write_file(os.path.join(dir_work, "POSCAR"))
    potcar.write_file(os.path.join(dir_work, "POTCAR"))
    incar.write_file(os.path.join(dir_work, "INCAR"))
    kpoints.write_file(os.path.join(dir_work, "KPOINTS"))


def vasp_run(path_poscar, cfg):
    name = pathlib.Path(path_poscar).stem
    structure = Structure.from_file(path_poscar)
    dir_work = os.path.join(cfg.sad.save_dir, name)
    _vasp_sad(structure, os.path.join(cfg.sad.save_dir, name), cfg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    with open(cfg.sad.get("filelist"), "r") as f:
        filelist = [line.strip() for line in f if line.strip()]

    nproc = min(multiprocessing.cpu_count(), cfg.run.nproc)
    with multiprocessing.Pool(processes=nproc) as pool:
        pool.starmap(vasp_run, [(path_poscar, cfg) for path_poscar in filelist])


main()
