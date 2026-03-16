"""-----------------------------------------------------------------------------

Generate Superposition of atomic charge from CHGCAR or json.gz (Materials Project)

Usage:

    make sure you have set "PMG_VASP_PSP_DIR"

    python vasp_sad.py --config config_vasp.yaml



-----------------------------------------------------------------------------"""

import os

from omegaconf import OmegaConf
import numpy as np
import pathlib, glob, multiprocessing, argparse, os
from pymatgen.io.vasp import Chgcar
from pymatgen.core import Structure
from pymatgen.io.vasp.inputs import Incar, Poscar, Potcar, Kpoints

from edengnn.data.basis_aug import pseudo_map


def _vasp_sad(structure, dir_work, cfg):
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
        # "ISYM": -1,
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
    print(name, flush=True)
    structure = Structure.from_file(path_poscar)
    dir_work = os.path.join(cfg.sad.save_dir, name)
    os.makedirs(dir_work, exist_ok=True)
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
