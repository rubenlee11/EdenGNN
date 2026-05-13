"""-----------------------------------------------------------------------------

Generate VASP calculations from CHGCAR

Usage:

    make sure you have set "PMG_VASP_PSP_DIR"

    python vasp_band.py --config config_vasp.yaml

-----------------------------------------------------------------------------"""

import os
import numpy as np
from omegaconf import OmegaConf
from pymatgen.io.vasp.inputs import Incar, Poscar, Potcar, Kpoints
from pymatgen.io.vasp.outputs import Chgcar
import pathlib, argparse, json, glob, multiprocessing, shutil

KAPPA = 900
DK_BAND = 0.037

from edengnn.data.basis_vasp import pseudo_map


def _vasp_nscf(structure, chgcar, dir_work, cfg):
    os.makedirs(dir_work, exist_ok=True)

    poscar = Poscar(structure)
    potcar_symbols = list(
        dict.fromkeys([pseudo_map[i] for i in structure.atomic_numbers])
    )
    potcar = Potcar(potcar_symbols, functional="PBE")

    incar_dict = {
        "SYSTEM": "Non-SCF",
        "ICHARG": 11,
        "IBRION": -1,
        "NSW": 0,
        "ALGO": cfg.incar.get("algo", "Normal"),
        "ISMEAR": cfg.incar.get("ismear", -5),
        "SIGMA": cfg.incar.get("sigma", 0.05),
        "PREC": cfg.incar.get("prec", "Normal"),
        "NCORE": cfg.incar.get("ncore", 4),
        "LMAXMIX": cfg.incar.get("lmaxmix", 2),
        "LWAVE": False,
        "LCHARG": False,
        "LREAL": "Auto",
        "NELM": 100,
        "LASPH": True,
        "ADDGRID": True,
    }

    if cfg.incar.get("encut"):
        incar_dict["ENCUT"] = cfg.incar.get("encut")
    incar = Incar(incar_dict)

    kpoints = Kpoints.automatic_density(structure, kppa=KAPPA)

    poscar.write_file(os.path.join(dir_work, "POSCAR"))
    potcar.write_file(os.path.join(dir_work, "POTCAR"))
    incar.write_file(os.path.join(dir_work, "INCAR"))
    kpoints.write_file(os.path.join(dir_work, "KPOINTS"))
    chgcar.write_file(os.path.join(dir_work, "CHGCAR"))


def _vasp_scf(structure, chgcar, dir_work, cfg):
    os.makedirs(dir_work, exist_ok=True)

    poscar = Poscar(structure)
    potcar_symbols = list(
        dict.fromkeys([pseudo_map[i] for i in structure.atomic_numbers])
    )
    potcar = Potcar(potcar_symbols, functional="PBE")

    incar_dict = {
        "SYSTEM": "Non-SCF",
        "ICHARG": 1,
        "IBRION": -1,
        "NSW": 0,
        "ALGO": cfg.incar.get("algo", "Normal"),
        "ISMEAR": cfg.incar.get("ismear", -5),
        "SIGMA": cfg.incar.get("sigma", 0.05),
        "PREC": cfg.incar.get("prec", "Normal"),
        "NCORE": cfg.incar.get("ncore", 4),
        "LMAXMIX": cfg.incar.get("lmaxmix", 2),
        "LWAVE": False,
        "LCHARG": True,
        "LREAL": "Auto",
        "NELM": 100,
        "LASPH": True,
        "ADDGRID": True,
    }

    if cfg.incar.get("encut"):
        incar_dict["ENCUT"] = cfg.incar.get("encut")
    incar = Incar(incar_dict)

    kpoints = Kpoints.automatic_density(structure, kppa=KAPPA)

    poscar.write_file(os.path.join(dir_work, "POSCAR"))
    potcar.write_file(os.path.join(dir_work, "POTCAR"))
    incar.write_file(os.path.join(dir_work, "INCAR"))
    kpoints.write_file(os.path.join(dir_work, "KPOINTS"))
    chgcar.write_file(os.path.join(dir_work, "CHGCAR"))


def vasp_run(path_chgcar, save_dir, task, cfg):
    """ """
    path = pathlib.Path(path_chgcar)
    name = path.stem if path.suffix else path.name
    chgcar = Chgcar.from_file(path_chgcar)
    structure = chgcar.structure
    dir_work = os.path.join(save_dir, name)
    os.makedirs(dir_work, exist_ok=True)

    _vasp_nscf(structure, chgcar, os.path.join(dir_work, "nscf"), cfg)
    _vasp_scf(structure, chgcar, os.path.join(dir_work, "scf"), cfg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    path = cfg.band.get("path", None)
    dir = cfg.band.get("dir")
    save_dir = cfg.band.get("save_dir")

    path_chgcars = []
    if path:
        with open(path, "r") as f:
            names = json.load(f)
        for name in names:
            path_chgcars.append(os.path.join(dir, name))
    else:
        path_chgcars = glob.glob(dir)

    nproc = min(multiprocessing.cpu_count(), cfg.run.nproc)
    with multiprocessing.Pool(processes=nproc) as pool:
        pool.starmap(
            vasp_run,
            [
                (path_chgcar, save_dir, cfg.band.get("task", 0), cfg)
                for path_chgcar in path_chgcars
            ],
        )


main()
