"""-----------------------------------------------------------------------------

Generate VASP dataset

Usage:

    make sure you have set "PMG_VASP_PSP_DIR"

    python vasp_dataset.py --config config_vasp.yaml

-----------------------------------------------------------------------------"""

import os

from omegaconf import OmegaConf
from pymatgen.io.vasp.inputs import Incar, Poscar, Potcar, Kpoints
from pymatgen.core import Structure
import pathlib, argparse, multiprocessing

KAPPA = 900

from edengnn.data.basis_vasp import pseudo_map


def _vasp_scf(structure, dir_work, cfg):
    os.makedirs(dir_work, exist_ok=True)
    # vis = MPRelaxSet(structure)
    # vis.write_input(dir_work)

    poscar = Poscar(structure)
    potcar_symbols = list(
        dict.fromkeys([pseudo_map[i] for i in structure.atomic_numbers])
    )
    potcar = Potcar(potcar_symbols, functional="PBE")

    incar_dict = {
        "SYSTEM": "SCF",
        "ICHARG": 2,
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
        "SYMPREC": 1e-8,
        "ISYM": -1,
    }

    if cfg.incar.get("encut"):
        incar_dict["ENCUT"] = cfg.incar.get("encut")
    incar = Incar(incar_dict)

    # kpoints = Kpoints.monkhorst_automatic([6, 6, 6])
    # kpoints = Kpoints.gamma_automatic(kpts=[6, 6, 6])
    kpoints = Kpoints.automatic_density(structure, kppa=KAPPA)

    poscar.write_file(os.path.join(dir_work, "POSCAR"))
    potcar.write_file(os.path.join(dir_work, "POTCAR"))
    incar.write_file(os.path.join(dir_work, "INCAR"))
    kpoints.write_file(os.path.join(dir_work, "KPOINTS"))


def vasp_run(path_poscar, save_dir, cfg):
    name = pathlib.Path(path_poscar).stem
    print(name, flush=True)
    structure = Structure.from_file(path_poscar)

    dir_work = os.path.join(save_dir, name)
    os.makedirs(dir_work, exist_ok=True)
    _vasp_scf(structure, dir_work, cfg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    with open(cfg.dataset.get("filelist"), "r") as f:
        filelist = [line.strip() for line in f if line.strip()]
    save_dir = cfg.dataset.get("save_dir")

    nproc = min(multiprocessing.cpu_count(), cfg.run.nproc)
    with multiprocessing.Pool(processes=nproc) as pool:
        pool.starmap(
            vasp_run,
            [(path_poscar, save_dir, cfg) for path_poscar in filelist],
        )


main()
