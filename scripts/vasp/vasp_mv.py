"""-----------------------------------------------------------------------------

move CHGCAR to training dir

Usage: python vasp_mv.py --config config_vasp.yaml

Setting:
    mv.filelist: where you performed vasp calculations.
        file structure:
            /path/to/dir_vasp/
                name/
                    POSCAR
                    CHGCAR
                    ...
-----------------------------------------------------------------------------"""

import os, multiprocessing, argparse, pathlib
from pymatgen.io.vasp import Chgcar
import numpy as np

def parse_chgcar(dir_work):
    name = pathlib.Path(dir_work).stem
    print(f"moving {name}", flush=True)
    if os.path.exists(f"{dir_work}/density.npy"):
        return None
    try:
        chgcar = Chgcar.from_file(os.path.join(dir_work, "CHGCAR"))
        density = chgcar.data["total"]
    except:
        print(f"Fail to move {name}", flush=True)
        return None
    np.save(f"{dir_work}/density.npy", density)
    np.savez(f"{dir_work}/aug.npz", data_aug=chgcar.data_aug)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filelist", type=str, required=True, help="Path to the filelist"
    )
    args = parser.parse_args()
    filelist = args.filelist

    with open(filelist, "r") as f:
        dirs = [line.strip() for line in f]

    nproc = multiprocessing.cpu_count() - 1
    with multiprocessing.Pool(processes=nproc) as pool:
        results = pool.starmap(parse_chgcar, [(d,) for d in dirs])

main()