"""-----------------------------------------------------------------------------

Test the possibility of out-of-memory in the training stage, by examining the
number of points in the back propagation.

n_point = n_atom * grid_per_atom

Usage:
python /root/research/e3super-resolution/EdenGNN_git/scripts/tools/test_oom.py \
    --radius 4.0 \
    --encut 100 \
    --mode "pw" \
    --filelist "/root/dataset/universal_abacus/mc3d/cifs_debug.txt" \
    --path_out "/root/dataset/universal_abacus/mc3d/npb.json" \
    --save_dir "/root/dataset/universal_abacus/mc3d/cifs_debug"
    
python /root/research/e3super-resolution/EdenGNN_git/scripts/tools/test_oom.py \
    --radius 4.0 \
    --encut 100 \
    --mode "pw" \
    --filelist "/root/dataset/universal_abacus/mc3d/cifs_nonmag.txt" \
    --path_out "/root/dataset/universal_abacus/mc3d/npb_radius4.json" \
    --save_dir "/root/dataset/universal_abacus/mc3d/cifs_work"    
-----------------------------------------------------------------------------"""

import numpy as np
from pymatgen.core import Structure
import os, json, multiprocessing, pathlib, argparse
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from edengnn.data.io_openmx import _set_grid, _round_for_fft

# from edengnn.data.dataload import get_mask_r

BOHR = 0.5291772109
DL = 0.15 * BOHR
# SAVE_DIR = "/root/dataset/universal_abacus/mc3d/cifs_work"


def set_grid_fft(cell, cutoff):
    """
    set fft grid according to the cutoff energy

    |G| < E_cut

    using atomic unit

    """

    cell_G = np.linalg.inv(cell.T)

    pf_ = np.sqrt(cutoff)

    m1_max = int(np.ceil(pf_ * np.linalg.norm(cell[0])))
    m2_max = int(np.ceil(pf_ * np.linalg.norm(cell[1])))
    m3_max = int(np.ceil(pf_ * np.linalg.norm(cell[2])))

    m1_range = np.arange(-m1_max, m1_max)
    m2_range = np.arange(-m2_max, m2_max)
    m3_range = np.arange(-m3_max, m3_max)

    M1, M2, M3 = np.meshgrid(m1_range, m2_range, m3_range, indexing="ij")
    mask_grid = np.stack([M1.ravel(), M2.ravel(), M3.ravel()], axis=1)
    g_grid = np.dot(mask_grid, cell_G)

    gg_grid = np.sum(g_grid**2, axis=1)
    mask = gg_grid <= cutoff

    mask_fft = mask_grid[mask]
    mask_all = np.vstack(mask_fft)

    nfft1 = _round_for_fft(np.max(mask_all[:, 0]))
    nfft2 = _round_for_fft(np.max(mask_all[:, 1]))
    nfft3 = _round_for_fft(np.max(mask_all[:, 2]))

    return nfft1, nfft2, nfft3


def get_lll(structure):
    a, b, c = structure.lattice.abc
    return a * b * c


def get_npoint(structure, mode, encut, radius):

    cell = structure.lattice.matrix / BOHR
    omega = structure.lattice.volume
    a, b, c = structure.lattice.abc

    if mode == "lcao":
        n1, n2, n3 = _set_grid(cell, encut)
    elif mode == "pw":
        # n1, n2, n3 = set_grid_fft(cell, encut)
        n1 = a / DL
        n2 = b / DL
        n3 = c / DL

    z = structure.atomic_numbers
    npb = len(z) * 4.0 * np.pi / 3.0 * radius**3 * n1 * n2 * n3 / omega
    return npb, n1, n2, n3


def analyze_cif(path_cif, save_dir, mode, encut, radius):
    name = pathlib.Path(path_cif).stem
    s = Structure.from_file(path_cif)
    sga = SpacegroupAnalyzer(s, symprec=1e-5)

    # s_refined = sga.get_refined_structure()
    s_prim = sga.get_primitive_standard_structure()
    s_conv = sga.get_conventional_standard_structure()

    structures = {
        # "original": s,
        # "refined": s_refined,
        "primitive": s_prim,
        "conventional": s_conv,
    }

    info = {}
    for k, struct in structures.items():
        npb, n1, n2, n3 = get_npoint(struct, mode, encut, radius)

        lll = get_lll(struct)
        v_lll_ratio = struct.lattice.volume / lll

        info[k] = {"v_lll": v_lll_ratio, "npb": npb, "n1": n1, "n2": n2, "n3": n3}

    if info["conventional"]["npb"] < info["primitive"]["npb"] - 1.0:
        min_key = "conventional"
    else:
        min_key = "primitive"
    structure_to_use = structures[min_key]
    structure_to_use.to(filename=os.path.join(save_dir, f"{name}.cif"), fmt="cif")

    return {
        "name": name,
        "info": info,
        "npb": info[min_key]["npb"],
        "n_atom": len(structures[min_key]),
        "cell": min_key,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--radius",
        type=float,
        required=True,
        help="radius of atomic density, in Angstrom",
    )
    parser.add_argument(
        "--encut",
        type=float,
        required=False,
        default=None,
        help="cutoff energy for FFT grid, in Rydberg",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        help="lcao or pw",
    )
    parser.add_argument(
        "--filelist",
        type=str,
        required=True,
        help="path to the filelist file of cif paths",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="path to save final structure",
    )
    parser.add_argument(
        "--path_out",
        type=str,
        required=True,
        help="path to the output json file",
    )
    args = parser.parse_args()

    with open(args.filelist, "r") as f:
        paths = [line.strip() for line in f]

    with multiprocessing.Pool(
        processes=max(1, multiprocessing.cpu_count() - 1)
    ) as pool:
        tasks = [
            (path, args.save_dir, args.mode, args.encut, args.radius) for path in paths
        ]
        results = pool.starmap(analyze_cif, tasks)

    results.sort(key=lambda x: x["npb"], reverse=True)
    with open(args.path_out, "w") as f:
        json.dump(results, f, indent=2)


main()
