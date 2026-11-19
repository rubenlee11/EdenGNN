r"""Compare VASP NSCF/SCF results and save statistics and band comparisons.

Usage (activate the eqgnn environment first):
    python scripts/vasp/stat.py \
        --filelist /path/to/calculation_dirs.txt \
        --save-dir /path/to/statistics --nproc 4
    # Optional separate reference root:
    python scripts/vasp/stat.py \
        --filelist /path/to/calculation_dirs.txt \
        --dir-tar /path/to/reference --save-dir /path/to/statistics

Inputs:
    --filelist Required list of structure calculation directories, one per line.
               Each contains nscf/ and, by default, scf/.
    --save-dir Required directory for stat_mae.json.
    --dir-tar  Optional reference root; read <root>/<structure name>/{scf,band_scf}/.
               Without this option, read scf/ alongside each nscf/ directory.
    --nproc    Parallel analysis processes (default: 1).

SCF and NSCF directories need vasprun.xml, KPOINTS and CHGCAR.
Retains band-energy, density, augmentation, energy-per-atom, band-gap and force
statistics. band/ and band_scf/ provide high-symmetry comparison bands.
Per-structure comparison images are saved in the listed directories.
"""

import os, pathlib, json, argparse, multiprocessing, re
import numpy as np
from pymatgen.io.vasp.outputs import Vasprun
from edengnn.data.io.vasp.parse_density import _read_chgcar
from pymatgen.electronic_structure.core import Spin
import matplotlib
from matplotlib.lines import Line2D

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_band import get_band_data, plot_band_dos

def aug2irreps(aug, aug_l, lmix_max):
    aug_irreps = []
    aug_tensor = []
    ibase = 0
    jbase = 0

    # e3nn calls in spherical harmonics as (y, z, x)
    # a translation of RETRIEVE_RHOLM in VASP source code
    # e3nn and VASP has the same convention for spherical tensor ():
    #   l = 1:  -1, 0, 1
    #   l = 2: -2, -1, 0, 1, 2
    #   ......
    for i, l_i in enumerate(aug_l):
        for l_j in aug_l[i:]:
            l_min = abs(l_j - l_i)
            l_max = min(abs(l_i + l_j), lmix_max)

            jbase = ibase
            for lmain in range(l_min, l_max + 1, 2):
                aug_tensor.extend(aug[jbase : jbase + 2 * lmain + 1])
                aug_irreps.append(lmain)
                jbase += 2 * lmain + 1

            for lmain in range(l_min, abs(l_i + l_j) + 1, 2):
                ibase += 2 * lmain + 1

    return aug_irreps, np.array(aug_tensor)


def parse_chgcar(path, lmix_max=2):
    """Read density in electrons/Å³ and flattened PAW augmentation occupancies."""
    _, _, _, _, density, lines = _read_chgcar(os.fspath(path))

    aug_num = []
    aug_list_str = []
    augs_str = None
    aug_re = r"augmentation\s+occupancies\s*(\d+)\s+(\d+)"
    for line in lines:
        if line.startswith("augment"):
            m = re.search(aug_re, line)
            aug_num.append(int(m.group(2)))
            if augs_str:
                aug_list_str.append(augs_str)
            augs_str = []
        else:
            augs_str.append(line.strip())
    aug_list_str.append(augs_str)
    aug_list = []
    for i, aug_str in enumerate(aug_list_str):
        aug = []
        for line in aug_str:
            aug.extend(map(float, line.split()))
        aug_list.extend(np.array(aug[0 : aug_num[i]]))
    return (density, np.array(aug_list))


def stat_vasprun(work_path, band=False):
    vasprun_path = os.path.join(work_path, "vasprun.xml")

    try:
        vr = Vasprun(
            vasprun_path, parse_dos=True, parse_eigen=True, parse_potcar_file=False
        )
        if not vr.converged:
            return None

        final_structure = vr.structures[-1]

        kpoints = None
        energies = None
        band_gap = None

        if band:
            bs = vr.get_band_structure(line_mode=True)
        else:
            bs = vr.get_band_structure(line_mode=False)

        kpoints = bs.kpoints

        if bs.is_metal():
            e0 = bs.efermi
        else:
            e0 = bs.get_vbm()["energy"]

        # energies = bs.bands[Spin.up] - e0
        energies = bs.bands[Spin.up]
        band_gap = bs.get_band_gap()["energy"]

        forces = []
        if vr.ionic_steps:
            forces = vr.ionic_steps[-1].get("forces", [])

        return {
            "energy_per_atom": float(vr.final_energy) / len(final_structure),
            "band_gap": band_gap,
            "kpoints": kpoints,
            "band_energies": energies,
            "forces": np.array(forces),
            "e0": e0,
        }, bs

    except Exception as e:
        print(f"error: {work_path} {str(e)}")
        return None


def plot_band_comparison(predicted_dir, target_dir, save_path):
    """Overlay high-symmetry bands using the target VBM/Ef as energy zero."""
    predicted = get_band_data(predicted_dir)
    target = get_band_data(target_dir)
    energy_zero = target["e_vbm"]
    fig = plot_band_dos(
        target["distances"], [energy - energy_zero for energy in target["energies"]],
        target["tick_positions"], target["tick_labels"], target["e_cbm"] - energy_zero,
    )
    try:
        for ax in fig.axes:
            for distance, energy in zip(predicted["distances"], predicted["energies"]):
                ax.plot(distance, energy - energy_zero, color="red", linewidth=0.9,
                        linestyle="--")
        fig.axes[0].legend(handles=[Line2D([], [], color="#245c91", label="SCF target"),
                                   Line2D([], [], color="red", linestyle="--", label="Prediction")])
        fig.savefig(save_path, dpi=300)
    finally:
        plt.close(fig)


def stat_structure(dir_work, args):
    name = pathlib.Path(dir_work).stem
    try:
        path_nscf = os.path.join(dir_work, "nscf")
        if args.dir_tar is not None:
            path_scf = os.path.join(args.dir_tar, name, "scf")
        else:
            path_scf = os.path.join(dir_work, "scf")

        data_nscf, bs = stat_vasprun(path_nscf, band=False)
        data_scf, bs = stat_vasprun(path_scf, band=False)

        reference_root = pathlib.Path(args.dir_tar) / name if args.dir_tar else pathlib.Path(dir_work)
        plot_band_comparison(pathlib.Path(dir_work) / "band", reference_root / "band_scf",
                             pathlib.Path(dir_work) / "band_compare.png")
        stat_mae = {"name": name}

        chg_nscf, aug_nscf = parse_chgcar(
            os.path.join(path_nscf, "CHGCAR")
        )
        chg_scf, aug_scf = parse_chgcar(
            os.path.join(path_scf, "CHGCAR")
        )
        den_pre = chg_nscf.flatten()
        den_tar = chg_scf.flatten()

        stat_mae["mae_density"] = np.sum(np.abs(den_pre - den_tar)) / np.sum(den_tar)
        stat_mae["mae_aug"] = np.abs(aug_nscf - aug_scf).mean()
        stat_mae["r2_aug"] = float(
            1.0 - np.sum((aug_nscf - aug_scf) ** 2)
            / np.sum((aug_nscf - aug_scf.mean()) ** 2)
        )
        # energy, forces, band energies, band gaps
        stat_mae["mae_" + "band_energies"] = np.abs(
            data_nscf["band_energies"] - data_scf["band_energies"]
        ).mean()
        stat_mae["mae_" + "energy_per_atom"] = np.abs(
            data_nscf["energy_per_atom"] - data_scf["energy_per_atom"]
        ).mean()
        stat_mae["mae_" + "band_gap"] = np.abs(
            data_nscf["band_gap"] - data_scf["band_gap"]
        ).mean()
        stat_mae["gap_predict_ev"] = float(data_nscf["band_gap"])
        stat_mae["gap_target_ev"] = float(data_scf["band_gap"])
        stat_mae["mae_" + "forces"] = np.abs(
            data_nscf["forces"] - data_scf["forces"]
        ).mean()

        return stat_mae

    except Exception as e:
        print("Caught exception:", e)
        return None


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--filelist", required=True, type=pathlib.Path, help="List of structure calculation directories")
    parser.add_argument("--save-dir", required=True, type=pathlib.Path, help="Statistics output directory")
    parser.add_argument("--dir-tar", type=pathlib.Path, help="Optional reference root containing <name>/scf")
    parser.add_argument("--nproc", type=int, default=1, help="Parallel analysis processes (default: 1)")
    args = parser.parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    with args.filelist.open() as f:
        dirs_work = [line.strip() for line in f if line.strip()]

    if args.nproc == 1:
        results = [stat_structure(path, args) for path in dirs_work]
    else:
        nproc = min(multiprocessing.cpu_count(), args.nproc)
        with multiprocessing.Pool(processes=nproc) as pool:
            results = pool.starmap(stat_structure, [(path, args) for path in dirs_work])

    maes = [result for result in results if result is not None]

    if not maes:
        raise ValueError("No structures were processed; statistics cannot be calculated.")

    with open(os.path.join(args.save_dir, "stat_mae.json"), "w") as f:
        json.dump(maes, f, indent=2)



if __name__ == "__main__":
    main()
