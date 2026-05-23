"""-----------------------------------------------------------------------------

Analysis VASP calculations

Usage: python vasp_stat.py --config config_vasp.yaml

-----------------------------------------------------------------------------"""

import os, glob, pathlib, json, argparse, multiprocessing, re
import numpy as np
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.io.vasp import Chgcar
from pymatgen.electronic_structure.core import Spin
from omegaconf import OmegaConf
import matplotlib
from matplotlib.lines import Line2D

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# colors = plt.cm.tab10.colors
colors = ["#6ca4a7", "#f0dfa7", "#7d7a7a", "#c199a2", "#995d6b"]
color_baseline = "darkolivegreen"
color_parity = "darkblue"

sample_density = 1000
xlabel_dict = {
    "band_energy": r"$\varepsilon_{nk}$ (eV)",
    "density": r"$\tilde{n}(\mathbf{r})$ $\mathrm{(e\ \AA^{-3})}$",
    "aug": r"$\rho^a_{NLM}$",
    # "energy_per_atom": r"$E$ (eV per atom)",
    "energy_per_atom": r"$E$ $(\mathrm{eV\cdot atom}^{-1})$",
    "band_gap": r"$E_g$ (eV)",
    "forces": r"$F$ $(eV\ \AA^{-1})$",
}
ylabel_dict = {
    "band_energy": r"$\hat{\varepsilon}_{nk}$ (eV)",
    "density": r"$\hat{\tilde{n}}(\mathbf{r})$ $\mathrm{(e\ \AA^{-3})}$",
    "aug": r"$\hat{\rho}^a_{NLM}$",
    # "energy_per_atom": r"$\hat{E}$ (eV per atom)",
    "energy_per_atom": r"$\hat{E}$ $(\mathrm{eV\cdot atom}^{-1})$",
    "band_gap": r"$\hat{E}_g$ (eV)",
    "forces": r"$\hat{F}$ $(eV\ \AA^{-1})$",
}
range_error_dict = {
    "band_energy": [-0.2, 0.2],
    "density": [-0.001, 0.001],
    "aug": [-0.001, 0.001],
    "energy_per_atom": [-0.1, 0.1],
    "band_gap": [-0.1, 0.1],
    "forces": [-0.1, 0.1],
}
mae_unit_dict = {
    "band_energy": r"$\mathrm{MAE}$",
    "density": r"$\varepsilon_{\tilde{n}}$",
    "aug": r"$\mathrm{MAE}$",
    "energy_per_atom": r"$\mathrm{MAE}$",
    "band_gap": r"$\mathrm{MAE}$",
    "forces": r"$\mathrm{MAE}$",
}
name_to_mae_dict = {
    "band_energy": "mae_band_energies",
    "density": "mae_density",
    "aug": "mae_aug",
    "energy_per_atom": "mae_energy_per_atom",
    "band_gap": "mae_band_gap",
    "forces": "mae_forces",
}


def plot_parity(pre, tar, save_path, mae=None, label=None):
    r2 = 1.0 - np.sum((pre - tar) ** 2) / np.sum((pre - tar.mean()) ** 2)
    err = pre - tar

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.plot(
        [tar.min(), tar.max()],
        [tar.min(), tar.max()],
        color=color_baseline,
        linestyle="--",
        zorder=10,
    )
    ax.scatter(
        tar,
        pre,
        s=12,
        color=color_parity,
        zorder=2,
        alpha=0.4,
        edgecolors="none",
        rasterized=True,
    )

    # set label
    if label is not None:
        ax.set_xlabel(xlabel_dict[label])
        ax.set_ylabel(ylabel_dict[label])
        # axins.set_xlim(range_error_dict[label][0], range_error_dict[label][1])
    else:
        ax.set_xlabel("True")
        ax.set_ylabel("Pred")

    if mae is not None:
        mae_text = mae_unit_dict[label]
        if label == "band_energy":
            mae_text += f" = {mae:.3f} eV"
        elif label == "density":
            mae_text += f" = {mae * 100:.2f} %"
        elif label == "aug":
            mae_text += f" = {mae:.4f}"
        elif label == "energy_per_atom":
            if mae > 0.001:
                mae_text += f" = {mae:.3f} " + r"$\mathrm{eV\cdot atom}^{-1}$"
            else:
                mae_text += f" < 1 " + r"$\mathrm{meV\cdot atom}^{-1}$"
        elif label == "band_gap":
            mae_text += f" = {mae:.3f} eV"
        elif label == "forces":
            mae_text += f" = {mae:.3f} " + r"$(eV\ \AA^{-1})$"
    else:
        mae_text = ""

    if r2 > 0.99999:
        r2_text = rf"$R^2 > 0.99999$"
    else:
        r2_text = rf"$R^2 = {r2:.5f}$"

    ax.text(
        0.05,
        0.95,
        r2_text + "\n" + mae_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    return r2


def plot_band_energy(energy_pre, energy_tar, e0, save_path):
    nb, nk = energy_pre.shape

    xk = np.linspace(0, 1, nk)

    plt.figure(figsize=(6, 3))
    for ik in range(nk):
        plt.plot(
            np.full(nb, xk[ik]),
            energy_pre[:, ik],
            marker="*",
            ms=4,
            linestyle="None",
            color="r",
            alpha=0.3,
        )
    for ik in range(nk):
        plt.plot(
            np.full(nb, xk[ik]),
            energy_tar[:, ik],
            marker="o",
            ms=2,
            linestyle="None",
            color="b",
            alpha=0.3,
        )

    plt.axhline(0.0, ls="--")
    plt.ylim(-5 + e0, 5 + e0)
    plt.xticks([])
    plt.ylabel("Energy (eV)")

    legend_elements = [
        Line2D([0], [0], color="blue", lw=2, label="VASP"),
        Line2D([0], [0], color="red", lw=2, label="E3SR"),
    ]

    plt.legend(handles=legend_elements)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def fermi_window_indices(Ek, n=4):
    """
    find the closest neighbor eigenstates for vbm or fermi surface
    """
    idx = np.searchsorted(Ek, 0.0)

    lo = max(0, idx - n)
    hi = min(len(Ek), idx + n)

    return lo, hi


def fermi_window_mae(Epre, Etar, n=4):
    """
    Epre, Etar: (nb, nk)
    """
    nb, nk = Epre.shape
    err = []

    for ik in range(nk):
        ep = Epre[:, ik]
        et = Etar[:, ik]

        lo_p, hi_p = fermi_window_indices(ep, n)
        lo_t, hi_t = fermi_window_indices(et, n)

        m = min(hi_p - lo_p, hi_t - lo_t)

        err.append(np.abs(ep[lo_p : lo_p + m] - et[lo_t : lo_t + m]))

    return np.mean(np.concatenate(err))


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
    chgcar = Chgcar.from_file(path)
    structure = chgcar.structure
    density = chgcar.data["total"] / structure.lattice.volume
    z = [site.specie.number for site in structure]

    # parse aug. pymatgen has bug reading aug, I need to modify a little bit
    lines = chgcar.data_aug["total"]

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


def stat_structure(dir_work, cfg):
    name = pathlib.Path(dir_work).stem
    try:
        path_nscf = os.path.join(dir_work, "nscf")
        if isinstance(cfg.stat.tar_dir, str) and cfg.stat.tar_dir:
            name = pathlib.Path(dir_work)
            path_scf = os.path.join(cfg.stat.tar_dir, name, "scf")
        else:
            path_scf = os.path.join(dir_work, "scf")

        data_nscf, bs = stat_vasprun(path_nscf, band=False)
        data_scf, bs = stat_vasprun(path_scf, band=False)

        stat_mae = {"name": name}

        # plot band
        plot_band_energy(
            data_nscf["band_energies"],
            data_scf["band_energies"],
            data_scf["e0"],
            os.path.join(dir_work, "band_energy_error.png"),
        )
        band_energy_pre = np.array(data_nscf["band_energies"]).flatten()
        band_energy_tar = np.array(data_scf["band_energies"]).flatten()
        r2_band = plot_parity(
            band_energy_pre,
            band_energy_tar,
            os.path.join(dir_work, "band_energy_parity.png"),
            label="band_energy",
        )

        stat_mae["eigenenergy_2"] = fermi_window_mae(
            data_nscf["band_energies"], data_scf["band_energies"], n=2
        )

        # plot density
        chg_nscf, aug_nscf = parse_chgcar(
            os.path.join(path_nscf, "CHGCAR"), cfg.incar.lmaxmix
        )
        chg_scf, aug_scf = parse_chgcar(
            os.path.join(path_scf, "CHGCAR"), cfg.incar.lmaxmix
        )
        den_pre = chg_nscf.flatten()
        den_tar = chg_scf.flatten()
        ngrid = len(den_tar)
        i_samp = np.random.choice(ngrid, sample_density, replace=False)

        stat_mae["mae_density"] = np.sum(np.abs(den_pre - den_tar)) / np.sum(den_tar)
        stat_mae["mae_aug"] = np.abs(aug_nscf - aug_scf).mean()
        r2_aug = plot_parity(
            aug_nscf, aug_scf, os.path.join(dir_work, "aug_error.png"), label="aug"
        )
        stat_mae["r2_aug"] = r2_aug
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
        stat_mae["mae_" + "forces"] = np.abs(
            data_nscf["forces"] - data_scf["forces"]
        ).mean()

        return {
            "mae": stat_mae,
            "pre": {
                "band_energy": band_energy_pre.tolist(),
                "density": den_pre[i_samp].tolist(),
                "aug": aug_nscf.tolist(),
                "energy_per_atom": [data_nscf["energy_per_atom"]],
                "band_gap": [data_nscf["band_gap"]],
                "forces": data_nscf["forces"].flatten().tolist(),
            },
            "tar": {
                "band_energy": band_energy_tar.tolist(),
                "density": den_tar[i_samp].tolist(),
                "aug": aug_scf.tolist(),
                "energy_per_atom": [data_scf["energy_per_atom"]],
                "band_gap": [data_scf["band_gap"]],
                "forces": data_scf["forces"].flatten().tolist(),
            },
        }

    except Exception as e:
        print("Caught exception:", e)
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    run_save_dir = cfg.band.get("save_dir")
    dirs_work = glob.glob(os.path.join(run_save_dir, "*"))

    pre_dict = {
        "band_energy": [],
        "density": [],
        "aug": [],
        "energy_per_atom": [],
        "band_gap": [],
        "forces": [],
    }
    tar_dict = {
        "band_energy": [],
        "density": [],
        "aug": [],
        "energy_per_atom": [],
        "band_gap": [],
        "forces": [],
    }

    nproc = min(multiprocessing.cpu_count(), cfg.run.nproc)
    with multiprocessing.Pool(processes=nproc) as pool:
        results = pool.starmap(stat_structure, [(d, cfg) for d in dirs_work])

    maes = []
    for result in results:
        if result is not None:
            maes.append(result["mae"])
            for key, value in result["pre"].items():
                pre_dict[key].extend(value)

            for key, value in result["tar"].items():
                tar_dict[key].extend(value)

    with open(os.path.join(cfg.stat.save_dir, "stat_mae.json"), "w") as f:
        json.dump(maes, f, indent=2)

    # calculate mae and plot parity plot

    for key in pre_dict.keys():
        values = [d[name_to_mae_dict[key]] for d in maes]

        plot_parity(
            np.array(pre_dict[key]),
            np.array(tar_dict[key]),
            os.path.join(cfg.stat.save_dir, f"parity_{key}.pdf"),
            mae=sum(values) / len(values),
            label=key,
        )


if __name__ == "__main__":
    main()
