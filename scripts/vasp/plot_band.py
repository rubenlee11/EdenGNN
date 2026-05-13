import glob
import json
import multiprocessing
import os
import pathlib

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from ase.data.colors import jmol_colors
from ase.io import read
from ase.visualize.plot import plot_atoms
from matplotlib.lines import Line2D
from pymatgen.electronic_structure.core import Spin
from pymatgen.electronic_structure.plotter import BSPlotter
from pymatgen.io.vasp.outputs import Vasprun

color_tar = "red"
color_pre = "blue"
color_parity = "#AAA486"
color_baseline = "black"

dirs = glob.glob("/root/dataset/gnome/compare_edengnn_hamgnn/vasp_run_predict/*")
path_save = "/root/dataset/gnome/compare_edengnn_hamgnn/stat_band_edengnn.json"


def get_band_index_window(evbm, eigen, width_lower=4.0):
    threshold = evbm - width_lower
    fully_above = np.all(eigen > threshold, axis=1)
    indices = np.where(fully_above)[0]
    if len(indices) == 0:
        return 0, eigen.shape[0]
    return int(indices[0]), eigen.shape[0]


def _append_tick(tick_positions, tick_labels, position, label):
    if label is None:
        return

    label = label.replace("GAMMA", r"$\Gamma$")
    if tick_positions and np.isclose(tick_positions[-1], position):
        prev = tick_labels[-1]
        if prev != label:
            left = prev.split("|")
            right = label.split("|")
            merged = left[:]
            for item in right:
                if item not in merged:
                    merged.append(item)
            tick_labels[-1] = "|".join(merged)
        return

    tick_positions.append(position)
    tick_labels.append(label)


def _parse_explicit_kpoints(kpoints_path, structure):
    with open(kpoints_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) < 4:
        raise ValueError(f"invalid KPOINTS file: {kpoints_path}")

    try:
        header = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise ValueError("KPOINTS is not explicit JSON-header format") from exc

    if not isinstance(header, list) or not header:
        raise ValueError(f"invalid explicit KPOINTS header: {kpoints_path}")

    nk_total = int(lines[1])
    coord_mode = lines[2].lower()
    if coord_mode not in {"reciprocal", "cartesian"}:
        raise ValueError(f"unsupported KPOINTS coordinate mode: {lines[2]}")

    coords = []
    for line in lines[3 : 3 + nk_total]:
        toks = line.split()
        if len(toks) < 3:
            raise ValueError(f"invalid KPOINTS coordinate line: {line}")
        coords.append([float(toks[0]), float(toks[1]), float(toks[2])])
    coords = np.array(coords, dtype=float)

    if len(coords) != nk_total:
        raise ValueError(
            f"KPOINTS count mismatch in {kpoints_path}: expected {nk_total}, got {len(coords)}"
        )

    if coord_mode == "reciprocal":
        coords_cart = coords @ structure.lattice.reciprocal_lattice.matrix
    else:
        coords_cart = coords

    distances = []
    tick_positions = []
    tick_labels = []

    cursor = 0
    cumulative = 0.0
    for start_label, end_label, nk in header:
        nk = int(nk)
        seg_frac = coords[cursor : cursor + nk]
        seg_cart = coords_cart[cursor : cursor + nk]
        if len(seg_frac) != nk:
            raise ValueError(
                f"KPOINTS segment mismatch in {kpoints_path}: expected {nk} points"
            )

        seg_dist = np.zeros(nk, dtype=float)
        for idx in range(1, nk):
            seg_dist[idx] = seg_dist[idx - 1] + np.linalg.norm(
                seg_cart[idx] - seg_cart[idx - 1]
            )
        seg_dist += cumulative

        distances.append(seg_dist)
        _append_tick(tick_positions, tick_labels, seg_dist[0], start_label)
        _append_tick(tick_positions, tick_labels, seg_dist[-1], end_label)

        cumulative = float(seg_dist[-1])
        cursor += nk

    if cursor != nk_total:
        raise ValueError(
            f"KPOINTS segment total mismatch in {kpoints_path}: used {cursor}, expected {nk_total}"
        )

    return {
        "distances": distances,
        "tick_positions": tick_positions,
        "tick_labels": tick_labels,
        "segment_lengths": [int(item[2]) for item in header],
    }


def _reshape_explicit_energies(bands, segment_lengths):
    energies = []
    cursor = 0
    for nk in segment_lengths:
        energies.append(bands[:, cursor : cursor + nk].T)
        cursor += nk
    if cursor != bands.shape[1]:
        raise ValueError(
            f"band/kpoint mismatch: used {cursor} k-points, but bands have {bands.shape[1]}"
        )
    return energies


def get_band_data(workdir):
    vr = Vasprun(
        os.path.join(workdir, "vasprun.xml"),
        parse_dos=True,
        parse_eigen=True,
        parse_potcar_file=False,
    )
    structure = vr.structures[-1]
    bs_uniform = vr.get_band_structure(line_mode=False)

    if bs_uniform.is_metal():
        e_vbm = bs_uniform.efermi
        e_cbm = bs_uniform.efermi
    else:
        e_vbm = bs_uniform.get_vbm()["energy"]
        e_cbm = bs_uniform.get_cbm()["energy"]

    bands = bs_uniform.bands[Spin.up]

    kpoints_path = os.path.join(workdir, "KPOINTS")
    try:
        explicit = _parse_explicit_kpoints(kpoints_path, structure)
        distances = explicit["distances"]
        energies = _reshape_explicit_energies(bands, explicit["segment_lengths"])
        tick_positions = explicit["tick_positions"]
        tick_labels = explicit["tick_labels"]
    except ValueError:
        bs_line = vr.get_band_structure(line_mode=True)
        plot_data = BSPlotter(bs_line).bs_plot_data(zero_to_efermi=False)
        distances = plot_data["distances"]
        energies = [np.array(item) for item in plot_data["energy"]["1"]]
        tick_info = BSPlotter(bs_line).get_ticks()
        tick_positions = []
        tick_labels = []
        for position, label in zip(
            tick_info["distance"], tick_info["label"], strict=True
        ):
            _append_tick(tick_positions, tick_labels, position, label)

    return {
        "bands": bands,
        "distances": distances,
        "energies": energies,
        "e_vbm": e_vbm,
        "e_cbm": e_cbm,
        "tick_positions": tick_positions,
        "tick_labels": tick_labels,
    }


def get_energy_per_atom(workdir):
    vr = Vasprun(
        os.path.join(workdir, "vasprun.xml"),
        parse_dos=False,
        parse_eigen=False,
        parse_potcar_file=False,
    )
    return float(vr.final_energy) / len(vr.structures[-1])


def plot_parity_band(ax, pre, tar):
    pre = np.asarray(pre).ravel()
    tar = np.asarray(tar).ravel()
    r2 = 1.0 - np.sum((pre - tar) ** 2) / np.sum((pre - tar.mean()) ** 2)

    ax.plot(
        [tar.min(), tar.max()],
        [tar.min(), tar.max()],
        color=color_baseline,
        linestyle="--",
        lw=1.0,
        zorder=10,
    )
    ax.scatter(
        tar,
        pre,
        s=12,
        color=color_parity,
        zorder=2,
        alpha=0.2,
        edgecolors="none",
    )

    ax.set_xlabel(r"$\varepsilon_{nk}$ $(\mathrm{eV})$")
    ax.set_ylabel(r"$\hat{\varepsilon}_{nk}$ $(\mathrm{eV})$")
    ax.tick_params(direction="in")
    ax.text(
        0.05,
        0.95,
        rf"$R^2 = {r2:.5f}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
    )


def plot_atoms_ase(ax, atoms):
    plot_atoms(atoms, ax, radii=0.5, rotation=("45x,45y,0z"))
    ax.set_axis_off()

    unique_elements = {}
    for atom in atoms:
        if atom.symbol not in unique_elements:
            unique_elements[atom.symbol] = atom.number
    sorted_symbols = sorted(unique_elements.keys())

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=sym,
            markerfacecolor=jmol_colors[unique_elements[sym]],
            markersize=10,
            markeredgecolor="black",
            markeredgewidth=0.5,
        )
        for sym in sorted_symbols
    ]

    return legend_elements


def plot_band(dir_work):
    try:
        dir_pre = os.path.join(dir_work, "band")
        dir_tar = os.path.join(dir_work, "band_scf")
        dir_energy_pre = os.path.join(dir_work, "nscf")
        dir_energy_tar = os.path.join(dir_work, "scf")

        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.5)

        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])

        atoms = read(os.path.join(dir_pre, "POSCAR"))
        legend_elements = plot_atoms_ase(ax2, atoms)
        fig.legend(
            handles=legend_elements,
            loc="lower center",
            bbox_to_anchor=(0.3, -0.02),
            ncol=len(legend_elements),
            frameon=False,
            handletextpad=0.2,
            columnspacing=1.0,
            handlelength=1.5,
        )

        band_pre = get_band_data(dir_pre)
        band_tar = get_band_data(dir_tar)

        e_ref = band_tar["e_vbm"]
        e_gap_pre = band_pre["e_cbm"] - band_pre["e_vbm"]
        e_gap_tar = band_tar["e_cbm"] - band_tar["e_vbm"]

        eg_pre = [segment - e_ref for segment in band_pre["energies"]]
        eg_tar = [segment - e_ref for segment in band_tar["energies"]]
        # eg_pre = band_pre["energies"]
        # eg_tar = band_tar["energies"]

        index_lower, index_upper = get_band_index_window(
            band_tar["e_vbm"], band_tar["bands"], width_lower=4.0
        )
        mae_band = np.abs(
            band_pre["bands"][index_lower:index_upper, :]
            - band_tar["bands"][index_lower:index_upper, :]
        ).mean()

        ax1.set_xticks(band_tar["tick_positions"])
        ax1.set_xticklabels(band_tar["tick_labels"])
        for xpos in band_tar["tick_positions"]:
            ax1.axvline(xpos, color="grey", ls="-", lw=0.5)

        for i, (dist, ene) in enumerate(
            zip(band_pre["distances"], eg_pre, strict=True)
        ):
            lines = ax1.plot(
                dist,
                ene,
                c=color_pre,
                zorder=10,
                alpha=1.0,
                linewidth=1,
                rasterized=True,
            )
            if i == 0:
                lines[0].set_label("EdenGNN")

        for i, (dist, ene) in enumerate(
            zip(band_tar["distances"], eg_tar, strict=True)
        ):
            lines = ax1.plot(
                dist,
                ene,
                c=color_tar,
                zorder=1,
                alpha=1.0,
                linewidth=1.0,
                rasterized=True,
                linestyle="dotted",
            )
            if i == 0:
                lines[0].set_label("DFT")

        ymax = max(
            float(np.max(np.concatenate(eg_pre))),
            float(np.max(np.concatenate(eg_tar))),
            4.0,
        )
        ax1.axhline(0.0, lw=0.5, ls="-", color="grey")
        ax1.set_ylabel("Energy (eV)")
        ax1.set_xlim(0, band_tar["distances"][-1][-1])
        ax1.set_ylim(-4.0, 4.0)
        ax1.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=3,
            frameon=False,
            fontsize=10,
        )

        plot_parity_band(ax3, np.concatenate(eg_pre), np.concatenate(eg_tar))

        energy_pre = get_energy_per_atom(dir_energy_pre)
        energy_tar = get_energy_per_atom(dir_energy_tar)

        fig.text(
            0.3,
            -0.05,
            r"$\Delta E$"
            + f"= {abs(energy_tar - energy_pre) * 1000:.1f} "
            + r"$\mathrm{meV\cdot atom}^{-1}$",
            ha="center",
            va="bottom",
            fontsize=10,
        )

        fig.text(
            0.7,
            -0.05,
            r"$\mathrm{MAE_{band}}$" + f" = {mae_band:.3f} eV",
            ha="center",
            va="bottom",
            fontsize=10,
        )

        plt.tight_layout()
        plt.savefig(os.path.join(dir_work, "band.png"), dpi=600, bbox_inches="tight")
        # plt.show()
        plt.close()

        name = pathlib.Path(dir_work).stem
        return {name: {"mae_band": mae_band, "mae_gap": abs(e_gap_pre - e_gap_tar)}}
    except Exception:
        print(f"error with {dir_work}")
        return None


nproc = multiprocessing.cpu_count()
with multiprocessing.Pool(processes=nproc) as pool:
    results = pool.map(plot_band, dirs)

mae_list = [res for res in results if res is not None]

with open(path_save, "w") as f:
    json.dump(mae_list, f, indent=4)
