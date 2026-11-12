r"""Plot VASP bands and, when nscf/ exists, the total density of states.

Usage (activate the eqgnn environment first):
    python scripts/vasp/plot_band.py \
        --filelist /path/to/structure_dirs.txt --ylim -4 4 --dpi 300 --nproc 4

Inputs:
    --filelist Required list of structure calculation directories, one per line.
    --band-dir Band calculation subdirectory (default: band).
    --ylim     Energy limits in eV (default: -4 4).
    --dpi      Image resolution (default: 300).
    --nproc    Parallel plotting processes (default: 1).

Reads band/vasprun.xml and band/KPOINTS. Supports explicit JSON-header KPOINTS
from vasp_sad.py and standard VASP Line-mode KPOINTS. If nscf/ exists, reads
nscf/vasprun.xml and draws the total DOS beside the bands, sharing the energy
axis. Both use the band calculation's VBM (Fermi energy for metals) as zero.
Saves band.png in each listed directory. Use stat.py for target comparisons.
"""

import argparse
import json
import multiprocessing
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pymatgen.electronic_structure.core import Spin
from pymatgen.electronic_structure.plotter import BSPlotter
from pymatgen.io.vasp.outputs import Vasprun


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
        energies = [np.array(item).T for item in plot_data["energy"]["1"]]
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



def plot_band(dir_work, args):
    """Save bands with an optional total DOS panel from the NSCF calculation."""
    dir_work = Path(dir_work)
    band = get_band_data(dir_work / args.band_dir)
    energy_zero = band["e_vbm"]
    path_nscf = dir_work / "nscf"
    has_dos = path_nscf.is_dir()
    dos = None
    if has_dos:
        dos = Vasprun(
            path_nscf / "vasprun.xml", parse_dos=True, parse_eigen=False,
            parse_projected_eigen=False, parse_potcar_file=False,
        ).tdos
        fig, (ax, ax_dos) = plt.subplots(
            1, 2, figsize=(6, 4), sharey=True,
            gridspec_kw={"width_ratios": [3, 1], "wspace": 0.05},
        )
    else:
        fig, ax = plt.subplots(figsize=(4, 4))
    try:
        for distance, energy in zip(band["distances"], band["energies"], strict=True):
            ax.plot(distance, energy - energy_zero, color="blue", linewidth=1)
        ax.set_xticks(band["tick_positions"])
        ax.set_xticklabels(band["tick_labels"])
        for position in band["tick_positions"]:
            ax.axvline(position, color="grey", linewidth=0.5)
        ax.axhline(0, color="grey", linewidth=0.5)
        ax.set_ylabel("Energy (eV)")
        ax.set_xlim(0, band["distances"][-1][-1])
        ax.set_ylim(*args.ylim)
        if has_dos:
            ax_dos.plot(dos.get_densities(), dos.energies - energy_zero, color="blue", linewidth=1)
            ax_dos.axhline(0, color="grey", linewidth=0.5)
            ax_dos.set_xlabel("DOS (states/eV)")
            ax_dos.set_xlim(left=0)
            ax_dos.tick_params(axis="y", left=False)
            fig.subplots_adjust(left=0.12, right=0.97, bottom=0.18, top=0.97)
        else:
            fig.tight_layout()
        fig.savefig(dir_work / "band.png", dpi=args.dpi)
    finally:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--filelist", required=True, type=Path, help="List of calculation directories")
    parser.add_argument("--band-dir", default="band", help="Band calculation subdirectory (default: band)")
    parser.add_argument("--ylim", nargs=2, type=float, default=(-4, 4), metavar=("MIN", "MAX"), help="Energy limits in eV (default: -4 4)")
    parser.add_argument("--dpi", type=int, default=300, help="Image resolution (default: 300)")
    parser.add_argument("--nproc", type=int, default=1, help="Parallel plotting processes (default: 1)")
    args = parser.parse_args()
    with args.filelist.open() as f:
        paths = [line.strip() for line in f if line.strip()]
    if args.nproc == 1:
        for path in paths:
            plot_band(path, args)
    else:
        with multiprocessing.Pool(min(multiprocessing.cpu_count(), args.nproc)) as pool:
            pool.starmap(plot_band, [(path, args) for path in paths])


if __name__ == "__main__":
    main()
