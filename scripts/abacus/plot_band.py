r"""Plot ABACUS bands and, when nscf/ exists, the total density of states.

Usage (activate the eqgnn environment first):
    python scripts/abacus/plot_band.py \
        --filelist /path/to/structure_dirs.txt \
        --prefix EDENGNN --ylim -4 4 --dpi 300 --nproc 4

Inputs:
    --filelist Required list of structure calculation directories, one per line.
    --band-dir Band calculation subdirectory (default: band).
    --prefix   OUT.<suffix> suffix for band and NSCF outputs (default: EDENGNN).
    --ylim     Energy limits relative to the band Fermi energy, in eV (default: -4 4).
    --dpi      Image resolution (default: 300).
    --nproc    Parallel plotting processes (default: 1).

Reads band/STRU, band/KPT (Line mode), and
band/OUT.<suffix>/{BANDS_1.dat,running_nscf.log}.
If nscf/ exists, reads nscf/OUT.<suffix>/TDOS.dat (LCAO out_dos=2) and draws
the total DOS beside the bands. Spin channels in TDOS.dat are summed.
Both panels use the band calculation's Fermi energy as zero.
Saves band.png in each listed directory. Use stat.py for target comparisons.
"""

import argparse
import multiprocessing
import os
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ase.io import read


def get_fermi_energy(log_path):
    ef = 0.0
    with open(log_path, "r") as f:
        for line in f:
            if "EFERMI" in line.upper():
                ef = float(line.split()[-2])
    return ef


def parse_kpt(kpt_path):
    kpt_data = []
    with open(kpt_path, "r") as f:
        lines = f.readlines()

    for line in lines[3:]:
        line = line.strip()
        if not line:
            continue

        parts = line.split("//")
        coords_npts = parts[0].split()

        if len(coords_npts) >= 4:
            k = np.array(
                [float(coords_npts[0]), float(coords_npts[1]), float(coords_npts[2])]
            )
            npts = int(coords_npts[3])
            label = parts[1].strip() if len(parts) > 1 else ""
            kpt_data.append((k, npts, label))

    return kpt_data


def parse_band(dir, prefix, ax, color="b"):
    data = np.loadtxt(os.path.join(dir, f"OUT.{prefix}", "BANDS_1.dat"))
    bands = data[:, 2:]

    ef = get_fermi_energy(os.path.join(dir, f"OUT.{prefix}", "running_nscf.log"))
    bands -= ef

    atoms = read(os.path.join(dir, "STRU"), format="abacus")
    recip_cell = atoms.cell.reciprocal() * 2 * np.pi

    kpt_data = parse_kpt(os.path.join(dir, "KPT"))

    x_path = []
    x_ticks = []
    x_labels = []
    current_x = 0.0
    continuous_segments = []
    start_idx = 0
    current_idx = 0

    for i in range(len(kpt_data) - 1):
        k1, npts, label1 = kpt_data[i]
        k2, _, label2 = kpt_data[i + 1]

        if label1.upper() == "GAMMA":
            label1 = r"$\Gamma$"

        if label1:
            if len(x_ticks) == 0 or abs(x_ticks[-1] - current_x) > 1e-5:
                x_ticks.append(current_x)
                x_labels.append(label1)
            else:
                existing_labels = x_labels[-1].split("|")
                if label1 not in existing_labels:
                    x_labels[-1] += f"|{label1}"

        if npts == 0:
            if current_idx > start_idx:
                continuous_segments.append((start_idx, current_idx))
            start_idx = current_idx
            continue

        k1_cart = np.dot(k1, recip_cell)
        k2_cart = np.dot(k2, recip_cell)
        dist = np.linalg.norm(k2_cart - k1_cart)

        segment_x = np.linspace(current_x, current_x + dist, npts, endpoint=False)
        x_path.extend(segment_x)

        current_x += dist
        current_idx += npts

    # Handle the very last k-point
    last_k, last_npts, last_label = kpt_data[-1]
    if last_label.upper() == "GAMMA":
        last_label = r"$\Gamma$"

    if last_npts == 1 and last_label:
        x_path.append(current_x)
        current_idx += 1
        if len(x_ticks) == 0 or abs(x_ticks[-1] - current_x) > 1e-5:
            x_ticks.append(current_x)
            x_labels.append(last_label)
        else:
            existing_labels = x_labels[-1].split("|")
            if last_label not in existing_labels:
                x_labels[-1] += f"|{last_label}"

    if current_idx > start_idx:
        continuous_segments.append((start_idx, current_idx))

    x_path = np.array(x_path)

    min_len = min(len(x_path), len(bands))
    x_path = x_path[:min_len]
    bands = bands[:min_len]

    # 6. Plotting
    num_bands = bands.shape[1]

    for start, end in continuous_segments:
        if start >= min_len:
            break
        end = min(end, min_len)

        for b in range(num_bands):
            ax.plot(x_path[start:end], bands[start:end, b], color=color, linewidth=1.5)

    ax.set_ylabel(r"$E - E_f$ (eV)", fontsize=14)
    ax.set_xlim(x_path[0], x_path[-1])
    ax.set_ylim(-4, 4)

    for xt in x_ticks:
        ax.axvline(x=xt, color="black", linewidth=0.8, linestyle="--")

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=14)
    ax.axhline(y=0, color="gray", linewidth=1.0, linestyle=":")

    return bands, ef


def plot_band(dir_work, args):
    """Save bands and optional total DOS with a shared Fermi-energy reference."""
    dir_work = pathlib.Path(dir_work)
    path_nscf = dir_work / "nscf"
    has_dos = path_nscf.is_dir()
    if has_dos:
        dos = np.loadtxt(path_nscf / f"OUT.{args.prefix}" / "TDOS.dat")
        fig, (ax, ax_dos) = plt.subplots(
            1, 2, figsize=(6, 4), sharey=True,
            gridspec_kw={"width_ratios": [3, 1], "wspace": 0.05},
        )
    else:
        fig, ax = plt.subplots(figsize=(4, 4))
    try:
        _, ef = parse_band(dir_work / args.band_dir, args.prefix, ax, color="b")
        ax.set_ylim(*args.ylim)
        if has_dos:
            ax_dos.plot(dos[:, 1:].sum(axis=1), dos[:, 0] - ef, color="b", linewidth=1)
            ax_dos.axhline(0, color="gray", linewidth=0.5)
            ax_dos.set_xlabel("DOS (states/eV)")
            ax_dos.set_xlim(left=0)
            ax_dos.tick_params(axis="y", left=False)
            fig.subplots_adjust(left=0.14, right=0.97, bottom=0.18, top=0.97)
        else:
            fig.tight_layout()
        fig.savefig(dir_work / "band.png", dpi=args.dpi)
    finally:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--filelist", required=True, type=pathlib.Path, help="List of structure calculation directories")
    parser.add_argument("--band-dir", default="band", help="Band calculation subdirectory (default: band)")
    parser.add_argument("--prefix", default="EDENGNN", help="Calculation suffix (default: EDENGNN)")
    parser.add_argument("--ylim", nargs=2, type=float, default=(-4, 4), metavar=("MIN", "MAX"), help="Energy limits in eV relative to Ef (default: -4 4)")
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
