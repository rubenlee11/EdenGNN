r"""Plot ABACUS bands and, when nscf/ exists, the total density of states.

Usage (activate the eqgnn environment first):
    python scripts/abacus/plot_band.py \
        --filelist /path/to/structure_dirs.txt \
        --prefix EDENGNN --ylim -4 4 --dpi 300 --nproc 4

Inputs:
    --filelist Required list of structure calculation directories, one per line.
    --band-dir Band calculation subdirectory (default: band).
    --prefix   OUT.<suffix> suffix for band and NSCF outputs (default: EDENGNN).
    --plot-pdos Also overlay PDOS (default: disabled).
    --ylim     Offsets below VBM and above CBM, in eV (default: -4 4).
    --gap-break Split gaps larger than this value in eV (default: 2).
    --dpi      Image resolution (default: 300).
    --nproc    Parallel plotting processes (default: 1).

Reads band/STRU, band/KPT (Line mode), and
band/OUT.<suffix>/{BANDS_1.dat,running_nscf.log}.
If nscf/ exists, reads nscf/OUT.<suffix>/TDOS (LCAO out_dos=2) and draws
the total DOS beside the bands. Spin channels in TDOS are summed.
With --plot-pdos, read nscf/OUT.<suffix>/PDOS and overlay projections by element
and angular momentum (s/p/d/f), including all atoms and spin channels.
Both panels use VBM (Ef for metals) as zero; gaps above 2 eV are broken.
Saves band.png in each listed directory. Use stat.py for target comparisons.
"""

import argparse
import sys
import xml.etree.ElementTree as ET
import multiprocessing
import os
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ase.io import read


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from plot_band_utils import plot_band_dos


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


def get_band_data(dir, prefix):
    """Read path segments and find band edges; crossing bands use Ef."""
    data = np.loadtxt(os.path.join(dir, f"OUT.{prefix}", "BANDS_1.dat"))
    bands = data[:, 2:]

    ef = get_fermi_energy(os.path.join(dir, f"OUT.{prefix}", "running_nscf.log"))

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

    band_min = bands.min(axis=0)
    band_max = bands.max(axis=0)
    is_metal = np.any((band_min < ef - 1e-3) & (band_max > ef + 1e-3))
    if is_metal:
        e_vbm = e_cbm = ef
    else:
        e_vbm = band_max[band_max <= ef + 1e-3].max()
        e_cbm = band_min[band_min > ef + 1e-3].min()
    return {
        "distances": [x_path[start:min(end, min_len)] for start, end in continuous_segments if start < min_len],
        "energies": [bands[start:min(end, min_len)] for start, end in continuous_segments if start < min_len],
        "tick_positions": x_ticks, "tick_labels": x_labels,
        "e_vbm": e_vbm, "e_cbm": e_cbm, "efermi": ef,
    }


def parse_band(dir, prefix, ax, color="b"):
    """Plot bands relative to Ef on an existing axis; return bands and Ef."""
    band = get_band_data(dir, prefix)
    for distance, energy in zip(band["distances"], band["energies"], strict=True):
        ax.plot(distance, energy - band["efermi"], color=color, linewidth=1.5)
    ax.set_xticks(band["tick_positions"], band["tick_labels"])
    ax.set_ylabel(r"$E - E_f$ (eV)")
    return np.concatenate(band["energies"]) - band["efermi"], band["efermi"]


def read_pdos(path):
    """Read ABACUS XML PDOS, summing atoms, radial functions and spins by species/l."""
    root = ET.parse(path).getroot()
    energy_ev = np.fromstring(root.findtext("energy_values"), sep=" ")
    n_spin = int(root.findtext("nspin"))
    projections = {}
    for orbital in root.findall("orbital"):
        element = orbital.attrib["species"].strip()
        angular_momentum = int(orbital.attrib["l"])
        label = f"{element}-{('spdfgh'[angular_momentum])}"
        density = np.fromstring(orbital.findtext("data"), sep=" ")
        density = density.reshape(len(energy_ev), n_spin).sum(axis=1)
        if label in projections:
            projections[label] += density
        else:
            projections[label] = density
    return energy_ev, projections


def plot_band(dir_work, args):
    """Save a band/DOS plot spanning VBM-4 eV to CBM+4 eV by default."""
    dir_work = pathlib.Path(dir_work)
    band = get_band_data(dir_work / args.band_dir, args.prefix)
    energy_zero = band["e_vbm"]
    dos_energy_ev = dos_density = None
    pdos_energy_ev = None
    projections = None
    path_nscf = dir_work / "nscf"
    if path_nscf.is_dir():
        dir_output = path_nscf / f"OUT.{args.prefix}"
        dos = np.loadtxt(dir_output / "TDOS")
        dos_energy_ev = dos[:, 0] - energy_zero
        dos_density = dos[:, 1:].sum(axis=1)
        if args.plot_pdos:
            pdos_energy_ev, projections = read_pdos(dir_output / "PDOS")
            pdos_energy_ev -= energy_zero
    fig = plot_band_dos(
        band["distances"], [energy - energy_zero for energy in band["energies"]],
        band["tick_positions"], band["tick_labels"], band["e_cbm"] - energy_zero,
        dos_energy_ev, dos_density, args.ylim, args.gap_break,
        pdos_energy_ev=pdos_energy_ev, projections=projections,
    )
    try:
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
    parser.add_argument("--plot-pdos", action="store_true", help="Overlay element/orbital PDOS (default: disabled)")
    parser.add_argument("--ylim", nargs=2, type=float, default=(-4, 4), metavar=("MIN", "MAX"), help="Offsets below VBM and above CBM in eV (default: -4 4)")
    parser.add_argument("--gap-break", type=float, default=2, help="Split gaps larger than this value in eV (default: 2)")
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
