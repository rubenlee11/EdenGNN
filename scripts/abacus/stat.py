r"""
Compare predicted and reference ABACUS bands and save band MAE statistics.

Usage:
    python scripts/abacus/stat.py \
        --filelist /path/to/predicted_calculations.txt \
        --dir-tar /path/to/reference_calculations \
        --mae-output /path/to/stat.json \
        --prefix EDENGNN --prefix-tar aiida

Inputs:
    --filelist   Required list of predicted calculation directories, one per line.
    --dir-tar    Required reference root; references use each predicted directory's name.
    --mae-output Required output JSON path.
    --prefix     Predicted OUT.<suffix> suffix (default: EDENGNN).
    --prefix-tar Reference OUT.<suffix> suffix (default: aiida).
    --use-shift / --no-use-shift
                 Align predicted and reference mean band energies (default: enabled).

Each calculation needs KPT and OUT.<suffix>/{BANDS_1.dat,running_nscf.log,STRU.cif}.
Saves band.png in each predicted calculation directory and summary MAE in eV.
"""

import argparse
import json
import os
import pathlib

import numpy as np
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


def parse_band(dir, prefix, ax, color="b", line=True, shift_to_mean_ev=None):
    data = np.loadtxt(os.path.join(dir, f"OUT.{prefix}", "BANDS_1.dat"))
    bands = data[:, 2:]

    ef = get_fermi_energy(os.path.join(dir, f"OUT.{prefix}", "running_nscf.log"))

    atoms = read(os.path.join(dir, f"OUT.{prefix}", "STRU.cif"))
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

    if shift_to_mean_ev is not None:
        bands = bands + shift_to_mean_ev - np.mean(bands)

    # 6. Plotting
    num_bands = bands.shape[1]

    for start, end in continuous_segments:
        if start >= min_len:
            break
        end = min(end, min_len)

        for b in range(num_bands):
            if line:
                ax.plot(x_path[start:end], bands[start:end, b], color=color, linewidth=1.5)
            else:
                ax.scatter(x_path[start:end], bands[start:end, b], color=color, s = 0.2)

    ax.set_ylabel(r"$E - E_f$ (eV)", fontsize=14)
    ax.set_xlim(x_path[0], x_path[-1])

    for xt in x_ticks:
        ax.axvline(x=xt, color="black", linewidth=0.8, linestyle="--")

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=14)
    ax.axhline(y=0, color="gray", linewidth=1.0, linestyle=":")

    return bands, ef


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--filelist", required=True, type=pathlib.Path, help="List of predicted calculation directories")
    parser.add_argument("--dir-tar", required=True, type=pathlib.Path, help="Reference calculation root")
    parser.add_argument("--mae-output", required=True, type=pathlib.Path, help="Output JSON path")
    parser.add_argument("--prefix", default="EDENGNN", help="Prediction suffix (default: EDENGNN)")
    parser.add_argument("--prefix-tar", default="aiida", help="Reference suffix (default: aiida)")
    parser.add_argument(
        "--use-shift", action=argparse.BooleanOptionalAction, default=True,
        help="Align mean band energies (default: enabled)",
    )
    args = parser.parse_args()

    with args.filelist.open() as f:
        paths = [line.strip() for line in f if line.strip()]

    structure_metrics = []
    for structure_index, path in enumerate(paths):
        try:
            fig, ax = plt.subplots(figsize=(4, 4))
            plt.title("Electronic Band Structure", fontsize=16)
            plt.tight_layout()
            print(path)
            name = pathlib.Path(path).stem
            bands_tar, ef_tar = parse_band(
                os.path.join(args.dir_tar, name), args.prefix_tar, ax, color="r", line=False
            )
            shift_to_mean_ev = np.mean(bands_tar) if args.use_shift else None
            bands_pre, ef_pre = parse_band(
                path,
                args.prefix,
                ax,
                color="b",
                shift_to_mean_ev=shift_to_mean_ev,
            )

            band_mae_ev = np.mean(np.abs(bands_pre - bands_tar))
            print(f"Structure {structure_index}: band MAE = {band_mae_ev:.6f} eV")
            structure_metrics.append(
                {
                    "structure_index": structure_index,
                    "path": path,
                    "band_mae_ev": float(band_mae_ev),
                }
            )
            ax.set_ylim(-4 + ef_tar, 4 + ef_tar)

            plt.savefig(os.path.join(path, "band.png"), dpi=300)
            plt.close()
        except Exception as error:
            plt.close()
            print(f"Error processing {path}: {error}")

    if not structure_metrics:
        raise ValueError("No structures were processed; band MAE cannot be calculated.")

    mean_band_mae_ev = float(
        np.mean([metrics["band_mae_ev"] for metrics in structure_metrics])
    )
    mae_output = {
        "unit": "eV",
        "use_shift": args.use_shift,
        "mean_band_mae_ev": mean_band_mae_ev,
        "num_structures": len(structure_metrics),
        "structures": structure_metrics,
    }
    with args.mae_output.open("w") as output_file:
        json.dump(mae_output, output_file, indent=2)

    shift_label = "shifted" if args.use_shift else "unshifted"
    print(f"Mean {shift_label} band MAE = {mean_band_mae_ev:.6f} eV")
    print(f"Band MAE metrics saved to {args.mae_output}")


if __name__ == "__main__":
    main()
