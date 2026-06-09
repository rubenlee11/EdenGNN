import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np


color_tar = "red"
color_pre = "blue"
color_parity = "#AAA486"
color_baseline = "black"


def _format_label(label):
    label = label.strip().strip("'\"").lstrip("\\")
    if label.lower() in {"gamma", "gam"}:
        return r"$\Gamma$"
    return label


def _merge_tick(tick_positions, tick_labels, position, label):
    if tick_positions and np.isclose(tick_positions[-1], position, atol=1e-8):
        labels = tick_labels[-1].split("|")
        if label not in labels:
            labels.append(label)
        tick_labels[-1] = "|".join(labels)
        return

    tick_positions.append(float(position))
    tick_labels.append(label)


def read_bands(path):
    lines = Path(path).read_text().splitlines()
    fermi = float(lines[0].split()[0])
    nband, nspin, nk = [int(x) for x in lines[3].split()[:3]]

    values_per_k = 1 + nband * nspin
    cursor = 4
    k_values = []
    energies = []

    for _ in range(nk):
        values = []
        while len(values) < values_per_k:
            values.extend(float(x) for x in lines[cursor].split())
            cursor += 1

        k_values.append(values[0])
        row = np.array(values[1:], dtype=float)
        if nspin > 1:
            row = row.reshape(nspin, nband)[0]
        energies.append(row)

    energies = np.asarray(energies, dtype=float)
    if energies.shape != (nk, nband):
        energies = energies.reshape(nk, nband)
    energies -= fermi

    nticks = int(lines[cursor].split()[0])
    cursor += 1
    tick_positions = []
    tick_labels = []
    tick_re = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)\s+(.+?)\s*$")
    for line in lines[cursor : cursor + nticks]:
        match = tick_re.match(line)
        if match is None:
            continue
        tick_positions.append(float(match.group(1)))
        tick_labels.append(_format_label(match.group(2)))

    return {
        "fermi": fermi,
        "k": np.asarray(k_values, dtype=float),
        "energies": energies,
        "tick_positions": tick_positions,
        "tick_labels": tick_labels,
    }


def read_bandline_counts(fdf_path):
    fdf_path = Path(fdf_path)
    if not fdf_path.exists():
        return None

    entries = []
    in_block = False
    for raw in fdf_path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        lower = line.lower()
        if not line:
            continue
        if lower.startswith("%block bandlines"):
            in_block = True
            continue
        if lower.startswith("%endblock bandlines"):
            break
        if not in_block:
            continue

        parts = line.split()
        if len(parts) >= 5:
            entries.append((int(parts[0]), _format_label(parts[-1])))

    return entries or None


def compress_discontinuous_kpath(band, fdf_path):
    entries = read_bandline_counts(fdf_path)
    k = np.asarray(band["k"], dtype=float)
    tick_positions = list(band["tick_positions"])
    tick_labels = list(band["tick_labels"])

    if not entries or len(entries) != len(tick_positions):
        return {
            "k": k,
            "segments": [slice(0, len(k))],
            "tick_positions": tick_positions,
            "tick_labels": tick_labels,
        }

    compressed_k = k.copy()
    compressed_ticks = np.asarray(tick_positions, dtype=float).copy()
    segments = []
    start = 0
    shift = 0.0

    for i, (npoints, _label) in enumerate(entries):
        if i == 0 or npoints != 1:
            compressed_ticks[i] -= shift
            continue

        prev_tick = tick_positions[i - 1]
        jump_tick = tick_positions[i]
        end = int(np.searchsorted(k, prev_tick, side="right"))
        if start < end:
            segments.append(slice(start, end))

        gap = jump_tick - prev_tick
        shift += gap
        compressed_k[k >= jump_tick - 1e-10] -= gap
        compressed_ticks[i] = tick_positions[i] - shift
        start = int(np.searchsorted(k, jump_tick, side="left"))

    if start < len(k):
        segments.append(slice(start, len(k)))

    merged_positions = []
    merged_labels = []
    for position, label in zip(compressed_ticks, tick_labels, strict=True):
        _merge_tick(merged_positions, merged_labels, position, label)

    return {
        "k": compressed_k,
        "segments": segments,
        "tick_positions": merged_positions,
        "tick_labels": merged_labels,
    }


def infer_nocc_from_fdf(fdf_path):
    fdf_path = Path(fdf_path)
    if not fdf_path.exists():
        return None

    valence_by_z = {
        1: 1,
        3: 1,
        4: 2,
        5: 3,
        6: 4,
        7: 5,
        8: 6,
        9: 7,
        11: 1,
        12: 2,
        13: 3,
        14: 4,
        15: 5,
        16: 6,
        17: 7,
        31: 3,
        32: 4,
        33: 5,
        34: 6,
        35: 7,
        49: 3,
        50: 4,
        51: 5,
        52: 6,
        53: 7,
        81: 3,
        82: 4,
        83: 5,
    }
    species_z = {}
    species_counts = {}
    in_species = False
    in_coords = False

    for raw in fdf_path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        lower = line.lower()
        if not line:
            continue
        if lower.startswith("%block chemicalspecieslabel"):
            in_species = True
            continue
        if lower.startswith("%endblock chemicalspecieslabel"):
            in_species = False
            continue
        if lower.startswith("%block atomiccoordinatesandatomicspecies"):
            in_coords = True
            continue
        if lower.startswith("%endblock atomiccoordinatesandatomicspecies"):
            in_coords = False
            continue

        if in_species:
            parts = line.split()
            if len(parts) >= 2:
                species_z[int(parts[0])] = int(parts[1])
        elif in_coords:
            parts = line.split()
            if len(parts) >= 4:
                species = int(parts[3])
                species_counts[species] = species_counts.get(species, 0) + 1

    total_valence = 0
    for species, count in species_counts.items():
        z = species_z.get(species)
        valence = valence_by_z.get(z)
        if valence is None:
            return None
        total_valence += count * valence

    if total_valence <= 0 or total_valence % 2:
        return None
    return total_valence // 2


def band_edges(energies, nocc=None):
    energies = np.asarray(energies, dtype=float)
    if nocc is not None:
        if nocc <= 0 or nocc >= energies.shape[1]:
            raise ValueError(f"invalid nocc={nocc} for {energies.shape[1]} bands")
        return float(energies[:, nocc - 1].max()), float(energies[:, nocc].min())

    flat = energies.ravel()
    below = flat[flat <= 0.0]
    above = flat[flat > 0.0]
    if below.size and above.size:
        return float(below.max()), float(above.min())

    fallback_nocc = energies.shape[1] // 2
    return float(energies[:, fallback_nocc - 1].max()), float(energies[:, fallback_nocc].min())


def get_band_index_window(evbm, eigen, width_lower=4.0):
    threshold = evbm - width_lower
    fully_above = np.all(eigen.T > threshold, axis=1)
    indices = np.where(fully_above)[0]
    if len(indices) == 0:
        return 0, eigen.shape[1]
    return int(indices[0]), eigen.shape[1]


def plot_parity_band(ax, pre, tar):
    pre = np.asarray(pre).ravel()
    tar = np.asarray(tar).ravel()
    denom = np.sum((pre - tar.mean()) ** 2)
    r2 = np.nan if denom == 0 else 1.0 - np.sum((pre - tar) ** 2) / denom

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


def plot_bands(args):
    band_pre = read_bands(args.pre)
    band_tar = read_bands(args.tar)

    if band_pre["energies"].shape != band_tar["energies"].shape:
        raise ValueError(
            f"band shape mismatch: {args.pre} {band_pre['energies'].shape}, "
            f"{args.tar} {band_tar['energies'].shape}"
        )

    nocc = args.nocc if args.nocc is not None else infer_nocc_from_fdf(args.fdf)
    evbm_pre, ecbm_pre = band_edges(band_pre["energies"], nocc)
    evbm_tar, ecbm_tar = band_edges(band_tar["energies"], nocc)
    e_ref = evbm_tar

    eg_pre = band_pre["energies"] - e_ref
    eg_tar = band_tar["energies"] - e_ref
    kpath = compress_discontinuous_kpath(band_tar, args.fdf)

    if len(band_pre["k"]) != len(kpath["k"]):
        raise ValueError("pre/tar k-point count mismatch after k-path compression")

    index_lower, index_upper = get_band_index_window(evbm_tar, band_tar["energies"])
    mae_band = np.abs(
        eg_pre[:, index_lower:index_upper]
        - eg_tar[:, index_lower:index_upper]
    ).mean()

    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.5)

    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax2.axis("off")

    ax1.set_xticks(kpath["tick_positions"])
    ax1.set_xticklabels(kpath["tick_labels"])
    for xpos in kpath["tick_positions"]:
        ax1.axvline(xpos, color="grey", ls="-", lw=0.5)

    for i in range(eg_pre.shape[1]):
        for j, segment in enumerate(kpath["segments"]):
            lines = ax1.plot(
                kpath["k"][segment],
                eg_pre[segment, i],
                c=color_pre,
                zorder=10,
                alpha=1.0,
                linewidth=1,
                rasterized=True,
            )
            if i == 0 and j == 0:
                lines[0].set_label("EdenGNN")

    for i in range(eg_tar.shape[1]):
        for j, segment in enumerate(kpath["segments"]):
            lines = ax1.plot(
                kpath["k"][segment],
                eg_tar[segment, i],
                c=color_tar,
                zorder=1,
                alpha=1.0,
                linewidth=1.0,
                rasterized=True,
                linestyle="dotted",
            )
            if i == 0 and j == 0:
                lines[0].set_label("DFT")

    ax1.axhline(0.0, lw=0.5, ls="-", color="grey")
    ax1.set_ylabel("Energy (eV)")
    ax1.set_xlim(float(kpath["k"][0]), float(kpath["k"][-1]))
    ax1.set_ylim(args.ylim)
    ax1.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        frameon=False,
        fontsize=10,
    )

    plot_parity_band(ax3, eg_pre[:, index_lower:index_upper], eg_tar[:, index_lower:index_upper])

    ax2.text(
        0.0,
        0.95,
        rf"$E_g^{{pre}}$ = {ecbm_pre - evbm_pre:.3f} eV\n"
        rf"$E_g^{{DFT}}$ = {ecbm_tar - evbm_tar:.3f} eV",
        transform=ax2.transAxes,
        ha="left",
        va="top",
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

    plt.savefig(args.output, dpi=600, bbox_inches="tight")
    plt.close()

    return {
        "nocc": nocc,
        "segments": len(kpath["segments"]),
        "ticks": kpath["tick_labels"],
        "mae_band": mae_band,
        "gap_pre": ecbm_pre - evbm_pre,
        "gap_tar": ecbm_tar - evbm_tar,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare two SIESTA .bands files.")
    parser.add_argument("--pre", default="si.bands", help="Predicted/non-SCF bands file.")
    parser.add_argument("--tar", default="si_scf.bands", help="Target/DFT bands file.")
    parser.add_argument("--output", default="band_compare.png", help="Output image path.")
    parser.add_argument(
        "--ylim",
        type=float,
        nargs=2,
        default=(-4.0, 4.0),
        help="Y-axis limits after VBM alignment.",
    )
    parser.add_argument(
        "--nocc",
        type=int,
        default=None,
        help="Number of occupied bands. If omitted, infer from si.fdf when possible.",
    )
    parser.add_argument("--fdf", default="si.fdf", help="FDF file used to infer --nocc.")
    args = parser.parse_args()

    result = plot_bands(args)

    print(f"saved: {args.output}")
    print(f"nocc = {result['nocc'] if result['nocc'] is not None else 'fermi-threshold'}")
    print(f"segments = {result['segments']}")
    print(f"ticks = {', '.join(result['ticks'])}")
    print(f"MAE_band = {result['mae_band']:.6f} eV")
    print(f"gap_pred = {result['gap_pre']:.6f} eV")
    print(f"gap_DFT = {result['gap_tar']:.6f} eV")


if __name__ == "__main__":
    main()
