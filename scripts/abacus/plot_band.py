import os, pathlib
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


def parse_band(dir, prefix, ax, color="b"):
    name = pathlib.Path(dir).stem
    try:
        data = np.loadtxt(os.path.join(dir, f"OUT.{prefix}", "BANDS_1.dat"))
    except:
        print(f"error with {name}")
        return None, None
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


path_dir = "/public/home/lixiwen/dataset/universal_abacus/mc3d/dataset_test_gnome.txt"
with open(path_dir, "r") as f:
    paths = [line.strip() for line in f]

for path in paths:
    fig, ax = plt.subplots(figsize=(4, 4))
    plt.title("Electronic Band Structure", fontsize=16)
    plt.tight_layout()

    bands_pre, ef_pre = parse_band(
        path,
        "aiida",
        ax,
        color="b",
    )
    plt.show()
    plt.savefig(os.path.join(path, "band.png"), dpi=300)
    plt.close()
