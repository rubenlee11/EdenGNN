#!/usr/bin/env python3

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HARTREE_TO_EV = 27.211386245988


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare two OpenMX calculations by overlaying band structures and "
            "reporting total-energy / eigenvalue errors."
        )
    )
    parser.add_argument("pred_dir", type=Path, help="Predicted OpenMX run directory")
    parser.add_argument("scf_dir", type=Path, help="Self-consistent OpenMX run directory")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save plots and reports. Default: <pred_dir>/band_compare",
    )
    parser.add_argument(
        "--pred-label",
        default="Predicted",
        help="Legend label for the predicted structure",
    )
    parser.add_argument(
        "--scf-label",
        default="SCF",
        help="Legend label for the self-consistent structure",
    )
    parser.add_argument(
        "--window-ev",
        type=float,
        default=6.0,
        help="Energy window around the SCF Fermi level for plotting, in eV",
    )
    return parser.parse_args()


def find_single_file(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one '{pattern}' in {directory}, found {len(matches)}"
        )
    return matches[0]


def parse_out_file(path: Path) -> dict:
    text = path.read_text()

    total_energy_matches = re.findall(r"^\s*Utot\.\s+([-+0-9.Ee]+)\s*$", text, flags=re.M)
    if not total_energy_matches:
        total_energy_matches = re.findall(
            r"Total energy \(Hartree\).*?\n\s*([-+0-9.Ee]+)", text, flags=re.S
        )
    if not total_energy_matches:
        raise ValueError(f"Failed to parse total energy from {path}")
    total_energy_ha = float(total_energy_matches[-1])

    chem_matches = re.findall(
        r"Chemical potential \(Hartree\)\s+([-+0-9.Ee]+)", text, flags=re.I
    )
    if not chem_matches:
        raise ValueError(f"Failed to parse chemical potential from {path}")
    chemical_potential_ha = float(chem_matches[-1])

    num_states_match = re.search(
        r"Number of States\s*=\s*([-+0-9.Ee]+)", text, flags=re.I
    )
    homo_match = re.search(r"HOMO\s*=\s*(\d+)", text, flags=re.I)
    if not num_states_match or not homo_match:
        raise ValueError(f"Failed to parse eigenvalue metadata from {path}")
    num_states = int(round(float(num_states_match.group(1))))
    homo = int(homo_match.group(1))

    start = text.find("Eigenvalues (Hartree) for SCF KS-eq.")
    if start < 0:
        raise ValueError(f"Failed to locate eigenvalue table in {path}")
    lines = text[start:].splitlines()

    eigenvalues = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 3 and parts[0].isdigit() and int(parts[0]) == len(eigenvalues) + 1:
            eigenvalues.append(float(parts[1]))
            if len(eigenvalues) == num_states:
                break
        elif eigenvalues:
            break

    if len(eigenvalues) != num_states:
        raise ValueError(
            f"Parsed {len(eigenvalues)} eigenvalues from {path}, expected {num_states}"
        )

    return {
        "path": str(path),
        "total_energy_ha": total_energy_ha,
        "chemical_potential_ha": chemical_potential_ha,
        "num_states": num_states,
        "homo": homo,
        "eigenvalues_ha": np.array(eigenvalues, dtype=float),
    }


def _normalize_label(label: str) -> str:
    return "Γ" if label.upper() == "GAMMA" else label


def parse_band_file(path: Path) -> dict:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    cursor = 0

    header = lines[cursor].split()
    cursor += 1
    num_bands = int(header[0])
    spin_flag = int(header[1])
    fermi_ha = float(header[2])

    reciprocal_lattice = np.array([float(x) for x in lines[cursor].split()], dtype=float)
    reciprocal_lattice = reciprocal_lattice.reshape(3, 3)
    cursor += 1

    nkpath = int(lines[cursor])
    cursor += 1

    path_segments = []
    total_kpoints = 0
    for _ in range(nkpath):
        parts = lines[cursor].split()
        cursor += 1
        npts = int(parts[0])
        start_k = np.array([float(x) for x in parts[1:4]], dtype=float)
        end_k = np.array([float(x) for x in parts[4:7]], dtype=float)
        start_label = _normalize_label(parts[7])
        end_label = _normalize_label(parts[8])
        total_kpoints += npts
        path_segments.append(
            {
                "npts": npts,
                "start_k": start_k,
                "end_k": end_k,
                "start_label": start_label,
                "end_label": end_label,
            }
        )

    spin_channels = 1 if spin_flag == 0 else 2
    kpoints = []
    energies = np.zeros((spin_channels, total_kpoints, num_bands), dtype=float)

    for ik in range(total_kpoints):
        parts = lines[cursor].split()
        cursor += 1
        band_count = int(parts[0])
        if band_count != num_bands:
            raise ValueError(
                f"Band count mismatch in {path}: header says {num_bands}, block says {band_count}"
            )
        kpoints.append([float(x) for x in parts[1:4]])

        for ispin in range(spin_channels):
            band_values = np.array([float(x) for x in lines[cursor].split()], dtype=float)
            cursor += 1
            if band_values.size != num_bands:
                raise ValueError(
                    f"Expected {num_bands} band values in {path}, got {band_values.size}"
                )
            energies[ispin, ik, :] = band_values

    kpoints = np.array(kpoints, dtype=float)
    distances, tick_positions, tick_labels = build_k_axis(kpoints, reciprocal_lattice, path_segments)

    return {
        "path": str(path),
        "num_bands": num_bands,
        "spin_channels": spin_channels,
        "fermi_ha": fermi_ha,
        "reciprocal_lattice": reciprocal_lattice,
        "segments": path_segments,
        "kpoints_frac": kpoints,
        "distances": distances,
        "tick_positions": tick_positions,
        "tick_labels": tick_labels,
        "energies_ha": energies,
    }


def build_k_axis(kpoints_frac: np.ndarray, reciprocal_lattice: np.ndarray, segments: list):
    kpoints_cart = kpoints_frac @ reciprocal_lattice
    distances = np.zeros(len(kpoints_frac), dtype=float)
    for i in range(1, len(kpoints_frac)):
        distances[i] = distances[i - 1] + np.linalg.norm(kpoints_cart[i] - kpoints_cart[i - 1])

    tick_positions = [distances[0]]
    tick_labels = [_normalize_label(segments[0]["start_label"])]
    cursor = 0
    for segment in segments:
        cursor += segment["npts"] - 1
        tick_positions.append(distances[cursor])
        tick_labels.append(_normalize_label(segment["end_label"]))

    merged_positions = [tick_positions[0]]
    merged_labels = [tick_labels[0]]
    for pos, label in zip(tick_positions[1:], tick_labels[1:]):
        if math.isclose(pos, merged_positions[-1], rel_tol=0.0, abs_tol=1e-12):
            if label != merged_labels[-1]:
                merged_labels[-1] = f"{merged_labels[-1]}|{label}"
        else:
            merged_positions.append(pos)
            merged_labels.append(label)
    return distances, merged_positions, merged_labels


def compare_shapes(pred: dict, scf: dict):
    if pred["energies_ha"].shape != scf["energies_ha"].shape:
        raise ValueError(
            "Band data shape mismatch: "
            f"{pred['energies_ha'].shape} vs {scf['energies_ha'].shape}"
        )
    if pred["tick_labels"] != scf["tick_labels"]:
        raise ValueError(
            "High-symmetry labels differ between band files: "
            f"{pred['tick_labels']} vs {scf['tick_labels']}"
        )
    if not np.allclose(pred["kpoints_frac"], scf["kpoints_frac"], atol=1e-10, rtol=0.0):
        raise ValueError("K-point grids differ between the two .Band files")


def summarize_errors(pred_out: dict, scf_out: dict, pred_band: dict, scf_band: dict) -> dict:
    eig_diff_ha = pred_out["eigenvalues_ha"] - scf_out["eigenvalues_ha"]
    eig_aligned_diff_ha = (
        pred_out["eigenvalues_ha"]
        - pred_out["chemical_potential_ha"]
        - (scf_out["eigenvalues_ha"] - scf_out["chemical_potential_ha"])
    )

    band_diff_ha = pred_band["energies_ha"] - scf_band["energies_ha"]
    band_aligned_diff_ha = (
        pred_band["energies_ha"]
        - pred_band["fermi_ha"]
        - (scf_band["energies_ha"] - scf_band["fermi_ha"])
    )

    homo = scf_out["homo"]
    metrics = {
        "total_energy_pred_ha": pred_out["total_energy_ha"],
        "total_energy_scf_ha": scf_out["total_energy_ha"],
        "total_energy_error_ha": pred_out["total_energy_ha"] - scf_out["total_energy_ha"],
        "total_energy_error_ev": (pred_out["total_energy_ha"] - scf_out["total_energy_ha"]) * HARTREE_TO_EV,
        "chemical_potential_pred_ha": pred_out["chemical_potential_ha"],
        "chemical_potential_scf_ha": scf_out["chemical_potential_ha"],
        "chemical_potential_error_ha": pred_out["chemical_potential_ha"] - scf_out["chemical_potential_ha"],
        "chemical_potential_error_ev": (pred_out["chemical_potential_ha"] - scf_out["chemical_potential_ha"]) * HARTREE_TO_EV,
        "eigenvalue_mae_ev": np.mean(np.abs(eig_diff_ha)) * HARTREE_TO_EV,
        "eigenvalue_rmse_ev": np.sqrt(np.mean(eig_diff_ha**2)) * HARTREE_TO_EV,
        "eigenvalue_max_abs_ev": np.max(np.abs(eig_diff_ha)) * HARTREE_TO_EV,
        "eigenvalue_fermi_aligned_mae_ev": np.mean(np.abs(eig_aligned_diff_ha)) * HARTREE_TO_EV,
        "eigenvalue_fermi_aligned_rmse_ev": np.sqrt(np.mean(eig_aligned_diff_ha**2)) * HARTREE_TO_EV,
        "occupied_eigenvalue_mae_ev": np.mean(np.abs(eig_diff_ha[:homo])) * HARTREE_TO_EV,
        "unoccupied_eigenvalue_mae_ev": np.mean(np.abs(eig_diff_ha[homo:])) * HARTREE_TO_EV,
        "band_mae_ev": np.mean(np.abs(band_diff_ha)) * HARTREE_TO_EV,
        "band_rmse_ev": np.sqrt(np.mean(band_diff_ha**2)) * HARTREE_TO_EV,
        "band_max_abs_ev": np.max(np.abs(band_diff_ha)) * HARTREE_TO_EV,
        "band_fermi_aligned_mae_ev": np.mean(np.abs(band_aligned_diff_ha)) * HARTREE_TO_EV,
        "band_fermi_aligned_rmse_ev": np.sqrt(np.mean(band_aligned_diff_ha**2)) * HARTREE_TO_EV,
        "band_fermi_aligned_max_abs_ev": np.max(np.abs(band_aligned_diff_ha)) * HARTREE_TO_EV,
    }
    return metrics


def write_report(path: Path, metrics: dict, pred_label: str, scf_label: str):
    lines = [
        f"{pred_label} vs {scf_label} OpenMX comparison",
        "",
        "Total energy",
        f"  {pred_label}: {metrics['total_energy_pred_ha']:.12f} Ha",
        f"  {scf_label}: {metrics['total_energy_scf_ha']:.12f} Ha",
        f"  Error ({pred_label} - {scf_label}): {metrics['total_energy_error_ha']:.12e} Ha = {metrics['total_energy_error_ev']:.6f} eV",
        "",
        "Chemical potential / Fermi level",
        f"  {pred_label}: {metrics['chemical_potential_pred_ha']:.12f} Ha",
        f"  {scf_label}: {metrics['chemical_potential_scf_ha']:.12f} Ha",
        f"  Error: {metrics['chemical_potential_error_ha']:.12e} Ha = {metrics['chemical_potential_error_ev']:.6f} eV",
        "",
        "Eigenvalue error from .out",
        f"  MAE: {metrics['eigenvalue_mae_ev']:.6f} eV",
        f"  RMSE: {metrics['eigenvalue_rmse_ev']:.6f} eV",
        f"  Max |error|: {metrics['eigenvalue_max_abs_ev']:.6f} eV",
        f"  Fermi-aligned MAE: {metrics['eigenvalue_fermi_aligned_mae_ev']:.6f} eV",
        f"  Fermi-aligned RMSE: {metrics['eigenvalue_fermi_aligned_rmse_ev']:.6f} eV",
        f"  Occupied-state MAE: {metrics['occupied_eigenvalue_mae_ev']:.6f} eV",
        f"  Unoccupied-state MAE: {metrics['unoccupied_eigenvalue_mae_ev']:.6f} eV",
        "",
        "Band error from .Band",
        f"  MAE: {metrics['band_mae_ev']:.6f} eV",
        f"  RMSE: {metrics['band_rmse_ev']:.6f} eV",
        f"  Max |error|: {metrics['band_max_abs_ev']:.6f} eV",
        f"  Fermi-aligned MAE: {metrics['band_fermi_aligned_mae_ev']:.6f} eV",
        f"  Fermi-aligned RMSE: {metrics['band_fermi_aligned_rmse_ev']:.6f} eV",
        f"  Fermi-aligned Max |error|: {metrics['band_fermi_aligned_max_abs_ev']:.6f} eV",
        "",
    ]
    path.write_text("\n".join(lines))


def plot_bands(path: Path, pred_band: dict, scf_band: dict, pred_label: str, scf_label: str, window_ev: float):
    x = scf_band["distances"]
    pred_ev = (pred_band["energies_ha"][0] - pred_band["fermi_ha"]) * HARTREE_TO_EV
    scf_ev = (scf_band["energies_ha"][0] - scf_band["fermi_ha"]) * HARTREE_TO_EV

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    for ib in range(scf_ev.shape[1]):
        ax.plot(
            x,
            scf_ev[:, ib],
            color="#1f77b4",
            lw=1.2,
            alpha=0.9,
            label=scf_label if ib == 0 else None,
        )
        ax.plot(
            x,
            pred_ev[:, ib],
            color="#d62728",
            lw=1.0,
            alpha=0.65,
            ls="--",
            label=pred_label if ib == 0 else None,
        )

    for xpos in scf_band["tick_positions"]:
        ax.axvline(xpos, color="0.75", lw=0.8)
    ax.axhline(0.0, color="0.2", lw=1.0)

    ax.set_xticks(scf_band["tick_positions"])
    ax.set_xticklabels(scf_band["tick_labels"])
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(-window_ev, window_ev)
    ax.set_ylabel("Energy - Ef (eV)")
    ax.set_title(f"{pred_label} vs {scf_label} band structure")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()

    pred_dir = args.pred_dir.resolve()
    scf_dir = args.scf_dir.resolve()
    output_dir = (args.output_dir or (pred_dir / "band_compare")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_out = parse_out_file(find_single_file(pred_dir, "*.out"))
    scf_out = parse_out_file(find_single_file(scf_dir, "*.out"))
    pred_band = parse_band_file(find_single_file(pred_dir, "*.Band"))
    scf_band = parse_band_file(find_single_file(scf_dir, "*.Band"))

    compare_shapes(pred_band, scf_band)
    metrics = summarize_errors(pred_out, scf_out, pred_band, scf_band)

    report_txt = output_dir / "comparison_report.txt"
    report_json = output_dir / "comparison_report.json"
    plot_png = output_dir / "band_overlay.png"

    write_report(report_txt, metrics, args.pred_label, args.scf_label)
    report_json.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    plot_bands(
        plot_png,
        pred_band,
        scf_band,
        pred_label=args.pred_label,
        scf_label=args.scf_label,
        window_ev=args.window_ev,
    )

    print(f"Saved band plot: {plot_png}")
    print(f"Saved text report: {report_txt}")
    print(f"Saved JSON report: {report_json}")
    print("")
    print(f"Total energy error: {metrics['total_energy_error_ev']:.6f} eV")
    print(f"Eigenvalue MAE (.out): {metrics['eigenvalue_mae_ev']:.6f} eV")
    print(f"Fermi-aligned band MAE (.Band): {metrics['band_fermi_aligned_mae_ev']:.6f} eV")


if __name__ == "__main__":
    main()
