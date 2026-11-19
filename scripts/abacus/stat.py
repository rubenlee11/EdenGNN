r"""Compare ABACUS uniform-grid eigenvalues/gaps and high-symmetry bands.

python stat.py --filelist structures.txt --save-dir statistics --prefix EDENGNN

Each listed structure directory contains nscf/, scf/, band/, band_scf/.
--dir-tar optionally supplies <root>/<name>/{scf,band_scf} instead.
Uniform-grid calculations must write BANDS_1.dat (out_band=1).
Outputs stat_mae.json and per-structure band_compare.png. Eigenvalue MAE
uses raw energies unless --use-shift aligns their means. Gaps are unshifted.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from plot_band import get_band_data, get_fermi_energy, plot_band_dos


def read_uniform_bands(path, prefix, mode):
    """Read eigenvalues in eV and the gap from a uniform-grid calculation."""
    output = path / f'OUT.{prefix}'
    bands = np.loadtxt(output / 'BANDS_1.dat')[:, 2:]
    efermi = get_fermi_energy(output / f'running_{mode}.log')
    band_min = bands.min(axis=0)
    band_max = bands.max(axis=0)
    is_metal = np.any((band_min < efermi - 1e-3) & (band_max > efermi + 1e-3))
    gap_ev = 0.0 if is_metal else float(
        band_min[band_min > efermi + 1e-3].min()
        - band_max[band_max <= efermi + 1e-3].max()
    )
    return bands, gap_ev


def plot_comparison(predicted_dir, target_dir, prefix, prefix_tar, save_path):
    """Overlay path bands using target VBM/Ef as a common energy zero."""
    predicted = get_band_data(predicted_dir, prefix)
    target = get_band_data(target_dir, prefix_tar)
    zero_ev = target['e_vbm']
    fig = plot_band_dos(
        target['distances'], [energy - zero_ev for energy in target['energies']],
        target['tick_positions'], target['tick_labels'], target['e_cbm'] - zero_ev,
    )
    try:
        for ax in fig.axes:
            for distance, energy in zip(predicted['distances'], predicted['energies']):
                ax.plot(distance, energy - zero_ev, color='red', linestyle='--', linewidth=0.9)
        fig.axes[0].legend(handles=[
            Line2D([], [], color='#245c91', label='SCF target'),
            Line2D([], [], color='red', linestyle='--', label='Prediction'),
        ])
        fig.savefig(save_path, dpi=300)
    finally:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--filelist', required=True, type=Path)
    parser.add_argument('--save-dir', type=Path)
    parser.add_argument('--mae-output', type=Path, help='Explicit JSON path (legacy option)')
    parser.add_argument('--dir-tar', type=Path)
    parser.add_argument('--prefix', default='EDENGNN')
    parser.add_argument('--prefix-tar', default='EDENGNN')
    parser.add_argument('--use-shift', action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    if args.save_dir is None and args.mae_output is None:
        parser.error('provide --save-dir or --mae-output')
    output = args.mae_output if args.mae_output else args.save_dir / 'stat_mae.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics = []
    for line in args.filelist.read_text().splitlines():
        if not line.strip():
            continue
        path = Path(line.strip())
        target = args.dir_tar / path.name if args.dir_tar else path
        predicted_bands, predicted_gap = read_uniform_bands(path / 'nscf', args.prefix, 'nscf')
        target_bands, target_gap = read_uniform_bands(target / 'scf', args.prefix_tar, 'scf')
        if args.use_shift:
            predicted_bands = predicted_bands + target_bands.mean() - predicted_bands.mean()
        metric = {
            'name': path.name,
            'mae_band_energies': float(np.abs(predicted_bands - target_bands).mean()),
            'gap_predict_ev': predicted_gap,
            'gap_target_ev': target_gap,
            'mae_band_gap': abs(predicted_gap - target_gap),
        }
        plot_comparison(path / 'band', target / 'band_scf', args.prefix, args.prefix_tar,
                        path / 'band_compare.png')
        metrics.append(metric)
        print(json.dumps(metric), flush=True)
    if not metrics:
        raise ValueError('No structures supplied for comparison')
    result = {
        'unit': 'eV', 'use_shift': args.use_shift, 'num_structures': len(metrics),
        'mean_band_mae_ev': float(np.mean([m['mae_band_energies'] for m in metrics])),
        'mean_gap_mae_ev': float(np.mean([m['mae_band_gap'] for m in metrics])),
        'structures': metrics,
    }
    output.write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
