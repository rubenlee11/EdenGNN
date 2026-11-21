"""Prepare PyATB optical inputs or batch-plot its dielectric/conductivity tensors."""
import argparse
import json
import os
from pathlib import Path
import re


def prepare(calculation, args):
    """Read an ABACUS NSCF calculation and write pyatb/Input with absolute CSR paths."""
    fields = {}
    for line in (calculation / 'INPUT').read_text().splitlines()[1:]:
        parts = line.split('#', 1)[0].split()
        if len(parts) >= 2:
            fields[parts[0]] = parts[1]
    nspin = int(fields.get('nspin', 1))
    if nspin not in (1, 4):
        raise ValueError(f'{calculation}: PyATB optical conductivity requires nspin 1 or 4')
    output = calculation / f"OUT.{fields.get('suffix', 'ABACUS')}"
    log = output / f"running_{fields['calculation']}.log"
    matches = re.findall(r'EFERMI\s*=\s*([-+\d.Ee]+)', log.read_text())
    if not matches:
        raise ValueError(f'{log}: missing EFERMI in eV')
    matrices = [output / name for name in (
        'data-HR-sparse_SPIN0.csr', 'data-SR-sparse_SPIN0.csr', 'data-rR-sparse.csr')]
    for matrix in matrices:
        if not matrix.is_file():
            raise FileNotFoundError(matrix)
    work = calculation / 'pyatb'
    work.mkdir(exist_ok=True)
    # PyATB initializes Out in the working directory when imported.
    os.chdir(work)
    from pyatb.easy_use.stru_analyzer import read_abacus_stru
    with (calculation / 'STRU').open() as stream:
        atoms = read_abacus_stru(stream)
    lattice = '\n'.join('    ' + ' '.join(map(str, row)) for row in atoms.cell.array)
    text = f'''INPUT_PARAMETERS
{{
    nspin {nspin}
    package ABACUS
    fermi_energy {matches[-1]}
    fermi_energy_unit eV
    HR_route {matrices[0]}
    SR_route {matrices[1]}
    rR_route {matrices[2]}
    HR_unit Ry
    rR_unit Bohr
    max_kpoint_num 256
}}
LATTICE
{{
    lattice_constant 1.0
    lattice_constant_unit Angstrom
    lattice_vector
{lattice}
}}
OPTICAL_CONDUCTIVITY
{{
    occ_band -1
    omega {args.omega[0]} {args.omega[1]}
    domega {args.domega}
    eta {args.eta}
    grid {' '.join(map(str, args.grid))}
}}
'''
    (work / 'Input').write_text(text)
    print(work)


def plot(calculation, target=None):
    """Save separate dielectric and real-conductivity figures for xx, yy, zz."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    spectra = {}
    references = {}
    metrics = {}
    for quantity in ('dielectric_function', 'optical_conductivity'):
        for part in ('real', 'imag'):
            key = f'{quantity}_{part}'
            filename = f'{key}_part.dat'
            data = np.loadtxt(calculation / 'pyatb/Out/Optical_Conductivity' / filename)
            if not np.isfinite(data).all():
                raise ValueError(f'{calculation}: non-finite {key} spectrum')
            spectra[key] = data
            if target is None:
                continue
            reference = np.loadtxt(target / 'pyatb/Out/Optical_Conductivity' / filename)
            if not np.isfinite(reference).all():
                raise ValueError(f'{target}: non-finite {key} spectrum')
            if data.shape != reference.shape or not np.allclose(
                    data[:, 0], reference[:, 0], rtol=0, atol=1e-8):
                raise ValueError(f'{calculation}, {target}: use identical photon energy grids')
            references[key] = reference
            error = np.abs(data[:, 1:] - reference[:, 1:])
            metrics[key] = {
                'unit': 'S/m' if quantity == 'optical_conductivity' else 'dimensionless',
                'mae_diagonal': dict(zip(('xx', 'yy', 'zz'),
                                         error[:, [0, 4, 8]].mean(axis=0).tolist())),
                'mae_tensor': float(error.mean()),
            }

    suffix = '_compare' if target is not None else ''
    for column, component in ((1, 'xx'), (5, 'yy'), (9, 'zz')):
        for quantity in ('dielectric_function', 'optical_conductivity'):
            fig, ax = plt.subplots(figsize=(6, 6))
            parts = ('real', 'imag') if quantity == 'dielectric_function' else ('real',)
            for part, color in zip(parts, ('tab:blue', 'tab:orange')):
                key = f'{quantity}_{part}'
                data = spectra[key]
                label = 'Re' if part == 'real' else 'Im'
                ax.plot(data[:, 0], data[:, column], color=color,
                        label=f'{label} prediction' if target is not None else label)
                if target is not None:
                    reference = references[key]
                    ax.plot(reference[:, 0], reference[:, column], '--', color=color,
                            label=f'{label} DFT')
            ax.set_xlabel('Photon energy (eV)')
            ax.set_xlim(0, 6)
            ax.set_ylabel(r'$\epsilon_{%s}$' % component if quantity == 'dielectric_function'
                          else r'Re $\sigma_{%s}$ (S/m)' % component)
            ax.set_title(f'{calculation.name} — {component}')
            ax.legend()
            ax.grid(alpha=0.2)
            fig.tight_layout()
            destination = calculation / f'{quantity}_{component}{suffix}.png'
            fig.savefig(destination, dpi=180)
            plt.close(fig)
            print(destination)
    if target is not None:
        (calculation / 'optical_compare.json').write_text(json.dumps({
            'prediction': str(calculation), 'target': str(target),
            'energy_range_ev': [float(data[0, 0]), float(data[-1, 0])],
            'metrics': metrics,
        }, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'plot', 'compare'))
    parser.add_argument('--filelist', required=True, type=Path)
    parser.add_argument('--target-filelist', type=Path,
                        help='DFT directories, paired by line with --filelist')
    parser.add_argument('--grid', nargs=3, type=int, default=(12, 12, 12))
    parser.add_argument('--omega', nargs=2, type=float, default=(0.1, 10.0))
    parser.add_argument('--domega', type=float, default=0.05)
    parser.add_argument('--eta', type=float, default=0.1)
    args = parser.parse_args()
    if args.mode == 'compare' and args.target_filelist is None:
        parser.error('compare requires --target-filelist')
    calculations = [Path(line.strip()).expanduser().resolve()
                    for line in args.filelist.read_text().splitlines() if line.strip()]
    targets = [None] * len(calculations)
    if args.mode == 'compare':
        targets = [Path(line.strip()).expanduser().resolve()
                   for line in args.target_filelist.read_text().splitlines() if line.strip()]
        if len(targets) != len(calculations):
            parser.error('prediction and target filelists must have the same length')
    for calculation, target in zip(calculations, targets):
        if args.mode == 'prepare':
            prepare(calculation, args)
        else:
            plot(calculation, target)


if __name__ == '__main__':
    main()
