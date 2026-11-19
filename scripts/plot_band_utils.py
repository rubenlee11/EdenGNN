"""Shared band/DOS layout for the VASP and ABACUS plotting scripts."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


def plot_band_dos(distances, energies, tick_positions, tick_labels, gap_ev,
                  dos_energy_ev=None, dos_density=None, ylim=(-4, 4), gap_break=2,
                  pdos_energy_ev=None, projections=None):
    """Return a figure with energies relative to VBM (Ef for metals).

    ``ylim`` gives offsets below VBM and above CBM. Gaps larger than
    ``gap_break`` eV are split, retaining 0.35 eV of gap beside each edge.
    DOS limits use only the displayed energy windows; DOS is filled to zero.
    Optional ``projections`` maps legend labels to PDOS on ``pdos_energy_ev``.
    """
    windows = [(ylim[0], gap_ev + ylim[1])]
    if gap_ev > gap_break:
        windows = [(gap_ev - 0.35, gap_ev + ylim[1]), (ylim[0], 0.35)]
    has_dos = dos_energy_ev is not None
    fig = plt.figure(figsize=(6.8 if has_dos else 5.2, 5.2))
    grid = fig.add_gridspec(
        len(windows), 2 if has_dos else 1,
        width_ratios=[3.2, 1.2] if has_dos else [1],
        height_ratios=[upper - lower for lower, upper in windows],
        hspace=0.06, wspace=0.06,
    )
    dos_max = 0
    if has_dos:
        visible = np.zeros(len(dos_energy_ev), dtype=bool)
        for lower, upper in windows:
            visible |= (dos_energy_ev >= lower) & (dos_energy_ev <= upper)
        dos_max = np.max(dos_density[visible], initial=0)
        if projections:
            pdos_visible = np.zeros(len(pdos_energy_ev), dtype=bool)
            for lower, upper in windows:
                pdos_visible |= (pdos_energy_ev >= lower) & (pdos_energy_ev <= upper)
            for density in projections.values():
                dos_max = max(dos_max, np.max(density[pdos_visible], initial=0))
    band_axes = []
    dos_axes = []
    for row, (lower, upper) in enumerate(windows):
        ax = fig.add_subplot(grid[row, 0])
        band_axes.append(ax)
        for distance, energy in zip(distances, energies, strict=True):
            ax.plot(distance, energy, color="#245c91", linewidth=1.05)
        ax.set_xticks(tick_positions, tick_labels)
        ax.set_xlim(distances[0][0], distances[-1][-1])
        ax.set_ylim(lower, upper)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.grid(axis="x", color="0.86", linewidth=0.6)
        ax.axhline(0, color="0.5", linewidth=0.6, linestyle="--")
        if has_dos:
            ax_dos = fig.add_subplot(grid[row, 1], sharey=ax)
            dos_axes.append(ax_dos)
            ax_dos.fill_betweenx(dos_energy_ev, 0, dos_density, color="#6da9be",
                                 alpha=0.25 if projections else 0.65)
            ax_dos.plot(dos_density, dos_energy_ev, color="#245c91", linewidth=0.9,
                        label="Total")
            if projections:
                for index, (label, density) in enumerate(projections.items()):
                    ax_dos.plot(density, pdos_energy_ev, color=plt.cm.tab10(index % 10),
                                linewidth=1.1, label=label)
            ax_dos.set_xlim(0, dos_max * 1.05 if dos_max > 0 else 1)
            ax_dos.xaxis.set_major_locator(MaxNLocator(nbins=3, prune="lower"))
            ax_dos.tick_params(axis="y", left=False, labelleft=False)
        if row == len(windows) - 1 and has_dos:
            ax_dos.set_xlabel("DOS (states/eV)")
    if len(windows) == 2:
        for axes in [band_axes, dos_axes] if has_dos else [band_axes]:
            top, bottom = axes
            top.spines["bottom"].set_visible(False)
            bottom.spines["top"].set_visible(False)
            top.tick_params(axis="x", bottom=False, labelbottom=False)
            for ax, y in [(top, 0), (bottom, 1)]:
                for x in [0, 1]:
                    ax.plot([x - 0.012, x + 0.012], [y - 0.018, y + 0.018],
                            transform=ax.transAxes, color="0.25", linewidth=1,
                            clip_on=False)
    fig.supylabel("Energy relative to VBM / Ef (eV)", x=0.015)
    if projections:
        handles, labels = dos_axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
                   fontsize=9, columnspacing=1.2)
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.12,
                        top=0.88 if projections else 0.97)
    return fig
