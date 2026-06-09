import numpy as np
import math
from edengnn.data.io.utils_f import utils

BOHR = 0.5291772109
BOHR3 = BOHR**3
RYDBERG = 13.6057039763


def _round_for_fft(n):
    n = math.ceil(n)
    while True:
        temp = n
        for p in [2, 3, 5]:
            while temp % p == 0:
                temp //= p
        if temp == 1:
            return n
        n += 1


def set_grid_fft(cell, encut):
    """
    set fft grid according to the cutoff energy

    |G| < E_cut

    unit: rydberg
    """
    cell = cell / BOHR
    cell_G = np.linalg.inv(cell.T)
    Gcut = np.sqrt(encut)

    ng, nm1, nm2, nm3 = utils.count_grid(Gcut, cell_G.T)
    gcart, gfrac = utils.set_grid(Gcut, cell_G.T, ng, nm1, nm2, nm3)
    mill = gfrac.T
    n1 = _round_for_fft(2 * np.max(mill[:, 0]) + 1)
    n2 = _round_for_fft(2 * np.max(mill[:, 1]) + 1)
    n3 = _round_for_fft(2 * np.max(mill[:, 2]) + 1)
    return n1, n2, n3


def set_grid_lcao(cell, encut=220):
    """
    Set automatically the real space grid for integration based on the cell and
    the cutoff energy.
    """
    # encut: cutoff energy for integration in rydberg.
    cell_G = np.linalg.inv(cell.T)
    tmp = np.sqrt(encut) / np.pi
    n1 = _round_for_fft(tmp / np.linalg.norm(cell_G[0]))
    n2 = _round_for_fft(tmp / np.linalg.norm(cell_G[1]))
    n3 = _round_for_fft(tmp / np.linalg.norm(cell_G[2]))

    return n1, n2, n3


def get_mask_r(cell, grid_shape, radius=4.0):
    N1, N2, N3 = grid_shape
    grid = np.array([cell[0] / N1, cell[1] / N2, cell[2] / N3])
    ng, nm1, nm2, nm3 = utils.count_grid(radius, grid.T)
    gcart, gfrac = utils.set_grid(radius, grid.T, ng, nm1, nm2, nm3)
    return gfrac.T, gcart.T


def pos2n(cell, grid_shape, pos):
    N1, N2, N3 = grid_shape
    a1, a2, a3 = cell[0], cell[1], cell[2]
    omega = np.inner(np.cross(a1, a2), a3)

    grid = np.array([cell[0] / N1, cell[1] / N2, cell[2] / N3])

    b1 = np.cross(a2, a3) / omega
    b2 = np.cross(a3, a1) / omega
    b3 = np.cross(a1, a2) / omega

    n_atom = len(pos)
    n_grid = np.zeros(pos.shape)
    for i in range(n_atom):
        n_grid[i, 0] = np.inner(pos[i], b1 * N1)
        n_grid[i, 1] = np.inner(pos[i], b2 * N2)
        n_grid[i, 2] = np.inner(pos[i], b3 * N3)

    n_grid = np.round(n_grid).astype(int)

    return n_grid, grid


def init_basic_irreps(basis, mode="onsite"):
    """
    transform operator into irreps
    """
    irreps = []
    num_basis = len(basis)
    block_start = [
        [0] * num_basis for _ in range(num_basis)
    ]  # map from basis index to irreps tensor start position

    count = 0

    for i, l_i in enumerate(basis):
        start_j = i if mode == "onsite" else 0
        for j in range(start_j, num_basis):
            l_j = basis[j]

            if mode == "onsite":
                block_start[i][j] = count
                block_start[j][i] = count
                interval = 2
            elif mode == "offsite":
                block_start[i][j] = count
                interval = 1

            l_min = abs(l_j - l_i)
            l_max = l_i + l_j
            p = (-1) ** (l_max)

            for lmain in range(l_min, l_max + 1, interval):
                irreps.append((1, (lmain, p)))
                count += 2 * lmain + 1
    len_tensor = count
    return irreps, block_start, len_tensor
