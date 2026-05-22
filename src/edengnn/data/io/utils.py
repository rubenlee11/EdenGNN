import numpy as np
from edengnn.data.io.utils_f import utils

BOHR = 0.5291772109


def _round_for_fft(n):
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


def get_mask_r_deprecated(cell, grid_shape, radius=4.0):
    """
    Specify probe index. The algorithm can be improved.

    reminicent of setting k grid in PW DFT. use the same technique to bound the mask index

    cell[0,:] [1,:], [2,:] is a1, a2, a3 vectors
    grid_shape: (n1, n2, n3) is the number of grid points in each direction
    """

    N1, N2, N3 = grid_shape
    grid = np.array([cell[0] / N1, cell[1] / N2, cell[2] / N3])

    cell_G = np.linalg.inv(grid.T)
    m1_max = int(np.ceil(radius * np.linalg.norm(cell_G[0, :]) + 0.5))
    m2_max = int(np.ceil(radius * np.linalg.norm(cell_G[1, :]) + 0.5))
    m3_max = int(np.ceil(radius * np.linalg.norm(cell_G[2, :]) + 0.5))

    m1_range = np.arange(-m1_max, m1_max)
    m2_range = np.arange(-m2_max, m2_max)
    m3_range = np.arange(-m3_max, m3_max)

    M1, M2, M3 = np.meshgrid(m1_range, m2_range, m3_range, indexing="ij")
    mask0_all = np.stack([M1.ravel(), M2.ravel(), M3.ravel()], axis=1)
    edge_vec_all = np.dot(mask0_all, grid)

    vecl_sq = np.sum(edge_vec_all**2, axis=1)
    radius_sq = radius**2
    mask = vecl_sq <= radius_sq

    mask0 = mask0_all[mask]
    edge_vec = edge_vec_all[mask]

    # avoid NaN for r = 0
    center_index = np.where(np.all(mask0 == 0, axis=1))[0]
    edge_vec[center_index] += [0, 1e-6, 0]
    return np.array(mask0), np.array(edge_vec)
