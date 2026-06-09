import numpy as np
import math
from edengnn.data.io.utils_f import utils
from dataclasses import dataclass
from typing import List, Any

BOHR = 0.5291772109
BOHR3 = BOHR**3
RYDBERG = 13.6057039763


"""-----------------------------------------------------------------------------

Utils for DFT

-----------------------------------------------------------------------------"""


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


"""-----------------------------------------------------------------------------

Utils for density

-----------------------------------------------------------------------------"""


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


"""-----------------------------------------------------------------------------

Utils for operator

-----------------------------------------------------------------------------"""


@dataclass
class BasisConfig:
    r"""dataclass for describing atomic orbitals

    Parameters
    ----------
    basis: list of int
        Irreps covering the orbitals of trained elements in ascending order of
        angular momentum quantum number

    size: int
        Length of irreps tensors of ``basis`` shape.

    basis_start: list of int
        Start indices of each irrep in irreps tensors of ``basis`` shape.

    l_max: int
        Maximum angular momentum quantum number in ``basis``.

    irreps_onsite: list of tuple
        Irreps of (``basis`` $\otimes$ ``basis``) for onsite operators.

    i1i2_start_onsite:
        Start indices of onsite irreps tensor, given the indices of irrep in the
        direct product representation.

    size_onsite: int
        Length of irreps tensors of ``irreps_onsite`` shape.

    i1i2_size_onsite:
        Length of irreps coupled from l1 $\otimes$ l2.

    irreps_offsite: list of tuple
        Irreps of (``basis`` $\otimes$ ``basis``) for offsite operators.

    i1i2_start_offsite:
        Start indices of offsite irreps tensor, given the indices of irrep in the
        direct product representation.

    size_offsite: int
        Length of irreps tensors of ``irreps_offsite`` shape.

    i1i2_size_offsite:
        Length of irreps coupled from l1 $\otimes$ l2.

    index_dft2e3nn: list of int
        Index which transforms the order of magnetic quantum number of
        DFT convention into that of e3nn.

    index_e3nn2dft: list of int
        Inverse of ``index_dft2e3nn``.

    atom_irreps: dict
        The key is the atomic number, and the value is the irreps of atomic
        orbitals in the DFT convention.

    atom_irreps_idx: dict
        The key is the atomic number, and the value is the index of the irreps in
        the ``basis`` list.

    """

    basis: List[int]
    size: int
    basis_start: List[int]
    l_max: int
    # onsite
    irreps_onsite: Any = None
    i1i2_start_onsite: Any = None
    size_onsite: int = None
    i1i2_size_onsite: Any = None
    # offsite
    irreps_offsite: Any = None
    i1i2_start_offsite: Any = None
    size_offsite: int = None
    i1i2_size_offsite: int = None
    # index change
    index_dft2e3nn: List[int] = None
    index_e3nn2dft: List[int] = None
    # basis dict
    atom_irreps: Any = None
    atom_irreps_idx: Any = None


def init_e3nn_irreps(basis, mode="onsite"):
    """
    transform direct product of irreps into direct sum
    """
    irreps = []
    num_basis = len(basis)
    i1i2_start = [
        [0] * num_basis for _ in range(num_basis)
    ]  # map from basis index to irreps tensor start position
    i1i2_size = [[0] * num_basis for _ in range(num_basis)]

    count = 0

    for i, l_i in enumerate(basis):
        start_j = i if mode == "onsite" else 0
        for j in range(start_j, num_basis):
            l_j = basis[j]

            if mode == "onsite":
                i1i2_start[i][j] = count
                i1i2_start[j][i] = count
                interval = 2
            elif mode == "offsite":
                i1i2_start[i][j] = count
                interval = 1

            l_min = abs(l_j - l_i)
            l_max = l_i + l_j
            p = (-1) ** (l_max)

            block_len = 0
            for lmain in range(l_min, l_max + 1, interval):
                irreps.append((1, (lmain, p)))
                block_len += 2 * lmain + 1
                count += 2 * lmain + 1

            if mode == "onsite":
                i1i2_size[i][j] = block_len
                i1i2_size[j][i] = block_len
            elif mode == "offsite":
                i1i2_size[i][j] = block_len

    len_tensor = count
    return irreps, i1i2_start, i1i2_size, len_tensor
