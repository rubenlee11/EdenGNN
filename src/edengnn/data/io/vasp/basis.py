"""PAW projector bases read from the selected POTCAR files."""

from collections import defaultdict

from pymatgen.core import Element
from edengnn.data.io.utils import init_e3nn_irreps


AUG_BASIS = [0, 0, 0, 1, 1, 2, 2, 3, 3]
AUG_IRREPS, I1I2_IDX, _, LEN_AUG_TENSOR = init_e3nn_irreps(AUG_BASIS)


def read_basis(pseudo_files):
    """Return PAW angular momenta, embedding indices and occupancy lengths by Z.

    ``pseudo_files`` maps element symbols to dictionaries with ``path_pp``
    pointing to single-species POTCAR files. Projector order follows POTCAR.
    """
    positions = defaultdict(list)
    for index, l in enumerate(AUG_BASIS):
        positions[l].append(index)

    basis = {}
    for element, entry in pseudo_files.items():
        angular_momenta = []
        with open(entry["path_pp"]) as f:
            for line in f:
                if "Non local Part" in line:
                    l, n_projectors = map(int, next(f).split()[:2])
                    angular_momenta.extend([l] * n_projectors)

        used_count = defaultdict(int)
        indices = []
        for l in angular_momenta:
            indices.append(positions[l][used_count[l]])
            used_count[l] += 1
        n_aug = sum(
            2 * lmain + 1
            for i, l_i in enumerate(angular_momenta)
            for l_j in angular_momenta[i:]
            for lmain in range(abs(l_i - l_j), l_i + l_j + 1, 2)
        )
        basis[Element(element).Z] = {
            "angular_momenta": angular_momenta,
            "indices": indices,
            "n_aug": n_aug,
        }
    return basis
