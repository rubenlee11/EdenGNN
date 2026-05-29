import numpy as np
import pathlib


class IO_Abacus_Operator:
    def __init__(self, threshold=1e-7, unit=1.0):
        """
        threshold:
            truncate blocks whose matrix elements are smaller than the threshold
        """
        self.threshold = threshold  # [unit]
        self.unit = unit
        return None

    def read_data(self, path):
        """
        z:
        cell:
        pos:
        edge_index:
        edge_vec:
        nbr_shift:

        edge information are read for offsite blocks
        """
        name = pathlib.Path(path).stem
        npz_data = np.load(os.path.join(path, "output_HR0.npz"), allow_pickle=True)
        keys = npz_data.files

        operator_onsite = []
        operator_onsite_mask = []
        operator_offsite = []
        operator_offsite_mask = []

        edge_index = []
        cell_shift = []

        onsite_index = []

        # get cell info
        cell = npz_data[keys[0]].reshape(3, 3) * BOHR
        del keys[0]
        # get atom info
        atom_info = npz_data[keys[0]]
        z = atom_info[:, 1].astype(int)
        type_dict = dict(zip(atom_info[:, 0].astype(int), z))
        pos = atom_info[:, 2:5] @ cell
        del keys[0]
        # ignore orbital info
        del keys[: len(type_dict)]
        # get operator info
        for key in keys:
            array = npz_data[key]
            # ignore operator blocks smaller than threshold
            if np.max(np.abs(array)) >= self.threshold:
                # read the edge index and the cell shift of the operator block
                _edge = [int(x) for x in key.split("_")[1:]]
                i, j = _edge[:2]
                shift = _edge[2:]

                # reshape the operator block into (BASIS, BASIS)
                _data = array * self.unit

                lis = basis_irreps[z[i]]
                lis_idx = basis_idx[z[i]]
                ljs = basis_irreps[z[j]]
                ljs_idx = basis_idx[z[j]]

                block = np.zeros((BASIS_SIZE, BASIS_SIZE))
                block_mask = np.zeros((BASIS_SIZE, BASIS_SIZE))

                _idx_i = 0
                for i, li in enumerate(lis):
                    _idx_j = 0
                    for j, lj in enumerate(ljs):
                        idx_i = lis_idx[i]
                        idx_j = ljs_idx[j]
                        block[
                            idx_i : idx_i + 2 * li + 1, idx_j : idx_j + 2 * lj + 1
                        ] = _data[
                            _idx_i : _idx_i + 2 * li + 1, _idx_j : _idx_j + 2 * lj + 1
                        ]
                        block_mask[
                            idx_i : idx_i + 2 * li + 1, idx_j : idx_j + 2 * lj + 1
                        ] = 1
                        _idx_j += 2 * lj + 1
                    _idx_i += 2 * li + 1

                if i == j and shift == 0:
                    # onsite
                    operator_onsite.append(block)
                    operator_onsite_mask.append(block_mask)
                    onsite_index.append(i)
                else:
                    # offsite
                    edge_index.append([i, j])
                    cell_shift.append(shift)
                    operator_offsite.append(block)
                    operator_offsite_mask.append(block_mask)

        npz_data.close()

        # onsite information
        operator_onsite = np.array(operator_onsite)
        # change index
        operator_onsite = operator_onsite[:, BASIS_INDEX_ABACUS2E3NN, :][
            :, :, BASIS_INDEX_ABACUS2E3NN
        ]
        operator_onsite_mask = operator_onsite_mask[:, BASIS_INDEX_ABACUS2E3NN, :][
            :, :, BASIS_INDEX_ABACUS2E3NN
        ]
        # sort onsite blocks
        onsite_index = np.array(onsite_index)
        _onsite_index = np.arange(len(onsite_index))
        if not np.array_equal(onsite_index, _onsite_index):
            sort_indices = np.argsort(onsite_index)
            operator_onsite = operator_onsite[sort_indices]
            operator_onsite_mask = operator_onsite_mask[sort_indices]

        # offsite information
        edge_index = np.array(edge_index)
        cell_shift = np.array(cell_shift)
        operator_offsite = np.array(operator_offsite)
        # change index
        operator_offsite = operator_offsite[:, BASIS_INDEX_ABACUS2E3NN, :][
            :, :, BASIS_INDEX_ABACUS2E3NN
        ]
        operator_offsite_mask = operator_offsite_mask[:, BASIS_INDEX_ABACUS2E3NN, :][
            :, :, BASIS_INDEX_ABACUS2E3NN
        ]

        # calculate cell shift vectors
        nbr_shift = cell_shift @ cell
        # calculate edge vectors
        i = edge_index[:, 0]
        j = edge_index[:, 1]
        edge_vec = pos[j] - pos[i] + nbr_shift

        return (
            name,
            cell,
            z,
            pos,
            edge_index,
            edge_vec,
            nbr_shift,
            operator_onsite,
            operator_onsite_mask,
            operator_offsite,
            operator_offsite_mask,
        )

    def write_data(self):
        return None
