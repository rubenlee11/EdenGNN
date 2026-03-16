"""-----------------------------------------------------------------------------

Parse CHGCAR or binary files then create dataloader

-----------------------------------------------------------------------------"""

import torch, os, re, pathlib
import numpy as np
from vesin import NeighborList
from pymatgen.io.vasp import Chgcar

from edengnn.data import io_chgcar
from pymatgen.core import Structure
from pymatgen.io.vasp.inputs import Poscar
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from edengnn.data.basis_aug import (
    AUG_BASIS,
    I1I2_IDX,
    LEN_AUG_TENSOR,
    aug_basis_dict,
    aug_basis_idx_dict,
    aug_basis_len_dict,
    pseudo_map,
)

"""-----------------------------------------------------------------------------

Create PyTorch dataset and dataloader.

Both in-memory and on-the-fly loading are supported.
On-the-fly loading is recommended when memory is bound, while in-memory loading 
is recommended when network speed is bound.

-----------------------------------------------------------------------------"""


def get_loader(cfg, stage):
    path = getattr(cfg.data, f"path_{stage}")
    if cfg.data.graphdata is True:
        dataset = GraphDataset(path)
    else:
        if cfg.model.task == 1:
            dataset = AugDataset(
                cfg.data.dir,
                path,
                cfg.model.atom.edge.cutoff,
                cfg.model.probe.edge.cutoff,
                lmix_max=cfg.model.lmix_max,
                stage=stage,
                use_bin=cfg.data.use_bin,
            )
        else:
            dataset = DensityDataset(
                cfg.data.dir,
                path,
                cfg.model.atom.edge.cutoff,
                cfg.model.probe.edge.cutoff,
                lmix_max=cfg.model.lmix_max,
                stage=stage,
                use_bin=cfg.data.use_bin,
            )
    return DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.get("num_workers", 0),
        pin_memory=True,
        persistent_workers=True if cfg.data.get("num_workers", 0) > 0 else False,
    )


class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.dataset = torch.load(path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


class AugDataset(torch.utils.data.Dataset):
    """
    When train aug seperately, set use_bin True to relieve IO
    """

    def __init__(
        self, dir, split, cutoff, radius, lmix_max=6, stage="train", use_bin=True
    ):
        self.cutoff = cutoff
        self.radius = radius

        with open(split, "r") as f:
            self.paths = [line.strip() for line in f]

        self.dir = dir
        self.lmix_max = lmix_max
        self.use_bin = use_bin
        self.stage_predict = True if stage == "predict" else False
        self.dtype = torch.get_default_dtype()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path_sad = self.paths[index]
        name = pathlib.Path(self.paths[index]).stem

        # load structure and density
        if self.use_bin:
            structure = Structure.from_file(os.path.join(path_sad, "POSCAR"))
            aug_lines_in = np.load(
                os.path.join(path_sad, "aug.npz"), allow_pickle=True
            )["data_aug"].item()["total"]
            if not self.stage_predict:
                path_scf = os.path.join(self.dir, f"{name}")
                aug_lines = np.load(
                    os.path.join(path_scf, "aug.npz"), allow_pickle=True
                )["data_aug"].item()["total"]
        else:
            chgcar_sad = Chgcar.from_file(os.path.join(path_sad, "CHGCAR"))
            structure = chgcar_sad.structure
            aug_lines_in = chgcar_sad.data_aug["total"]
            if not self.stage_predict:
                chgcar = Chgcar.from_file(os.path.join(self.dir, f"{name}", "CHGCAR"))
                aug_lines = chgcar.data_aug["total"]

        z = structure.atomic_numbers
        pos = structure.cart_coords
        cell = structure.lattice.matrix

        # use vesin to determin the neighbor list
        nl = NeighborList(cutoff=self.cutoff, full_list=True)
        edge_index_atoms, edge_vec_atoms, cell_shift = nl.compute(
            points=pos, box=cell, periodic=True, quantities="PDS"
        )
        nbr_shift = cell_shift @ cell  # convert to Cartesian

        if not self.stage_predict:
            aug_tensor, aug_mask = parse_aug(aug_lines, z, self.lmix_max)
        else:
            aug_tensor = None
        aug_tensor_in, aug_mask = parse_aug(aug_lines_in, z, self.lmix_max)

        return Data(
            z=torch.tensor(z, dtype=torch.long),
            cell=torch.tensor(np.array(cell), dtype=self.dtype),
            pos=torch.tensor(pos, dtype=self.dtype),
            edge_index=torch.LongTensor(edge_index_atoms.T),
            edge_vec_atoms=torch.tensor(edge_vec_atoms, dtype=self.dtype),
            nbr_shift=torch.tensor(nbr_shift, dtype=self.dtype),
            aug_mask=torch.LongTensor(aug_mask).bool().flatten(),
            aug_tensor=(
                torch.tensor(aug_tensor, dtype=self.dtype)
                if not self.stage_predict
                else None
            ),
            aug_tensor_in=torch.tensor(aug_tensor_in, dtype=self.dtype),
            npb_total=0,
            name=name,
            nat=len(z),
            lmix_max=self.lmix_max,
            volume=structure.lattice.volume,
        )


class DensityDataset(torch.utils.data.Dataset):
    def __init__(
        self, dir, split, cutoff, radius, lmix_max=6, stage="train", use_bin=False
    ):
        self.cutoff = cutoff
        self.radius = radius

        with open(split, "r") as f:
            self.paths = [line.strip() for line in f]

        self.dir = dir
        self.lmix_max = lmix_max
        self.use_bin = use_bin
        self.stage_predict = True if stage == "predict" else False
        self.dtype = torch.get_default_dtype()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path_sad = self.paths[index]
        name = pathlib.Path(self.paths[index]).stem

        # load structure and density
        if self.use_bin:
            structure = Structure.from_file(os.path.join(path_sad, "POSCAR"))
            density_sad = (
                np.load(os.path.join(path_sad, "density.npy"))
                / structure.lattice.volume
            )
            aug_lines_in = np.load(
                os.path.join(path_sad, "aug.npz"), allow_pickle=True
            )["data_aug"].item()["total"]
            if not self.stage_predict:
                path_scf = os.path.join(self.dir, f"{name}")
                density = (
                    np.load(os.path.join(path_scf, "density.npy"))
                    / structure.lattice.volume
                )
                aug_lines = np.load(
                    os.path.join(path_scf, "aug.npz"), allow_pickle=True
                )["data_aug"].item()["total"]
        else:
            chgcar_sad = Chgcar.from_file(os.path.join(path_sad, "CHGCAR"))
            structure = chgcar_sad.structure
            density_sad = chgcar_sad.data["total"] / structure.lattice.volume
            aug_lines_in = chgcar_sad.data_aug["total"]
            if not self.stage_predict:
                chgcar = Chgcar.from_file(os.path.join(self.dir, f"{name}", "CHGCAR"))
                density = chgcar.data["total"] / structure.lattice.volume
                aug_lines = chgcar.data_aug["total"]

        z = structure.atomic_numbers
        pos = structure.cart_coords
        cell = structure.lattice.matrix

        # use vesin to determin the neighbor list
        nl = NeighborList(cutoff=self.cutoff, full_list=True)
        edge_index_atoms, edge_vec_atoms, cell_shift = nl.compute(
            points=pos, box=cell, periodic=True, quantities="PDS"
        )
        nbr_shift = cell_shift @ cell  # convert to Cartesian
        grid_shape = density_sad.shape
        pos_n, grid = pos2n(cell, grid_shape, pos)
        map_probe, edge_vec_probes = get_mask_r_optimized(cell, grid_shape, self.radius)
        npb = len(map_probe)

        if not self.stage_predict:
            aug_tensor, aug_mask = parse_aug(aug_lines, z, self.lmix_max)
        else:
            aug_tensor = None
        aug_tensor_in, aug_mask = parse_aug(aug_lines_in, z, self.lmix_max)

        return Data(
            z=torch.tensor(z, dtype=torch.long),
            cell=torch.tensor(np.array(cell), dtype=self.dtype),
            pos=torch.tensor(pos, dtype=self.dtype),
            edge_index=torch.LongTensor(edge_index_atoms.T),
            edge_vec_atoms=torch.tensor(edge_vec_atoms, dtype=self.dtype),
            nbr_shift=torch.tensor(nbr_shift, dtype=self.dtype),
            pos_n=torch.LongTensor(pos_n),
            grid=torch.tensor(grid, dtype=self.dtype),
            map_probe=torch.LongTensor(map_probe),
            edge_vec_probes=torch.tensor(edge_vec_probes, dtype=self.dtype),
            grid_func_out=(
                torch.tensor(density, dtype=self.dtype)
                if not self.stage_predict
                else None
            ),
            grid_func_in=(torch.tensor(density_sad, dtype=self.dtype)),
            grid_shape=torch.LongTensor(grid_shape),
            aug_mask=torch.LongTensor(aug_mask).bool().flatten(),
            aug_tensor=(
                torch.tensor(aug_tensor, dtype=self.dtype)
                if not self.stage_predict
                else None
            ),
            aug_tensor_in=torch.tensor(aug_tensor_in, dtype=self.dtype),
            name=name,
            nat=len(z),
            npb_total=len(z) * npb,
            npb=npb,
            lmix_max=self.lmix_max,
            volume=structure.lattice.volume,
        )


"""-----------------------------------------------------------------------------

The following functions are not optimally efficient, they are used to specify
probe index when we are not using DM formulation. Future development is to use a
unified C function for both DM and non-DM formulation.

-----------------------------------------------------------------------------"""


def get_mask_r_optimized(cell, grid_shape, radius=4.0):
    """
    reminicent of setting k grid in PW DFT. use the same technique to bound the mask index

    cell[0,:] [1,:], [2,:] is a1, a2, a3 vectors
    grid_shape: (nx, ny, nz) is the number of grid points in each direction
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

    # pos_res = pos - n_grid @ grid
    # return n_grid, pos_res

    return n_grid, grid


"""-----------------------------------------------------------------------------

Read and parse augmentation occupancies. 

I abandoned varied LMAX_MIX for structures in dataset, the sampling probability 
of large L might be too low.

-----------------------------------------------------------------------------"""


def aug2irreps(
    z,
    aug_vasp,
    lmix_max,
):
    """-------------------------------------------------------------------------
    Description:

        a translation of RETRIEVE_RHOLM in VASP source code

        e3nn calls in spherical harmonics as (y, z, x)
        e3nn and VASP use the same convention for spherical tensor:
            l = 1:  -1, 0, 1
            l = 2: -2, -1, 0, 1, 2
            ......
        consistent with the real spherical harmonics defined in
        https://en.wikipedia.org/wiki/Table_of_spherical_harmonics

        VASP only outputs the irreducible representations of the upper triangle part
        of the augmentation occupancy matrix in the basis of PAW projectors, with
        L larger than Lmix_max filled with 0.

    Variables:
        aug_basis_idx: [n_basis]
            index of l in AUG_BASIS. For example, we have AUG_BASIS
            [0,0,0,1,1,2,2,3,3], Si has aug_basis [0,0,1,1], then its aug_basis_idx is
            [0,1,3,4]
        I1I2_IDX: [N_AUG_BASIS][N_AUG_BASIS]
            map from AUG_BASIS index to irreps tensor start position

        jbase:
            counter for haw many element have been embedded into aug_tensor
    -------------------------------------------------------------------------"""

    aug_basis = aug_basis_dict[pseudo_map[z]]
    aug_basis_idx = aug_basis_idx_dict[pseudo_map[z]]

    aug_tensor_compact = []
    ibase = 0
    jbase = 0

    for i, l_i in enumerate(aug_basis):
        for l_j in aug_basis[i:]:

            l_min = abs(l_j - l_i)
            l_max = min(abs(l_i + l_j), lmix_max)

            jbase = ibase
            for lmain in range(l_min, l_max + 1, 2):
                aug_tensor_compact.extend(aug_vasp[jbase : jbase + 2 * lmain + 1])
                jbase += 2 * lmain + 1

            for lmain in range(l_min, abs(l_i + l_j) + 1, 2):
                ibase += 2 * lmain + 1

    aug_tensor = np.zeros(LEN_AUG_TENSOR)
    aug_mask = np.zeros(LEN_AUG_TENSOR)
    jbase = 0
    for i, li_idx in enumerate(aug_basis_idx):
        for lj_idx in aug_basis_idx[i:]:
            li = AUG_BASIS[li_idx]
            lj = AUG_BASIS[lj_idx]
            start = I1I2_IDX[li_idx][lj_idx]

            l_min = abs(li - lj)
            l_max = min(abs(li + lj), lmix_max)

            basis_pointer = 0
            for lmain in range(l_min, l_max + 1, 2):
                aug_tensor[
                    start + basis_pointer : start + basis_pointer + 2 * lmain + 1
                ] = aug_tensor_compact[jbase : jbase + 2 * lmain + 1]
                aug_mask[
                    start + basis_pointer : start + basis_pointer + 2 * lmain + 1
                ] = 1
                basis_pointer += 2 * lmain + 1
                jbase += 2 * lmain + 1
    return aug_tensor, aug_mask


def irreps2aug(aug_tensor_list, z, lmix_max=2):
    """-------------------------------------------------------------------------

    Description:

        Inverse of aug2irreps

    -------------------------------------------------------------------------"""

    aug_list = []
    nele_list = []
    for i, z_atom in enumerate(z):
        aug_tensor = aug_tensor_list[i]

        pseudo = pseudo_map[z_atom]
        aug_basis_idx = aug_basis_idx_dict[pseudo]
        aug_vasp_len = aug_basis_len_dict[pseudo]

        aug_vasp = [0] * aug_vasp_len

        ibase = 0
        jbase = 0
        for i, li_idx in enumerate(aug_basis_idx):
            for lj_idx in aug_basis_idx[i:]:
                li = AUG_BASIS[li_idx]
                lj = AUG_BASIS[lj_idx]
                start = I1I2_IDX[li_idx][lj_idx]

                l_min = abs(li - lj)
                l_max = min(abs(li + lj), lmix_max)

                jbase = 0
                for lmain in range(l_min, l_max + 1, 2):
                    aug_vasp[ibase + jbase : ibase + jbase + 2 * lmain + 1] = (
                        aug_tensor[start + jbase : start + jbase + 2 * lmain + 1]
                    )
                    jbase += 2 * lmain + 1

                for lmain in range(l_min, abs(li + lj) + 1, 2):
                    ibase += 2 * lmain + 1
        aug_list.extend(aug_vasp)
        nele_list.append(len(aug_vasp))
    return aug_list, nele_list


def parse_aug(lines, z, lmix_max=2):
    # parse aug. pymatgen has bug reading aug, I need to modify a little bit
    aug_num = []
    aug_list_str = []
    augs_str = None
    aug_re = r"augmentation\s+occupancies\s*(\d+)\s+(\d+)"
    for line in lines:
        if line.startswith("augment"):
            m = re.search(aug_re, line)
            aug_num.append(int(m.group(2)))
            if augs_str:
                aug_list_str.append(augs_str)
            augs_str = []
        else:
            augs_str.append(line.strip())
    aug_list_str.append(augs_str)
    aug_list = []
    for i, aug_str in enumerate(aug_list_str):
        aug = []
        for line in aug_str:
            aug.extend(map(float, line.split()))
        aug_list.append(np.array(aug[0 : aug_num[i]]))

    # reshape aug into irreps
    aug_tensor_list = []
    aug_mask_list = []
    for i in range(len(z)):
        aug_tensor, aug_mask = aug2irreps(z[i], aug_list[i], lmix_max)
        aug_tensor_list.append(aug_tensor)
        aug_mask_list.append(aug_mask)
    return np.array(aug_tensor_list), np.array(aug_mask_list)


"""-----------------------------------------------------------------------------



-----------------------------------------------------------------------------"""


def write_chgcar(save_dir, name, aug_tensor, density, z, pos, cell, volume, lmix_max):
    lmix_max = int(lmix_max)
    path = os.path.join(save_dir, name)
    structure = Structure(
        lattice=cell,
        species=z,
        coords=pos,
        coords_are_cartesian=True,
    )
    poscar = Poscar(structure)

    if density is None:
        pass
    else:
        # write header
        with open(path, "w") as f:
            lines = f"CHGCAR generated by E3SR\n"
            lines += "   1.00000000000000\n"
            for vec in structure.lattice.matrix:
                lines += f" {vec[0]:12.6f}{vec[1]:12.6f}{vec[2]:12.6f}\n"
            lines += "".join(f"{s:5}" for s in poscar.site_symbols) + "\n"
            lines += "".join(f"{x:6}" for x in poscar.natoms) + "\n"
            lines += "Direct\n"
            for site in structure:
                a, b, c = site.frac_coords
                lines += f"{a:10.6f}{b:10.6f}{c:10.6f}\n"
            lines += " \n"
            f.write(lines)

            nx, ny, nz = density.shape
            f.write(f"   {nx}   {ny}   {nz}\n")

        # write density
        data = (density * volume).flatten(order="F")
        io_chgcar.write_density(data, nx * ny * nz, path)

    # write augmentation occupancy
    if aug_tensor is None:
        pass
    else:
        aug_list, nele_list = irreps2aug(aug_tensor, z, lmix_max)
        io_chgcar.write_aug(aug_list, path, nele_list)

    return None
