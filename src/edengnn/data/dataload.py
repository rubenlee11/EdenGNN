import torch, os, pathlib
import numpy as np
from vesin import NeighborList
from pymatgen.io.vasp import Chgcar
from pymatgen.core import Structure
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from edengnn.data.io_vasp import parse_aug

"""-----------------------------------------------------------------------------

Create PyTorch dataset and dataloader.

Both in-memory and on-the-fly loading are supported.
On-the-fly loading is recommended when memory is bound, while in-memory loading 
is recommended when network speed is bound.

-----------------------------------------------------------------------------"""


def get_loader(cfg, stage, io_dft):
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
                io_dft=io_dft,
                stage=stage,
                dft_software=cfg.data.dft_software,
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
        path = self.paths[index]
        name = pathlib.Path(self.paths[index]).stem

        # load structure and density
        if self.use_bin:
            structure = Structure.from_file(os.path.join(path, "POSCAR"))
            aug_lines_in = np.load(os.path.join(path, "aug.npz"), allow_pickle=True)[
                "data_aug"
            ].item()["total"]
            if not self.stage_predict:
                path_scf = os.path.join(self.dir, f"{name}")
                aug_lines = np.load(
                    os.path.join(path_scf, "aug.npz"), allow_pickle=True
                )["data_aug"].item()["total"]
        else:
            chgcar_sad = Chgcar.from_file(os.path.join(path, "CHGCAR"))
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
        self,
        dir,
        split,
        cutoff,
        radius,
        io_dft,
        stage="train",
        dft_software="vasp",
    ):
        self.cutoff = cutoff
        self.radius = radius
        self.io_dft = io_dft

        with open(split, "r") as f:
            self.paths = [line.strip() for line in f]

        self.dir = dir
        self.dft_software = dft_software
        self.stage = stage
        self.dtype = torch.get_default_dtype()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]

        # load structure and density
        if self.dft_software == "vasp":
            name, structure, density_in, density, aug_tensor, aug_mask = (
                self.io_dft.read_data(path)
            )
        elif self.dft_software == "openmx":
            name, structure, density_in, density = self.io_dft.read_data(path)

        z = structure.atomic_numbers
        pos = structure.cart_coords
        cell = structure.lattice.matrix

        # use vesin to determin the neighbor list
        nl = NeighborList(cutoff=self.cutoff, full_list=True)
        edge_index_atoms, edge_vec_atoms, cell_shift = nl.compute(
            points=pos, box=cell, periodic=True, quantities="PDS"
        )
        nbr_shift = cell_shift @ cell  # convert to Cartesian
        grid_shape = density_in.shape
        pos_n, grid = pos2n(cell, grid_shape, pos)
        map_probe, edge_vec_probes = get_mask_r(cell, grid_shape, self.radius)
        npb = len(map_probe)

        data = Data(
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
                if (self.stage == "train" or self.stage == "val")
                else None
            ),
            grid_func_in=(torch.tensor(density_in, dtype=self.dtype)),
            grid_shape=torch.LongTensor(grid_shape),
            name=name,
            nat=len(z),
            npb_total=len(z) * npb,
            npb=npb,
            volume=structure.lattice.volume,
        )

        if self.dft_software == "vasp":
            data["aug_mask"] = torch.LongTensor(aug_mask).bool().flatten()
            data["aug_tensor"] = (
                torch.tensor(aug_tensor, dtype=self.dtype)
                if (self.stage == "train" or self.stage == "val")
                else None
            )
            data["lmix_max"] = self.io_dft.lmix_max

        return data


def get_mask_r(cell, grid_shape, radius=4.0):
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
