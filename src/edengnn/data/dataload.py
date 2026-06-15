import torch, os, pathlib
import numpy as np
from vesin import NeighborList
from pymatgen.core import Structure
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from edengnn.data.io.vasp.parse_density import parse_aug
from edengnn.data.io.utils import get_mask_r, pos2n

"""-----------------------------------------------------------------------------

Create PyTorch dataset and dataloader.

-----------------------------------------------------------------------------"""


def get_loader(cfg, stage, io_dft):
    path = getattr(cfg.data, f"path_{stage}")
    if cfg.model.task == 1:
        dataset = AugDataset(
            cfg.data.vasp.dir,
            path,
            cfg.model.atom.edge.cutoff,
            cfg.model.probe.edge.cutoff,
            lmix_max=cfg.model.lmix_max,
            stage=stage,
            use_bin=cfg.data.use_bin,
        )
    elif cfg.model.task == 0 or cfg.model.task == 2:
        dataset = DensityDataset(
            path,
            cfg.model.atom.edge.cutoff,
            cfg.model.probe.edge.cutoff,
            io_dft=io_dft,
            stage=stage,
            dft_software=cfg.data.dft_software,
        )
    elif cfg.model.task == 3:
        dataset = OperatorDataset(path, cfg.model.atom.edge.cutoff, io_dft=io_dft)
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
    """ """

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

        self.dft_software = dft_software
        self.stage = stage
        self.dtype = torch.get_default_dtype()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]

        # load structure and density
        if self.dft_software == "vasp":
            (
                name,
                cell,
                z,
                pos,
                density,
                density_sad,
                grid_shape,
                nelec,
                volume,
                aug_tensor,
                aug_mask,
            ) = self.io_dft.read_data(path)
        else:
            name, cell, z, pos, density, grid_shape, nelec, volume = (
                self.io_dft.read_data(path)
            )

        # use vesin to determin the neighbor list
        nl = NeighborList(cutoff=self.cutoff, full_list=True)
        edge_index_atoms, edge_vec_atoms, cell_shift = nl.compute(
            points=pos, box=cell, periodic=True, quantities="PDS"
        )
        nbr_shift = cell_shift @ cell  # convert to Cartesian
        pos_n, grid = pos2n(cell, grid_shape, pos)
        map_probe, edge_vec_probes = get_mask_r(cell, grid_shape, self.radius)
        npb = len(map_probe)

        n1, n2, n3 = grid_shape
        dvolume = volume / (n1 * n2 * n3)

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
            grid_shape=torch.LongTensor(grid_shape),
            name=name,
            nat=len(z),
            npb_total=len(z) * npb,
            npb=npb,
            nelec=torch.tensor(nelec, dtype=self.dtype),
            volume=volume,
            dvolume=torch.tensor(dvolume, dtype=self.dtype),
        )

        if self.dft_software == "vasp":
            data["aug_mask"] = torch.LongTensor(aug_mask).bool().flatten()
            data["aug_tensor"] = (
                torch.tensor(aug_tensor, dtype=self.dtype)
                if (self.stage == "train" or self.stage == "val")
                else None
            )
            data["grid_func_in"] = (
                (
                    torch.tensor(density_sad, dtype=self.dtype)
                    if (self.stage == "predict")
                    else None
                ),
            )
            data["lmix_max"] = self.io_dft.lmix_max

        return data


class OperatorDataset(torch.utils.data.Dataset):
    def __init__(self, split, cutoff, io_dft):
        self.cutoff = cutoff
        self.io_dft = io_dft
        with open(split, "r") as f:
            self.paths = [line.strip() for line in f]

        self.dtype = torch.get_default_dtype()
        return None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]

        # load operator
        (
            name,
            cell,
            z,
            pos,
            edge_index_operator,
            edge_vec_operator,
            operator_onsite,
            operator_onsite_mask,
            operator_offsite,
            operator_offsite_mask,
        ) = self.io_dft.read_data(path)

        # use vesin to determin the neighbor list
        nl = NeighborList(cutoff=self.cutoff, full_list=True)
        edge_index_atoms, edge_vec_atoms, cell_shift = nl.compute(
            points=pos, box=cell, periodic=True, quantities="PDS"
        )
        nbr_shift = cell_shift @ cell  # convert to Cartesian

        return Data(
            z=torch.LongTensor(z),
            cell=torch.tensor(np.array(cell), dtype=self.dtype),
            pos=torch.tensor(pos, dtype=self.dtype),
            edge_index=torch.LongTensor(edge_index_atoms.T),
            edge_vec_atoms=torch.tensor(edge_vec_atoms, dtype=self.dtype),
            nbr_shift=torch.tensor(nbr_shift, dtype=self.dtype),
            operator_onsite=torch.tensor(operator_onsite, dtype=self.dtype),
            operator_onsite_mask=torch.LongTensor(operator_onsite_mask)
            .bool()
            .flatten(),
            operator_offsite=torch.tensor(operator_offsite, dtype=self.dtype),
            operator_offsite_mask=torch.LongTensor(operator_offsite_mask)
            .bool()
            .flatten(),
            edge_index_operator=torch.LongTensor(edge_index_operator.T),
            edge_vec_operator=torch.tensor(edge_vec_operator, dtype=self.dtype),
            name=name,
        )
