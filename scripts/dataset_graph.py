"""-----------------------------------------------------------------------------

Prepare graph file for system specific dataset to avoid IO bound when loading.

Usage: python dataset_graph.py

-----------------------------------------------------------------------------"""

import os, pathlib, multiprocessing, torch, argparse
import numpy as np
from pymatgen.io.vasp import Chgcar
from vesin import NeighborList
from torch_geometric.data import Data

from edengnn.data.io_vasp import parse_aug
from edengnn.data.dataload import get_mask_r, pos2n

DTYPE = torch.float32

parser = argparse.ArgumentParser(description="generate graph to avoid io bound")
parser.add_argument("--dir_total", help="VASP working dir for total density")
parser.add_argument(
    "--dir_sad", help="VASP working dir for superposition of atomic density"
)
parser.add_argument(
    "--filelist_train", help="Paths to sad directories for train", default=None
)
parser.add_argument(
    "--filelist_val", help="Paths to sad directories for val", default=None
)
parser.add_argument("--path_graph_train", help="Path to train graph file", default=None)
parser.add_argument("--path_graph_val", help="Path to val graph file", default=None)
parser.add_argument("--cutoff", type=float, default=4.0, help="atom graph cutoff")
parser.add_argument("--radius", type=float, default=4.0, help="probe radius")
parser.add_argument("--lmix_max", type=int, default=2, help="LMAXMIX in INCAR")
args = parser.parse_args()

# set filelist
dir_total = args.dir_total
dir_sad = args.dir_sad
filelist_train = args.filelist_train
filelist_val = args.filelist_val
path_graph_train = args.path_graph_train
path_graph_val = args.path_graph_val
# set graph structure
lmix_max = args.lmix_max
cutoff = args.cutoff
radius = args.radius


def create_graph(idx, name, dir_total, dir_sad, cutoff, radius, lmix_max):
    print(f"set {name}", flush=True)
    path_sad = os.path.join(dir_sad, name, "CHGCAR")
    path_total = os.path.join(dir_total, name, "CHGCAR")
    # load structure and density
    chgcar_sad = Chgcar.from_file(path_sad)
    structure = chgcar_sad.structure
    density_sad = chgcar_sad.data["total"] / structure.lattice.volume
    aug_lines_in = chgcar_sad.data_aug["total"]
    chgcar = Chgcar.from_file(path_total)
    density = chgcar.data["total"] / structure.lattice.volume
    aug_lines = chgcar.data_aug["total"]

    z = structure.atomic_numbers
    pos = structure.cart_coords
    cell = structure.lattice.matrix

    # use vesin to determin the neighbor list
    nl = NeighborList(cutoff=cutoff, full_list=True)
    edge_index_atoms, edge_vec_atoms, cell_shift = nl.compute(
        points=pos, box=cell, periodic=True, quantities="PDS"
    )
    nbr_shift = cell_shift @ cell  # convert to Cartesian
    grid_shape = density_sad.shape
    pos_n, grid = pos2n(cell, grid_shape, pos)
    map_probe, edge_vec_probes = get_mask_r(cell, grid_shape, radius)
    npb = len(map_probe)

    aug_tensor, aug_mask = parse_aug(aug_lines, z, lmix_max)
    aug_tensor_in, aug_mask = parse_aug(aug_lines_in, z, lmix_max)

    return idx, Data(
        z=torch.tensor(z, dtype=torch.long),
        cell=torch.tensor(np.array(cell), dtype=DTYPE),
        pos=torch.tensor(pos, dtype=DTYPE),
        edge_index=torch.LongTensor(edge_index_atoms.T),
        edge_vec_atoms=torch.tensor(edge_vec_atoms, dtype=DTYPE),
        nbr_shift=torch.tensor(nbr_shift, dtype=DTYPE),
        pos_n=torch.LongTensor(pos_n),
        grid=torch.tensor(grid, dtype=DTYPE),
        map_probe=torch.LongTensor(map_probe),
        edge_vec_probes=torch.tensor(edge_vec_probes, dtype=DTYPE),
        grid_func_out=torch.tensor(density, dtype=DTYPE),
        grid_func_in=torch.tensor(density_sad, dtype=DTYPE),
        grid_shape=torch.LongTensor(grid_shape),
        aug_mask=torch.LongTensor(aug_mask).bool().flatten(),
        aug_tensor=torch.FloatTensor(aug_tensor),
        aug_tensor_in=torch.FloatTensor(aug_tensor_in),
        name=name,
        nat=len(z),
        npb_total=len(z) * npb,
        npb=npb,
        lmix_max=lmix_max,
        volume=structure.lattice.volume,
    )


def create_file(filelist, path_save):
    names = []
    with open(filelist, "r") as f:
        dirs_sad = [line.strip() for line in f]

    for dir in dirs_sad:
        names.append(pathlib.Path(dir).stem)
    n = len(dirs_sad)
    results = [None] * n
    nproc = multiprocessing.cpu_count() - 1
    with multiprocessing.Pool(processes=nproc) as pool:
        for idx, data in pool.starmap(
            create_graph,
            [
                (
                    i,
                    names[i],
                    dir_total,
                    dir_sad,
                    cutoff,
                    radius,
                    lmix_max,
                )
                for i in range(n)
            ],
        ):
            results[idx] = data
    print("saveing......", flush=True)
    torch.save(results, path_save)


def main():
    # train
    if filelist_train is not None:
        create_file(filelist_train, path_graph_train)
    # val
    if filelist_val is not None:
        create_file(filelist_val, path_graph_val)


if __name__ == "__main__":
    main()
