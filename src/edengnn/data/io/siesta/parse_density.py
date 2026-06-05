"""-----------------------------------------------------------------------------

Siesta IO

-----------------------------------------------------------------------------"""

import os, pathlib, seekpath
import numpy as np
from pymatgen.core import Structure, Lattice
import netCDF4 as nc
from edengnn.data.io.utils import BOHR, BOHR3, set_grid_lcao


class IO_Siesta:
    def __init__(
        self,
        stage="train",
        save_dir="",
        prefix="",
        filename_out="",
        meshcutoff=300,  # rydberg
        dk_bz=0.35,
        dk_band=0.05,
        plot_band=True,
    ):
        self.stage = stage
        if self.stage == "predict":
            path_predict = os.path.join(save_dir, "predict")
            os.makedirs(path_predict, exist_ok=True)
            self.save_dir = path_predict
        self.filename_out = filename_out
        self.meshcutoff = meshcutoff
        self.dk_bz = dk_bz
        self.dk_band = dk_band
        self.plot_band = plot_band
        self.prefix = prefix

    def read_data(self, path):
        name = pathlib.Path(path).stem
        # ----------------------------------------------------------------------
        # read structure
        # ----------------------------------------------------------------------
        if self.stage == "predict":
            structure_ = Structure.from_file(path)
            # ------------------------------------------------------------------
            # write structure
            # ------------------------------------------------------------------
            path_save = os.path.join(self.save_dir, name)
            os.makedirs(path_save, exist_ok=True)

            if self.plot_band:
                # --------------------------------------------------------------
                # use the standard primitive cell
                # --------------------------------------------------------------
                cell = structure_.lattice.matrix
                positions = structure_.frac_coords
                numbers = [site.specie.number for site in structure_]
                sp_res = seekpath.get_path(
                    (cell, positions, numbers),
                )
                structure = Structure(
                    lattice=Lattice(sp_res["primitive_lattice"]),
                    species=sp_res["primitive_types"],
                    coords=sp_res["primitive_positions"],
                    coords_are_cartesian=False,
                )
                # set the real space grid
                n1, n2, n3 = set_grid_lcao(
                    (structure.lattice.matrix) / BOHR, self.encut
                )

            else:
                structure = structure_
                # set the real space grid
                n1, n2, n3 = set_grid_lcao(
                    (structure.lattice.matrix) / BOHR, self.encut
                )

            density = None
            nelec = 0.0
        else:
            z, cell, pos = _read_xv(os.path.join(path, f"{self.prefix}.XV"))
            density = _read_nc(os.path.join(path, "DeltaRho.grid.nc"))
            nelec = _read_elec(os.path.join(path, self.filename_out))
            n1, n2, n3 = density.shape

        volume = np.linalg.det(cell)

        return name, cell, z, pos, density, (n1, n2, n3), nelec, volume

    def write_density():
        return None

    def write_input():
        return None


def _read_elec(path):
    # read number of valence electrons
    with open(path, "r") as f:
        for line in f:
            if "Total number of electrons" in line:
                return float(line.split()[-1])


def _read_xv(path):
    cell = np.loadtxt(path, max_rows=3, usecols=(0, 1, 2)) * BOHR
    atomic_data = np.loadtxt(path, skiprows=4)
    z = atomic_data[:, 1].astype(int)
    pos = atomic_data[:, 2:5] * BOHR
    return z, cell, pos


def _read_nc(path):
    # Open the NetCDF file in read-only mode
    dataset = nc.Dataset(path, "r")
    # cell = np.array(dataset.variables['cell'][:]) * BOHR

    gridfunc = dataset.variables["gridfunc"][:]
    gridfunc = np.array(gridfunc)
    dataset.close()

    charge_density = np.transpose(gridfunc[0, :, :, :], (2, 1, 0)) / BOHR**3

    return charge_density
