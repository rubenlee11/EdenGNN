"""-----------------------------------------------------------------------------

Abacus IO

This interface supports norm conserving pseudopotentials used by Abacus.

-----------------------------------------------------------------------------"""

import os, pathlib, seekpath
from ase.calculators.abacus import Abacus, AbacusProfile
import numpy as np
from pymatgen.core import Structure, Lattice
from pymatgen.io.ase import AseAtomsAdaptor
from edengnn.data.io.utils import BOHR, set_grid_fft
from edengnn.data.io.io_cube import io_cube
from edengnn.data.io.abacus.pseudo import PP_dict, BASIS_dict

BOHR3 = BOHR**3


class IO_Abacus:
    def __init__(
        self,
        stage="train",
        save_dir="",
        prefix="",
        path_template="",
        ecutwfc=100,  # rydberg
        dk_bz=0.35,
        dk_band=0.05,
        plot_band=True,
    ):
        self.stage = stage
        if self.stage == "predict":
            path_predict = os.path.join(save_dir, "predict")
            os.makedirs(path_predict, exist_ok=True)
            self.save_dir = path_predict
        self.ecutwfc = ecutwfc
        self.dk_bz = dk_bz
        self.dk_band = dk_band
        self.plot_band = plot_band
        self.path_template = path_template
        self.prefix = prefix

        if self.path_template is not None:
            with open(self.path_template, "r") as f:
                self.template = f.read()
        else:
            self.template = ""

    def read_data(self, path):
        name = pathlib.Path(path).stem
        if self.stage == "train":
            z, charges, cell, pos, density = _read_cube(
                os.path.join(path, f"OUT.{self.prefix}", "chgdelta.cube")
            )
            n1, n2, n3 = density.shape
            nelec = np.sum(charges)
        elif self.stage == "predict":
            structure_ = Structure.from_file(path)
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
            else:
                structure = structure_
            n1, n2, n3 = set_grid_fft(
                structure.lattice.matrix, 4.0 * self.ecutwfc / (np.pi * 2) ** 2
            )
            z = structure.atomic_numbers
            pos = structure.cart_coords
            cell = structure.lattice.matrix
            density = None
            nelec = 0.0

            self.write_input(name, structure)

        volume = np.linalg.det(cell)
        return name, cell, z, pos, density, (n1, n2, n3), nelec, volume

    def write_density(self, name, z, cell, pos, density):
        # write density
        dir_out = os.path.join(self.save_dir, name, f"OUT.{self.prefix}")
        os.makedirs(dir_out, exist_ok=True)
        path = os.path.join(dir_out, f"SPIN1_CHG.cube")
        _write_cube(path, z, np.zeros(len(z)), cell, pos, density)

    def write_input(self, name, structure):
        """
        generate input files
        """

        r_cell = structure.lattice.reciprocal_lattice.matrix

        # automatic specify uniform K grid with fixed sampling density
        Nka = np.round(np.sqrt(np.inner(r_cell[0], r_cell[0])) / self.dk_bz)
        if Nka < 1:
            Nka = 1
        Nkb = np.round(np.sqrt(np.inner(r_cell[1], r_cell[1])) / self.dk_bz)
        if Nkb < 1:
            Nkb = 1
        Nkc = np.round(np.sqrt(np.inner(r_cell[2], r_cell[2])) / self.dk_bz)
        if Nkc < 1:
            Nkc = 1

        dir_work = os.path.join(self.save_dir, name)
        # os.makedirs(dir_work, exist_ok=True)
        atoms = AseAtomsAdaptor.get_atoms(structure)
        profile = AbacusProfile(command="abacus")

        try:
            elements = set(atoms.get_chemical_symbols())
            pp = {e: PP_dict[e] for e in elements}
            basis = {e: BASIS_dict[e] for e in elements}

            calc = Abacus(
                profile=profile,
                directory=dir_work,
                pp=pp,
                basis=basis,
                calculation="scf",
                basis_type="lcao",
                out_chg=-1,
                scf_nmax=1,
                init_chg="drho",
                smearing_method="gaussian",
                smearing_sigma=0.015,
                ecutwfc=self.ecutwfc,
                kpts=[int(Nka), int(Nkb), int(Nkc)],
                suffix=self.prefix,
                out_band=1,
            )
            atoms.calc = calc
            calc.write_inputfiles(
                atoms,
                properties=["energy"],
            )
        except:
            print(f"Could not find pseudo or basis files for {name}")


def _read_cube(filename):
    """
    parse abacus cube files
    """
    natoms, nx, ny, nz = io_cube.get_cube_info(filename)
    z, charges, cell, pos, density_1d = io_cube.read_cube_data(
        filename, natoms, nx, ny, nz
    )
    cell = cell * BOHR
    pos = pos.T * BOHR
    density = density_1d.reshape((nx, ny, nz)) / BOHR3

    return z, charges, cell, pos, density


def _write_cube(filename, numbers, charges, cell, pos, density_3d):
    cell = cell / BOHR
    nx, ny, nz = density_3d.shape
    pos_f = np.asfortranarray(pos.T) / BOHR
    density_1d = np.asarray(density_3d, dtype=np.float64).ravel(order="C") * BOHR3
    io_cube.write_cube_data(
        filename, nx, ny, nz, numbers, charges, cell, pos_f, density_1d
    )
