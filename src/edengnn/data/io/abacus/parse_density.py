"""-----------------------------------------------------------------------------

Abacus IO

This interface supports norm conserving pseudopotentials used by Abacus.

-----------------------------------------------------------------------------"""

import os, pathlib, seekpath
from ase.calculators.abacus import Abacus, AbacusProfile
import numpy as np
from pymatgen.core import Structure, Lattice
from pymatgen.io.ase import AseAtomsAdaptor
from edengnn.data.io.utils import BOHR, BOHR3, set_grid_fft
from edengnn.data.io.io_cube import io_cube
from edengnn.data.io.abacus.pseudo import PP_dict, BASIS_dict


class IO_Abacus:
    def __init__(
        self,
        stage="train",
        save_dir="",
        prefix="",
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
        self.prefix = prefix

    def read_data(self, path):
        name = pathlib.Path(path).stem
        if self.stage == "train":
            z, charges, cell, pos, density = _read_cube(
                os.path.join(path, f"OUT.{self.prefix}", "chgdelta.cube")
            )
            n1, n2, n3 = density.shape
            nelec = np.sum(charges)
        elif self.stage == "predict":
            os.makedirs(os.path.join(self.save_dir, name), exist_ok=True)
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
                self.write_kpath(
                    os.path.join(self.save_dir, name, "KPT_BAND"),
                    structure,
                    sp_res["point_coords"],
                    sp_res["path"],
                )
            else:
                structure = structure_
            # write kpt
            r_cell = structure.lattice.reciprocal_lattice.matrix
            k_grid = np.maximum(
                np.round(np.linalg.norm(r_cell, axis=1) / self.dk_bz), 1
            ).astype(int)
            self.write_input(name, structure, k_grid)

            n1, n2, n3 = set_grid_fft(
                structure.lattice.matrix, 4.0 * self.ecutwfc / (np.pi * 2) ** 2
            )
            z = structure.atomic_numbers
            pos = structure.cart_coords
            cell = structure.lattice.matrix
            density = None
            nelec = 0.0

        volume = np.linalg.det(cell)
        return name, cell, z, pos, density, (n1, n2, n3), nelec, volume

    def write_density(self, name, z, cell, pos, density):
        # write density
        dir_out = os.path.join(self.save_dir, name, f"OUT.{self.prefix}")
        os.makedirs(dir_out, exist_ok=True)
        path = os.path.join(dir_out, f"SPIN1_CHG.cube")
        _write_cube(path, z, np.zeros(len(z)), cell, pos, density)

    def write_input(self, name, structure, k_grid):
        """
        generate input files
        """

        dir_work = os.path.join(self.save_dir, name)
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
                calculation="nscf",
                basis_type="lcao",
                out_chg=-1,
                init_chg="drho",
                scf_thr=1e-6,
                smearing_method="gaussian",
                smearing_sigma=0.015,
                ecutwfc=self.ecutwfc,
                suffix=self.prefix,
                out_band=1,
                kpts=k_grid,
            )
            atoms.calc = calc
            calc.write_inputfiles(
                atoms,
                properties=["energy"],
            )
            # write KPT file
        except:
            print(f"Could not find pseudo or basis files for {name}")

    def write_kpath(self, path, structure, k_coords, k_path):
        kpts_lines = []
        kpt_labels = []
        r_cell = structure.lattice.reciprocal_lattice.matrix
        for i, (start_label, end_label) in enumerate(k_path):
            k1_frac = np.array(k_coords[start_label])
            k2_frac = np.array(k_coords[end_label])

            # Calculate distance in reciprocal space
            k1_cart = np.dot(k1_frac, r_cell)
            k2_cart = np.dot(k2_frac, r_cell)
            dist = np.linalg.norm(k2_cart - k1_cart)

            # Determine number of points for this segment
            npts = max(2, int(np.ceil(dist / self.dk_band)))

            if i == 0:
                kpts_lines.append(
                    [k1_frac[0], k1_frac[1], k1_frac[2], npts, start_label]
                )
                kpt_labels.append(start_label)
            else:
                prev_end_label = k_path[i - 1][1]
                if start_label != prev_end_label:
                    # Handle path jump: set npts of the previous point to 0
                    kpts_lines[-1][3] = 0
                    kpts_lines.append(
                        [k1_frac[0], k1_frac[1], k1_frac[2], npts, start_label]
                    )
                    kpt_labels.append(start_label)

            kpts_lines.append([k2_frac[0], k2_frac[1], k2_frac[2], npts, end_label])
            kpt_labels.append(end_label)

        # The last k-point in ABACUS Line mode should have npts=1
        kpts_lines[-1][3] = 1
        with open(path, "w") as f:
            f.write("K_POINTS\n")
            f.write(f"{len(kpts_lines)}\n")
            f.write("Line\n")
            for kpt in kpts_lines:
                f.write(
                    f"{kpt[0]:.6f} {kpt[1]:.6f} {kpt[2]:.6f} {kpt[3]} // {kpt[4]}\n"
                )


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
