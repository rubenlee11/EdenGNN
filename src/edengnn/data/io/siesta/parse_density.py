"""-----------------------------------------------------------------------------

Siesta IO

-----------------------------------------------------------------------------"""

import os, pathlib, seekpath
import numpy as np
from pymatgen.core import Structure, Lattice
import netCDF4 as nc
from edengnn.data.io.utils import BOHR, BOHR3, set_grid_lcao
from ase.calculators.siesta import Siesta
from pymatgen.io.ase import AseAtomsAdaptor
from ase.units import Ry


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
                block_band = self.write_kpath(
                    structure,
                    sp_res["point_coords"],
                    sp_res["path"],
                )
                # set the real space grid
                n1, n2, n3 = set_grid_lcao(
                    (structure.lattice.matrix) / BOHR, self.meshcutoff
                )

            else:
                structure = structure_
                # set the real space grid
                n1, n2, n3 = set_grid_lcao(
                    (structure.lattice.matrix) / BOHR, self.meshcutoff
                )

            # write input file
            r_cell = structure.lattice.reciprocal_lattice.matrix
            k_grid = np.maximum(
                np.round(np.linalg.norm(r_cell, axis=1) / self.dk_bz), 1
            ).astype(int)
            self.write_input(name, structure, k_grid)
            if self.plot_band:
                with open(os.path.join(path_save, f"{name}.fdf"), "a") as f:
                    f.write("\n")
                    f.write(block_band)
                    f.write("\n")

            # get structure info
            z = structure.atomic_numbers
            pos = structure.cart_coords
            cell = structure.lattice.matrix
            density = None
            nelec = 0.0

        else:
            z, cell, pos = _read_xv(os.path.join(path, f"{self.prefix}.XV"))
            density = _read_nc(os.path.join(path, "DeltaRho.grid.nc"))
            nelec = _read_elec(os.path.join(path, self.filename_out))
            n1, n2, n3 = density.shape

        volume = np.linalg.det(cell)

        return name, cell, z, pos, density, (n1, n2, n3), nelec, volume

    def write_density(self, name, cell, density):
        path = os.path.join(self.save_dir, name, "DeltaRho.IN.grid.nc")
        _write_nc(path, cell, density)

    def write_input(self, name, structure, k_grid):
        """
        generate input files
        """

        dir_work = os.path.join(self.save_dir, name)
        atoms = AseAtomsAdaptor.get_atoms(structure)

        calc = Siesta(
            directory=dir_work,
            label=name,
            xc="PBE",
            mesh_cutoff=self.meshcutoff * Ry,
            energy_shift=0.01 * Ry,
            kpts=k_grid,
            fdf_arguments={
                "MaxSCFIterations": 1,
                "SolutionMethod": "diagon",
                "SCF.Read.Deformation.Charge.NetCDF": True,
                "SCFMustConverge": False,
                "Write.DM": False,
            },
        )
        atoms.calc = calc
        calc.write_input(atoms, properties=["energy"])

        # except:
        #    print(f"Could not find pseudo or basis files for {name}")

    def write_kpath(self, structure, k_coords, k_path):
        """
        Write K points along high-symmetry lines for SIESTA %block BandLines
        """
        r_cell = structure.lattice.reciprocal_lattice.matrix
        lines = []
        lines.append(
            "# The high symmetry lines are generated presuming the input structure is a standard primitive cell !"
        )

        # Use fractional coordinates (reciprocal lattice vectors)
        lines.append("BandLinesScale ReciprocalLatticeVectors")
        lines.append("%block BandLines")

        last_end_label = None

        for start_label, end_label in k_path:
            k1_frac = np.array(k_coords[start_label])
            k2_frac = np.array(k_coords[end_label])

            # Calculate Cartesian distance to determine number of points
            k1_cart = np.dot(k1_frac, r_cell)
            k2_cart = np.dot(k2_frac, r_cell)
            dist = np.linalg.norm(k2_cart - k1_cart)
            n_points = max(2, int(np.round(dist / self.dk_band)))

            # Format labels (replace 'GAMMA' with '\Gamma' for better plotting compatibility if desired)
            s_label = r"\Gamma" if start_label.upper() == "GAMMA" else start_label
            e_label = r"\Gamma" if end_label.upper() == "GAMMA" else end_label

            # If this is the first path segment or there is a break in the path
            if start_label != last_end_label:
                # A '1' indicates the start of a new path sequence in SIESTA
                line = f"    1  {k1_frac[0]:8.5f} {k1_frac[1]:8.5f} {k1_frac[2]:8.5f}  {s_label}"
                lines.append(line)

            # Write the end point of the current segment with the calculated number of points
            line = f"  {n_points:<3}  {k2_frac[0]:8.5f} {k2_frac[1]:8.5f} {k2_frac[2]:8.5f}  {e_label}"
            lines.append(line)

            # Update the last end label to check for continuity in the next iteration
            last_end_label = end_label

        lines.append("%endblock BandLines")

        return "\n".join(lines)


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


def _write_nc(filename, cell, density_3d):
    # density_3d has shape (n1, n2, n3)
    n1, n2, n3 = density_3d.shape

    # Convert density units back to Bohr^-3 and transpose axes to (n3, n2, n1)
    gridfunc_3d = np.transpose(density_3d * (BOHR**3), (2, 1, 0))

    # Add the spin dimension to form a 4D array (1, n3, n2, n1)
    gridfunc_4d = gridfunc_3d[np.newaxis, :, :, :]

    # Open the NetCDF file in write mode with NETCDF3_CLASSIC format
    dataset = nc.Dataset(filename, "w", format="NETCDF3_CLASSIC")

    # Create dimensions exactly as Siesta expects
    dataset.createDimension("xyz", 3)
    dataset.createDimension("abc", 3)
    dataset.createDimension("spin", 1)
    dataset.createDimension("n1", n1)
    dataset.createDimension("n2", n2)
    dataset.createDimension("n3", n3)

    # Create variables with float32 (f4) data type
    var_cell = dataset.createVariable("cell", "f4", ("abc", "xyz"))
    var_gridfunc = dataset.createVariable("gridfunc", "f4", ("spin", "n3", "n2", "n1"))

    var_cell[:] = cell / BOHR
    var_gridfunc[:] = gridfunc_4d

    # Close the dataset
    dataset.close()
