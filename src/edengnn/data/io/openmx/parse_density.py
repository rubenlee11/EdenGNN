import math, pathlib, os, seekpath
from pymatgen.core import Structure, Lattice
import numpy as np

from edengnn.data.io.openmx.basis import spin_set, PAO_dict, PBE_dict
from edengnn.data.io.utils import BOHR
from pymatgen.core import Element

nelec_dict = np.zeros((119))
for sym, spin in spin_set.items():
    z = Element(sym).Z
    nelec_dict[z] = spin[0] + spin[1]


BOHR3 = BOHR**3


class IO_OpenMX:
    def __init__(
        self,
        stage="train",
        save_dir="",
        path_template="",
        encut=220,
        num_proc=64,
        dk_bz=0.35,
        dk_band=0.05,
        plot_band=True,
    ):
        self.stage = stage
        if self.stage == "predict":
            path_predict = os.path.join(save_dir, "predict")
            os.makedirs(path_predict, exist_ok=True)
            self.save_dir = path_predict
        self.encut = encut
        self.dk_bz = dk_bz
        self.dk_band = dk_band
        self.num_proc = num_proc
        self.plot_band = plot_band
        self.path_template = path_template

        if self.path_template is not None:
            with open(self.path_template, "r") as f:
                self.template = f.read()
        else:
            self.template = ""

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
            dat = self.template

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
                n1, n2, n3 = _set_grid((structure.lattice.matrix) / BOHR, self.encut)
                dat += self.write_dat(structure, name, n1, n2, n3)
                dat += self.write_kpath(
                    structure, sp_res["point_coords"], sp_res["path"]
                )
            else:
                structure = structure_
                # set the real space grid
                n1, n2, n3 = _set_grid((structure.lattice.matrix) / BOHR, self.encut)
                dat += self.write_dat(structure, name, n1, n2, n3)

            with open(os.path.join(path_save, f"{name}.dat"), "w") as f:
                f.write(dat)

            density = None
        else:
            structure = Structure.from_file(os.path.join(path, f"{name}.cif"))
            density, n1, n2, n3 = _read_rst(path, name)

        z = structure.atomic_numbers
        pos = structure.cart_coords
        cell = structure.lattice.matrix
        nelec = nelec_dict[structure.atomic_numbers].sum(axis=0)
        volume = np.linalg.det(cell)
        return name, cell, z, pos, density, (n1, n2, n3), nelec, volume

    def write_density(
        self,
        name,
        density,
    ):
        """
        write the restart files of difference charge density
        """

        spin = 0
        path = os.path.join(self.save_dir, name)
        path_rst = os.path.join(path, f"{name}_rst")
        os.makedirs(path_rst, exist_ok=True)

        grid_shape = density.shape
        density = (density.reshape(-1, order="C")) * BOHR3 / 2.0
        density = np.asarray(density, dtype=np.float64)
        n_grid = density.size
        chunk_size = n_grid // self.num_proc

        # charge density restart file
        for i in range(self.num_proc):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < self.num_proc - 1 else n_grid
            density_chunk = density[start:end]

            out_path = f"{path_rst}/{name}.crst{spin}_{i}_0"  # the last 0 index means the newest charge density
            with open(out_path, "wb") as f:
                density_chunk.tofile(f)

        # charge density check file
        with open(f"{path_rst}/{name}.crst_check", "w") as f:
            f.write(
                f"{self.num_proc} {grid_shape[0]} {grid_shape[1]} {grid_shape[2]} {spin}\n"
            )

    def write_dat(self, structure, name, n1, n2, n3):
        """
        generate the dat file for restart calculations in openmx
        """
        chemical_symbols = [site.specie.symbol for site in structure]
        species = list(dict.fromkeys(chemical_symbols))
        positions = structure.cart_coords
        cell = structure.lattice.matrix
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

        openmx = "System.Name                      {}\n".format(name)
        openmx += "#\n# Definition of Atomic Species\n#\n"
        openmx += f"Species.Number       {len(species)}\n"
        openmx += "<Definition.of.Atomic.Species\n"
        try:
            for s in species:
                openmx += f"{s}   {PAO_dict[s]}       {PBE_dict[s]}\n"
        except Exception as e:
            print("failed to find the element of {}".format(name))
            return

        openmx += "Definition.of.Atomic.Species>\n\n"
        openmx += "#\n# Atoms\n#\n"
        openmx += "Atoms.Number%12d" % len(chemical_symbols)
        openmx += "\nAtoms.SpeciesAndCoordinates.Unit   Ang # Ang|AU"
        openmx += "\n<Atoms.SpeciesAndCoordinates           # Unit=Ang."
        for num, sym in enumerate(chemical_symbols):
            openmx += "\n%3d  %s  %10.7f  %10.7f  %10.7f   %.2f   %.2f" % (
                num + 1,
                sym,
                *positions[num],
                *spin_set[chemical_symbols[num]],
            )
        openmx += "\nAtoms.SpeciesAndCoordinates>"
        openmx += "\nAtoms.UnitVectors.Unit             Ang #  Ang|AU"
        openmx += "\n<Atoms.UnitVectors                     # unit=Ang."
        openmx += (
            "\n      %10.7f  %10.7f  %10.7f\n      %10.7f  %10.7f  %10.7f\n      %10.7f  %10.7f  %10.7f"
            % (*cell[0], *cell[1], *cell[2])
        )
        openmx += "\nAtoms.UnitVectors>"
        openmx += "\n"
        openmx += f"scf.energycutoff              {self.encut}\n"
        openmx += "scf.Kgrid                     {:.0f}  {:.0f}  {:.0f}\n".format(
            Nka, Nkb, Nkc
        )
        openmx += f"scf.Ngrid                     {n1}  {n2}  {n3}\n"
        openmx += "scf.fixed.grid   0.0   0.0   0.0\n\n"
        return openmx

    def write_kpath(self, structure, k_coords, k_path):
        """
        write K points along high-symmetry lines
        """

        r_cell = structure.lattice.reciprocal_lattice.matrix

        lines = []
        lines.append(
            "# The high symmetry lines are generated presuming the input structure is a standard primitive cell !"
        )
        if self.plot_band:
            lines.append("Band.dispersion               on")
        else:
            lines.append("Band.dispersion               off")
        lines.append(f"Band.Nkpath                   {len(k_path)}")
        lines.append("<Band.kpath")

        for start_label, end_label in k_path:
            k1_frac = np.array(k_coords[start_label])
            k2_frac = np.array(k_coords[end_label])

            k1_cart = np.dot(k1_frac, r_cell)
            k2_cart = np.dot(k2_frac, r_cell)

            dist = np.linalg.norm(k2_cart - k1_cart)

            n_points = max(2, int(np.round(dist / self.dk_band)))

            line = f"  {n_points:<3} "
            line += f"{k1_frac[0]:8.6f} {k1_frac[1]:8.6f} {k1_frac[2]:8.6f}   "
            line += f"{k2_frac[0]:8.6f} {k2_frac[1]:8.6f} {k2_frac[2]:8.6f}   "
            line += f"{start_label} {end_label}"
            lines.append(line)

        lines.append("Band.kpath>")

        return "\n".join(lines)


def _read_rst(dir, name):
    # ------------------------------------------------------------------
    # read the real space grid
    # ------------------------------------------------------------------
    with open(os.path.join(dir, f"{name}_rst", f"{name}.crst_check"), "r") as f:
        line = f.readline().split()
        num_proc, n1, n2, n3, spin = map(int, line)
    # ------------------------------------------------------------------
    # read the difference charge density from crst files
    # ------------------------------------------------------------------
    files = [
        os.path.join(dir, f"{name}_rst", f"{name}.crst{spin}_{i}_0")
        for i in range(num_proc)
    ]
    rho_list = []
    for fname in files:
        data = np.fromfile(fname, dtype=np.float64)
        rho_list.append(data)
    # the 2.0 factor is due to OpenMX's convention
    density = (np.concatenate(rho_list)).reshape((n1, n2, n3), order="C") / BOHR3 * 2.0
    return density, n1, n2, n3


def _set_grid(cell, encut=220):
    """
    Set automatically the real space grid for integration based on the cell and
    the cutoff energy.
    """
    # encut: cutoff energy for integration in rydberg.
    cell_G = np.linalg.inv(cell.T)
    tmp = np.sqrt(encut) / np.pi
    n1 = _round_for_fft(tmp / np.linalg.norm(cell_G[0]))
    n2 = _round_for_fft(tmp / np.linalg.norm(cell_G[1]))
    n3 = _round_for_fft(tmp / np.linalg.norm(cell_G[2]))

    return n1, n2, n3


def _round_for_fft(n):
    n = math.ceil(n)
    if n % 2 != 0:
        n += 1
    while True:
        temp = n
        for p in [2, 3, 5]:
            while temp % p == 0:
                temp //= p
        if temp == 1:
            return n
        n += 2
