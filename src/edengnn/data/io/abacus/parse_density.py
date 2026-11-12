"""-----------------------------------------------------------------------------

Abacus IO

This interface supports norm conserving pseudopotentials used by Abacus.

-----------------------------------------------------------------------------"""

import os, pathlib, seekpath, json, re, shutil
import numpy as np
from pymatgen.core import Structure, Lattice
from edengnn.data.io.utils import BOHR, BOHR3, set_grid_fft
from edengnn.data.io.io_cube import io_cube


class IO_Abacus:
    def __init__(
        self,
        stage="train",
        save_dir="",
        prefix="",
        dk_bz=0.35,
        dk_band=0.05,
        plot_band=True,
        path_pp=None,
        path_input=None,
    ):
        self.stage = stage
        if self.stage == "predict":
            path_predict = os.path.join(save_dir, "predict")
            os.makedirs(path_predict, exist_ok=True)
            self.save_dir = path_predict
        self.dk_bz = dk_bz
        self.dk_band = dk_band
        self.plot_band = plot_band
        self.prefix = "EDENGNN" if self.stage == "predict" else prefix
        self.input_parameters = {}
        self.pseudo_files = {}
        if self.stage == "predict":
            self.input_parameters = _read_input(path_input)
            self.ecutwfc = float(self.input_parameters["ecutwfc"])  # rydberg
            self.path_pp = pathlib.Path(path_pp).expanduser().resolve()
            with self.path_pp.open() as f:
                self.pseudo_files = json.load(f)

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
            structure, band_path = read_structure(path, self.plot_band)
            if self.plot_band:
                self.write_kpath(
                    os.path.join(self.save_dir, name, "KPT_BAND"),
                    structure,
                    band_path["point_coords"],
                    band_path["path"],
                )
            k_grid = get_k_grid(structure, self.dk_bz)
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
        """Write ABACUS inputs with prediction-specific density settings."""
        parameters = {
            "calculation": "nscf",
            "out_band": 1,
            **self.input_parameters,
            "suffix": self.prefix,
            "basis_type": "lcao",
            "init_chg": "drho",
            "out_chg": -1,
        }
        write_input_files(
            pathlib.Path(self.save_dir) / name,
            structure,
            k_grid,
            parameters,
            self.pseudo_files,
            self.path_pp,
        )

    def write_kpath(self, path, structure, k_coords, k_path):
        write_kpath(path, structure, k_coords, k_path, self.dk_band)


def write_kpath(path, structure, k_coords, k_path, dk_band):
    """Write an ABACUS Line-mode band path with spacing in inverse angstrom."""
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
        npts = max(2, int(np.ceil(dist / dk_band)))

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


def read_structure(path, use_primitive=False):
    """Read a structure, optionally returning its standard primitive cell/path."""
    structure = Structure.from_file(path)
    if not use_primitive:
        return structure, None
    band_path = seekpath.get_path(
        (structure.lattice.matrix, structure.frac_coords, structure.atomic_numbers)
    )
    structure = Structure(
        lattice=Lattice(band_path["primitive_lattice"]),
        species=band_path["primitive_types"],
        coords=band_path["primitive_positions"],
        coords_are_cartesian=False,
    )
    return structure, band_path


def get_k_grid(structure, dk_bz):
    """Return Gamma mesh dimensions for reciprocal spacing in inverse angstrom."""
    reciprocal = structure.lattice.reciprocal_lattice.matrix
    return np.maximum(
        np.ceil(np.round(np.linalg.norm(reciprocal, axis=1) / dk_bz, 5)), 1
    ).astype(int)


def write_input_files(dir_work, structure, k_grid, parameters, pseudo_files, path_pp):
    """Write INPUT/STRU/KPT and copy UPF/orbital files into ``dir_work``.

    ``path_pp`` is the JSON path; relative species file paths resolve against
    its directory. ``parameters`` contains the calculation-specific settings.
    """
    path_pp = pathlib.Path(path_pp)
    species_sites = {}
    for site in structure:
        species_sites.setdefault(site.specie.symbol, []).append(site)
    elements = list(species_sites)
    species_files = {}
    copy_files = {}
    for element in elements:
        entry = pseudo_files.get(element)
        if not isinstance(entry, dict):
            raise ValueError(f"Missing file mapping for {element} in {path_pp}")
        files = []
        for key in ("path_pp", "path_basis"):
            value = entry.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Missing {key} for {element} in {path_pp}")
            source = pathlib.Path(value).expanduser()
            if not source.is_absolute():
                source = path_pp.parent / source
            source = source.resolve()
            if not source.is_file():
                raise FileNotFoundError(f"{element} {key}: {source}")
            copy_files[source.name] = source
            files.append(source.name)
        species_files[element] = files

    parameters = {
        **parameters,
        "ntype": len(elements),
        "pseudo_dir": ".",
        "orbital_dir": ".",
    }
    dir_work = pathlib.Path(dir_work)
    dir_work.mkdir(parents=True, exist_ok=True)
    for filename, source in copy_files.items():
        target = dir_work / filename
        if source != target.resolve():
            shutil.copy2(source, target)

    with (dir_work / "INPUT").open("w") as f:
        f.write("INPUT_PARAMETERS\n")
        for key, value in parameters.items():
            f.write(f"{key:<20} {value}\n")

    with (dir_work / "STRU").open("w") as f:
        f.write("ATOMIC_SPECIES\n")
        for element in elements:
            mass = float(species_sites[element][0].specie.atomic_mass)
            f.write(f"{element} {mass:.10f} {species_files[element][0]}\n")
        f.write("\nNUMERICAL_ORBITAL\n")
        for element in elements:
            f.write(f"{species_files[element][1]}\n")
        # ABACUS lattice vectors are scaled by LATTICE_CONSTANT in bohr.
        f.write(f"\nLATTICE_CONSTANT\n{1.0 / BOHR:.16f}\n")
        f.write("\nLATTICE_VECTORS\n")
        for vector in structure.lattice.matrix:
            f.write(" ".join(f"{value:.16f}" for value in vector) + "\n")
        f.write("\nATOMIC_POSITIONS\nDirect\n")
        for element in elements:
            sites = species_sites[element]
            f.write(f"\n{element}\n0.0\n{len(sites)}\n")
            for site in sites:
                coords = " ".join(f"{value:.16f}" for value in site.frac_coords)
                f.write(f"{coords} 1 1 1\n")

    with (dir_work / "KPT").open("w") as f:
        f.write("K_POINTS\n0\nGamma\n")
        f.write(" ".join(str(int(value)) for value in k_grid) + " 0 0 0\n")


def _read_input(path):
    """Read parameter-only ABACUS input, ignoring #, // and ! comments."""
    parameters = {}
    with open(path) as f:
        for line_number, line in enumerate(f, 1):
            line = re.split(r"#|//|!", line, maxsplit=1)[0].strip()
            if not line:
                continue
            fields = line.split(maxsplit=1)
            if len(fields) != 2:
                raise ValueError(f"{path}:{line_number}: expected a keyword and value")
            parameters[fields[0].lower()] = fields[1]
    return parameters


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
