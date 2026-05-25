"""-----------------------------------------------------------------------------

VASP IO

I abandoned varied LMAX_MIX for structures in dataset, the sampling probability
of large L might be too low.

-----------------------------------------------------------------------------"""

import os, re, pathlib
from pymatgen.core import Structure
import numpy as np
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.core import Element
from edengnn.data.io.io_chgcar import io_chgcar
from edengnn.data.io.vasp.basis import (
    AUG_BASIS,
    I1I2_IDX,
    LEN_AUG_TENSOR,
    aug_basis_dict,
    aug_basis_idx_dict,
    aug_basis_len_dict,
    pseudo_map,
    nelec_dict,
)


class IO_VASP:
    def __init__(
        self,
        stage="train",
        save_dir="",
        dir="",
        use_bin=False,
        path_template="",
        encut=220,
        lmix_max=6,
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
        self.lmix_max = lmix_max
        self.plot_band = plot_band
        self.path_template = path_template
        self.dir = dir
        self.use_bin = use_bin

        if self.path_template is not None:
            with open(self.path_template, "r") as f:
                self.incar_head = f.read()
        else:
            self.incar_head = ""
        return

    def read_data(self, path):
        name = pathlib.Path(path).stem
        """
        if self.use_bin:
            structure = Structure.from_file(os.path.join(path, "POSCAR"))
            density_in = (
                np.load(os.path.join(path, "density.npy")) / structure.lattice.volume
            )
            aug_lines_in = np.load(os.path.join(path, "aug.npz"), allow_pickle=True)[
                "data_aug"
            ].item()["total"]
            if self.stage == "train":
                path_scf = os.path.join(self.dir, f"{name}")
                density = (
                    np.load(os.path.join(path_scf, "density.npy"))
                    / structure.lattice.volume
                )
                aug_lines = np.load(
                    os.path.join(path_scf, "aug.npz"), allow_pickle=True
                )["data_aug"].item()["total"]

            z = structure.atomic_numbers
        else:
            chgcar_sad = Chgcar.from_file(os.path.join(path, "CHGCAR"))
            structure = chgcar_sad.structure
            z = structure.atomic_numbers
            density_in = chgcar_sad.data["total"] / structure.lattice.volume
            aug_lines_in = chgcar_sad.data_aug["total"]
            aug_tensor_in, aug_mask = parse_aug(aug_lines_in, z, self.lmix_max)
            aug_tensor = None
            density = None
            if self.stage == "train":
                chgcar = Chgcar.from_file(os.path.join(self.dir, f"{name}", "CHGCAR"))
                density = chgcar.data["total"] / structure.lattice.volume
                aug_lines = chgcar.data_aug["total"]
                aug_tensor, aug_mask = parse_aug(aug_lines, z, self.lmix_max)

        pos = structure.cart_coords
        cell = structure.lattice.matrix
        nelec = sum(nelec_dict[pseudo_map[z]] for z in structure.atomic_numbers)
        """
        if self.stage == "train":
            path_td = os.path.join(self.dir, f"{name}", "CHGCAR")
            z, cell, pos, volume, density_td, aug_str = _read_chgcar(path_td)
            z, cell, pos, volume, density_sad, _aug_str = _read_chgcar(
                os.path.join(path, "CHGCAR")
            )

            density = density_td - density_sad
            n1, n2, n3 = density.shape
            nelec = np.sum(density_sad) * volume / (n1 * n2 * n3)

            aug_tensor, aug_mask = parse_aug(aug_str, z, self.lmix_max)

        elif self.stage == "predict":
            z, cell, pos, volume, density_sad, aug_str = _read_chgcar(
                os.path.join(path, "CHGCAR")
            )
            aug_tensor, aug_mask = parse_aug(aug_str, z, self.lmix_max)
            nelec = 0.0
            n1, n2, n3 = density_sad.shape
            density = None
        return (
            name,
            cell,
            z,
            pos,
            density,
            density_sad,
            (n1, n2, n3),
            nelec,
            volume,
            aug_tensor,
            aug_mask,
        )

    def write_density(self, name, aug_tensor, density, z, pos, cell, volume):
        path = os.path.join(self.save_dir, name)
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
                lines = f"CHGCAR generated by EdenGNN\n"
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

                n1, n2, n3 = density.shape
                f.write(f"   {n1}   {n2}   {n3}\n")

            # write density
            data = (density * volume).flatten(order="F")
            io_chgcar.write_density(data, n1 * n2 * n3, path)

        # write augmentation occupancy
        if aug_tensor is None:
            pass
        else:
            aug_list, nele_list = irreps2aug(aug_tensor, z, self.lmix_max)
            io_chgcar.write_aug(aug_list, path, nele_list)

        return None


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
    # Read and parse augmentation occupancies.
    # pymatgen has bug reading aug, I need to modify a little bit
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


def _read_chgcar(filename):
    natoms, nx, ny, nz, naug, ntypes = io_chgcar.get_chgcar_info(filename)
    atom_counts, elements_str, cell, pos, density_1d, aug_str = io_chgcar.read_chgcar(
        filename, natoms, nx, ny, nz, naug, ntypes
    )

    # Parse element symbols
    if isinstance(elements_str, bytes):
        elements_str = elements_str.decode("utf-8")
    elements = elements_str.split()
    # Parse atomic numbers
    atomic_numbers = [Element(el).Z for el in elements]
    z = np.repeat(atomic_numbers, atom_counts)

    cell = cell.T
    pos = pos.T
    volume = np.linalg.det(cell)
    density = density_1d.reshape((nx, ny, nz), order="F") / volume

    aug_str = [line.decode("utf-8").rstrip() + "\n" for line in aug_str]
    return z, cell, pos @ cell, volume, density, aug_str


def cal_sacd():
    return None
