"""
Descripttion: Script for converting poscar to openmx input file
version: 0.1
Author: Yang Zhong
Date: 2022-11-24 19:03:36
LastEditors: Yang Zhong
LastEditTime: 2023-07-18 03:24:04
"""

from pymatgen.core.structure import Structure
from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor
import os, pathlib
import numpy as np
import argparse
import seekpath

K_SPACING = 0.35


def ordered_set(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]


def set_band_kpath(r_cell, special_points, path_segments, kpath_spacing=0.02):
    """
    generating K points along high-symmetry lines
    """
    lines = []
    lines.append("Band.dispersion               off")
    lines.append(f"Band.Nkpath                   {len(path_segments)}")
    lines.append("<Band.kpath")

    for start_label, end_label in path_segments:
        k1_frac = np.array(special_points[start_label])
        k2_frac = np.array(special_points[end_label])

        k1_cart = np.dot(k1_frac, r_cell)
        k2_cart = np.dot(k2_frac, r_cell)

        dist = np.linalg.norm(k2_cart - k1_cart)

        n_points = max(2, int(np.round(dist / kpath_spacing)))

        line = f"  {n_points:<3} "
        line += f"{k1_frac[0]:8.6f} {k1_frac[1]:8.6f} {k1_frac[2]:8.6f}   "
        line += f"{k2_frac[0]:8.6f} {k2_frac[1]:8.6f} {k2_frac[2]:8.6f}   "
        line += f"{start_label} {end_label}"
        lines.append(line)

    lines.append("Band.kpath>")

    return "\n".join(lines)


spin_set = {
    "H": [0.5, 0.5],  # 1
    "He": [1.0, 1.0],  # 2
    "Li": [1.5, 1.5],  # 3
    "Be": [1.0, 1.0],  # 4
    "B": [1.5, 1.5],  # 5
    "C": [2.0, 2.0],  # 6
    "N": [2.5, 2.5],  # 7
    "O": [3.0, 3.0],  # 8
    "F": [3.5, 3.5],  # 9
    "Ne": [4.0, 4.0],  # 10
    "Na": [4.5, 4.5],  # 11
    "Mg": [4.0, 4.0],  # 12
    "Al": [1.5, 1.5],  # 13
    "Si": [2.0, 2.0],  # 14
    "P": [2.5, 2.5],  # 15
    "S": [3.0, 3.0],  # 16
    "Cl": [3.5, 3.5],  # 17
    "Ar": [4.0, 4.0],  # 18
    "K": [4.5, 4.5],  # 19
    "Ca": [5.0, 5.0],  # 20
    "Sc": [5.5, 5.5],
    "Ti": [6.0, 6.0],
    "V": [6.5, 6.5],
    "Cr": [7.0, 7.0],
    "Mn": [7.5, 7.5],
    "Fc": [8.0, 8.0],
    "Co": [8.5, 8.5],
    "Ni": [9.0, 9.0],
    "Cu": [9.5, 9.5],
    "Zn": [10.0, 10.0],
    "Ga": [6.5, 6.5],
    "Ge": [2.0, 2.0],
    "As": [7.5, 7.5],
    "Se": [3.0, 3.0],
    "Br": [3.5, 3.5],
    "Kr": [4.0, 4.0],
    "Rb": [4.5, 4.5],
    "Sr": [5.0, 5.0],
    "Y": [5.5, 5.5],
    "Zr": [6.0, 6.0],
    "Nb": [6.5, 6.5],
    "Mo": [7.0, 7.0],
    "Tc": [7.5, 7.5],
    "Ru": [7.0, 7.0],
    "Rh": [7.5, 7.5],
    "Pd": [8.0, 8.0],
    "Ag": [8.5, 8.5],
    "Cd": [6.0, 6.0],
    "In": [6.5, 6.5],
    "Sn": [7.0, 7.0],
    "Sb": [7.5, 7.5],
    "Te": [8.0, 8.0],
    "I": [3.5, 3.5],
    "Xe": [4.0, 4.0],
    "Cs": [4.5, 4.5],
    "Ba": [5.0, 5.0],
    "La": [5.5, 5.5],
    "Ce": [6.0, 6.0],
    "Pr": [6.5, 6.5],
    "Nd": [7.0, 7.0],
    "Pm": [7.5, 7.5],
    "Sm": [8.0, 8.0],
    "Dy": [10.0, 10.0],
    "Ho": [10.5, 10.5],
    "Lu": [5.5, 5.5],
    "Hf": [6.0, 6.0],
    "Ta": [6.5, 6.5],
    "W": [6.0, 6.0],
    "Re": [7.5, 7.5],
    "Os": [7.0, 7.0],
    "Ir": [7.5, 7.5],
    "Pt": [8.0, 8.0],
    "Au": [8.5, 8.5],
    "Hg": [9.0, 9.0],
    "Tl": [9.5, 9.5],
    "Pb": [7.0, 7.0],
    "Bi": [7.5, 7.5],
    "Fe": [8.0, 8.0],
    "Gd": [9.0, 9.0],
    "Tb": [9.5, 9.5],
}

PAO_dict = {
    "H": "H6.0-s2p1",
    "He": "He8.0-s2p1",
    "Li": "Li8.0-s3p2",
    "Be": "Be7.0-s2p2",
    "B": "B7.0-s2p2d1",
    "C": "C6.0-s2p2d1",
    "N": "N6.0-s2p2d1",
    "O": "O6.0-s2p2d1",
    "F": "F6.0-s2p2d1",
    "Ne": "Ne9.0-s2p2d1",
    "Na": "Na9.0-s3p2d1",
    "Mg": "Mg9.0-s3p2d1",
    "Al": "Al7.0-s2p2d1",
    "Si": "Si7.0-s2p2d1",
    "P": "P7.0-s2p2d1",
    "S": "S7.0-s2p2d1",
    "Cl": "Cl7.0-s2p2d1",
    "Ar": "Ar9.0-s2p2d1",
    "K": "K10.0-s3p2d1",
    "Ca": "Ca9.0-s3p2d1",
    "Sc": "Sc9.0-s3p2d1",
    "Ti": "Ti7.0-s3p2d1",
    "V": "V6.0-s3p2d1",
    "Cr": "Cr6.0-s3p2d1",
    "Mn": "Mn6.0-s3p2d1",
    "Fe": "Fe5.5H-s3p2d1",
    "Co": "Co6.0H-s3p2d1",
    "Ni": "Ni6.0H-s3p2d1",
    "Cu": "Cu6.0H-s3p2d1",
    "Zn": "Zn6.0H-s3p2d1",
    "Ga": "Ga7.0-s3p2d2",
    "Ge": "Ge7.0-s3p2d2",
    "As": "As7.0-s3p2d2",
    "Se": "Se7.0-s3p2d2",
    "Br": "Br7.0-s3p2d2",
    "Kr": "Kr10.0-s3p2d2",
    "Rb": "Rb11.0-s3p2d2",
    "Sr": "Sr10.0-s3p2d2",
    "Y": "Y10.0-s3p2d2",
    "Zr": "Zr7.0-s3p2d2",
    "Nb": "Nb7.0-s3p2d2",
    "Mo": "Mo7.0-s3p2d2",
    "Tc": "Tc7.0-s3p2d2",
    "Ru": "Ru7.0-s3p2d2",
    "Rh": "Rh7.0-s3p2d2",
    "Pd": "Pd7.0-s3p2d2",
    "Ag": "Ag7.0-s3p2d2",
    "Cd": "Cd7.0-s3p2d2",
    "In": "In7.0-s3p2d2",
    "Sn": "Sn7.0-s3p2d2",
    "Sb": "Sb7.0-s3p2d2",
    "Te": "Te7.0-s3p2d2f1",
    "I": "I7.0-s3p2d2f1",
    "Xe": "Xe11.0-s3p2d2",
    "Cs": "Cs12.0-s3p2d2",
    "Ba": "Ba10.0-s3p2d2",
    "La": "La8.0-s3p2d2f1",
    "Ce": "Ce8.0-s3p2d2f1",
    "Pr": "Pr8.0-s3p2d2f1",
    "Nd": "Nd8.0-s3p2d2f1",
    "Pm": "Pm8.0-s3p2d2f1",
    "Sm": "Sm8.0-s3p2d2f1",
    "Dy": "Dy8.0-s3p2d2f1",
    "Ho": "Ho8.0-s3p2d2f1",
    "Lu": "Lu8.0-s3p2d2f1",
    "Hf": "Hf9.0-s3p2d2f1",
    "Ta": "Ta7.0-s3p2d2f1",
    "W": "W7.0-s3p2d2f1",
    "Re": "Re7.0-s3p2d2f1",
    "Os": "Os7.0-s3p2d2f1",
    "Ir": "Ir7.0-s3p2d2f1",
    "Pt": "Pt7.0-s3p2d2f1",
    "Au": "Au7.0-s3p2d2f1",
    "Hg": "Hg8.0-s3p2d2f1",
    "Tl": "Tl8.0-s3p2d2f1",
    "Pb": "Pb8.0-s3p2d2f1",
    "Bi": "Bi8.0-s3p2d2f1",
    "Gd": "Gd8.0-s3p2d2f1",
    "Tb": "Tb8.0-s3p2d2f1",
}

PBE_dict = {
    "H": "H_PBE19",
    "He": "He_PBE19",
    "Li": "Li_PBE19",
    "Be": "Be_PBE19",
    "B": "B_PBE19",
    "C": "C_PBE19",
    "N": "N_PBE19",
    "O": "O_PBE19",
    "F": "F_PBE19",
    "Ne": "Ne_PBE19",
    "Na": "Na_PBE19",
    "Mg": "Mg_PBE19",
    "Al": "Al_PBE19",
    "Si": "Si_PBE19",
    "P": "P_PBE19",
    "S": "S_PBE19",
    "Cl": "Cl_PBE19",
    "Ar": "Ar_PBE19",
    "K": "K_PBE19",
    "Ca": "Ca_PBE19",
    "Sc": "Sc_PBE19",
    "Ti": "Ti_PBE19",
    "V": "V_PBE19",
    "Cr": "Cr_PBE19",
    "Mn": "Mn_PBE19",
    "Fe": "Fe_PBE19H",
    "Co": "Co_PBE19H",
    "Ni": "Ni_PBE19H",
    "Cu": "Cu_PBE19H",
    "Zn": "Zn_PBE19H",
    "Ga": "Ga_PBE19",
    "Ge": "Ge_PBE19",
    "As": "As_PBE19",
    "Se": "Se_PBE19",
    "Br": "Br_PBE19",
    "Kr": "Kr_PBE19",
    "Rb": "Rb_PBE19",
    "Sr": "Sr_PBE19",
    "Y": "Y_PBE19",
    "Zr": "Zr_PBE19",
    "Nb": "Nb_PBE19",
    "Mo": "Mo_PBE19",
    "Tc": "Tc_PBE19",
    "Ru": "Ru_PBE19",
    "Rh": "Rh_PBE19",
    "Pd": "Pd_PBE19",
    "Ag": "Ag_PBE19",
    "Cd": "Cd_PBE19",
    "In": "In_PBE19",
    "Sn": "Sn_PBE19",
    "Sb": "Sb_PBE19",
    "Te": "Te_PBE19",
    "I": "I_PBE19",
    "Xe": "Xe_PBE19",
    "Cs": "Cs_PBE19",
    "Ba": "Ba_PBE19",
    "La": "La_PBE19",
    "Ce": "Ce_PBE19",
    "Pr": "Pr_PBE19",
    "Nd": "Nd_PBE19",
    "Pm": "Pm_PBE19",
    "Sm": "Sm_PBE19",
    "Dy": "Dy_PBE19",
    "Ho": "Ho_PBE19",
    "Lu": "Lu_PBE19",
    "Hf": "Hf_PBE19",
    "Ta": "Ta_PBE19",
    "W": "W_PBE19",
    "Re": "Re_PBE19",
    "Os": "Os_PBE19",
    "Ir": "Ir_PBE19",
    "Pt": "Pt_PBE19",
    "Au": "Au_PBE19",
    "Hg": "Hg_PBE19",
    "Tl": "Tl_PBE19",
    "Pb": "Pb_PBE19",
    "Bi": "Bi_PBE19",
    "Gd": "Gd_PBE19",
    "Tb": "Tb_PBE19",
}


def write_openmx_xyz_file(atoms, name, filename):
    chemical_symbols = atoms.get_chemical_symbols()
    species = ordered_set(chemical_symbols)
    positions = atoms.get_array(name="positions")
    cell = atoms.get_cell().array
    r_cell = atoms.cell.reciprocal() * 2 * np.pi

    # specify Kgrid
    Nka = np.round(np.sqrt(np.inner(r_cell[0], r_cell[0])) / K_SPACING)
    if Nka < 1:
        Nka = 1
    Nkb = np.round(np.sqrt(np.inner(r_cell[1], r_cell[1])) / K_SPACING)
    if Nkb < 1:
        Nkb = 1
    Nkc = np.round(np.sqrt(np.inner(r_cell[2], r_cell[2])) / K_SPACING)
    if Nkc < 1:
        Nkc = 1

    # specify K point along high-symmetry lines
    numbers = atoms.get_atomic_numbers()
    structure = (cell, positions, numbers)

    res = seekpath.get_path(structure)
    special_points = res["point_coords"]
    path_segments = res["path"]

    std_cell = np.array(res["conv_lattice"])
    r_cell = np.linalg.inv(std_cell).T * 2 * np.pi

    openmx_band_block = set_band_kpath(
        r_cell=r_cell,
        special_points=special_points,
        path_segments=path_segments,
        kpath_spacing=0.02,
    )

    openmx = "#\n#  This is an OpenMX input file\n#\n"

    openmx += "\n"
    openmx += "System.CurrrentDirectory         ./\n"
    openmx += "System.Name                      {}\n".format(name)
    openmx += "DATA.PATH          /public/home/lixiwen/software/openmx3.9/DFT_DATA19\n"
    openmx += "level.of.stdout                   1\n"
    openmx += "level.of.fileout                  1\n"
    openmx += "\n"

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
    openmx += "scf.XcType                    GGA-PBE\n"
    openmx += "scf.partialCoreCorrection        on\n"
    openmx += "scf.SpinPolarization          off\n"
    openmx += "scf.ElectronicTemperature     300.0\n"
    openmx += "scf.energycutoff              220.0\n"
    openmx += "scf.maxIter                   200\n"
    openmx += "scf.EigenvalueSolver          Band\n"
    openmx += "scf.Kgrid                     {:.0f}  {:.0f}  {:.0f}\n".format(
        Nka, Nkb, Nkc
    )
    openmx += "scf.fixed.grid   0.0   0.0   0.0\n"
    openmx += "scf.Mixing.Type               rmm-diisk\n"
    openmx += "scf.Init.Mixing.Weight        0.05\n"
    openmx += "scf.Min.Mixing.Weight         0.01\n"
    openmx += "scf.Max.Mixing.Weight         0.30\n"
    openmx += "scf.Mixing.History            25\n"
    openmx += "scf.Mixing.StartPulay         15\n"
    openmx += "scf.Mixing.EveryPulay        1\n"
    openmx += "scf.criterion                 1.0e-7\n"

    openmx += "\n"
    openmx += "MD.Type                       Nomd\n"
    openmx += "MD.maxIter                    1\n"
    openmx += "MD.TimeStep                   1.0\n"
    openmx += "MD.Opt.criterion              0.0003\n"

    openmx += "\n"

    openmx += "Dos.fileout                   off\n"
    openmx += "Dos.Erange                    -25.0  20.0\n"
    openmx += "Dos.Kgrid                     10  8  8"

    openmx += "\n"
    # openmx += openmx_band_block

    with open(filename, "w") as wf:
        wf.write(openmx)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir_in", type=str, required=True, help="Path to the cif or vasp files"
    )
    parser.add_argument("--dir_out", type=str, required=True, help="Dir out")
    args = parser.parse_args()

    # openmx file directory to save
    filepath = args.dir_out

    os.makedirs(filepath, exist_ok=True)

    dir_in = args.dir_in
    for entry in os.listdir(dir_in):
        # full path
        entry_path = os.path.join(dir_in, entry)
        if os.path.isfile(entry_path) and (
            entry.endswith(".cif") or entry.endswith(".vasp")
        ):
            name = pathlib.Path(entry).stem
            try:
                crystal = Structure.from_file(
                    entry_path
                )  # return a structure class. pymatgen
                ase_atoms = AseAtomsAdaptor.get_atoms(crystal)
                os.makedirs(os.path.join(filepath, name), exist_ok=True)
                filename = os.path.join(filepath, name, name + ".dat")
            except Exception as e:
                print("failed to convert {}".format(name), e)
                continue

            write_openmx_xyz_file(ase_atoms, name, filename)

            print("{} is converted successfully!\n".format(name))


if __name__ == "__main__":
    main()
