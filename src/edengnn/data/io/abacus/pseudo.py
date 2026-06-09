from edengnn.data.io.utils import init_e3nn_irreps, BasisConfig
import sys

PP_dict = {
    "Hf": "Hf.upf",
    "Cd": "Cd.upf",
    "Ta": "Ta.upf",
    "K": "K.upf",
    "N": "N.upf",
    "Pb": "Pb.upf",
    "In": "In.upf",
    "Mo": "Mo.upf",
    "Ag": "Ag.upf",
    "P": "P.upf",
    "He": "He.upf",
    "Mn": "Mn.upf",
    "Kr": "Kr.upf",
    "Y": "Y.upf",
    "W": "W.upf",
    "Mg": "Mg.upf",
    "Be": "Be.upf",
    "Ba": "Ba.upf",
    "Bi": "Bi.upf",
    "Ga": "Ga.upf",
    "Ca": "Ca.upf",
    "Sr": "Sr.upf",
    "Al": "Al.upf",
    "As": "As.upf",
    "Br": "Br.upf",
    "Te": "Te.upf",
    "Rh": "Rh.upf",
    "Cu": "Cu.upf",
    "C": "C.upf",
    "Fe": "Fe.upf",
    "Ni": "Ni.upf",
    "Rb": "Rb.upf",
    "O": "O.upf",
    "Se": "Se.upf",
    "Pd": "Pd.upf",
    "Ne": "Ne.upf",
    "Ir": "Ir.upf",
    "Li": "Li.upf",
    "La": "La.upf",
    "Zr": "Zr.upf",
    "Hg": "Hg.upf",
    "Au": "Au.upf",
    "Cr": "Cr.upf",
    "Cl": "Cl.upf",
    "Sn": "Sn.upf",
    "Ti": "Ti.upf",
    "Tl": "Tl.upf",
    "Ge": "Ge.upf",
    "Sc": "Sc.upf",
    "B": "B.upf",
    "Si": "Si.upf",
    "S": "S.upf",
    "Co": "Co.upf",
    "Na": "Na.upf",
    "Pt": "Pt.upf",
    "Os": "Os.upf",
    "Tc": "Tc.upf",
    "Re": "Re.upf",
    "Ar": "Ar.upf",
    "Xe": "Xe.upf",
    "Cs": "Cs.upf",
    "V": "V.upf",
    "H": "H.upf",
    "Sb": "Sb.upf",
    "Zn": "Zn.upf",
    "F": "F.upf",
    "Nb": "Nb.upf",
    "Ru": "Ru.upf",
    "I": "I.upf",
}

BASIS_dict = {
    "He": "He_gga_6au_100Ry_2s1p.orb",
    "Ni": "Ni_gga_8au_100Ry_4s2p2d1f.orb",
    "Al": "Al_gga_7au_100Ry_4s4p1d.orb",
    "Sr": "Sr_gga_9au_100Ry_4s2p1d.orb",
    "P": "P_gga_7au_100Ry_2s2p1d.orb",
    "Co": "Co_gga_8au_100Ry_4s2p2d1f.orb",
    "Re": "Re_gga_7au_100Ry_4s2p2d1f.orb",
    "Cr": "Cr_gga_8au_100Ry_4s2p2d1f.orb",
    "Ti": "Ti_gga_8au_100Ry_4s2p2d1f.orb",
    "Se": "Se_gga_8au_100Ry_2s2p1d.orb",
    "Ca": "Ca_gga_9au_100Ry_4s2p1d.orb",
    "Zr": "Zr_gga_8au_100Ry_4s2p2d1f.orb",
    "Nb": "Nb_gga_8au_100Ry_4s2p2d1f.orb",
    "Tl": "Tl_gga_7au_100Ry_2s2p2d1f.orb",
    "S": "S_gga_7au_100Ry_2s2p1d.orb",
    "Y": "Y_gga_8au_100Ry_4s2p2d1f.orb",
    "Rb": "Rb_gga_10au_100Ry_4s2p1d.orb",
    # "Ta": "Ta_gga_8au_100Ry_4s2p2d2f1g.orb",
    "Cs": "Cs_gga_10au_100Ry_4s2p1d.orb",
    "K": "K_gga_9au_100Ry_4s2p1d.orb",
    # "W": "W_gga_8au_100Ry_4s2p2d2f1g.orb",
    "Ar": "Ar_gga_7au_100Ry_2s2p1d.orb",
    "Mn": "Mn_gga_8au_100Ry_4s2p2d1f.orb",
    "As": "As_gga_7au_100Ry_2s2p1d.orb",
    "Mo": "Mo_gga_7au_100Ry_4s2p2d1f.orb",
    "Mg": "Mg_gga_8au_100Ry_4s2p1d.orb",
    "B": "B_gga_8au_100Ry_2s2p1d.orb",
    "F": "F_gga_7au_100Ry_2s2p1d.orb",
    "Pb": "Pb_gga_7au_100Ry_2s2p2d1f.orb",
    "N": "N_gga_7au_100Ry_2s2p1d.orb",
    "Ga": "Ga_gga_8au_100Ry_2s2p2d1f.orb",
    "Si": "Si_gga_7au_100Ry_2s2p1d.orb",
    "Ba": "Ba_gga_10au_100Ry_4s2p2d1f.orb",
    "In": "In_gga_7au_100Ry_2s2p2d1f.orb",
    "Tc": "Tc_gga_7au_100Ry_4s2p2d1f.orb",
    "Sn": "Sn_gga_7au_100Ry_2s2p2d1f.orb",
    "Cl": "Cl_gga_7au_100Ry_2s2p1d.orb",
    "Ag": "Ag_gga_7au_100Ry_4s2p2d1f.orb",
    "Cu": "Cu_gga_8au_100Ry_4s2p2d1f.orb",
    "V": "V_gga_8au_100Ry_4s2p2d1f.orb",
    # "Hf": "Hf_gga_7au_100Ry_4s2p2d2f1g.orb",
    "Ne": "Ne_gga_6au_100Ry_2s2p1d.orb",
    "I": "I_gga_7au_100Ry_2s2p2d1f.orb",
    "O": "O_gga_7au_100Ry_2s2p1d.orb",
    "Be": "Be_gga_7au_100Ry_4s1p.orb",
    "Bi": "Bi_gga_7au_100Ry_2s2p2d1f.orb",
    "Sc": "Sc_gga_8au_100Ry_4s2p2d1f.orb",
    "Br": "Br_gga_7au_100Ry_2s2p1d.orb",
    "Ir": "Ir_gga_7au_100Ry_4s2p2d1f.orb",
    "Au": "Au_gga_7au_100Ry_4s2p2d1f.orb",
    "Te": "Te_gga_7au_100Ry_2s2p2d1f.orb",
    "Fe": "Fe_gga_8au_100Ry_4s2p2d1f.orb",
    "Hg": "Hg_gga_9au_100Ry_4s2p2d1f.orb",
    "Rh": "Rh_gga_7au_100Ry_4s2p2d1f.orb",
    "Xe": "Xe_gga_8au_100Ry_2s2p2d1f.orb",
    "Pt": "Pt_gga_7au_100Ry_4s2p2d1f.orb",
    "H": "H_gga_6au_100Ry_2s1p.orb",
    "C": "C_gga_7au_100Ry_2s2p1d.orb",
    "Ru": "Ru_gga_7au_100Ry_4s2p2d1f.orb",
    "Sb": "Sb_gga_7au_100Ry_2s2p2d1f.orb",
    "Os": "Os_gga_7au_100Ry_4s2p2d1f.orb",
    "Na": "Na_gga_8au_100Ry_4s2p1d.orb",
    "Zn": "Zn_gga_8au_100Ry_4s2p2d1f.orb",
    "Kr": "Kr_gga_7au_100Ry_2s2p1d.orb",
    "Ge": "Ge_gga_8au_100Ry_2s2p2d1f.orb",
    "Cd": "Cd_gga_7au_100Ry_4s2p2d1f.orb",
    "Li": "Li_gga_7au_100Ry_4s1p.orb",
    "Pd": "Pd_gga_7au_100Ry_4s2p2d1f.orb",
}


# basis_irreps stores the angular momentum quantum numbers
BASIS_IRREPS = {
    1: [0, 0, 1],  # H
    14: [0, 0, 1, 1, 2],  # Si
    31: [0, 0, 1, 1, 2, 2, 3],  # Ga
    33: [0, 0, 1, 1, 2],  # As
}


def build_basis(basis):
    # Check if BASIS is sorted in ascending order
    if basis != sorted(basis):
        sys.exit("Error: BASIS is not sorted in ascending order. Exiting program.")

    count = 0
    basis_start = []
    for l in basis:
        basis_start.append(count)
        count += 2 * l + 1
    basis_size = count

    irreps_onsite, i1i2_start_onsite, i1i2_size_onsite, size_onsite = init_e3nn_irreps(
        basis
    )
    irreps_offsite, i1i2_start_offsite, i1i2_size_offsite, size_offsite = (
        init_e3nn_irreps(basis, mode="offsite")
    )

    index_dft2e3nn = []
    count_m = 0
    for l in basis:
        for m in range(l):
            index_dft2e3nn.append(count_m + 2 * l - 2 * m)
        index_dft2e3nn.append(count_m)
        for m in range(l):
            index_dft2e3nn.append(count_m + 2 * m + 1)
        count_m += 2 * l + 1

    index_e3nn2dft = []
    count_m = 0
    for l in basis:
        index_e3nn2dft.append(count_m + l)
        for m in range(l):
            index_e3nn2dft.append(count_m + l + m + 1)
            index_e3nn2dft.append(count_m + l - m - 1)
        count_m += 2 * l + 1

    # atom_irreps_idx stores the start positions in tensors of basis
    atom_irreps = BASIS_IRREPS
    atom_irreps_idx = {}
    l_to_starts = {}
    for l, start in zip(basis, basis_start):
        if l not in l_to_starts:
            l_to_starts[l] = []
        l_to_starts[l].append(start)

    # Map atom_irreps to atom_irreps_idx
    for atom, irreps in atom_irreps.items():
        atom_idx_list = []
        l_counts = {}

        for l in irreps:
            count_l = l_counts.get(l, 0)
            # Fetch the start position based on the occurrence count
            atom_idx_list.append(l_to_starts[l][count_l])
            l_counts[l] = count_l + 1

        atom_irreps_idx[atom] = atom_idx_list

    return BasisConfig(
        basis=basis,
        size=basis_size,
        basis_start=basis_start,
        l_max=max(basis),
        irreps_onsite=irreps_onsite,
        i1i2_start_onsite=i1i2_start_onsite,
        size_onsite=size_onsite,
        i1i2_size_onsite=i1i2_size_onsite,
        irreps_offsite=irreps_offsite,
        i1i2_start_offsite=i1i2_start_offsite,
        size_offsite=size_offsite,
        i1i2_size_offsite=i1i2_size_offsite,
        index_dft2e3nn=index_dft2e3nn,
        index_e3nn2dft=index_e3nn2dft,
        atom_irreps=atom_irreps,
        atom_irreps_idx=atom_irreps_idx,
    )
