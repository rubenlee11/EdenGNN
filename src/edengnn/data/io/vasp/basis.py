"""
Set the VASP PAW basis here

search 'Non local Part' in POTCAR file to retrieve the basis dict

"""

from edengnn.data.io.utils import init_e3nn_irreps


AUG_BASIS = [0, 0, 0, 1, 1, 2, 2, 3, 3]
AUG_IRREPS, I1I2_IDX, _, LEN_AUG_TENSOR = init_e3nn_irreps(AUG_BASIS)

aug_basis_dict = {
    "Ac": [0, 0, 1, 1, 2, 2, 3, 3],  # 390 element
    "Ag": [2, 2, 0, 0, 1, 1],  # 138 element
    "Al": [0, 0, 1, 1],  # 33 element
    "Ar": [0, 0, 1],  # 33 element
    "As": [0, 0, 1, 1, 2],  # 78 element
    "Au": [2, 2, 0, 0, 1, 1],  # 138 element
    "B": [0, 0, 1, 1],  # 33 element
    "Ba_sv": [0, 0, 1, 1, 2, 2],  # 138 element
    "Be_sv": [0, 0, 0, 1, 1],  # 42 element
    "Bi": [0, 0, 1, 1, 2],  # 78 element
    "Br": [0, 0, 1, 1, 2],  # 78 element
    "C": [0, 0, 1, 1],  # 33 element
    "Ca_sv": [0, 0, 1, 1, 2, 2],  # 138 element
    "Cd": [2, 2, 0, 0, 1, 1],  # 138 element
    "Ce": [0, 0, 1, 1, 2, 2, 3, 3],  # 390 element
    "Cl": [0, 0, 1, 1],  # 33 element
    "Co": [2, 2, 0, 0, 1, 1],  # 138 element
    "Cr_pv": [1, 1, 2, 2, 0, 0],  # 138 element
    "Cs_sv": [0, 0, 1, 1, 2],  # 78 element
    "Cu_pv": [1, 1, 2, 2, 0, 0],  # 138 element
    "Dy_3": [0, 0, 1, 1, 2, 2],  # 138 element
    "Er_3": [0, 0, 1, 1, 2, 2],  # 138 element
    "Eu": [0, 0, 1, 1, 2, 2, 3, 3],  # 390 element
    "F": [0, 0, 1, 1],  # 33 element
    "Fe_pv": [1, 1, 2, 2, 0, 0],  # 138 element
    "Ga_d": [2, 2, 0, 0, 1, 1],  # 138 element
    "Gd": [0, 0, 1, 1, 2, 2, 3, 3],  # 390 element
    "Ge_d": [2, 2, 0, 0, 1, 1],  # 138 element
    "H": [0, 0, 1],  # 15 element
    "He": [0, 0, 1],
    "Hf_pv": [1, 1, 2, 2, 0, 0],  # 138 element
    "Hg": [2, 2, 0, 0, 1, 1],  # 138 element
    "Ho_3": [0, 0, 1, 1, 2, 2],  # 138 element
    "I": [0, 0, 1, 1, 2],  # 78 element
    "In_d": [2, 2, 0, 0, 1, 1],  # 138 element
    "Ir": [2, 2, 0, 0, 1, 1],  # 138 element
    "K_sv": [0, 0, 1, 1, 2],  # 78 element
    "Kr": [0, 0, 1],  # 33 element
    "La": [0, 0, 1, 1, 2, 2, 3, 3],  # 390 element
    "Li_sv": [0, 0, 1],  # 15 element
    "Lu_3": [0, 0, 1, 1, 2, 2],  # 138 element
    "Mg_pv": [1, 1, 0, 0, 2],  # 78 element
    "Mn_pv": [1, 1, 2, 2, 0, 0],  # 138 element
    "Mo_pv": [1, 1, 2, 2, 0, 0],  # 138 element
    "N": [0, 0, 1, 1],  # 33 element
    "Na_pv": [1, 1, 0, 0],  # 33 element
    "Nb_pv": [1, 1, 2, 2, 0, 0],  # 138 element
    "Nd_3": [0, 0, 1, 1, 2, 2],  # 138 element
    "Ne": [0, 0, 1],  # 33 element
    "Ni_pv": [1, 1, 2, 2, 0, 0],  # 138 element
    "Np": [0, 0, 1, 1, 2, 2, 3, 3],  # 390 element
    "O": [0, 0, 1, 1],  # 33 element
    "Os_pv": [1, 1, 2, 2, 0, 0],  # 138 element
    "P": [0, 0, 1, 1],  # 33 element
    "Pa": [0, 0, 1, 1, 2, 2, 3, 3],  # 390 element
    "Pb_d": [2, 2, 0, 0, 1, 1],  # 138 element
    "Pd": [2, 2, 0, 0, 1, 1],  # 138 element
    "Pm_3": [0, 0, 1, 1, 2, 2],  # 138 element
    "Pr_3": [0, 0, 1, 1, 2, 2],  # 138 element
    "Pt": [2, 2, 0, 0, 1, 1],  # 138 element
    "Pu": [0, 0, 1, 1, 2, 2, 3, 3],  # 390 element
    "Rb_sv": [0, 0, 1, 1, 2],  # 78 element
    "Re_pv": [1, 1, 2, 2, 0, 0],  # 138 element
    "Rh_pv": [1, 1, 2, 2, 0, 0],  # 138 element
    "Ru_pv": [1, 1, 2, 2, 0, 0],  # 138 element
    "S": [0, 0, 1, 1],  # 33 element
    "Sb": [0, 0, 1, 1, 2],  # 78 element
    "Sc_sv": [0, 0, 1, 1, 2, 2],  # 138 element
    "Se": [0, 0, 1, 1, 2],  # 78 element
    "Si": [0, 0, 1, 1],  # 33 element
    "Sm_3": [0, 0, 1, 1, 2, 2],  # 138 element
    "Sn_d": [2, 2, 0, 0, 1, 1],  # 138 element
    "Sr_sv": [0, 0, 1, 1, 2, 2],  # 138 element
    "Ta_pv": [1, 1, 2, 2, 0, 0],  # 138 element
    "Tb_3": [0, 0, 1, 1, 2, 2],  # 138 element
    "Tc_pv": [1, 1, 2, 2, 0, 0],  # 138 element
    "Te": [0, 0, 1, 1, 2],  # 78 element
    "Th": [0, 0, 1, 1, 2, 2, 3, 3],  # 390 element
    "Ti_pv": [1, 1, 2, 2, 0, 0],  # 138 element
    "Tl_d": [2, 2, 0, 0, 1, 1],  # 138 element
    "Tm_3": [0, 0, 1, 1, 2, 2],  # 138 element
    "U": [0, 0, 1, 1, 2, 2, 3, 3],  # 390 element
    "V_pv": [1, 1, 2, 2, 0, 0],  # 138 element
    "W_pv": [1, 1, 2, 2, 0, 0],  # 138 element
    "Xe": [0, 0, 1],  # 15 element
    "Y_sv": [0, 0, 1, 1, 2, 2, 3],  # 250 element
    "Yb_3": [0, 0, 1, 1, 2, 2],  # 138 element
    "Zn": [2, 2, 0, 0, 1, 1],  # 138 element
    "Zr_sv": [0, 0, 1, 1, 2, 2],  # 138 element
}

aug_basis_idx_dict = {
    "Ac": [0, 1, 3, 4, 5, 6, 7, 8],
    "Ag": [5, 6, 0, 1, 3, 4],
    "Al": [0, 1, 3, 4],
    "Ar": [0, 1, 3],
    "As": [0, 1, 3, 4, 5],
    "Au": [5, 6, 0, 1, 3, 4],
    "B": [0, 1, 3, 4],
    "Ba_sv": [0, 1, 3, 4, 5, 6],
    "Be_sv": [0, 1, 2, 3, 4],
    "Bi": [0, 1, 3, 4, 5],
    "Br": [0, 1, 3, 4, 5],
    "C": [0, 1, 3, 4],
    "Ca_sv": [0, 1, 3, 4, 5, 6],
    "Cd": [5, 6, 0, 1, 3, 4],
    "Ce": [0, 1, 3, 4, 5, 6, 7, 8],
    "Cl": [0, 1, 3, 4],
    "Co": [5, 6, 0, 1, 3, 4],
    "Cr_pv": [3, 4, 5, 6, 0, 1],
    "Cs_sv": [0, 1, 3, 4, 5],
    "Cu_pv": [3, 4, 5, 6, 0, 1],
    "Dy_3": [0, 1, 3, 4, 5, 6],
    "Er_3": [0, 1, 3, 4, 5, 6],
    "Eu": [0, 1, 3, 4, 5, 6, 7, 8],
    "F": [0, 1, 3, 4],
    "Fe_pv": [3, 4, 5, 6, 0, 1],
    "Ga_d": [5, 6, 0, 1, 3, 4],
    "Gd": [0, 1, 3, 4, 5, 6, 7, 8],
    "Ge_d": [5, 6, 0, 1, 3, 4],
    "H": [0, 1, 3],
    "He": [0, 1, 3],
    "Hf_pv": [3, 4, 5, 6, 0, 1],
    "Hg": [5, 6, 0, 1, 3, 4],
    "Ho_3": [0, 1, 3, 4, 5, 6],
    "I": [0, 1, 3, 4, 5],
    "In_d": [5, 6, 0, 1, 3, 4],
    "Ir": [5, 6, 0, 1, 3, 4],
    "K_sv": [0, 1, 3, 4, 5],
    "Kr": [0, 1, 3],
    "La": [0, 1, 3, 4, 5, 6, 7, 8],
    "Li_sv": [0, 1, 3],
    "Lu_3": [0, 1, 3, 4, 5, 6],
    "Mg_pv": [3, 4, 0, 1, 5],
    "Mn_pv": [3, 4, 5, 6, 0, 1],
    "Mo_pv": [3, 4, 5, 6, 0, 1],
    "N": [0, 1, 3, 4],
    "Na_pv": [3, 4, 0, 1],
    "Nb_pv": [3, 4, 5, 6, 0, 1],
    "Nd_3": [0, 1, 3, 4, 5, 6],
    "Ne": [0, 1, 3],
    "Ni_pv": [3, 4, 5, 6, 0, 1],
    "Np": [0, 1, 3, 4, 5, 6, 7, 8],
    "O": [0, 1, 3, 4],
    "Os_pv": [3, 4, 5, 6, 0, 1],
    "P": [0, 1, 3, 4],
    "Pa": [0, 1, 3, 4, 5, 6, 7, 8],
    "Pb_d": [5, 6, 0, 1, 3, 4],
    "Pd": [5, 6, 0, 1, 3, 4],
    "Pm_3": [0, 1, 3, 4, 5, 6],
    "Pr_3": [0, 1, 3, 4, 5, 6],
    "Pt": [5, 6, 0, 1, 3, 4],
    "Pu": [0, 1, 3, 4, 5, 6, 7, 8],
    "Rb_sv": [0, 1, 3, 4, 5],
    "Re_pv": [3, 4, 5, 6, 0, 1],
    "Rh_pv": [3, 4, 5, 6, 0, 1],
    "Ru_pv": [3, 4, 5, 6, 0, 1],
    "S": [0, 1, 3, 4],
    "Sb": [0, 1, 3, 4, 5],
    "Sc_sv": [0, 1, 3, 4, 5, 6],
    "Se": [0, 1, 3, 4, 5],
    "Si": [0, 1, 3, 4],
    "Sm_3": [0, 1, 3, 4, 5, 6],
    "Sn_d": [5, 6, 0, 1, 3, 4],
    "Sr_sv": [0, 1, 3, 4, 5, 6],
    "Ta_pv": [3, 4, 5, 6, 0, 1],
    "Tb_3": [0, 1, 3, 4, 5, 6],
    "Tc_pv": [3, 4, 5, 6, 0, 1],
    "Te": [0, 1, 3, 4, 5],
    "Th": [0, 1, 3, 4, 5, 6, 7, 8],
    "Ti_pv": [3, 4, 5, 6, 0, 1],
    "Tl_d": [5, 6, 0, 1, 3, 4],
    "Tm_3": [0, 1, 3, 4, 5, 6],
    "U": [0, 1, 3, 4, 5, 6, 7, 8],
    "V_pv": [3, 4, 5, 6, 0, 1],
    "W_pv": [3, 4, 5, 6, 0, 1],
    "Xe": [0, 1, 3],
    "Y_sv": [0, 1, 3, 4, 5, 6, 7],
    "Yb_3": [0, 1, 3, 4, 5, 6],
    "Zn": [5, 6, 0, 1, 3, 4],
    "Zr_sv": [0, 1, 3, 4, 5, 6],
}

aug_basis_len_dict = {
    "Ac": 390,  # 390 element
    "Ag": 138,  # 138 element
    "Al": 33,  # 33 element
    "Ar": 15,  # 15 element
    "As": 78,  # 78 element
    "Au": 138,  # 138 element
    "B": 33,  # 33 element
    "Ba_sv": 138,  # 138 element
    "Be_sv": 42,  # 42 element
    "Bi": 78,  # 78 element
    "Br": 78,  # 78 element
    "C": 33,  # 33 element
    "Ca_sv": 138,  # 138 element
    "Cd": 138,  # 138 element
    "Ce": 390,  # 390 element
    "Cl": 33,  # 33 element
    "Co": 138,  # 138 element
    "Cr_pv": 138,  # 138 element
    "Cs_sv": 78,  # 78 element
    "Cu_pv": 138,  # 138 element
    "Dy_3": 138,  # 138 element
    "Er_3": 138,  # 138 element
    "Eu": 390,  # 390 element
    "F": 33,  # 33 element
    "Fe_pv": 138,  # 138 element
    "Ga_d": 138,  # 138 element
    "Gd": 390,  # 390 element
    "Ge_d": 138,  # 138 element
    "H": 15,  # 15 element
    "He": 15,
    "Hf_pv": 138,  # 138 element
    "Hg": 138,  # 138 element
    "Ho_3": 138,  # 138 element
    "I": 78,  # 78 element
    "In_d": 138,  # 138 element
    "Ir": 138,  # 138 element
    "K_sv": 78,  # 78 element
    "Kr": 15,
    "La": 390,  # 390 element
    "Li_sv": 15,  # 15 element
    "Lu_3": 138,  # 138 element
    "Mg_pv": 78,  # 78 element
    "Mn_pv": 138,  # 138 element
    "Mo_pv": 138,  # 138 element
    "N": 33,  # 33 element
    "Ne": 15,
    "Na_pv": 33,  # 33 element
    "Nb_pv": 138,  # 138 element
    "Nd_3": 138,  # 138 element
    "Ni_pv": 138,  # 138 element
    "Np": 390,  # 390 element
    "O": 33,  # 33 element
    "Os_pv": 138,  # 138 element
    "P": 33,  # 33 element
    "Pa": 390,  # 390 element
    "Pb_d": 138,  # 138 element
    "Pd": 138,  # 138 element
    "Pm_3": 138,  # 138 element
    "Pr_3": 138,  # 138 element
    "Pt": 138,  # 138 element
    "Pu": 390,  # 390 element
    "Rb_sv": 78,  # 78 element
    "Re_pv": 138,  # 138 element
    "Rh_pv": 138,  # 138 element
    "Ru_pv": 138,  # 138 element
    "S": 33,  # 33 element
    "Sb": 78,  # 78 element
    "Sc_sv": 138,  # 138 element
    "Se": 78,  # 78 element
    "Si": 33,  # 33 element
    "Sm_3": 138,  # 138 element
    "Sn_d": 138,  # 138 element
    "Sr_sv": 138,  # 138 element
    "Ta_pv": 138,  # 138 element
    "Tb_3": 138,  # 138 element
    "Tc_pv": 138,  # 138 element
    "Te": 78,  # 78 element
    "Th": 390,  # 390 element
    "Ti_pv": 138,  # 138 element
    "Tl_d": 138,  # 138 element
    "Tm_3": 138,  # 138 element
    "U": 390,  # 390 element
    "V_pv": 138,  # 138 element
    "W_pv": 138,  # 138 element
    "Xe": 15,
    "Y_sv": 250,  # 250 element
    "Yb_3": 138,
    "Zn": 138,  # 138 element
    "Zr_sv": 138,  # 138 element
}

pseudo_map = {
    1: "H",
    2: "He",
    3: "Li_sv",
    4: "Be_sv",
    5: "B",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    10: "Ne",
    11: "Na_pv",
    12: "Mg_pv",
    13: "Al",
    14: "Si",
    15: "P",
    16: "S",
    17: "Cl",
    18: "Ar",
    19: "K_sv",
    20: "Ca_sv",
    21: "Sc_sv",
    22: "Ti_pv",
    23: "V_pv",
    24: "Cr_pv",
    25: "Mn_pv",
    26: "Fe_pv",
    27: "Co",
    28: "Ni_pv",
    29: "Cu_pv",
    30: "Zn",
    31: "Ga_d",
    32: "Ge_d",
    33: "As",
    34: "Se",
    35: "Br",
    36: "Kr",
    37: "Rb_sv",
    38: "Sr_sv",
    39: "Y_sv",
    40: "Zr_sv",
    41: "Nb_pv",
    42: "Mo_pv",
    43: "Tc_pv",
    44: "Ru_pv",
    45: "Rh_pv",
    46: "Pd",
    47: "Ag",
    48: "Cd",
    49: "In_d",
    50: "Sn_d",
    51: "Sb",
    52: "Te",
    53: "I",
    54: "Xe",
    55: "Cs_sv",
    56: "Ba_sv",
    57: "La",
    58: "Ce",
    59: "Pr_3",
    60: "Nd_3",
    61: "Pm_3",
    62: "Sm_3",
    63: "Eu",
    64: "Gd",
    65: "Tb_3",
    66: "Dy_3",
    67: "Ho_3",
    68: "Er_3",
    69: "Tm_3",
    70: "Yb_3",
    71: "Lu_3",
    72: "Hf_pv",
    73: "Ta_pv",
    74: "W_pv",
    75: "Re_pv",
    76: "Os_pv",
    77: "Ir",
    78: "Pt",
    79: "Au",
    80: "Hg",
    81: "Tl_d",
    82: "Pb_d",
    83: "Bi",
    89: "Ac",
    90: "Th",
    91: "Pa",
    92: "U",
    93: "Np",
    94: "Pu",
}

nelec_dict = {
    "Ac": 11,
    "Ag": 11,
    "Al": 3,
    "Ar": 8,
}
