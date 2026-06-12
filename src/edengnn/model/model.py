import torch, e3nn
from torch.utils.checkpoint import checkpoint

from .utils.layers import CosineCutoff, BesselBasis
from .nequip.data import AtomicDataDict
from .nequip.nn.embedding import (
    OneHotAtomEncoding,
    RadialBasisEdgeEncoding,
    SphericalHarmonicEdgeAttrs,
)
from .nequip.nn import (
    AtomwiseLinear,
    ConvNetLayer,
)
from .nequip.nn.nonlinearities import ShiftedSoftPlus
from edengnn.data.io.vasp.basis import AUG_IRREPS
from edengnn.model.utils.cg import ClebschGordan


class DensityLayer(torch.nn.Module):
    """

    Description:

    """

    def __init__(
        self,
        cutoff,
        l_max,
        irreps_node_attribute,
        irreps_node_features,
        n_channels=20,
        n_radial_basis=8,
        n_z_embedding_dim=8,
        n_z_embedding_neurons=64,
        n_z_embedding_layers=2,
        num_radial_filter_layers=2,
        num_radial_filter_neurons=64,
        spin=False,
        chunk_size_train=20000000,
        chunk_size_predict=40000000,
    ):
        super().__init__()

        self.cutoff = cutoff
        self.spin = spin
        self.l_max = l_max
        self.n_channels = n_channels
        self.dtype = torch.get_default_dtype()
        self.chunk_size_train = chunk_size_train
        self.chunk_size_predict = chunk_size_predict

        irreps_probe_features = e3nn.o3.Irreps(
            [
                (n_channels, (l, -1)) if l % 2 else (n_channels, (l, 1))
                for l in range(self.l_max + 1)
            ]
        )

        # l_index for spherical harmonics l wise product
        index = []
        for l in range(self.l_max + 1):
            index += [l] * (2 * l + 1)
        self.l_index = torch.tensor(index)

        # reshape_index for reshape irreps feature
        index = [[] for _ in range(n_channels)]
        idx = 0
        for l in range(self.l_max + 1):
            for c in range(n_channels):
                index[c] += list(range(idx, idx + 2 * l + 1, 1))
                idx += 2 * l + 1
        self.reshape_index = torch.tensor(index)

        # conv node feature to probe feature
        self.conv_to_output_node = AtomwiseLinear(
            out_field="node_charge",
            irreps_in={
                AtomicDataDict.NODE_FEATURES_KEY: e3nn.o3.Irreps(irreps_node_features)
            },
            irreps_out=e3nn.o3.Irreps(irreps_probe_features),
        )

        # probe edge attributes
        # important, because NequIP changed the coordinate index in atomic
        # representation block......
        self.coord_change = torch.LongTensor([1, 2, 0])
        self.spharm_edges_probe = e3nn.o3.SphericalHarmonics(
            irreps_out=e3nn.o3.Irreps(
                [(1, (l, -1)) if l % 2 else (1, (l, 1)) for l in range(self.l_max + 1)]
            ),
            normalization="component",
            normalize=True,
        )

        # probe edge embedding
        self.radial_basis_probe = BesselBasis(
            cutoff=cutoff,
            n_rbf=n_radial_basis,
            cutoff_func=CosineCutoff(cutoff=cutoff),
        )

        # radial filter
        self.z_net = e3nn.nn.FullyConnectedNet(
            [irreps_node_attribute.dim]
            + n_z_embedding_layers * [n_z_embedding_neurons]
            + [n_z_embedding_dim],
            ShiftedSoftPlus,
        )

        self.edge_net = e3nn.nn.FullyConnectedNet(
            [n_radial_basis + n_z_embedding_dim]
            + num_radial_filter_layers * [num_radial_filter_neurons]
            + [self.n_channels * (self.l_max + 1)],
            ShiftedSoftPlus,
        )

    def reshape_irreps(self, tensor):
        """---------------------------------------------------------------------
        change irreps from e3nn form [c * l1 + c * l2 + ...] to [c][l1+l2+...]
        ---------------------------------------------------------------------"""
        shape = tensor.shape
        tensor_flat = tensor.reshape(-1, shape[-1])
        out = tensor_flat[:, self.reshape_index]
        out_shape = shape[:-1] + (self.n_channels, self.reshape_index.shape[1])
        return out.reshape(out_shape)

    def compute_chunk(self, pe_c, pl_c, z_c, h_c):
        len_c = pe_c.shape[0]
        n_probe = pe_c.shape[1]

        Y_lm_c = self.spharm_edges_probe(pe_c)
        radial_embedding_c = torch.cat((self.radial_basis_probe(pl_c), z_c), dim=-1)
        radial_filters_c = self.edge_net(radial_embedding_c).reshape(
            len_c, n_probe, self.n_channels, self.l_max + 1
        )

        p_c = torch.einsum(
            "ipl,ipl->ip",
            (h_c[:, None, :, :] * radial_filters_c[..., self.l_index]).sum(dim=-2),
            Y_lm_c,
        )
        return p_c

    def forward(self, data, batch=None):
        """---------------------------------------------------------------------

        Description:

            probe layer

            do not support batch.

        Variables:

            h: [n_atom][dim_probe_irreps]

            Y_lm: [n_atom * n_probe][lm_dim]

            radial_embedding: [n_atom][n_probe][n_radial_basis]

            radial_filters: [n_atom][n_probe][n_channels][l_max + 1]

            probe_feature: [n_atom][n_probe]

            grid_feature: [n_grid]

            density:  [nx][ny][nz][spin]

        ---------------------------------------------------------------------"""
        self.conv_to_output_node(data)
        device = data["pos"].device

        # ----------------------------------------------------------------------
        # set strucuture
        # ----------------------------------------------------------------------
        # reshape node feature to (natom, n_channels, lm)
        h_clm = data["node_charge"]
        h_clm = self.reshape_irreps(h_clm)

        # ----------------------------------------------------------------------
        # basis: radial filters and spherical harmonics
        # ----------------------------------------------------------------------
        pos_res = data["pos"] - data["pos_n"].to(self.dtype) @ data["grid"].to(
            self.dtype
        )
        probe_edge = data["edge_vec_probes"][None, :, :] - pos_res[:, None, :]
        probe_edge = probe_edge[:, :, self.coord_change]
        probe_length = torch.linalg.norm(probe_edge, dim=-1)
        n_atom, n_probe = data["nat"], data["npb"]

        # probe radial filters
        z_embedding = self.z_net(data[AtomicDataDict.NODE_ATTRS_KEY])[
            :, None, :
        ].expand(-1, data["npb"], -1)
        # ----------------------------------------------------------------------
        # apply spherical harmonics expansion
        # ----------------------------------------------------------------------
        if self.training:
            chunk_size = self.chunk_size_train
        else:
            chunk_size = self.chunk_size_predict
        if data["npb_total"] <= chunk_size:
            Y_lm = self.spharm_edges_probe(probe_edge)
            # probe radial filters
            radial_embedding = torch.cat(
                (self.radial_basis_probe(probe_length), z_embedding), dim=-1
            )

            radial_filters = self.edge_net(radial_embedding).reshape(
                n_atom, n_probe, self.n_channels, self.l_max + 1
            )
            p = torch.einsum(
                "ipl,ipl->ip",
                (h_clm[:, None, :, :] * radial_filters[..., self.l_index]).sum(dim=-2),
                Y_lm,
            )
        else:
            atom_chunk_size = max(1, chunk_size // n_probe)
            chunks = []

            for start in range(0, n_atom, atom_chunk_size):
                end = min(start + atom_chunk_size, n_atom)

                pe_chunk = probe_edge[start:end]
                pl_chunk = probe_length[start:end]
                z_chunk = z_embedding[start:end]
                h_chunk = h_clm[start:end]

                if self.training:
                    p_chunk = checkpoint(
                        self.compute_chunk,
                        pe_chunk,
                        pl_chunk,
                        z_chunk,
                        h_chunk,
                        use_reentrant=False,
                    )
                else:
                    p_chunk = self.compute_chunk(pe_chunk, pl_chunk, z_chunk, h_chunk)

                chunks.append(p_chunk)

            p = torch.cat(chunks, dim=0)

        # ----------------------------------------------------------------------
        # wrap atom grids under PBC to density
        # ----------------------------------------------------------------------

        n1, n2, n3 = data["grid_shape"]
        grid_feature = torch.zeros((n1, n2, n3, 1), device=device, dtype=self.dtype)

        global_idx = data["pos_n"][:, None, :] + data["map_probe"][None, :, :]
        global_idx = (global_idx % data["grid_shape"]).reshape(-1, 3)
        grid_feature.index_put_(
            tuple(global_idx.T),
            p.reshape(-1, 1),
            accumulate=True,
        )

        # ----------------------------------------------------------------------
        # output
        # ----------------------------------------------------------------------

        grid_output = grid_feature
        if self.spin:
            total = torch.sum(grid_output, dim=-1)
            diff = torch.diff(grid_output, dim=-1)[:, :, :, 0]
            data["charge"] = torch.stack([total, diff], dim=-1)
        else:
            data["charge"] = grid_output[:, :, :, 0]

        return None


class AugmentationLayer(torch.nn.Module):
    def __init__(
        self,
        irreps_node_features,
    ):
        super().__init__()
        # conv node feature to output
        self.conv_to_output_node = AtomwiseLinear(
            out_field="node_aug",
            irreps_in={
                AtomicDataDict.NODE_FEATURES_KEY: e3nn.o3.Irreps(irreps_node_features)
            },
            irreps_out=e3nn.o3.Irreps(AUG_IRREPS),
        )

    def forward(self, data):
        self.conv_to_output_node(data)


class OperatorOnsiteLayer(torch.nn.Module):
    def __init__(
        self,
        irreps_node_features,
        irreps_out,
    ):
        super().__init__()
        # conv node feature to output
        self.conv_to_output_node = AtomwiseLinear(
            out_field="node_operator_onsite",
            irreps_in={
                AtomicDataDict.NODE_FEATURES_KEY: e3nn.o3.Irreps(irreps_node_features)
            },
            irreps_out=e3nn.o3.Irreps(irreps_out),
        )

    def forward(self, data):
        self.conv_to_output_node(data)


class OperatorOffsiteLayer(torch.nn.Module):
    def __init__(
        self,
        irreps_node_features,
        cutoff,
        basis_cfg,
        n_radial_basis,
        num_radial_filter_layers,
        num_radial_filter_neurons,
    ):
        super().__init__()

        self.basis_cfg = basis_cfg
        irreps_out = e3nn.o3.Irreps(self.basis_cfg.irreps_offsite)

        # offsite operator transpose index and coefficient.
        index_tensor = torch.empty(self.basis_cfg.size_offsite, dtype=torch.long)
        index_tensor_original = torch.arange(
            self.basis_cfg.size_offsite, dtype=torch.long
        )
        coeff = torch.empty(self.basis_cfg.size_offsite)

        for i, l_i in enumerate(self.basis_cfg.basis):
            for j, l_j in enumerate(self.basis_cfg.basis):
                block_len = self.basis_cfg.i1i2_size_offsite[i][j]
                start_ij = self.basis_cfg.i1i2_start_offsite[i][j]
                start_ji = self.basis_cfg.i1i2_start_offsite[j][i]

                # Map the block from (i, j) to (j, i)
                index_tensor[start_ji : start_ji + block_len] = index_tensor_original[
                    start_ij : start_ij + block_len
                ]

                lmin = abs(l_i - l_j)
                lmax = l_i + l_j
                count = 0
                for l_main in range(lmin, lmax + 1):
                    coeff[start_ji + count : start_ji + count + 2 * l_main + 1] = (
                        -1
                    ) ** (l_i + l_j - l_main)
                    count += 2 * l_main + 1

        self.register_buffer("transpose_index", index_tensor)
        self.register_buffer("transpose_coeff", coeff)

        irreps_node_features = e3nn.o3.Irreps(irreps_node_features)
        self.linear_1 = e3nn.o3.Linear(
            irreps_in=irreps_node_features,
            irreps_out=irreps_node_features,
        )

        # tensor product
        irreps_out = e3nn.o3.Irreps(irreps_out)
        irreps_edge_attr = [
            (1, (l, -1)) if l % 2 else (1, (l, 1))
            for l in range(self.basis_cfg.l_max * 2 + 1)
        ]
        edge_sh_parity = []
        for mul, (l, _) in irreps_edge_attr:
            edge_sh_parity.extend([(-1.0) ** l] * mul * (2 * l + 1))
        self.register_buffer("edge_sh_parity", torch.tensor(edge_sh_parity))

        irreps_mid = []
        instructions = []
        for i, (mul, ir_in) in enumerate(irreps_node_features):
            for j, (_, ir_edge) in enumerate(irreps_edge_attr):
                for ir_out in ir_in * ir_edge:
                    if ir_out in irreps_out:
                        k = len(irreps_mid)
                        irreps_mid.append((mul, ir_out))
                        instructions.append((i, j, k, "uvu", True))

        self.tp = e3nn.o3.TensorProduct(
            irreps_node_features,
            irreps_edge_attr,
            irreps_mid,
            instructions,
            shared_weights=False,
            internal_weights=False,
        )

        # edge embedding
        self.coord_change = torch.LongTensor([1, 2, 0])
        self.spharm_edges = e3nn.o3.SphericalHarmonics(
            irreps_out=e3nn.o3.Irreps(irreps_edge_attr),
            normalization="component",
            normalize=True,
        )

        self.radial_basis = BesselBasis(
            cutoff=cutoff,
            n_rbf=n_radial_basis,
            cutoff_func=CosineCutoff(cutoff=cutoff),
        )

        # radial filters
        self.edge_net = e3nn.nn.FullyConnectedNet(
            [n_radial_basis]
            + num_radial_filter_layers * [num_radial_filter_neurons]
            + [self.tp.weight_numel],
            ShiftedSoftPlus,
        )

        self.linear_2 = e3nn.o3.Linear(
            irreps_in=e3nn.o3.Irreps(irreps_mid),
            irreps_out=irreps_out,
        )

    def forward(self, data):
        node_features = self.linear_1(data["node_features"])
        # geometric information
        operator_edge_length = torch.linalg.norm(data["edge_vec_operator"], dim=-1)
        operator_edge = data["edge_vec_operator"][:, self.coord_change]
        operator_edge_embedding = self.radial_basis(operator_edge_length)

        Y_lm = self.spharm_edges(operator_edge)
        _Y_lm = Y_lm * self.edge_sh_parity
        iedge = data["edge_index_operator"][0, :]
        jedge = data["edge_index_operator"][1, :]

        edge_features = node_features[iedge] + node_features[jedge]

        # hermitian tensor product
        weight = self.edge_net(operator_edge_embedding)
        edge_tensors = self.tp(edge_features, Y_lm, weight)
        edge_tensors = self.linear_2(edge_tensors)

        _edge_tensors = self.tp(edge_features, _Y_lm, weight)
        _edge_tensors = self.linear_2(_edge_tensors)

        data["node_operator_offsite"] = 0.5 * (
            edge_tensors
            + (_edge_tensors[..., self.transpose_index]) * self.transpose_coeff
        )


class Encoder(torch.nn.Module):
    """
    See NequIP
    """

    def __init__(
        self,
        cutoff=4.0,
        l_max=3,
        num_types=100,
        n_radial_basis=8,
        irreps_node_features="64x0e+64x0o+32x1o+16x1e+12x2o+25x2e+18x3o+9x3e+4x4o+9x4e",
        num_conv_layers=3,
        num_radial_filter_layers=2,
        num_radial_filter_neurons=64,
    ):
        super().__init__()

        self.irreps_edge_sh = [
            (1, (l, -1)) if l % 2 else (1, (l, 1)) for l in range(l_max + 1)
        ]

        self.irreps_node_features = e3nn.o3.Irreps(irreps_node_features)

        self.num_interaction_layers = num_conv_layers
        # ----------------------------------------------------------------------
        # embedding block
        # ----------------------------------------------------------------------

        # The atomic number of the node is mapped to the one_hot encoding of "num_types*0e"
        self.one_hot = OneHotAtomEncoding(num_types=num_types, set_features=True)

        # Embed edges' directions as spherical harmonics
        self.spharm_edges = SphericalHarmonicEdgeAttrs(
            irreps_edge_sh=self.irreps_edge_sh,
            edge_sh_normalization="component",
            edge_sh_normalize=True,
        )

        # Embed edge distances as features of 'num_basis*0e'
        self.radial_basis = RadialBasisEdgeEncoding(
            basis=BesselBasis(cutoff=cutoff, n_rbf=n_radial_basis, cutoff_func=None),
            cutoff=CosineCutoff(cutoff=cutoff),
        )

        self.chemical_embedding = AtomwiseLinear(
            irreps_in={
                AtomicDataDict.NODE_FEATURES_KEY: self.one_hot.irreps_out["node_attrs"]
            },
            irreps_out=self.irreps_node_features,
        )

        # ----------------------------------------------------------------------
        # convolution block
        # ----------------------------------------------------------------------
        convolution_kwargs = {
            "invariant_layers": num_radial_filter_layers,
            "invariant_neurons": num_radial_filter_neurons,
        }
        self.convnet = torch.nn.ModuleList(
            [
                ConvNetLayer(
                    irreps_in={
                        AtomicDataDict.EDGE_ATTRS_KEY: self.irreps_edge_sh,
                        AtomicDataDict.EDGE_EMBEDDING_KEY: self.radial_basis.irreps_out[
                            AtomicDataDict.EDGE_EMBEDDING_KEY
                        ],
                        AtomicDataDict.NODE_FEATURES_KEY: self.irreps_node_features,
                        AtomicDataDict.NODE_ATTRS_KEY: self.one_hot.irreps_out[
                            AtomicDataDict.NODE_ATTRS_KEY
                        ],
                    },
                    feature_irreps_hidden=self.irreps_node_features,
                    convolution_kwargs=convolution_kwargs,
                    resnet=True,
                )
                for _ in range(num_conv_layers)
            ]
        )

    def forward(self, data):
        # embedding
        self.one_hot(data)
        self.spharm_edges(data)
        self.radial_basis(data)
        self.chemical_embedding(data)

        # orbital convolution
        for i in range(self.num_interaction_layers):
            self.convnet[i](data)


class EfficientDensity(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.representation = Encoder(
            cutoff=config.atom.edge.cutoff,
            l_max=config.atom.l_max,
            num_types=config.atom.num_types,
            n_radial_basis=config.atom.edge.n_radial_basis,
            irreps_node_features=config.atom.conv.irreps_node_features,
            num_conv_layers=config.atom.conv.num_conv_layers,
            num_radial_filter_layers=config.atom.conv.num_radial_filter_layers,
            num_radial_filter_neurons=config.atom.conv.num_radial_filter_neurons,
        )

        self.task = config.task
        self.use_sad = config.use_sad

        probe_params = {
            "cutoff": config.probe.edge.cutoff,
            "l_max": config.probe.l_max,
            "irreps_node_attribute": self.representation.one_hot.irreps_out[
                AtomicDataDict.NODE_ATTRS_KEY
            ],
            "irreps_node_features": config.atom.conv.irreps_node_features,
            "n_channels": config.probe.conv.n_channels,
            "n_radial_basis": config.probe.edge.n_radial_basis,
            "n_z_embedding_dim": config.probe.edge.n_z_embedding_dim,
            "n_z_embedding_neurons": config.probe.edge.n_z_embedding_neurons,
            "n_z_embedding_layers": config.probe.edge.n_z_embedding_layers,
            "num_radial_filter_layers": config.probe.conv.num_radial_filter_layers,
            "num_radial_filter_neurons": config.probe.conv.num_radial_filter_neurons,
            "spin": False,
            "chunk_size_train": config.chunk_size_train,
            "chunk_size_predict": config.chunk_size_predict,
        }
        self.probe = DensityLayer(**probe_params)
        self.aug = AugmentationLayer(
            irreps_node_features=config.atom.conv.irreps_node_features,
        )

    def _cal_density(self, data, output):
        self.probe(data)
        if self.use_sad:
            grid_func_out = data["charge"] + data["grid_func_in"][0]
        else:
            grid_func_out = data["charge"]
        output["grid_func_out"] = grid_func_out
        output["total_charge"] = grid_func_out.mean()

    def _cal_aug(self, data, output):
        self.aug(data)
        aug_tensor = data["node_aug"]
        output["aug_tensor"] = aug_tensor

    def forward(self, data, batch=None):
        self.representation(data)
        output = {}

        if self.task == 0:
            self._cal_density(data, output)
        else:
            self._cal_aug(data, output)
            if self.task == 2:
                self._cal_density(data, output)

        return output


class HermitianOperator(torch.nn.Module):
    def __init__(self, basis_cfg, config):
        super().__init__()
        self.basis_cfg = basis_cfg

        self.cg_cal = ClebschGordan()
        self.representation = Encoder(
            cutoff=config.atom.edge.cutoff,
            l_max=config.atom.l_max,
            num_types=config.atom.num_types,
            n_radial_basis=config.atom.edge.n_radial_basis,
            irreps_node_features=config.atom.conv.irreps_node_features,
            num_conv_layers=config.atom.conv.num_conv_layers,
            num_radial_filter_layers=config.atom.conv.num_radial_filter_layers,
            num_radial_filter_neurons=config.atom.conv.num_radial_filter_neurons,
        )
        self.onsite = OperatorOnsiteLayer(
            irreps_out=self.basis_cfg.irreps_onsite,
            irreps_node_features=config.atom.conv.irreps_node_features,
        )
        self.offsite = OperatorOffsiteLayer(
            irreps_node_features=config.atom.conv.irreps_node_features,
            cutoff=config.operator.offsite.cutoff,
            basis_cfg=basis_cfg,
            n_radial_basis=config.operator.offsite.n_radial_basis,
            num_radial_filter_layers=config.operator.offsite.num_radial_filter_layers,
            num_radial_filter_neurons=config.operator.offsite.num_radial_filter_neurons,
        )

    def sum2product(self, l1, l2, opeartor_irreps, operator_block, mode="onsite"):
        l_min = abs(l2 - l1)
        l_max = l1 + l2

        idx_element = 0
        if mode == "onsite":
            interval = 2
        elif mode == "offsite":
            interval = 1
        for lmain in range(l_min, l_max + 1, interval):
            # this 2L+1 coefficient is due to e3nn's convention
            cg = self.cg_cal(l1, l2, lmain) * (2 * lmain + 1)
            operator_block += torch.einsum(
                "ijk, mk->mij",
                cg,
                opeartor_irreps[:, idx_element : idx_element + 2 * lmain + 1],
            )
            idx_element += 2 * lmain + 1

        return

    def forward(self, data, batch=None):
        self.representation(data)
        self.onsite(data)
        self.offsite(data)

        device = data["pos"].device
        dtype = data["pos"].dtype
        #
        # transform irreps to operator
        #
        n_atom = len(data["z"])
        n_edge = len(data["edge_vec_operator"])
        operator_onsite = torch.zeros(
            (n_atom, self.basis_cfg.size, self.basis_cfg.size),
            device=device,
            dtype=dtype,
        )
        operator_offsite = torch.zeros(
            (n_edge, self.basis_cfg.size, self.basis_cfg.size),
            device=device,
            dtype=dtype,
        )
        for i, l_i in enumerate(self.basis_cfg.basis):
            for j, l_j in enumerate(self.basis_cfg.basis):
                istart_i = self.basis_cfg.basis_start[i]
                istart_j = self.basis_cfg.basis_start[j]

                self.sum2product(
                    l_i,
                    l_j,
                    data["node_operator_onsite"][
                        :, self.basis_cfg.i1i2_start_onsite[i][j] :
                    ],
                    operator_onsite[
                        :,
                        istart_i : istart_i + 2 * l_i + 1,
                        istart_j : istart_j + 2 * l_j + 1,
                    ],
                    mode="onsite",
                )
                self.sum2product(
                    l_i,
                    l_j,
                    data["node_operator_offsite"][
                        :, self.basis_cfg.i1i2_start_offsite[i][j] :
                    ],
                    operator_offsite[
                        :,
                        istart_i : istart_i + 2 * l_i + 1,
                        istart_j : istart_j + 2 * l_j + 1,
                    ],
                    mode="offsite",
                )

        return {
            "operator": torch.concatenate([operator_onsite, operator_offsite]),
        }
