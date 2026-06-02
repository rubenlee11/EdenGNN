import torch
import torch.nn as nn
import math
from math import pi
from e3nn import o3
from typing import Dict, Callable
from ..nequip.data import AtomicDataDict
from ..nequip.nn import GraphModuleMixin


class CosineCutoff(nn.Module):
    r"""Class of Behler cosine cutoff. From schnetpack

    .. math::
       f(r) = \begin{cases}
        0.5 \times \left[1 + \cos\left(\frac{\pi r}{r_\text{cutoff}}\right)\right]
          & r < r_\text{cutoff} \\
        0 & r \geqslant r_\text{cutoff} \\
        \end{cases}

    Args:
        cutoff (float, optional): cutoff radius.

    """

    def __init__(self, cutoff=5.0):
        super(CosineCutoff, self).__init__()
        self.register_buffer("cutoff", torch.FloatTensor([cutoff]))

    def forward(self, distances):
        """Compute cutoff.

        Args:
            distances (torch.Tensor): values of interatomic distances.

        Returns:
            torch.Tensor: values of cutoff function.

        """
        # Compute values of cutoff function
        cutoffs = 0.5 * (torch.cos(distances * pi / self.cutoff) + 1.0)
        # Remove contributions beyond the cutoff radius
        cutoffs *= (distances < self.cutoff).float()
        return cutoffs


class BesselBasis(nn.Module):
    """
    Modified 0th order Bessel basis expansion supporting r = 0.
    """

    def __init__(self, cutoff=5.0, n_rbf: int = None, cutoff_func: callable = None):
        super(BesselBasis, self).__init__()
        freqs = torch.arange(1, n_rbf + 1) * math.pi / cutoff
        self.register_buffer("freqs", freqs)
        self.cutoff_func = cutoff_func

    def forward(self, dist):
        r"""Computes the 0th order Bessel expansion of inter-atomic distances.

        Args:
            dist (torch.Tensor):
                inter-atomic distances with (N_edge,) shape

        Returns:
            rbf (torch.Tensor):
                the 0th order Bessel expansion of inter-atomic distances
                with (N_edge, n_rbf) shape.
        """
        a = self.freqs[None, :]

        # torch.sinc(x) computes sin(pi * x) / (pi * x)
        # To compute sin(a * dist) / dist, we use a * sinc(a * dist / pi)
        x = dist.unsqueeze(-1) * a / math.pi
        rbf = torch.sinc(x) * a

        if self.cutoff_func is not None:
            rbf = rbf * self.cutoff_func(dist.unsqueeze(-1))
        return rbf


class Edge_builder(GraphModuleMixin, torch.nn.Module):

    def __init__(
        self,
        irreps_in,
        irreps_out,
        invariant_layers=1,
        invariant_neurons=8,
        nonlinearity_scalars: Dict[int, Callable] = {"e": "ssp"},
    ) -> None:

        super().__init__()

        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[
                AtomicDataDict.EDGE_EMBEDDING_KEY,
                AtomicDataDict.EDGE_ATTRS_KEY,
                AtomicDataDict.NODE_FEATURES_KEY,
            ],
            my_irreps_in={
                AtomicDataDict.EDGE_EMBEDDING_KEY: o3.Irreps(
                    [
                        (
                            irreps_in[AtomicDataDict.EDGE_EMBEDDING_KEY].num_irreps,
                            (0, 1),
                        )
                    ]  # (0, 1) is even (invariant) scalars. We are forcing the EDGE_EMBEDDING to be invariant scalars so we can use a dense network
                )
            },
            irreps_out={AtomicDataDict.EDGE_FEATURES_KEY: irreps_out},
        )

        irreps_node_fea = self.irreps_in[AtomicDataDict.NODE_FEATURES_KEY]
        irreps_edge_attr = self.irreps_in[AtomicDataDict.EDGE_ATTRS_KEY]
        feature_irreps_out = self.irreps_out[AtomicDataDict.EDGE_FEATURES_KEY]

        # - Build modules -
        self.linear_node_src = Linear(
            irreps_in=irreps_node_fea,
            irreps_out=irreps_node_fea,
            internal_weights=True,
            shared_weights=True,
        )

        self.linear_node_dst = Linear(
            irreps_in=irreps_node_fea,
            irreps_out=irreps_node_fea,
            internal_weights=True,
            shared_weights=True,
        )

        irreps_mid = []
        instructions = []
        for i, (mul, ir_in1) in enumerate(irreps_node_fea):
            for j, (_, ir_in2) in enumerate(irreps_edge_attr):
                for ir_out in ir_in1 * ir_in2:
                    if ir_out in feature_irreps_out:
                        k = len(irreps_mid)
                        irreps_mid.append((mul, ir_out))
                        instructions.append((i, j, k, "uvu", True))

        # We sort the output irreps of the tensor product so that we can simplify them
        # when they are provided to the second o3.Linear
        irreps_mid = o3.Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()

        # Permute the output indexes of the instructions to match the sorted irreps:
        instructions = [
            (i_in1, i_in2, p[i_out], mode, train)
            for i_in1, i_in2, i_out, mode, train in instructions
        ]

        self.tp = TensorProduct(
            irreps_node_fea,
            irreps_edge_attr,
            irreps_mid,
            instructions,
            shared_weights=False,
            internal_weights=False,
        )

        # init_irreps already confirmed that the edge embeddding is all invariant scalars
        self.fc = FullyConnectedNet(
            [self.irreps_in[AtomicDataDict.EDGE_EMBEDDING_KEY].num_irreps]
            + invariant_layers * [invariant_neurons]
            + [self.tp.weight_numel],
            {
                "ssp": ShiftedSoftPlus,
                "silu": torch.nn.functional.silu,
            }[nonlinearity_scalars["e"]],
        )

        self.linear_edge = Linear(
            irreps_in=irreps_mid.simplify(),
            irreps_out=feature_irreps_out,
            internal_weights=True,
            shared_weights=True,
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:

        weight = self.fc(data[AtomicDataDict.EDGE_EMBEDDING_KEY])

        x = data[AtomicDataDict.NODE_FEATURES_KEY]
        edge_src = data[AtomicDataDict.EDGE_INDEX_KEY][1]  # i
        edge_dst = data[AtomicDataDict.EDGE_INDEX_KEY][0]  # j

        x_ij = self.linear_node_src(x[edge_src]) + self.linear_node_dst(x[edge_dst])

        edge_features = self.tp(x_ij, data[AtomicDataDict.EDGE_ATTRS_KEY], weight)

        edge_features = self.linear_edge(edge_features)

        data[AtomicDataDict.EDGE_FEATURES_KEY] = edge_features
        return data
