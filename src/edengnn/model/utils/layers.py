import torch
import torch.nn as nn
import math
from math import pi


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
    Sine for radial basis expansion with coulomb decay. (0th order Bessel from DimeNet)
    """

    def __init__(self, cutoff=5.0, n_rbf: int = None, cutoff_func: callable = None):
        """
        Args:
            cutoff: radial cutoff
            n_rbf: number of basis functions.
        """
        super(BesselBasis, self).__init__()
        # compute offset and width of Gaussian functions
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
        ax = dist.unsqueeze(-1) * a
        rbf = torch.sin(ax) / dist.unsqueeze(-1)
        if self.cutoff_func is not None:
            rbf = rbf * self.cutoff_func(dist.unsqueeze(-1))
        return rbf
