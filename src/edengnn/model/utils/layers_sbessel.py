import math
import torch
import torch.nn as nn
import numpy as np
from scipy.special import spherical_jn
from scipy.optimize import brentq


class ScipySphericalBessel_deprecated(torch.autograd.Function):
    """
    Custom PyTorch Autograd function wrapping SciPy's highly stable
    spherical Bessel implementation with a division-free backward pass.
    """

    @staticmethod
    def forward(ctx, x, l):
        ctx.l = l
        x_np = x.detach().cpu().numpy()

        j_l = spherical_jn(l, x_np)

        ctx.save_for_backward(x)
        return torch.from_numpy(j_l).to(x.device).type(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        l = ctx.l
        x_np = x.detach().cpu().numpy()

        # Bulletproof derivative formula: NO division by x!
        # j_l'(x) = (l / (2l + 1)) * j_{l-1}(x) - ((l + 1) / (2l + 1)) * j_{l+1}(x)
        if l == 0:
            dj_dx = -spherical_jn(1, x_np)
        else:
            term1 = (l / (2.0 * l + 1.0)) * spherical_jn(l - 1, x_np)
            term2 = ((l + 1.0) / (2.0 * l + 1.0)) * spherical_jn(l + 1, x_np)
            dj_dx = term1 - term2

        dj_dx_tensor = torch.from_numpy(dj_dx).to(x.device).type(x.dtype)

        return grad_output * dj_dx_tensor, None


class GPUSphericalBessel(torch.autograd.Function):
    """
    Pure GPU, highly stable spherical Bessel implementation.
    Uses float64 internally for stable forward recurrence and
    a division-free formula for backward pass.
    """

    @staticmethod
    def forward(ctx, x, l):
        ctx.l = l
        # Save original dtype to return the same type later
        orig_dtype = x.dtype

        # Cast to float64 for numerical stability during recurrence
        x_d = x.to(torch.float64)

        # Threshold for Taylor expansion
        eps = 0.1
        mask = x_d < eps
        x_safe = torch.where(mask, torch.ones_like(x_d), x_d)

        # --- Forward Recurrence ---
        j_prev = torch.sin(x_safe) / x_safe
        j_curr = torch.sin(x_safe) / (x_safe**2) - torch.cos(x_safe) / x_safe

        if l == 0:
            res_exact = j_prev
        elif l == 1:
            res_exact = j_curr
        else:
            for i in range(1, l):
                j_next = (2.0 * i + 1.0) / x_safe * j_curr - j_prev
                j_prev = j_curr
                j_curr = j_next
            res_exact = j_curr

        # --- Taylor Expansion for small x ---
        dfact = math.prod(range(1, 2 * l + 2, 2))
        term0 = (x_d**l) / dfact
        term1 = -(x_d**2) / (2.0 * (2.0 * l + 3.0))
        term2 = (x_d**4) / (8.0 * (2.0 * l + 3.0) * (2.0 * l + 5.0))
        res_taylor = term0 * (1.0 + term1 + term2)

        # Combine and cast back to original dtype (float32)
        j_l = torch.where(mask, res_taylor, res_exact).to(orig_dtype)

        ctx.save_for_backward(x)
        return j_l

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        l = ctx.l

        # We need j_{l-1} and j_{l+1} for the stable derivative.
        # We compute them quickly on the fly using the same stable forward method.
        # Since this is pure GPU tensor ops, it's extremely fast.

        def _get_j(order):
            if order < 0:
                # Only needed if formula asks for j_{-1}, but we handle l=0 separately
                return torch.zeros_like(x)
            # Re-use the forward logic (without autograd)
            with torch.no_grad():
                x_d = x.to(torch.float64)
                mask = x_d < 0.1
                x_safe = torch.where(mask, torch.ones_like(x_d), x_d)

                j_p = torch.sin(x_safe) / x_safe
                j_c = torch.sin(x_safe) / (x_safe**2) - torch.cos(x_safe) / x_safe

                if order == 0:
                    res_e = j_p
                elif order == 1:
                    res_e = j_c
                else:
                    for i in range(1, order):
                        j_n = (2.0 * i + 1.0) / x_safe * j_c - j_p
                        j_p = j_c
                        j_c = j_n
                    res_e = j_c

                dfact = math.prod(range(1, 2 * order + 2, 2))
                t0 = (x_d**order) / dfact
                t1 = -(x_d**2) / (2.0 * (2.0 * order + 3.0))
                t2 = (x_d**4) / (8.0 * (2.0 * order + 3.0) * (2.0 * order + 5.0))
                res_t = t0 * (1.0 + t1 + t2)

                return torch.where(mask, res_t, res_e).to(x.dtype)

        # Bulletproof derivative formula: NO division by x!
        if l == 0:
            dj_dx = -_get_j(1)
        else:
            term1 = (l / (2.0 * l + 1.0)) * _get_j(l - 1)
            term2 = ((l + 1.0) / (2.0 * l + 1.0)) * _get_j(l + 1)
            dj_dx = term1 - term2

        return grad_output * dj_dx, None


class CosineCutoff(nn.Module):
    def __init__(self, cutoff=5.0):
        super(CosineCutoff, self).__init__()
        self.register_buffer("cutoff", torch.tensor([cutoff], dtype=torch.float32))

    def forward(self, distances):
        cutoffs = 0.5 * (torch.cos(distances * math.pi / self.cutoff) + 1.0)
        cutoffs *= (distances < self.cutoff).float()
        return cutoffs


def get_all_spherical_bessel_roots(l_max: int, n_channels: int):
    roots = np.zeros((n_channels, l_max + 1))
    for l in range(l_max + 1):
        l_roots = []
        step = math.pi
        x1 = 0.1
        while len(l_roots) < n_channels:
            x2 = x1 + step
            while spherical_jn(l, x1) * spherical_jn(l, x2) > 0:
                x1 = x2
                x2 += step
            root = brentq(lambda x: spherical_jn(l, x), x1, x2)
            l_roots.append(root)
            x1 = root + 0.1
        roots[:, l] = l_roots
    return roots


class BesselRadialBasis(nn.Module):
    """
    Computes spherical Bessel radial basis for all l up to l_max simultaneously.
    Uses SciPy backend for absolute numerical stability at high l.
    """

    def __init__(self, l_max: int, cutoff: float, n_channels: int):
        super().__init__()
        self.l_max = l_max
        self.cutoff = cutoff
        self.n_channels = n_channels

        self.cutoff_func = CosineCutoff(cutoff=cutoff)

        roots = get_all_spherical_bessel_roots(l_max, n_channels)
        self.register_buffer("roots", torch.tensor(roots, dtype=torch.float32))

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        basis_list = []
        dist_expanded = dist.unsqueeze(-1)

        for l in range(self.l_max + 1):
            x_l = dist_expanded * self.roots[:, l] / self.cutoff

            # Apply the custom SciPy autograd function
            j_l_val = ScipySphericalBessel.apply(x_l, l)
            basis_list.append(j_l_val)

        basis = torch.stack(basis_list, dim=-1)
        env_vals = self.cutoff_func(dist).unsqueeze(-1).unsqueeze(-1)

        return basis * env_vals
