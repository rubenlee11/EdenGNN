"""
Helper class that stores Clebsch-Gordan coefficients
"""

from itertools import permutations
import numpy as np
import torch, os


class ClebschGordan_deprecated(torch.nn.Module):
    def __init__(self):
        super(ClebschGordan, self).__init__()
        tmp = np.load(
            os.path.join(
                "/root/research/ml_hybrid_functional/EdenGNN-hybrid/src/edengnn/model/utils/clebsch_gordan_coefficients_L10.npz"
            ),
            allow_pickle=True,
        )["cg"][()]
        # add permutations (the npz file only stores coefficients for l1 <= l2 <= l3) and register buffers
        for l123 in tmp.keys():
            for a, b, c in permutations((0, 1, 2)):
                name = "cg_{}_{}_{}".format(l123[a], l123[b], l123[c])
                if name not in dir(self):
                    self.register_buffer(
                        name, torch.tensor(tmp[l123].transpose(a, b, c))
                    )

    def forward(self, l1, l2, l3):
        return getattr(self, "cg_{}_{}_{}".format(l1, l2, l3))


class ClebschGordan(torch.nn.Module):
    def __init__(self):
        super().__init__()
        tmp = np.load(
            "/root/research/ml_hybrid_functional/EdenGNN-hybrid/src/edengnn/model/utils/clebsch_gordan_coefficients_L10.npz",
            allow_pickle=True,
        )["cg"][()]

        # 使用 ParameterDict 统一管理，Lightning 能 100% 追踪到它
        self.cg_dict = torch.nn.ParameterDict()

        for l123 in tmp.keys():
            for a, b, c in permutations((0, 1, 2)):
                name = "cg_{}_{}_{}".format(l123[a], l123[b], l123[c])
                if name not in self.cg_dict:
                    # 转换为 float32 并作为不可训练的 Parameter 存入
                    tensor_cg = torch.tensor(
                        tmp[l123].transpose(a, b, c), dtype=torch.float32
                    )
                    self.cg_dict[name] = torch.nn.Parameter(
                        tensor_cg, requires_grad=False
                    )

    def forward(self, l1, l2, l3):
        name = "cg_{}_{}_{}".format(l1, l2, l3)
        return self.cg_dict[name]
