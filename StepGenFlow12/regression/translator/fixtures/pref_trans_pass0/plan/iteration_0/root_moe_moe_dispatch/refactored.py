import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.rms_norm = RmsNormModel()
        self.moe_dispatch__root_moe_moe_dispatch = MoeDispatchRootMoeMoeDispatchModel()

    def forward(
        self,
        res_add_0,
        w_gate,
        w_up,
        w_down,
        expert_weights,
        expert_onehot,
    ):
        normed_2 = self.rms_norm(res_add_0)
        return self.moe_dispatch__root_moe_moe_dispatch(
            normed_2,
            w_gate,
            w_up,
            w_down,
            expert_weights,
            expert_onehot,
        )

def get_init_inputs(dims):
    return []

def compute_gold(dims):
    inputs = get_inputs(dims)
    return Model()(*inputs)