import torch
import torch.nn as nn

# The two child classes are automatically made available as
# PreAttentionAndQkvModel and PerHeadNormModel (derived from the child blocks).

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # Instantiate the sub‑modules.
        self.pre_attention_and_qkv = PreAttentionAndQkvModel()
        self.per_head_norm = PerHeadNormModel()

    def forward(self, input_tensor, q_proj, k_proj, v_proj):
        # ---------------------------------------------------------
        # 1️⃣ Pre‑attention RMSNorm + QKV projections (child 1)
        # ---------------------------------------------------------
        Q, K, V = self.pre_attention_and_qkv(
            input_tensor, q_proj, k_proj, v_proj
        )
        # ---------------------------------------------------------
        # 2️⃣ Per‑head RMSNorm on Q and K (child 2)
        # ---------------------------------------------------------
        Q_norm, K_norm = self.per_head_norm(Q, K)

        # Return the same tuple shape as the original model.
        return Q_norm, K_norm, V


def get_init_inputs(dims):
    return []


def compute_gold(dims):
    inputs = get_inputs(dims)
    return Model()(*inputs)
