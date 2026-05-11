import torch
import torch.nn as nn

def _rms_norm(x, eps=1e-6):
    """Root‑mean‑square layer‑norm applied per head."""
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K):
        """
        Apply per‑head RMSNorm to the Q and K tensors.
        """
        Q_norm = _rms_norm(Q)
        K_norm = _rms_norm(K)
        return Q_norm, K_norm

def get_inputs(dims):
    """
    Generates placeholder Q and K tensors of the correct shape for the
    current model configuration.  The actual values are irrelevant because
    the child only normalises them.
    """
    torch.manual_seed(203)   # distinct seed for this child

    model_name = dims["model_name"]
    seq_len = dims["seq_len"]
    is_small = dims.get("is_small", False)

    if model_name == "mixtral":
        from end_to_end.model_configs import SmallerMixtral8x7B, Mixtral8x7B
        mc = SmallerMixtral8x7B() if is_small else Mixtral8x7B()
    elif model_name == "qwen":
        from end_to_end.model_configs import SmallerQwen30B, Qwen30B
        mc = SmallerQwen30B() if is_small else Qwen30B()
    else:
        raise AssertionError(f"Unknown model_name: {model_name!r}")

    Q = torch.randn(seq_len, mc.num_heads, mc.head_dim)
    K = torch.randn(seq_len, mc.num_kv_heads, mc.head_dim)

    return [Q, K]


def get_init_inputs(dims):
    return []


def compute_gold(dims):
    inputs = get_inputs(dims)
    return Model()(*inputs)
