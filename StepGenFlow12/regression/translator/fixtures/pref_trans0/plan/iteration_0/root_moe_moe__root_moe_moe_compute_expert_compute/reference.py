import torch
import torch.nn as nn
import torch.nn.functional as F

def _rms_norm(x, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,
                normed_2: torch.Tensor,
                tok: torch.Tensor,
                w_gate: torch.Tensor,
                w_up: torch.Tensor,
                w_down: torch.Tensor):
        """
        Compute the expert's contribution for a given set of token indices.
        """
        # [N, INTER] = [N, DIM] @ [DIM, INTER]
        gate_out = normed_2[tok] @ w_gate
        # [N, INTER] = [N, DIM] @ [DIM, INTER]
        up_out = normed_2[tok] @ w_up
        # SiLU activation + element‑wise multiply
        hidden = F.silu(gate_out) * up_out
        # [N, DIM] = [N, INTER] @ [INTER, DIM]
        down_out = hidden @ w_down
        return down_out

def get_inputs(dims):
    # Produce dummy inputs that match the signature of the child.
    torch.manual_seed(201)   # distinct seed for this child

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

    # Dummy residual tensor → RMSNorm to obtain normed_2
    res_add_0 = torch.randn(seq_len, mc.dim)
    normed_2 = _rms_norm(res_add_0)

    # Random token indices for this expert (ensure non‑empty)
    N = max(1, seq_len // 4)   # arbitrary non‑zero size
    tok = torch.randint(0, seq_len, (N,), dtype=torch.long)

    # Single‑expert weight matrices
    w_gate = torch.randn(mc.dim, mc.moe_inter_dim)
    w_up   = torch.randn(mc.dim, mc.moe_inter_dim)
    w_down = torch.randn(mc.moe_inter_dim, mc.dim)

    return [normed_2, tok, w_gate, w_up, w_down]


def get_init_inputs(dims):
    return []


def compute_gold(dims):
    inputs = get_inputs(dims)
    return Model()(*inputs)
