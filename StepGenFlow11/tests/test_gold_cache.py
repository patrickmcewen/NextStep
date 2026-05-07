import torch
from src import gold_cache


def test_inject_then_get_returns_injected_tensor():
    fake_dims = {"M": 4}
    fake_gold = torch.zeros(4, 8)
    gold_cache._inject_gold("__synthetic_test__", fake_dims, fake_gold)
    out = gold_cache._get_gold("__synthetic_test__", fake_dims)
    assert torch.equal(out, fake_gold)


def test_inject_tuple():
    fake_dims = {"M": 7}
    fake_gold = (torch.zeros(3), torch.ones(3))
    gold_cache._inject_gold("__synthetic_tuple__", fake_dims, fake_gold)
    out = gold_cache._get_gold("__synthetic_tuple__", fake_dims)
    assert isinstance(out, tuple) and len(out) == 2
    assert torch.equal(out[0], fake_gold[0])
    assert torch.equal(out[1], fake_gold[1])


def test_gold_key_distinguishes_dims():
    k1 = gold_cache._gold_key("foo", {"M": 4})
    k2 = gold_cache._gold_key("foo", {"M": 8})
    k3 = gold_cache._gold_key("bar", {"M": 4})
    assert k1 != k2 and k1 != k3 and k2 != k3
