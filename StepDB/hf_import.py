"""HuggingFace model importer for StepDB.

Reads an HF model id, inlines every non-torch nn.Module class that's actually
used in the model's named_modules() tree (plus any module-level free helpers
they reference) into a single self-contained ``seed_kernels/hf_imported/<safe_id>/
reference.py``. After emission the kernel has no runtime dependency on
``transformers`` — the planner / refactor agent sees plain PyTorch.

Usage:
    python hf_import.py gpt2
    python hf_import.py facebook/opt-1.3b

What's transformed:
  - HF base classes (``PreTrainedModel``, ``GenerationMixin``,
    ``GradientCheckpointingLayer``, etc.) are rewritten to ``nn.Module``.
  - HF-internal class decorators (``@auto_docstring``, ``@can_return_tuple``,
    etc.) are stripped.
  - HF type hints in forward signatures that reference dataclasses /
    framework objects are stubbed as ``object``.

What's NOT yet handled (and will surface as runtime errors if encountered):
  - Free symbols in helper bodies that aren't classes-from-named_modules
    and aren't in the same modeling file (e.g., ``ACT2FN``, masking helpers).
    Add to ``_STUBS`` as they show up.
"""
from __future__ import annotations

import argparse
import ast
import inspect
import re
import sys
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM


STEPDB_DIR = Path(__file__).resolve().parent

# Class names that, when seen as a base class, are rewritten to nn.Module.
_BASE_REPLACEMENTS = {
    "PreTrainedModel",
    "GenerationMixin",
    "GradientCheckpointingLayer",
}
# We also detect any class whose name ends in ``PreTrainedModel`` (e.g.
# ``GPT2PreTrainedModel``) and rewrite to nn.Module.


# Decorator names that are HF-internal metadata and have no behavioral effect.
_STRIPPED_DECORATORS = {
    "auto_docstring",
    "can_return_tuple",
    "deprecate_kwarg",
    "add_start_docstrings",
    "add_start_docstrings_to_model_forward",
    "add_code_sample_docstrings",
    "replace_return_docstrings",
    "use_kernel_forward_from_hub",
    "capture_outputs",
    "install_all_output_capturing_hooks",
    "maybe_install_capturing_hooks",
    "merge_with_config_defaults",
}


# Names that appear as type-hint references but aren't classes/helpers we
# can inline. Each gets a do-nothing stub at file top. Object stubs accept
# any kwargs and store them; this is enough to satisfy isinstance checks
# and attribute reads on the return value.
_STUBS_PROLOGUE = '''\
# ----------------------------------------------------------------------
# Stubs for HF-internal symbols referenced by inlined source.
# These are placeholders just sufficient to make the file importable;
# none of them is called on the inference path of the wrapped Model.
# ----------------------------------------------------------------------

def _identity_decorator(*a, **kw):
    """Catch-all stub for HF decorators that only attach metadata.

    Handles both ``@dec`` (called with the function/class as the only arg)
    and ``@dec(...)`` (factory pattern — returns a wrapper). Distinguishing
    these requires guessing: if the single positional arg is a function or
    class, treat it as ``@dec``-style. Otherwise treat as factory.
    """
    if len(a) == 1 and not kw and (inspect.isfunction(a[0]) or inspect.isclass(a[0])):
        return a[0]
    def _wrap(obj):
        return obj
    return _wrap


# HF decorators / output-capture machinery that are stubbed as identity:
# they only attach metadata or install observability hooks and never affect
# the inference graph.
auto_docstring = _identity_decorator
can_return_tuple = _identity_decorator
deprecate_kwarg = _identity_decorator
add_start_docstrings = _identity_decorator
add_start_docstrings_to_model_forward = _identity_decorator
add_code_sample_docstrings = _identity_decorator
replace_return_docstrings = _identity_decorator
use_kernel_forward_from_hub = _identity_decorator
capture_outputs = _identity_decorator
install_all_output_capturing_hooks = _identity_decorator
maybe_install_capturing_hooks = _identity_decorator
merge_with_config_defaults = _identity_decorator


class _OpaqueStub:
    """Catch-all object used in place of HF dataclasses / utility singletons.

    Any attribute access returns a callable that returns 0; any method call
    therefore yields 0. Index/contain operations are no-op-friendly. This is
    enough to satisfy Cache.get_seq_length() etc. on the forward path.
    """
    def __init__(self, *a, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
    def __call__(self, *a, **kw):
        return self
    def __getitem__(self, key):
        return self
    def __contains__(self, key):
        return False
    def __getattr__(self, name):
        # Unknown attr → AttributeError so ``hasattr(stub, x)`` returns False,
        # letting forward-path ``if hasattr(...)`` branches fall through to
        # the no-cache / default code path.
        raise AttributeError(name)
    def __iter__(self):
        return iter(())
    def __bool__(self):
        return False
    def __len__(self):
        return 0


class _HFBase(nn.Module):
    """nn.Module shim used in place of HF base classes (PreTrainedModel etc.).

    HF base classes accept ``config`` (and sometimes other args) in their
    ``__init__`` and store ``self.config``; subclasses call
    ``super().__init__(config)``. This shim preserves that contract without
    pulling in the rest of the HF base-class machinery (weight loading,
    generation utilities, etc.).
    """
    def __init__(self, config=None, *args, **kwargs):
        super().__init__()
        if config is not None:
            self.config = config

    def _init_weights(self, module):  # called by some HF __init__ chains
        pass

    def post_init(self):  # called after __init__ in some HF classes
        pass

    def tie_weights(self):
        pass

    def get_input_embeddings(self):
        return None

    def gradient_checkpointing_enable(self, *a, **kw):
        pass

    def warn_if_padding_and_no_attention_mask(self, *a, **kw):
        pass

    @property
    def device(self):
        for p in self.parameters():
            return p.device
        return torch.device("cpu")

    @property
    def dtype(self):
        for p in self.parameters():
            return p.dtype
        return torch.float32


Cache = _OpaqueStub
DynamicCache = _OpaqueStub
EncoderDecoderCache = _OpaqueStub
StaticCache = _OpaqueStub
HybridCache = _OpaqueStub
SlidingWindowCache = _OpaqueStub
BaseModelOutput = _OpaqueStub
BaseModelOutputWithPast = _OpaqueStub
BaseModelOutputWithPastAndCrossAttentions = _OpaqueStub
CausalLMOutputWithPast = _OpaqueStub
CausalLMOutputWithCrossAttentions = _OpaqueStub
OutputRecorder = _OpaqueStub
BlockMask = _OpaqueStub
ALL_ATTENTION_FUNCTIONS = {}
ALL_MASK_ATTENTION_FUNCTIONS = {}


# ACT2FN: minimal map from activation-name to nn.Module class. Only entries
# the imported model actually uses need to resolve correctly. Anything else
# falls back to nn.Identity so the file imports cleanly.
class _ACT2FN(dict):
    def __missing__(self, key):
        return nn.Identity
ACT2FN = _ACT2FN(
    relu=nn.ReLU, gelu=nn.GELU, tanh=nn.Tanh, sigmoid=nn.Sigmoid,
    silu=nn.SiLU, swish=nn.SiLU,
)
'''


# Universal header: imports every inlined source can rely on, plus the
# decorator + type stubs.
_HEADER = '''\
"""AUTO-GENERATED by StepDB/hf_import.py — DO NOT EDIT BY HAND.

Self-contained PyTorch reference for {model_id}. All classes the model
actually instantiates have been inlined; the file has no `transformers`
dependency at run time. See StepDB/hf_import.py for the import flow.
"""
from __future__ import annotations

import contextlib
import inspect
import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache, wraps
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoConfig  # only for get_init_inputs(dims) → config


'''


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

def _decorator_name(node: ast.expr) -> str | None:
    """Top-level name of a decorator expression. `@auto_docstring(...)` → `auto_docstring`."""
    if isinstance(node, ast.Call):
        return _decorator_name(node.func)
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _base_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _should_replace_base(name: str | None) -> bool:
    if name is None:
        return False
    if name in _BASE_REPLACEMENTS:
        return True
    if name.endswith("PreTrainedModel"):  # e.g. GPT2PreTrainedModel
        return True
    return False


def _patch_class_source(src: str) -> str:
    """Rewrite HF bases to nn.Module on a class body. (Decorators are
    handled at runtime via identity stubs in the prologue.)"""
    tree = ast.parse(src)
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        new_bases: list[ast.expr] = []
        replaced = False
        for base in node.bases:
            if _should_replace_base(_base_name(base)):
                replaced = True
                continue
            new_bases.append(base)
        if replaced and not any(_base_name(b) == "_HFBase" for b in new_bases):
            new_bases.append(ast.Name(id="_HFBase", ctx=ast.Load()))
        node.bases = new_bases
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


# ---------------------------------------------------------------------------
# Class + helper collection
# ---------------------------------------------------------------------------

def _is_torch_class(cls: type) -> bool:
    return cls.__module__.startswith("torch.") or cls.__module__ == "torch"


def collect_inlined_classes(model: nn.Module) -> list[type]:
    """Walk model.named_modules() and return unique non-torch classes in dependency order.

    Order: dependencies first (e.g., ``Conv1D`` before ``GPT2Attention`` before
    ``GPT2Block`` before ``GPT2Model`` before ``GPT2LMHeadModel``). We achieve
    that by collecting in named_modules() order (root-first) then reversing.
    """
    seen: list[type] = []
    seen_set: set[type] = set()
    for _, sub in model.named_modules():
        cls = type(sub)
        if _is_torch_class(cls) or cls in seen_set:
            continue
        seen.append(cls)
        seen_set.add(cls)
    return list(reversed(seen))


def _free_names(src: str) -> set[str]:
    """Return all free-name references in ``src`` (best-effort via AST)."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            # Only the leftmost name in e.g. `foo.bar.baz` matters for resolution
            cur = node
            while isinstance(cur, ast.Attribute):
                cur = cur.value
            if isinstance(cur, ast.Name):
                names.add(cur.id)
    return names


def collect_module_helpers(classes: list[type], class_sources: dict[type, str]) -> list[tuple[str, str]]:
    """Recursively resolve free-name references in the inlined source.

    For every undefined name in the concatenated inlined source, search the
    namespaces of:
      1. each inlined class's defining module,
      2. that module's own ``vars()`` (its imports + module-level objects),
    and pull ``inspect.getsource`` for the resolved object if it's a function
    defined in ``transformers.*``. New helpers contribute new free names; we
    iterate until the fixpoint is reached.

    Returns ``[(name, source), ...]`` in collection order — leaf helpers
    first, so emit order is dependency-safe.
    """
    # Universally pre-defined names (header + stubs + class names + per-class params).
    pre_defined: set[str] = {
        "torch", "nn", "F", "Tensor", "math", "contextlib", "inspect",
        "Callable", "Optional", "Tuple", "Any", "Union",
        "dataclass", "AutoConfig", "lru_cache", "wraps",
        "Cache", "DynamicCache", "EncoderDecoderCache",
        "BaseModelOutput", "BaseModelOutputWithPast",
        "BaseModelOutputWithPastAndCrossAttentions",
        "CausalLMOutputWithPast", "CausalLMOutputWithCrossAttentions",
        "OutputRecorder", "ALL_ATTENTION_FUNCTIONS", "ACT2FN", "_OpaqueStub",
        "_identity_decorator", "_ACT2FN", "_HFBase",
        "StaticCache", "HybridCache", "SlidingWindowCache", "BlockMask",
        "ALL_MASK_ATTENTION_FUNCTIONS",
    }
    pre_defined |= _STRIPPED_DECORATORS  # stubbed identity decorators
    pre_defined |= {cls.__name__ for cls in classes}
    pre_defined |= set(dir(__builtins__))

    # Pool of candidate namespaces to search.
    src_mods = {sys.modules[c.__module__] for c in classes if c.__module__ in sys.modules}

    helpers: list[tuple[str, str]] = []
    seen_names: set[str] = set()
    helper_text = ""

    def all_text() -> str:
        return "\n".join(class_sources.values()) + "\n" + helper_text

    def find_in_mods(name: str):
        for mod in src_mods:
            if not hasattr(mod, name):
                continue
            obj = getattr(mod, name)
            if inspect.isclass(obj):
                continue
            if not callable(obj):
                continue
            # Only inline objects defined in transformers (skip stdlib/torch)
            owner_mod = getattr(obj, "__module__", "") or ""
            if not owner_mod.startswith("transformers"):
                continue
            return obj
        return None

    progress = True
    while progress:
        progress = False
        names = _free_names(all_text()) - pre_defined - seen_names
        for name in sorted(names):
            obj = find_in_mods(name)
            if obj is None:
                continue
            try:
                src = inspect.getsource(obj)
            except (OSError, TypeError):
                continue
            helpers.append((name, src))
            helper_text += "\n" + src + "\n"
            seen_names.add(name)
            # Also register the module that defined the new helper, so its
            # own dependencies become resolvable in the next iteration.
            owner_mod_name = getattr(obj, "__module__", "")
            if owner_mod_name in sys.modules:
                src_mods.add(sys.modules[owner_mod_name])
            progress = True
    return helpers


# ---------------------------------------------------------------------------
# reference.py emission
# ---------------------------------------------------------------------------

_TRAILER_TEMPLATE = '''
# ----------------------------------------------------------------------
# StepDB wrapper
# ----------------------------------------------------------------------

class Model(nn.Module):
    def __init__(self, model_name, config, state_dict):
        super().__init__()
        self.model = {top_class}(config)
        self.model.load_state_dict(state_dict, strict=True)

    def forward(self, x):
        out = self.model(x)
        return out.logits if hasattr(out, "logits") else out


def get_init_inputs(dims):
    from precompute import precompute_tensors
    t = precompute_tensors("{kernel_name}", dims)
    state_dict = {{k: v for k, v in t.items() if k != "input_ids"}}
    return [dims["model_name"],
            AutoConfig.from_pretrained(dims["model_name"]),
            state_dict]


def get_inputs(dims):
    from precompute import precompute_tensors
    return [precompute_tensors("{kernel_name}", dims)["input_ids"]]


def compute_gold(dims):
    with torch.no_grad():
        return Model(*get_init_inputs(dims))(*get_inputs(dims))
'''


def emit_reference(model_id: str, kernel_name: str, out_path: Path) -> None:
    cfg = AutoConfig.from_pretrained(model_id)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg)

    classes = collect_inlined_classes(model)
    class_sources = {cls: inspect.getsource(cls) for cls in classes}
    helpers = collect_module_helpers(classes, class_sources)

    patched_class_sources = [_patch_class_source(class_sources[cls]) for cls in classes]

    top_class = type(model).__qualname__

    parts = [
        _HEADER.format(model_id=model_id),
        _STUBS_PROLOGUE,
        "\n# ----------------------------------------------------------------------\n",
        "# Inlined module-level helpers\n",
        "# ----------------------------------------------------------------------\n",
        *(f"\n{src}\n" for _name, src in helpers),
        "\n# ----------------------------------------------------------------------\n",
        "# Inlined nn.Module classes (dependencies first)\n",
        "# ----------------------------------------------------------------------\n",
        *(f"\n{src}\n" for src in patched_class_sources),
        _TRAILER_TEMPLATE.format(top_class=top_class, kernel_name=kernel_name),
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(parts))


def _safe_model_id(model_id: str) -> str:
    return model_id.replace("/", "_").replace("-", "_").replace(".", "p").lower()


def main():
    parser = argparse.ArgumentParser(description="Inline a HuggingFace model into a self-contained StepDB reference.py")
    parser.add_argument("model_id", help="HF model id, e.g. 'gpt2' or 'facebook/opt-1.3b'")
    parser.add_argument("--kernel-name", default=None,
                        help="bench_config kernel name (default: hf__<safe_model_id>)")
    parser.add_argument("--out-dir", default=None,
                        help="output dir (default: seed_kernels/hf_imported/<safe_model_id>/)")
    args = parser.parse_args()

    safe = _safe_model_id(args.model_id)
    kernel_name = args.kernel_name or f"hf__{safe}"
    out_dir = Path(args.out_dir) if args.out_dir else (STEPDB_DIR / "seed_kernels" / "hf_imported" / safe)
    out_path = out_dir / "reference.py"

    print(f"[hf_import] model_id={args.model_id!r} kernel_name={kernel_name!r}")
    print(f"[hf_import] writing → {out_path}")
    emit_reference(args.model_id, kernel_name, out_path)
    print(f"[hf_import] done. lines: {sum(1 for _ in open(out_path))}")


if __name__ == "__main__":
    main()
