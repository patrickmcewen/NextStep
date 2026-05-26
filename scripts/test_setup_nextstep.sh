#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
script="$repo_root/scripts/setup_nextstep.sh"

output="$("$script" --dry-run)"

grep -F "conda env update -n testenv -f $repo_root/environment.yml" <<<"$output" >/dev/null
grep -F "install Miniforge to " <<<"$output" >/dev/null
grep -F "install Rust toolchain with rustup if cargo or rustc is missing" <<<"$output" >/dev/null
grep -F "python -m pip install maturin grpcio-tools" <<<"$output" >/dev/null
grep -F "openai-agents==0.13.6" "$repo_root/environment.yml" >/dev/null
grep -F "tiktoken==0.13.0" "$repo_root/environment.yml" >/dev/null
grep -F "torch==2.11.0" "$repo_root/environment.yml" >/dev/null
grep -F "transformers==4.57.6" "$repo_root/environment.yml" >/dev/null
grep -F "import torch, torch._refs" <<<"$output" >/dev/null
grep -F "libtorch_global_deps.so" <<<"$output" >/dev/null
grep -F "mkdir -p $repo_root/step_tl/src/proto" <<<"$output" >/dev/null
grep -F "cd $repo_root/step_tl/step_perf_ir/proto && python -m grpc_tools.protoc -I$repo_root/step_tl/step_perf_ir/proto --python_out=$repo_root/step_tl/src/proto datatype.proto func.proto ops.proto graph.proto" <<<"$output" >/dev/null
grep -F "cd $repo_root/step_tl && maturin develop --release" <<<"$output" >/dev/null
grep -F "create grpc_tools protoc wrapper at " <<<"$output" >/dev/null
grep -F "install native build dependencies if missing" <<<"$output" >/dev/null
grep -F "install MongoDB 7.0 and start local mongod if needed" <<<"$output" >/dev/null
grep -F "MongoClient('mongodb://127.0.0.1:27017'" <<<"$output" >/dev/null
grep -F "libgraphviz-dev" "$repo_root/scripts/setup_nextstep.sh" >/dev/null
grep -F "cd $repo_root/step_tl/step-perf && PROTOC=" <<<"$output" >/dev/null
grep -F "maturin develop --release" <<<"$output" >/dev/null
grep -F "python -c import step_tl, step_perf;" <<<"$output" >/dev/null
