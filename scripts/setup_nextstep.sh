#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/setup_nextstep.sh [--dry-run] [--skip-mongodb] [--env-name NAME] [--conda-prefix DIR]

Set up the NextStep step_tl toolchain on a fresh machine:
  1. create or update the conda environment
  2. install Python build/proto helpers
  3. regenerate Python protobuf bindings
  4. build step_tl and step_perf with maturin
  5. install/start local MongoDB for simulator event logging
  6. verify Python imports

Defaults:
  --env-name testenv
  --conda-prefix "$HOME/miniforge3"
USAGE
}

dry_run=0
setup_mongodb=1
env_name="testenv"
conda_prefix="${HOME:-/root}/miniforge3"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      dry_run=1
      shift
      ;;
    --skip-mongodb)
      setup_mongodb=0
      shift
      ;;
    --env-name)
      env_name="${2:-}"
      if [[ -z "$env_name" ]]; then
        echo "error: --env-name requires a value" >&2
        exit 2
      fi
      shift 2
      ;;
    --conda-prefix)
      conda_prefix="${2:-}"
      if [[ -z "$conda_prefix" ]]; then
        echo "error: --conda-prefix requires a value" >&2
        exit 2
      fi
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
step_tl_dir="$repo_root/step_tl"
step_perf_dir="$step_tl_dir/step-perf"
env_file="$repo_root/environment.yml"
proto_dir="$step_tl_dir/step_perf_ir/proto"
python_proto_out="$step_tl_dir/src/proto"
mongodb_uri="mongodb://127.0.0.1:27017"
mongodb_data_dir="$repo_root/.local/mongodb/db"
mongodb_log_dir="$repo_root/.local/mongodb/log"
mongodb_log_path="$mongodb_log_dir/mongod.log"

run() {
  printf '+ %s\n' "$*"
  if [[ "$dry_run" -eq 0 ]]; then
    "$@"
  fi
}

run_shell() {
  printf '+ %s\n' "$*"
  if [[ "$dry_run" -eq 0 ]]; then
    bash -lc "$*"
  fi
}

if [[ ! -f "$env_file" ]]; then
  echo "error: missing conda environment file: $env_file" >&2
  exit 1
fi

find_conda() {
  if command -v conda >/dev/null 2>&1; then
    command -v conda
  elif [[ -x "$conda_prefix/bin/conda" ]]; then
    printf '%s\n' "$conda_prefix/bin/conda"
  elif [[ -x "/root/miniconda3/bin/conda" ]]; then
    printf '%s\n' "/root/miniconda3/bin/conda"
  elif [[ -x "/opt/conda/bin/conda" ]]; then
    printf '%s\n' "/opt/conda/bin/conda"
  fi
}

install_miniforge() {
  local os arch installer url
  os="$(uname -s)"
  arch="$(uname -m)"

  case "$os:$arch" in
    Linux:x86_64) installer="Miniforge3-Linux-x86_64.sh" ;;
    Linux:aarch64) installer="Miniforge3-Linux-aarch64.sh" ;;
    Darwin:x86_64) installer="Miniforge3-MacOSX-x86_64.sh" ;;
    Darwin:arm64) installer="Miniforge3-MacOSX-arm64.sh" ;;
    *)
      echo "error: unsupported platform for automatic Miniforge install: $os $arch" >&2
      exit 1
      ;;
  esac

  url="https://github.com/conda-forge/miniforge/releases/latest/download/$installer"

  if [[ "$dry_run" -eq 1 ]]; then
    printf '+ install Miniforge to %s if conda is missing\n' "$conda_prefix"
    return
  fi

  if [[ -e "$conda_prefix" && ! -x "$conda_prefix/bin/conda" ]]; then
    echo "error: conda prefix exists but does not contain conda: $conda_prefix" >&2
    exit 1
  fi

  if [[ -x "$conda_prefix/bin/conda" ]]; then
    return
  fi

  local tmp
  tmp="$(mktemp -t miniforge.XXXXXX.sh)"
  if command -v curl >/dev/null 2>&1; then
    run curl -L "$url" -o "$tmp"
  elif command -v wget >/dev/null 2>&1; then
    run wget -O "$tmp" "$url"
  else
    echo "error: need curl or wget to install Miniforge" >&2
    exit 1
  fi
  run bash "$tmp" -b -p "$conda_prefix"
  rm -f "$tmp"
}

install_rust() {
  if [[ "$dry_run" -eq 1 ]]; then
    printf '+ install Rust toolchain with rustup if cargo or rustc is missing\n'
    return
  fi

  if command -v cargo >/dev/null 2>&1 && command -v rustc >/dev/null 2>&1; then
    return
  fi

  local tmp
  tmp="$(mktemp -t rustup.XXXXXX.sh)"
  if command -v curl >/dev/null 2>&1; then
    run curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o "$tmp"
  elif command -v wget >/dev/null 2>&1; then
    run wget -O "$tmp" https://sh.rustup.rs
  else
    echo "error: need curl or wget to install Rust with rustup" >&2
    exit 1
  fi
  run sh "$tmp" -y --profile minimal
  rm -f "$tmp"

  if [[ -f "${CARGO_HOME:-$HOME/.cargo}/env" ]]; then
    # shellcheck disable=SC1091
    source "${CARGO_HOME:-$HOME/.cargo}/env"
  fi
}

create_protoc_wrapper() {
  local wrapper python_bin

  if [[ "$dry_run" -eq 1 ]]; then
    wrapper="$conda_prefix/envs/$env_name/bin/protoc-grpc-tools"
    printf '+ create grpc_tools protoc wrapper at %s\n' "$wrapper"
    return
  fi

  python_bin="$(command -v python)"
  wrapper="${CONDA_PREFIX:?}/bin/protoc-grpc-tools"

  if ! python -c "import grpc_tools.protoc" >/dev/null 2>&1; then
    echo "error: grpcio-tools is required before creating the protoc wrapper" >&2
    exit 1
  fi

  printf '+ create grpc_tools protoc wrapper at %s\n' "$wrapper"
  cat >"$wrapper" <<EOF
#!/usr/bin/env bash
exec "$python_bin" -m grpc_tools.protoc "\$@"
EOF
  chmod +x "$wrapper"
}

install_native_build_deps() {
  if [[ "$dry_run" -eq 1 ]]; then
    printf '+ install native build dependencies if missing\n'
    return
  fi

  if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists openssl >/dev/null 2>&1 && pkg-config --exists libgvc >/dev/null 2>&1; then
    if command -v m4 >/dev/null 2>&1; then
      return
    fi
  fi

  if command -v apt-get >/dev/null 2>&1 && [[ "$(id -u)" -eq 0 ]]; then
    run apt-get update
    run apt-get install -y build-essential pkg-config libssl-dev libgraphviz-dev graphviz m4
  elif command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null && command -v apt-get >/dev/null 2>&1; then
    run sudo apt-get update
    run sudo apt-get install -y build-essential pkg-config libssl-dev libgraphviz-dev graphviz m4
  elif command -v brew >/dev/null 2>&1; then
    run brew install pkgconf openssl graphviz m4
  else
    echo "error: pkg-config, OpenSSL headers, and Graphviz headers are required" >&2
    exit 1
  fi
}

install_mongodb() {
  if [[ "$dry_run" -eq 1 ]]; then
    printf '+ install MongoDB 7.0 and start local mongod if needed\n'
    printf '+ python -c from pymongo import MongoClient; assert MongoClient('\''%s'\'').admin.command('\''ping'\'')['\''ok'\''] == 1.0\n' "$mongodb_uri"
    return
  fi

  if python -c "from pymongo import MongoClient; assert MongoClient('$mongodb_uri', serverSelectionTimeoutMS=1000).admin.command('ping')['ok'] == 1.0" >/dev/null 2>&1; then
    return
  fi

  if ! command -v apt-get >/dev/null 2>&1; then
    echo "error: automatic MongoDB setup currently requires apt-get" >&2
    exit 1
  fi
  if [[ ! -r /etc/os-release ]]; then
    echo "error: automatic MongoDB setup requires /etc/os-release" >&2
    exit 1
  fi

  local distro_id distro_codename
  distro_id="$(. /etc/os-release && printf '%s' "${ID:-}")"
  distro_codename="$(. /etc/os-release && printf '%s' "${VERSION_CODENAME:-}")"
  if [[ "$distro_id" != "ubuntu" || -z "$distro_codename" ]]; then
    echo "error: automatic MongoDB setup currently supports Ubuntu apt hosts; use --skip-mongodb on this platform" >&2
    exit 1
  fi

  local apt_prefix
  if [[ "$(id -u)" -eq 0 ]]; then
    apt_prefix=""
  elif command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
    apt_prefix="sudo"
  else
    echo "error: MongoDB setup requires root or passwordless sudo for apt-get" >&2
    exit 1
  fi

  run_shell "$apt_prefix apt-get update"
  run_shell "$apt_prefix DEBIAN_FRONTEND=noninteractive apt-get install -y ca-certificates curl gnupg"
  run_shell "curl -fsSL https://pgp.mongodb.com/server-7.0.asc | $apt_prefix gpg --batch --yes -o /usr/share/keyrings/mongodb-server-7.0.gpg --dearmor"
  run_shell "echo 'deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-7.0.gpg ] https://repo.mongodb.org/apt/ubuntu $distro_codename/mongodb-org/7.0 multiverse' | $apt_prefix tee /etc/apt/sources.list.d/mongodb-org-7.0.list >/dev/null"
  run_shell "$apt_prefix apt-get update"
  run_shell "$apt_prefix DEBIAN_FRONTEND=noninteractive apt-get install -y mongodb-org"

  run mkdir -p "$mongodb_data_dir" "$mongodb_log_dir"
  if ! command -v mongod >/dev/null 2>&1; then
    echo "error: mongod was not found after MongoDB installation" >&2
    exit 1
  fi
  if ! ss -ltn 2>/dev/null | grep -q ':27017'; then
    run mongod --dbpath "$mongodb_data_dir" --bind_ip 127.0.0.1 --port 27017 --logpath "$mongodb_log_path" --fork
  fi
  run python -c "from pymongo import MongoClient; assert MongoClient('$mongodb_uri', serverSelectionTimeoutMS=3000).admin.command('ping')['ok'] == 1.0"
}

install_miniforge
conda_bin="$(find_conda || true)"

if [[ "$dry_run" -eq 0 && -z "$conda_bin" ]]; then
  echo "error: conda was not found after Miniforge install attempt" >&2
  exit 1
fi

if [[ "$dry_run" -eq 0 ]]; then
  eval "$("$conda_bin" shell.bash hook)"
fi

install_rust

if [[ "$dry_run" -eq 0 ]]; then
  if [[ -f "${CARGO_HOME:-$HOME/.cargo}/env" ]]; then
    # shellcheck disable=SC1091
    source "${CARGO_HOME:-$HOME/.cargo}/env"
  fi
  if ! command -v cargo >/dev/null 2>&1 || ! command -v rustc >/dev/null 2>&1; then
    echo "error: cargo and rustc are required, but Rust installation did not provide them" >&2
    exit 1
  fi
fi

if [[ "$dry_run" -eq 1 ]]; then
  run conda env update -n "$env_name" -f "$env_file"
else
  if conda env list | awk '{print $1}' | grep -Fx "$env_name" >/dev/null; then
    run conda env update -n "$env_name" -f "$env_file"
  else
    run conda env create -n "$env_name" -f "$env_file"
  fi
fi

if [[ "$dry_run" -eq 0 ]]; then
  conda activate "$env_name"
fi

unset VIRTUAL_ENV

run python -m pip install maturin grpcio-tools

run python -c "import torch, torch._refs; from pathlib import Path; root = Path(torch.__file__).resolve().parent; assert (root / 'lib' / 'libtorch_global_deps.so').exists(), root; print('torch:', torch.__version__, torch.__file__)"

run mkdir -p "$python_proto_out"
run_shell "cd $proto_dir && python -m grpc_tools.protoc -I$proto_dir --python_out=$python_proto_out datatype.proto func.proto ops.proto graph.proto"

run_shell "cd $step_tl_dir && maturin develop --release"

create_protoc_wrapper
install_native_build_deps
if [[ "$setup_mongodb" -eq 1 ]]; then
  install_mongodb
fi

if [[ "$dry_run" -eq 0 ]]; then
  protoc_path="${CONDA_PREFIX:?}/bin/protoc-grpc-tools"
  if [[ -z "$protoc_path" ]]; then
    echo "error: could not find protoc after install attempt" >&2
    exit 1
  fi
else
  protoc_path="$conda_prefix/envs/$env_name/bin/protoc-grpc-tools"
fi

run_shell "cd $step_perf_dir && PROTOC=$protoc_path maturin develop --release"

run python -c "import step_tl, step_perf; print('step_tl:', step_tl.__file__); print('step_perf:', step_perf.__file__)"

cat <<EOF

Setup complete for conda env: $env_name

To use it in a new shell:
  conda activate $env_name
  cd $step_tl_dir
  source setup.sh
EOF
