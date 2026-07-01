#!/bin/bash
#
# Lambda Labs droplet setup script
# ================================
#
# Run this once on a gpu_1x_h100_sxm5 (or something similar
# Pretty much installs some dependencies, creates a Python venv, installs
# Python deps, builds stubs stubs, and clones+builds the NIST 
# assessment tool.
#
# Usage:
#   bash scripts/setup_lambda.sh
#
# After this completes, run benchmarks with:
#   bash scripts/run_all_benchmarks.sh
#
# Tested on:  Ubuntu 22.04 LTS, Lambda Labs gpu_1x_h100_sxm5
# ----------------------------------------------------------------------

set -e   # Exit on any error

# Color helpers
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log()    { echo -e "${GREEN}[setup]${NC} $1"; }
warn()   { echo -e "${YELLOW}[warn]${NC}  $1"; }
fatal()  { echo -e "${RED}[fatal]${NC} $1"; exit 1; }

cd "$(dirname "$0")/.."
ROOT_DIR=$(pwd)

log "Working directory: $ROOT_DIR"

# ----------------------------------------------------------------------
# 1. System packages
# ----------------------------------------------------------------------

log "Installing system packages (apt)..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3 python3-venv python3-pip python3-dev \
    build-essential git curl wget \
    libdivsufsort-dev \
    pkg-config

# ----------------------------------------------------------------------
# 2. Verify CUDA
# ----------------------------------------------------------------------

log "Verifying CUDA installation..."
if ! command -v nvcc &> /dev/null; then
    warn "nvcc not found. Lambda Labs images usually include CUDA already."
    warn "If this is a fresh non-Lambda image you may need to install CUDA 12.x."
    warn "Skipping nvcc check; CuPy will use the CUDA runtime libraries."
else
    nvcc --version | tail -n 1
fi

if ! command -v nvidia-smi &> /dev/null; then
    fatal "nvidia-smi not found. NVIDIA driver not installed."
fi

log "GPU information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# ----------------------------------------------------------------------
# 3. Python virtual environment
# ----------------------------------------------------------------------

if [ ! -d "venv" ]; then
    log "Creating Python venv..."
    python3 -m venv venv
else
    log "Python venv already exists; reusing."
fi

# shellcheck disable=SC1091
source venv/bin/activate

log "Upgrading pip..."
pip install --upgrade pip wheel setuptools

# ----------------------------------------------------------------------
# 4. Python dependencies
# ----------------------------------------------------------------------

log "Installing Python dependencies (this may take a few minutes)..."
pip install -r requirements.txt

# Verify CuPy can talk to the GPU
log "Verifying CuPy / GPU..."
python -c "
import cupy as cp
n_devs = cp.cuda.runtime.getDeviceCount()
print(f'  Detected {n_devs} CUDA device(s)')
for i in range(n_devs):
    props = cp.cuda.runtime.getDeviceProperties(i)
    name = props['name']
    if isinstance(name, bytes):
        name = name.decode('utf-8', errors='replace')
    mem_gb = props['totalGlobalMem'] / (1024**3)
    print(f'  Device {i}: {name} ({mem_gb:.1f} GB)')

# Quick functional test
arr = cp.arange(1000, dtype=cp.float32)
assert float(arr.sum()) == sum(range(1000)), 'CuPy basic test failed'
print('  CuPy basic test: OK')
"

# ----------------------------------------------------------------------
# 5. Generate gRPC stubs
# ----------------------------------------------------------------------

log "Building gRPC stubs from proto file..."
bash scripts/build_proto.sh

# Test that the proto stubs import cleanly
python -c "
from src.proto_generated import entropy_service_pb2
from src.proto_generated import entropy_service_pb2_grpc
print('  Proto stubs import OK')
"

# ----------------------------------------------------------------------
# 6. NIST SP 800-90B entropy assessment tool (optional but recommended)
# ----------------------------------------------------------------------

NIST_DIR="$HOME/SP800-90B_EntropyAssessment"
if [ ! -d "$NIST_DIR" ]; then
    log "Cloning NIST SP 800-90B EntropyAssessment tool..."
    git clone --depth 1 https://github.com/usnistgov/SP800-90B_EntropyAssessment.git "$NIST_DIR" || \
        warn "Failed to clone NIST tool — will fall back to built-in MCV estimator"
fi

if [ -d "$NIST_DIR" ] && [ ! -x "$NIST_DIR/cpp/ea_non_iid" ]; then
    log "Building NIST tool..."
    (
        cd "$NIST_DIR/cpp"
        make -j"$(nproc)" 2>&1 | tail -n 5 || warn "Build failed; will use fallback estimator"
    )
fi

if [ -x "$NIST_DIR/cpp/ea_non_iid" ]; then
    log "NIST tool built successfully:"
    log "  $NIST_DIR/cpp/ea_non_iid"
    # Make it discoverable
    sudo ln -sf "$NIST_DIR/cpp/ea_non_iid"  /usr/local/bin/ea_non_iid  2>/dev/null || true
    sudo ln -sf "$NIST_DIR/cpp/ea_iid"      /usr/local/bin/ea_iid      2>/dev/null || true
    sudo ln -sf "$NIST_DIR/cpp/ea_restart"  /usr/local/bin/ea_restart  2>/dev/null || true
fi

# ----------------------------------------------------------------------
# 7. Quick sanity check on the codebase
# ----------------------------------------------------------------------

log "Running CUDA kernel self-test..."
python -m src.cuda.kernels 2>&1 | tail -n 20 || warn "Kernel self-test had issues"

log "Running entropy simulator self-test..."
python -m src.simulation.entropy_sim 2>&1 | tail -n 20 || warn "Simulator self-test had issues"

log "Running attestation self-test..."
python -m src.server.attestation 2>&1 | tail -n 10 || warn "Attestation self-test had issues"

# ----------------------------------------------------------------------
# 8. Done
# ----------------------------------------------------------------------

log ""
log "============================================================"
log "Setup complete."
log "============================================================"
log ""
log "To run all benchmarks (paper-ready data):"
log "    source venv/bin/activate"
log "    bash scripts/run_all_benchmarks.sh"
log ""
log "To run the gRPC server:"
log "    source venv/bin/activate"
log "    python -m src.server.grpc_server"
log ""
log "To run the demo client:"
log "    source venv/bin/activate"
log "    python -m src.client.client"
log ""
log "Optional: expose server externally via CloudFlare Tunnel:"
log "    bash scripts/setup_cloudflare.sh"
log ""
