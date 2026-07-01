#!/bin/bash
#
# Run all benchmarks and generate paper-ready results.
# ====================================================
#
# Each benchmark is gonna write:
#   results/raw/<name>.json       
#   results/tables/<name>.md      —table for paper
#   results/figures/<name>.png    — figs for paper
#
# The paper/results_summary.md is generated from the JSON.
#
# Usage:
#   source venv/bin/activate
#   bash scripts/run_all_benchmarks.sh
#
# Approximate runtime on H100 SXM:  10–15 minutes total
# ----------------------------------------------------------------------

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[bench]${NC} $1"; }
warn() { echo -e "${YELLOW}[warn]${NC}  $1"; }

cd "$(dirname "$0")/.."

# Activate venv if not already active
if [ -z "$VIRTUAL_ENV" ] && [ -d "venv" ]; then
    log "Activating Python venv..."
    # shellcheck disable=SC1091
    source venv/bin/activate
fi

# Make sure results directories exist
mkdir -p results/raw results/tables results/figures paper

# Make sure proto stubs are built
if [ ! -f "src/proto_generated/entropy_service_pb2.py" ]; then
    log "Building gRPC stubs..."
    bash scripts/build_proto.sh
fi

# ----------------------------------------------------------------------
# Run each benchmark
# ----------------------------------------------------------------------

START_TS=$(date +%s)

run_benchmark() {
    local name="$1"
    local module="$2"
    log ""
    log "============================================================"
    log "Running benchmark: $name"
    log "============================================================"
    if python -m "$module"; then
        log "✓ $name complete"
    else
        warn "✗ $name failed (continuing with remaining benchmarks)"
    fi
}

run_benchmark "Throughput"            "src.benchmarks.throughput"
run_benchmark "Latency"               "src.benchmarks.latency"
run_benchmark "Health Sensitivity"    "src.benchmarks.health_sensitivity"
run_benchmark "Channel Independence"  "src.benchmarks.channel_independence"
run_benchmark "Buffer Dynamics"       "src.benchmarks.buffer_dynamics"
run_benchmark "NIST Assessment"       "src.benchmarks.nist_assessment"

# ----------------------------------------------------------------------
# Generate consolidated paper summary
# ----------------------------------------------------------------------

log ""
log "============================================================"
log "Generating consolidated paper summary..."
log "============================================================"

python -m scripts.generate_summary || warn "Summary generation failed"

# ----------------------------------------------------------------------
# Done
# ----------------------------------------------------------------------

END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))

log ""
log "============================================================"
log "All benchmarks complete in ${ELAPSED}s"
log "============================================================"
log ""
log "Paper-ready output:"
log "  Tables:   results/tables/*.md"
log "  Figures:  results/figures/*.png"
log "  Raw:      results/raw/*.json"
log "  Summary:  paper/results_summary.md"
log ""
log "Recommended next step:"
log "  cat paper/results_summary.md"
log ""
