# QRNG Distillation Pipeline — Paper Implementation

A GPU-accelerated certified entropy distillation and gRPC streaming delivery system, designed to produce reproducible benchmark results for academic publication.

## What This Codebase Demonstrates

1. **Persistent CUDA kernel** for continuous entropy distillation
2. **GF(2) Toeplitz hashing** for randomness extraction  
3. **NIST SP 800-90B health tests** (RCT, APT) running inline
4. **Per-channel cryptographic isolation** via independent Toeplitz seeds
5. **gRPC streaming delivery** with sub-microsecond application latency
6. **Real-time min-entropy attestation** signed per block
7. **Reproducible benchmarks** for technical paper publication

## Hardware Requirements

- NVIDIA H100 SXM (or any modern NVIDIA GPU with compute capability ≥ 7.0)
- Tested on Lambda Labs `gpu_1x_h100_sxm5` instance
- Minimum 24 GB GPU memory recommended (the H100 SXM's 80 GB gives headroom)
- Ubuntu 22.04 LTS



## Implementation Notes

- **CUDA kernels are inline** via CuPy `RawKernel`. This is intentional as it keeps deployment to a single
  `pip install -r requirements.txt`.
- **Persistent kernel pattern** is implemented at the Python orchestration
  layer (a continuously running thread that launches kernels on a dedicated
  CUDA stream). This avoids many
  grid-sizing problems while preserving the semantic the paper claims.
- **gRPC streaming** uses Python's standard `grpcio`
  
## Quick Start (Lambda Labs)

```bash
# 1. Spin up Lambda Labs gpu_1x_h100_sxm5 instance
# 2. SSH into the instance
# 3. Clone and setup
git clone <your-repo-url> qrng-distillation-paper
cd qrng-distillation-paper
bash scripts/setup_lambda.sh

# 4. Run all benchmarks (generates paper-ready data)
bash scripts/run_all_benchmarks.sh

# 5. Results are in results/ directory:
#    - results/tables/    Markdown tables for paper
#    - results/figures/   Matplotlib PNG/PDF figures
#    - results/raw/       JSON data for further analysis
#    - paper/results_summary.md   Auto-generated summary
```

## What Each Benchmark Proves

| Benchmark | File | Paper Claim |
|---|---|---|
| Throughput | `benchmarks/throughput.py` | Toeplitz extraction at ≥10 Gbps on H100 |
| Latency | `benchmarks/latency.py` | Application access at <1µs via buffer model |
| Health Sensitivity | `benchmarks/health_sensitivity.py` | RCT/APT detect failures within N bytes |
| Channel Independence | `benchmarks/channel_independence.py` | Statistically independent per-customer outputs |
| Buffer Dynamics | `benchmarks/buffer_dynamics.py` | Refill > drain across realistic loads |
| NIST Assessment | `benchmarks/nist_assessment.py` | H_min ≈ 7.9+ bits/byte after extraction |

## Running Individual Benchmarks

```bash
cd qrng-distillation-paper
source venv/bin/activate

python -m src.benchmarks.throughput
python -m src.benchmarks.latency
python -m src.benchmarks.health_sensitivity
python -m src.benchmarks.channel_independence
python -m src.benchmarks.buffer_dynamics
python -m src.benchmarks.nist_assessment
```

## Running the Server and Client (Live Demo)

```bash
# Terminal 1: Start the gRPC server
python -m src.server.grpc_server

# Terminal 2: Run the client
python -m src.client.client
```

## CloudFlare Tunnel Setup (for remote access)

```bash
bash scripts/setup_cloudflare.sh
# Follow prompts to authenticate with Cloudflare
# Tunnel URL will be printed; client uses this URL
```

## Architecture Overview

```
[Simulated Entropy Source]
    ↓ (raw bytes via async queue)
[Persistent CUDA Kernel]
    ├─ Health Tests (RCT, APT)
    ├─ Min-Entropy Estimation  
    └─ Toeplitz Extraction (GF(2))
        ↓
[Certified Pool in HBM]
    ↓ (per-customer streams)
[Channel Isolation Layer]
    ├─ Customer A: Toeplitz seed A → output A
    ├─ Customer B: Toeplitz seed B → output B
    └─ Customer C: Toeplitz seed C → output C
        ↓
[gRPC Streaming Server]
    ↓ (over CloudFlare Tunnel)
[Client Local Buffer]
    ↓ (memory read, ~100ns)
[Application]
```

## Citation

If you use this code in your research:

```bibtex
@misc{qrng_distillation_2026,
  title  = {GPU-Accelerated Certified Entropy Distillation with 
            Streaming Delivery for Sub-Microsecond Application Latency},
  author = {Your Name},
  year   = {2026}
}
```

## License

[Apache 2.0]
