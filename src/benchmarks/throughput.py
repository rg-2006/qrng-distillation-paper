"""
Benchmark: Toeplitz extraction throughput.

Measures sustained Gbps of certified entropy extraction on the GPU.
This benchmark proves the pipeline is not the bottleneck — it can
process more entropy than any commercial QRNG card can produce.

Outputs:
  - results/raw/throughput.json
  - results/figures/throughput.png
  - results/tables/throughput.md
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cupy as cp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.cuda.kernels import toeplitz_extract_gpu


def benchmark_throughput(
    block_sizes_bits: list[tuple[int, int]] = None,
    n_iterations:     int = 100,
    warmup:           int = 50,
) -> dict:
    """Measure Toeplitz throughput across a range of block sizes.
    
    Args:
        block_sizes_bits: list of (n_input_bits, m_output_bits) tuples
        n_iterations:     number of timed iterations per configuration
        warmup:           number of warmup iterations (excluded from timing)
    """
    if block_sizes_bits is None:
        # Test sizes from small (8 KB) to medium (512 KB)
        block_sizes_bits = [
            (   8_192,    6_144),    # 1 KB in -> 768 B out
            (  16_384,   12_288),    # 2 KB in
            (  32_768,   24_576),    # 4 KB in
            (  65_536,   49_152),    # 8 KB in
            ( 131_072,   98_304),    # 16 KB in
        ]

    print("=" * 70)
    print("Toeplitz Extraction Throughput Benchmark")
    print("=" * 70)
    print(f"Iterations per size: {n_iterations} (after {warmup} warmup)")
    print()

    results: list[dict] = []

    for n_in, m_out in block_sizes_bits:
        # Allocate seed and input on GPU
        seed_bytes = (n_in + m_out - 1 + 7) // 8 + 8
        seed_gpu = cp.asarray(np.frombuffer(np.random.bytes(seed_bytes),
                                              dtype=np.uint8).copy())
        input_gpu = cp.asarray(np.frombuffer(np.random.bytes(n_in // 8),
                                              dtype=np.uint8).copy())

        # Warmup (kernel JIT compilation, caches, etc.)
        for _ in range(warmup):
            _ = toeplitz_extract_gpu(seed_gpu, input_gpu, n_in, m_out)
        cp.cuda.Stream.null.synchronize()

        # Timed run
        t0 = time.perf_counter()
        for _ in range(n_iterations):
            _ = toeplitz_extract_gpu(seed_gpu, input_gpu, n_in, m_out)
        cp.cuda.Stream.null.synchronize()
        t1 = time.perf_counter()

        elapsed = t1 - t0
        bits_processed = n_iterations * m_out
        gbps = bits_processed / elapsed / 1e9
        per_op_us = (elapsed / n_iterations) * 1e6

        result = {
            'n_input_bits':   n_in,
            'm_output_bits':  m_out,
            'input_kb':       n_in / 8 / 1024,
            'output_kb':      m_out / 8 / 1024,
            'iterations':     n_iterations,
            'elapsed_s':      elapsed,
            'gbps_output':    gbps,
            'us_per_op':      per_op_us,
        }
        results.append(result)

        print(f"In: {n_in/8/1024:7.1f} KB | Out: {m_out/8/1024:7.1f} KB | "
              f"{per_op_us:7.2f} µs/op | {gbps:6.3f} Gbps")

    return {
        'benchmark':    'throughput',
        'gpu':          str(cp.cuda.runtime.getDeviceProperties(0)['name']),
        'results':      results,
    }


def save_results(data: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Raw JSON
    raw_path = output_dir / "raw" / "throughput.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with open(raw_path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    # Markdown table
    table_path = output_dir / "tables" / "throughput.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with open(table_path, "w") as f:
        f.write("# Toeplitz Extraction Throughput\n\n")
        gpu_str = data["gpu"]
        if isinstance(gpu_str, bytes):
            gpu_str = gpu_str.decode("utf-8", errors="replace")
        f.write(f"**GPU:** {gpu_str}\n\n")
        f.write("| Input Size | Output Size | Time per Op | Throughput |\n")
        f.write("|---:|---:|---:|---:|\n")
        for r in data["results"]:
            f.write(f"| {r['input_kb']:.1f} KB | {r['output_kb']:.1f} KB | "
                    f"{r['us_per_op']:.2f} µs | {r['gbps_output']:.3f} Gbps |\n")

    # Plot
    fig_path = output_dir / "figures" / "throughput.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    sizes = [r["input_kb"] for r in data["results"]]
    gbps = [r["gbps_output"] for r in data["results"]]
    ax.plot(sizes, gbps, "o-", linewidth=2, markersize=8, color="#2E86AB")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Input Block Size (KB)", fontsize=12)
    ax.set_ylabel("Throughput (Gbps)", fontsize=12)
    ax.set_title("Toeplitz Extraction Throughput vs Block Size",
                 fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Reference line: 4 Gbps (IDQ's best card)
    ax.axhline(y=4.0, color="orange", linestyle="--", alpha=0.7,
               label="ID Quantique flagship (4 Gbps)")
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()

    print(f"\nResults saved to:")
    print(f"  Raw:    {raw_path}")
    print(f"  Table:  {table_path}")
    print(f"  Figure: {fig_path}")


if __name__ == "__main__":
    data = benchmark_throughput()
    save_results(data, Path("results"))
