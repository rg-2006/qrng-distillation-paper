"""
Benchmark: per-channel cryptographic isolation.

Demonstrates that two customers (A and B) reading from the same shared
entropy pool but with different Toeplitz seeds produce statistically
independent outputs.

This is the strongest IP claim we identified — the patentable
combination of shared pool + per-customer seed re-extraction.

Tests performed:
  1. Pearson correlation between channels (should be ~0)
  2. Chi-square independence test
  3. Mutual information estimate (should be ~0)
  4. Direct byte-equality count (should match random chance)
  5. Bit-level Hamming distance (should be ~50%)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cupy as cp
import numpy as np
from scipy.stats import pearsonr, chi2_contingency
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.server.entropy_pool import EntropyPool
from src.server.distillation import DistillationPipeline, DistillationConfig
from src.server.channels      import ChannelManager
from src.simulation.entropy_sim import (
    EntropySimulator, SimulatorConfig, EntropyMode
)


def collect_channel_outputs(
    customer_ids: list[str],
    n_bytes_per_customer: int = 1_000_000,
) -> dict[str, bytes]:
    """Set up a shared pool, run distillation, collect bytes per customer."""
    pool = EntropyPool(pool_size_bytes=64 * 1024 * 1024, block_size=4096)
    sim  = EntropySimulator(SimulatorConfig(mode=EntropyMode.HIGH_QUALITY))
    pipe = DistillationPipeline(pool, sim, DistillationConfig())
    pipe.start()

    manager = ChannelManager(pool)
    channels = {cid: manager.create_channel(cid) for cid in customer_ids}

    # Wait for pool to start producing
    time.sleep(0.5)

    outputs: dict[str, bytearray] = {cid: bytearray() for cid in customer_ids}
    target_bytes = n_bytes_per_customer

    print(f"Collecting {target_bytes:,} bytes per customer...")
    while min(len(o) for o in outputs.values()) < target_bytes:
        for cid, ch in channels.items():
            if len(outputs[cid]) >= target_bytes:
                continue
            result = ch.read_block(timeout_s=2.0)
            if result is None:
                continue
            block_bytes, _meta, _seed_id = result
            outputs[cid].extend(block_bytes)

    pipe.stop()
    return {cid: bytes(o[:target_bytes]) for cid, o in outputs.items()}


def measure_independence(a: bytes, b: bytes) -> dict:
    """Statistical independence tests between two byte streams of equal length."""
    assert len(a) == len(b)
    a_arr = np.frombuffer(a, dtype=np.uint8).astype(np.float64)
    b_arr = np.frombuffer(b, dtype=np.uint8).astype(np.float64)

    # 1. Pearson correlation
    corr, p_corr = pearsonr(a_arr, b_arr)

    # 2. Byte equality count (P(A=B) should be ~ 1/256)
    eq_count = int(np.sum(a_arr == b_arr))
    expected_eq = len(a) / 256
    eq_ratio = eq_count / len(a)

    # 3. Bit-level Hamming distance
    a_bits = np.unpackbits(np.frombuffer(a, dtype=np.uint8))
    b_bits = np.unpackbits(np.frombuffer(b, dtype=np.uint8))
    hamming = int(np.sum(a_bits != b_bits))
    hamming_ratio = hamming / len(a_bits)

    # 4. Chi-square test on joint distribution (sampled to keep it cheap)
    sample = min(100_000, len(a))
    contingency = np.zeros((16, 16), dtype=np.int64)
    a_hi = (np.frombuffer(a[:sample], dtype=np.uint8) >> 4)
    b_hi = (np.frombuffer(b[:sample], dtype=np.uint8) >> 4)
    for ai, bi in zip(a_hi, b_hi):
        contingency[ai, bi] += 1
    chi2, p_chi2, _, _ = chi2_contingency(contingency)

    # 5. Mutual information lower bound (via histogram method)
    joint, _, _ = np.histogram2d(
        np.frombuffer(a[:sample], dtype=np.uint8),
        np.frombuffer(b[:sample], dtype=np.uint8),
        bins=64
    )
    p_joint = joint / joint.sum()
    p_a = p_joint.sum(axis=1, keepdims=True)
    p_b = p_joint.sum(axis=0, keepdims=True)
    mi = 0.0
    nz = p_joint > 0
    mi = float(np.sum(p_joint[nz] * np.log2(p_joint[nz] / (p_a * p_b)[nz])))

    return {
        'pearson_correlation':  float(corr),
        'pearson_p_value':      float(p_corr),
        'equality_count':       eq_count,
        'equality_ratio':       float(eq_ratio),
        'expected_equality':    float(expected_eq / len(a)),
        'hamming_distance_bits': hamming,
        'hamming_ratio':        float(hamming_ratio),
        'chi2_statistic':       float(chi2),
        'chi2_p_value':         float(p_chi2),
        'mutual_information_bits': float(mi),
        'n_bytes_compared':     len(a),
    }


def run_benchmark() -> dict:
    print("=" * 70)
    print("Per-Channel Independence Benchmark")
    print("=" * 70)

    customer_ids = ['customer_alpha', 'customer_beta', 'customer_gamma']
    outputs = collect_channel_outputs(customer_ids,
                                       n_bytes_per_customer=500_000)

    # Pairwise comparisons
    pairs = []
    for i in range(len(customer_ids)):
        for j in range(i + 1, len(customer_ids)):
            a_id, b_id = customer_ids[i], customer_ids[j]
            print(f"\nComparing {a_id} vs {b_id}...")
            indep = measure_independence(outputs[a_id], outputs[b_id])
            indep['customer_a'] = a_id
            indep['customer_b'] = b_id
            pairs.append(indep)
            print(f"  Pearson r:     {indep['pearson_correlation']:+.6f}")
            print(f"  Equality:      {indep['equality_count']} "
                  f"(expected ~{indep['expected_equality']*indep['n_bytes_compared']:.0f})")
            print(f"  Hamming ratio: {indep['hamming_ratio']:.6f} (expected 0.5)")
            print(f"  Chi^2 p-value: {indep['chi2_p_value']:.4f}")
            print(f"  MI estimate:   {indep['mutual_information_bits']:.6f} bits")

    return {
        'benchmark': 'channel_independence',
        'gpu':       str(cp.cuda.runtime.getDeviceProperties(0)['name']),
        'pairs':     pairs,
    }


def save_results(data: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = output_dir / "raw" / "channel_independence.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with open(raw_path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    table_path = output_dir / "tables" / "channel_independence.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with open(table_path, "w") as f:
        f.write("# Per-Channel Cryptographic Independence\n\n")
        gpu_str = data["gpu"]
        if isinstance(gpu_str, bytes):
            gpu_str = gpu_str.decode("utf-8", errors="replace")
        f.write(f"**GPU:** {gpu_str}\n\n")
        f.write("## Pairwise Independence Tests\n\n")
        f.write("| Pair | Pearson r | Hamming | Chi² p | MI (bits) |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for p in data["pairs"]:
            f.write(f"| {p['customer_a']} vs {p['customer_b']} | "
                    f"{p['pearson_correlation']:+.6f} | "
                    f"{p['hamming_ratio']:.6f} | "
                    f"{p['chi2_p_value']:.4f} | "
                    f"{p['mutual_information_bits']:.4f} |\n")
        f.write("\n## Interpretation\n\n")
        f.write("- **Pearson r ≈ 0**: no linear correlation between channels\n")
        f.write("- **Hamming ratio ≈ 0.5**: bits differ at random positions\n")
        f.write("- **Chi² p > 0.05**: cannot reject hypothesis of independence\n")
        f.write("- **MI ≈ 0**: no information shared between channels\n")
        f.write("\nThese together provide statistical evidence that "
                "per-customer Toeplitz re-extraction yields cryptographically "
                "independent outputs even from a shared entropy pool.\n")

    # Plot
    fig_path = output_dir / "figures" / "channel_independence.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    pair_labels = [f"{p['customer_a'][:8]}\nvs\n{p['customer_b'][:8]}"
                   for p in data["pairs"]]
    correlations = [p["pearson_correlation"] for p in data["pairs"]]
    hammings = [p["hamming_ratio"] for p in data["pairs"]]

    axes[0].bar(pair_labels, correlations, color="#2E86AB")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("Pearson Correlation", fontsize=12)
    axes[0].set_title("Channel-pair Correlation (target: 0)",
                       fontsize=13, fontweight="bold")
    axes[0].set_ylim(-0.01, 0.01)
    axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].bar(pair_labels, hammings, color="#E63946")
    axes[1].axhline(0.5, color="green", linestyle="--",
                    label="Ideal: 0.5")
    axes[1].set_ylabel("Hamming Distance Ratio", fontsize=12)
    axes[1].set_title("Bit-Level Difference Rate (target: 0.5)",
                       fontsize=13, fontweight="bold")
    axes[1].set_ylim(0.49, 0.51)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()

    print(f"\nResults saved to:")
    print(f"  Raw:    {raw_path}")
    print(f"  Table:  {table_path}")
    print(f"  Figure: {fig_path}")


if __name__ == "__main__":
    data = run_benchmark()
    save_results(data, Path("results"))
