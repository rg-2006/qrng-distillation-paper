"""
Benchmark: health test detection sensitivity.

Injects each failure mode (stuck, biased, periodic, gradual, intermittent)
and measures how many bytes/blocks pass before detection. This proves
the inline NIST 800-90B health tests are working as specified.
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

from src.cuda.kernels import rct_test_gpu, apt_test_gpu
from src.simulation.entropy_sim import (
    EntropySimulator, SimulatorConfig, EntropyMode, FailureMode
)
from src.server.distillation import calculate_rct_cutoff, calculate_apt_cutoff


def health_check_block(block: bytes, h_min_assumed: float = 4.0) -> dict:
    """Run RCT and APT on a block, return results dict."""
    samples_gpu = cp.asarray(np.frombuffer(block, dtype=np.uint8))
    rct_cutoff = calculate_rct_cutoff(h_min_assumed)
    apt_cutoff = calculate_apt_cutoff(h_min_assumed, W=512)

    rct_failed, longest_run = rct_test_gpu(samples_gpu, rct_cutoff)
    apt_failed = apt_test_gpu(samples_gpu, 512, apt_cutoff)

    return {
        'rct_failed':  rct_failed,
        'apt_failed':  apt_failed,
        'longest_run': int(longest_run),
        'rct_cutoff':  rct_cutoff,
        'apt_cutoff':  apt_cutoff,
    }


def measure_failure_detection(
    failure_mode: FailureMode,
    block_size:   int = 4096,
    max_blocks:   int = 1000,
    n_trials:     int = 30,
) -> dict:
    """For a given failure mode, measure detection latency across trials."""
    detection_blocks: list[int] = []
    detected_count = 0
    detector_counts = {'RCT': 0, 'APT': 0, 'BOTH': 0, 'NONE': 0}

    for trial in range(n_trials):
        sim = EntropySimulator(SimulatorConfig(
            mode=EntropyMode.FAILURE_INJECT,
            failure_mode=failure_mode,
            seed=trial,
        ))
        detected = False
        for block_idx in range(max_blocks):
            block = sim.generate(block_size)
            result = health_check_block(block)
            if result['rct_failed'] or result['apt_failed']:
                detection_blocks.append(block_idx + 1)
                detected_count += 1
                if result['rct_failed'] and result['apt_failed']:
                    detector_counts['BOTH'] += 1
                elif result['rct_failed']:
                    detector_counts['RCT'] += 1
                else:
                    detector_counts['APT'] += 1
                detected = True
                break
        if not detected:
            detection_blocks.append(max_blocks)
            detector_counts['NONE'] += 1

    detected_blocks = [b for b in detection_blocks if b < max_blocks]
    return {
        'failure_mode':       failure_mode.value,
        'n_trials':           n_trials,
        'block_size':         block_size,
        'detection_rate':     detected_count / n_trials,
        'mean_detection_blocks':   float(np.mean(detected_blocks)) if detected_blocks else None,
        'median_detection_blocks': float(np.median(detected_blocks)) if detected_blocks else None,
        'min_detection_blocks':    int(np.min(detected_blocks)) if detected_blocks else None,
        'max_detection_blocks':    int(np.max(detected_blocks)) if detected_blocks else None,
        'detection_bytes_mean':    float(np.mean(detected_blocks) * block_size) if detected_blocks else None,
        'detector_counts':    detector_counts,
        'all_detection_blocks': detection_blocks,
    }


def benchmark_false_positive_rate(
    block_size: int = 4096,
    n_blocks:   int = 10_000,
) -> dict:
    """Run health tests against high-quality input; measure FPR."""
    print(f"Measuring false-positive rate over {n_blocks} healthy blocks...")
    sim = EntropySimulator(SimulatorConfig(mode=EntropyMode.HIGH_QUALITY))

    rct_fp = 0
    apt_fp = 0
    for i in range(n_blocks):
        block = sim.generate(block_size)
        result = health_check_block(block)
        if result['rct_failed']:  rct_fp += 1
        if result['apt_failed']:  apt_fp += 1
        if i % 1000 == 0 and i > 0:
            print(f"  {i}/{n_blocks} blocks tested...")

    return {
        'n_blocks':    n_blocks,
        'rct_fpr':     rct_fp / n_blocks,
        'apt_fpr':     apt_fp / n_blocks,
        'rct_fps':     rct_fp,
        'apt_fps':     apt_fp,
        'expected_fpr': 2 ** -20,  # NIST target
    }


def run_benchmark() -> dict:
    print("=" * 70)
    print("Health Test Sensitivity Benchmark")
    print("=" * 70)

    # 1. Detection latency for each failure mode
    failure_modes = [
        FailureMode.STUCK_AT,
        FailureMode.BIASED,
        FailureMode.PERIODIC,
        FailureMode.GRADUAL,
        FailureMode.INTERMITTENT,
    ]
    detection_results = []
    for fmode in failure_modes:
        print(f"\nTesting failure mode: {fmode.value}")
        result = measure_failure_detection(fmode, n_trials=30, max_blocks=200)
        detection_results.append(result)
        if result['mean_detection_blocks'] is not None:
            print(f"  Detection rate: {result['detection_rate']:.1%}")
            print(f"  Mean blocks to detection: "
                  f"{result['mean_detection_blocks']:.1f} "
                  f"(~{result['detection_bytes_mean']/1024:.1f} KB)")
            print(f"  Detector counts: {result['detector_counts']}")
        else:
            print(f"  Not detected within {result['block_size']*200} bytes")

    # 2. False positive rate
    print()
    fpr_result = benchmark_false_positive_rate(n_blocks=5000)
    print(f"False positive rate (healthy input):")
    print(f"  RCT: {fpr_result['rct_fpr']:.6f} ({fpr_result['rct_fps']} / {fpr_result['n_blocks']})")
    print(f"  APT: {fpr_result['apt_fpr']:.6f} ({fpr_result['apt_fps']} / {fpr_result['n_blocks']})")
    print(f"  NIST target: 2^-20 = {2**-20:.6f}")

    return {
        'benchmark':         'health_sensitivity',
        'gpu':               str(cp.cuda.runtime.getDeviceProperties(0)['name']),
        'detection_results': detection_results,
        'false_positive':    fpr_result,
    }


def save_results(data: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    raw_path = output_dir / "raw" / "health_sensitivity.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with open(raw_path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    # Markdown
    table_path = output_dir / "tables" / "health_sensitivity.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with open(table_path, "w") as f:
        f.write("# Health Test Sensitivity\n\n")
        gpu_str = data["gpu"]
        if isinstance(gpu_str, bytes):
            gpu_str = gpu_str.decode("utf-8", errors="replace")
        f.write(f"**GPU:** {gpu_str}\n\n")
        f.write("## Detection Latency by Failure Mode\n\n")
        f.write("| Failure Mode | Detection Rate | Mean Blocks | Mean Bytes | Primary Detector |\n")
        f.write("|---|---:|---:|---:|---|\n")
        for r in data["detection_results"]:
            primary = max(r["detector_counts"], key=r["detector_counts"].get)
            mean_b = f"{r['mean_detection_blocks']:.1f}" if r['mean_detection_blocks'] else "—"
            mean_by = f"{r['detection_bytes_mean']/1024:.1f} KB" if r['detection_bytes_mean'] else "—"
            f.write(f"| {r['failure_mode']} | {r['detection_rate']:.1%} | "
                    f"{mean_b} | {mean_by} | {primary} |\n")

        fpr = data["false_positive"]
        f.write("\n## False Positive Rate (healthy input)\n\n")
        f.write(f"- Tested over {fpr['n_blocks']} healthy blocks\n")
        f.write(f"- RCT FPR: **{fpr['rct_fpr']:.6f}** ({fpr['rct_fps']} false positives)\n")
        f.write(f"- APT FPR: **{fpr['apt_fpr']:.6f}** ({fpr['apt_fps']} false positives)\n")
        f.write(f"- NIST target: 2^-20 = {2**-20:.6f}\n")

    # Plot detection latencies
    fig_path = output_dir / "figures" / "health_detection.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    modes = [r["failure_mode"] for r in data["detection_results"]]
    means = [r["mean_detection_blocks"] or 0 for r in data["detection_results"]]
    rates = [r["detection_rate"] for r in data["detection_results"]]
    colors = ["#2E86AB" if rate >= 0.95 else "#E63946"
              for rate in rates]
    bars = ax.bar(modes, means, color=colors)
    ax.set_xlabel("Failure Mode", fontsize=12)
    ax.set_ylabel("Mean Blocks to Detection", fontsize=12)
    ax.set_title("Health Test Detection Latency by Failure Mode",
                 fontsize=14, fontweight="bold")
    for bar, mean, rate in zip(bars, means, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, mean,
                f"{mean:.1f}\n({rate:.0%})", ha="center", va="bottom", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
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
