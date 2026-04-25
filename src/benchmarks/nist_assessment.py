"""
Benchmark: NIST SP 800-90B output entropy assessment.

Validates the entropy quality of our pipeline's output. Generates the
required 1M sample dataset, runs NIST's reference entropy assessment
tool against it, and parses the H_min result.

Two assessments:
  1. Output of Toeplitz extraction with high-quality input (should be ~7.99)
  2. Output with biased input (still should be high — Toeplitz amplifies)

If the NIST tool isn't installed, we fall back to a built-in estimator.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import re
from pathlib import Path

import cupy as cp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.cuda.kernels import toeplitz_extract_gpu, estimate_min_entropy_gpu
from src.simulation.entropy_sim import (
    EntropySimulator, SimulatorConfig, EntropyMode
)


def _has_nist_tool() -> str | None:
    """Return path to NIST ea_non_iid binary if available."""
    for name in ('ea_non_iid', 'ea_non_iid.exe'):
        p = shutil.which(name)
        if p:
            return p
    # Common build location
    candidates = [
        '/opt/SP800-90B_EntropyAssessment/cpp/ea_non_iid',
        '/usr/local/bin/ea_non_iid',
        os.path.expanduser('~/SP800-90B_EntropyAssessment/cpp/ea_non_iid'),
    ]
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def run_nist_assessment(input_file: Path, bits_per_sample: int = 8) -> dict:
    """Run NIST SP 800-90B non-IID assessment on a binary file."""
    tool_path = _has_nist_tool()

    if tool_path:
        print(f"Using NIST tool: {tool_path}")
        try:
            result = subprocess.run(
                [tool_path, str(input_file), str(bits_per_sample)],
                capture_output=True, text=True, timeout=600,
            )
            output = result.stdout + result.stderr

            # Parse min-entropy estimate. NIST output format varies; look
            # for "min(...)" or "min entropy = X" or "h_original" lines.
            patterns = [
                r"H[_\s]*original[:\s=]+([0-9.]+)",
                r"min[\s\-_]*entropy[:\s=]+([0-9.]+)",
                r"min\s*\(.*?\)\s*[:=]\s*([0-9.]+)",
            ]
            h_min = None
            for pat in patterns:
                m = re.search(pat, output, re.IGNORECASE)
                if m:
                    h_min = float(m.group(1))
                    break

            return {
                'tool': 'nist_official',
                'tool_path': tool_path,
                'h_min':     h_min,
                'output_excerpt': output[:1000],
                'returncode': result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {'tool': 'nist_official', 'error': 'timeout'}
        except Exception as e:
            return {'tool': 'nist_official', 'error': str(e)}

    # Fallback: use our built-in most-common-value estimator
    print("NIST tool not found; using built-in MCV estimator")
    with open(input_file, 'rb') as f:
        data = f.read()
    samples_gpu = cp.asarray(np.frombuffer(data, dtype=np.uint8))
    h_min = estimate_min_entropy_gpu(samples_gpu)
    return {
        'tool':  'builtin_mcv',
        'h_min': h_min,
        'note':  'Most-common-value estimator from NIST SP 800-90B §6.3.1',
    }


def generate_test_dataset(
    sim: EntropySimulator,
    n_samples: int = 1_000_000,
    output_path: Path = None,
) -> Path:
    if output_path is None:
        output_path = Path("/tmp") / "qrng_test_dataset.bin"
    data = sim.generate(n_samples)
    with open(output_path, 'wb') as f:
        f.write(data)
    return output_path


def extract_with_toeplitz(input_data: bytes,
                           ratio: float = 0.75) -> bytes:
    """Run Toeplitz extraction on a chunk of input bytes."""
    n_input_bits  = len(input_data) * 8
    m_output_bits = int(n_input_bits * ratio) // 8 * 8

    seed_bytes = (n_input_bits + m_output_bits - 1 + 7) // 8 + 8
    seed_gpu = cp.asarray(np.frombuffer(np.random.bytes(seed_bytes),
                                          dtype=np.uint8).copy())
    input_gpu = cp.asarray(np.frombuffer(input_data, dtype=np.uint8).copy())

    out_gpu = toeplitz_extract_gpu(seed_gpu, input_gpu,
                                    n_input_bits, m_output_bits)
    return bytes(out_gpu.get())


def run_benchmark() -> dict:
    print("=" * 70)
    print("NIST SP 800-90B Output Entropy Assessment")
    print("=" * 70)

    results = []

    # Scenario 1: high-quality input through pipeline
    print("\n[1] High-quality input (cryptographic random)...")
    sim_hq = EntropySimulator(SimulatorConfig(mode=EntropyMode.HIGH_QUALITY))

    # Sample raw input
    raw_input = sim_hq.generate(1_000_000)
    raw_path = generate_test_dataset(sim_hq,
                                       output_path=Path("/tmp/raw_hq.bin"))
    print(f"   Raw input written: {raw_path}")
    raw_assess = run_nist_assessment(raw_path)
    print(f"   Raw H_min: {raw_assess.get('h_min', 'N/A')}")

    # Extract via Toeplitz (process in chunks to handle 1MB input)
    print("   Extracting via Toeplitz...")
    extracted = bytearray()
    chunk_size = 8192
    for i in range(0, len(raw_input), chunk_size):
        chunk = raw_input[i:i + chunk_size]
        if len(chunk) < chunk_size:
            break
        out = extract_with_toeplitz(chunk, ratio=0.75)
        extracted.extend(out)
    ext_bytes = bytes(extracted)
    ext_path = Path("/tmp/extracted_hq.bin")
    with open(ext_path, 'wb') as f:
        f.write(ext_bytes)
    print(f"   Extracted output written: {ext_path} ({len(ext_bytes):,} B)")
    ext_assess = run_nist_assessment(ext_path)
    print(f"   Extracted H_min: {ext_assess.get('h_min', 'N/A')}")

    results.append({
        'scenario':    'high_quality',
        'raw_assessment': raw_assess,
        'extracted_assessment': ext_assess,
        'raw_path':    str(raw_path),
        'ext_path':    str(ext_path),
    })

    # Scenario 2: biased input (modeled laser noise with H_min ~0.85)
    print("\n[2] Modeled laser input (H_min target 0.85)...")
    sim_laser = EntropySimulator(SimulatorConfig(
        mode=EntropyMode.MODELED_LASER, target_h_min=0.85))
    raw_input = sim_laser.generate(1_000_000)
    raw_path = Path("/tmp/raw_laser.bin")
    with open(raw_path, 'wb') as f:
        f.write(raw_input)
    raw_assess = run_nist_assessment(raw_path)
    print(f"   Raw H_min: {raw_assess.get('h_min', 'N/A')}")

    extracted = bytearray()
    for i in range(0, len(raw_input), chunk_size):
        chunk = raw_input[i:i + chunk_size]
        if len(chunk) < chunk_size:
            break
        out = extract_with_toeplitz(chunk, ratio=0.75)
        extracted.extend(out)
    ext_bytes = bytes(extracted)
    ext_path = Path("/tmp/extracted_laser.bin")
    with open(ext_path, 'wb') as f:
        f.write(ext_bytes)
    ext_assess = run_nist_assessment(ext_path)
    print(f"   Extracted H_min: {ext_assess.get('h_min', 'N/A')}")

    results.append({
        'scenario':    'modeled_laser',
        'raw_assessment': raw_assess,
        'extracted_assessment': ext_assess,
        'raw_path':    str(raw_path),
        'ext_path':    str(ext_path),
    })

    return {
        'benchmark': 'nist_assessment',
        'gpu':       str(cp.cuda.runtime.getDeviceProperties(0)['name']),
        'tool_used': 'nist_official' if _has_nist_tool() else 'builtin_mcv',
        'results':   results,
    }


def save_results(data: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = output_dir / "raw" / "nist_assessment.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with open(raw_path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    table_path = output_dir / "tables" / "nist_assessment.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with open(table_path, "w") as f:
        f.write("# NIST SP 800-90B Entropy Assessment\n\n")
        gpu_str = data["gpu"]
        if isinstance(gpu_str, bytes):
            gpu_str = gpu_str.decode("utf-8", errors="replace")
        f.write(f"**GPU:** {gpu_str}\n\n")
        f.write(f"**Assessment Tool:** {data['tool_used']}\n\n")
        f.write("## Min-Entropy Before and After Extraction\n\n")
        f.write("| Scenario | Raw H_min | Extracted H_min | Ratio |\n")
        f.write("|---|---:|---:|---:|\n")
        for r in data["results"]:
            raw_h = r["raw_assessment"].get("h_min")
            ext_h = r["extracted_assessment"].get("h_min")
            raw_str = f"{raw_h:.4f}" if raw_h is not None else "—"
            ext_str = f"{ext_h:.4f}" if ext_h is not None else "—"
            ratio = f"{ext_h / raw_h:.3f}×" if (raw_h and ext_h and raw_h > 0) else "—"
            f.write(f"| {r['scenario']} | {raw_str} | {ext_str} | {ratio} |\n")
        f.write("\nIdeal extracted output: H_min ≈ 7.99 bits/byte (near uniform)\n")

    fig_path = output_dir / "figures" / "nist_assessment.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    scenarios = [r["scenario"] for r in data["results"]]
    raw_vals = [r["raw_assessment"].get("h_min", 0) or 0
                for r in data["results"]]
    ext_vals = [r["extracted_assessment"].get("h_min", 0) or 0
                for r in data["results"]]
    x = np.arange(len(scenarios))
    width = 0.35
    ax.bar(x - width/2, raw_vals, width, label="Raw input", color="#E76F51")
    ax.bar(x + width/2, ext_vals, width, label="After Toeplitz", color="#2E86AB")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel("Min-Entropy (bits/byte)", fontsize=12)
    ax.set_title("Toeplitz Extraction: H_min Before and After",
                 fontsize=14, fontweight="bold")
    ax.axhline(8.0, color="green", linestyle="--", alpha=0.6,
               label="Maximum (8 bits/byte)")
    ax.legend()
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
