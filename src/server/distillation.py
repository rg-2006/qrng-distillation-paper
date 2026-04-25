"""
Distillation pipeline orchestration.

Implements the "persistent kernel" pattern as a continuously-running
producer thread that:
  1. Pulls raw entropy from the simulator (in production: from FPGA/optical)
  2. Runs health tests on raw blocks (RCT, APT)
  3. Estimates min-entropy per block
  4. Performs Toeplitz extraction with a master seed
  5. Writes certified blocks to the entropy pool

This is the "Stage 1" of the pipeline — producing the master certified
pool. Per-customer channel re-extraction happens in channels.py, which
reads from this pool and re-extracts with customer-specific seeds for
cryptographic isolation.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass

import cupy as cp
import numpy as np

from src.cuda.kernels import (
    toeplitz_extract_gpu,
    rct_test_gpu,
    apt_test_gpu,
    estimate_min_entropy_gpu,
)
from src.server.entropy_pool import EntropyPool
from src.simulation.entropy_sim import (
    EntropySimulator, SimulatorConfig, EntropyMode
)


# ======================================================================
# Health test cutoff calculation (NIST SP 800-90B §4.4)
# ======================================================================

def calculate_rct_cutoff(h_min: float, alpha: float = 2 ** -20) -> int:
    """RCT cutoff C = ceil(1 + (-log2(alpha) / h_min))."""
    h_min = max(h_min, 0.1)   # Avoid division by zero
    return int(math.ceil(1 + (-math.log2(alpha) / h_min)))


def calculate_apt_cutoff(h_min: float, W: int = 512,
                          alpha: float = 2 ** -20) -> int:
    """APT cutoff via binomial approximation."""
    h_min = max(h_min, 0.1)
    p = 2 ** (-h_min)
    Z = 4.42   # Approximate Z-score for alpha = 2^-20
    return int(math.ceil(W * p + math.sqrt(W * p * (1 - p)) * Z))


# ======================================================================
# Distillation pipeline
# ======================================================================

@dataclass
class DistillationConfig:
    raw_block_size:    int   = 8192     # Bytes of raw input per Toeplitz operation
    output_block_size: int   = 4096     # Bytes after extraction (compression ~50 %)
    target_h_min:      float = 0.85     # Assumed source H_min for cutoff calculation
    apt_window:        int   = 512


class DistillationPipeline:
    """Continuously distills raw entropy into the certified pool."""

    def __init__(
        self,
        pool:      EntropyPool,
        simulator: EntropySimulator,
        config:    DistillationConfig | None = None,
    ):
        self.pool      = pool
        self.simulator = simulator
        self.config    = config or DistillationConfig()

        # Compute health test cutoffs
        self.rct_cutoff = calculate_rct_cutoff(self.config.target_h_min)
        self.apt_cutoff = calculate_apt_cutoff(self.config.target_h_min,
                                                self.config.apt_window)

        # Toeplitz parameters
        n_input_bits  = self.config.raw_block_size  * 8
        m_output_bits = self.config.output_block_size * 8
        seed_bytes    = (n_input_bits + m_output_bits - 1 + 7) // 8 + 8

        # Master Toeplitz seed (kept on GPU, fixed for production)
        self._master_seed_cpu = np.frombuffer(
            np.random.bytes(seed_bytes), dtype=np.uint8
        ).copy()
        self.master_seed_gpu = cp.asarray(self._master_seed_cpu)

        self.n_input_bits  = n_input_bits
        self.m_output_bits = m_output_bits

        # Control
        self._stop_event = threading.Event()
        self._thread:    threading.Thread | None = None

        # Stats
        self.blocks_produced = 0
        self.blocks_rejected = 0
        self._start_time:    float | None = None

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the persistent distillation thread."""
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._start_time = time.time()
        self._thread = threading.Thread(
            target=self._run_loop, name="distillation_pipeline", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the distillation thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    # ------------------------------------------------------------------
    # The persistent kernel loop
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        """Main distillation loop — runs continuously."""
        while not self._stop_event.is_set():
            try:
                self._process_one_block()
            except Exception as e:
                print(f"[distillation] error: {e}")
                time.sleep(0.001)

    def _process_one_block(self) -> None:
        """Process one raw block: health tests -> extraction -> pool write."""
        # 1. Get raw entropy from simulator
        raw_bytes = self.simulator.generate(self.config.raw_block_size)
        raw_gpu = cp.asarray(np.frombuffer(raw_bytes, dtype=np.uint8))

        # 2. Health tests on RAW data (NIST requires this on pre-conditioning)
        rct_failed, longest_run = rct_test_gpu(raw_gpu, self.rct_cutoff)
        apt_failed = apt_test_gpu(raw_gpu, self.config.apt_window, self.apt_cutoff)

        rct_passed = not rct_failed
        apt_passed = not apt_failed

        if rct_failed or apt_failed:
            # Health failure — reject this block
            self.pool.reject_block()
            self.blocks_rejected += 1
            return

        # 3. Min-entropy estimate (from raw data; output should be ~uniform)
        h_min = estimate_min_entropy_gpu(raw_gpu)

        # 4. Toeplitz extraction
        extracted_gpu = toeplitz_extract_gpu(
            self.master_seed_gpu, raw_gpu,
            self.n_input_bits, self.m_output_bits,
        )

        # 5. Write certified block to pool
        self.pool.write_block(
            block_data = extracted_gpu,
            h_min      = h_min,
            rct_passed = rct_passed,
            apt_passed = apt_passed,
        )
        self.blocks_produced += 1

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def throughput_gbps(self) -> float:
        """Output throughput in Gbps (averaged over runtime)."""
        if self._start_time is None or self.blocks_produced == 0:
            return 0.0
        elapsed = time.time() - self._start_time
        bits_out = self.blocks_produced * self.config.output_block_size * 8
        return bits_out / elapsed / 1e9

    def stats(self) -> dict:
        return {
            'blocks_produced': self.blocks_produced,
            'blocks_rejected': self.blocks_rejected,
            'throughput_gbps': self.throughput_gbps,
            'rct_cutoff':      self.rct_cutoff,
            'apt_cutoff':      self.apt_cutoff,
        }
