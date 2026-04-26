"""
Multi-mode entropy source simulator.

Generates simulated raw entropy data in three modes:
  1. HIGH_QUALITY  — cryptographic-quality random bytes (os.urandom)
  2. MODELED_LASER — DFB laser phase noise model (Gaussian + correlations)
  3. FAILURE_INJECT — deliberately compromised data for health test validation

This allows the paper to demonstrate:
  - Throughput with high-quality input
  - Min-entropy variation with realistic source modeling
  - Health test sensitivity with controlled failure injection
"""

from __future__ import annotations

import os
import time
import enum
from dataclasses import dataclass

import numpy as np


class EntropyMode(enum.Enum):
    HIGH_QUALITY     = "high_quality"
    MODELED_LASER    = "modeled_laser"
    FAILURE_INJECT   = "failure_inject"


class FailureMode(enum.Enum):
    NONE         = "none"
    STUCK_AT     = "stuck_at"        # Source stuck at constant value
    BIASED       = "biased"          # Heavily biased distribution
    PERIODIC     = "periodic"        # Periodic pattern (failed laser)
    GRADUAL      = "gradual"         # Gradual degradation (drift)
    INTERMITTENT = "intermittent"    # Random spikes of bad entropy


@dataclass
class SimulatorConfig:
    mode:           EntropyMode  = EntropyMode.HIGH_QUALITY
    failure_mode:   FailureMode  = FailureMode.NONE
    target_h_min:   float        = 0.85   # Target min-entropy for MODELED_LASER
    seed:           int          = 0      # Deterministic for reproducibility


class EntropySimulator:
    """Generates simulated raw entropy data in selectable modes."""

    def __init__(self, config: SimulatorConfig):
        self.config       = config
        self._rng         = np.random.default_rng(config.seed)
        self._sample_idx  = 0
        self._failure_active = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate(self, n_bytes: int) -> bytes:
        """Generate n_bytes of simulated raw entropy."""
        if self.config.mode == EntropyMode.HIGH_QUALITY:
            return self._generate_high_quality(n_bytes)
        elif self.config.mode == EntropyMode.MODELED_LASER:
            return self._generate_modeled_laser(n_bytes)
        elif self.config.mode == EntropyMode.FAILURE_INJECT:
            return self._generate_with_failure(n_bytes)
        else:
            raise ValueError(f"Unknown mode: {self.config.mode}")

    # ------------------------------------------------------------------
    # Mode 1: high-quality cryptographic randomness
    # ------------------------------------------------------------------

    def _generate_high_quality(self, n_bytes: int) -> bytes:
        """OS-level cryptographic randomness.
        
        Used as a proxy for an ideal QRNG output. Health tests should
        always pass; min-entropy should be very close to 8 bits/byte.
        """
        return os.urandom(n_bytes)

    # ------------------------------------------------------------------
    # Mode 2: modeled DFB laser phase noise
    # ------------------------------------------------------------------

    def _generate_modeled_laser(self, n_bytes: int) -> bytes:
        """Biased byte stream simulating non-ideal entropy source.
        
        Produces biased bytes where (1 - target_h_min) fraction are forced
        to zero, giving a measurable H_min reduction that Toeplitz extraction
        then amplifies back toward uniformity.
        """
        arr = self._rng.integers(0, 256, size=n_bytes, dtype=np.uint8)
        bias_mask = self._rng.random(n_bytes) < (1.0 - self.config.target_h_min)
        arr[bias_mask] = 0
        return bytes(arr)
    def _generate_with_failure(self, n_bytes: int) -> bytes:
        """Generate data with injected failure for health test validation."""
        fmode = self.config.failure_mode

        if fmode == FailureMode.NONE:
            return self._generate_high_quality(n_bytes)

        elif fmode == FailureMode.STUCK_AT:
            # Stuck at value 0xAA (10101010) — RCT must fire
            return bytes([0xAA] * n_bytes)

        elif fmode == FailureMode.BIASED:
            # 80 % of bytes are 0, 20 % uniform — APT must fire
            arr = np.zeros(n_bytes, dtype=np.uint8)
            mask = self._rng.random(n_bytes) > 0.8
            arr[mask] = self._rng.integers(1, 256, size=int(mask.sum()),
                                           dtype=np.uint8)
            return bytes(arr)

        elif fmode == FailureMode.PERIODIC:
            # Repeating pattern — both RCT and APT should fire
            pattern = np.array([0x00, 0xFF, 0x55, 0xAA] * (n_bytes // 4 + 1),
                               dtype=np.uint8)
            return bytes(pattern[:n_bytes])

        elif fmode == FailureMode.GRADUAL:
            # Gradually shift mean — slow degradation
            self._sample_idx += n_bytes
            shift = min(self._sample_idx / 1_000_000, 0.4)  # max 40% bias
            arr = self._rng.integers(0, 256, size=n_bytes, dtype=np.uint8)
            mask = self._rng.random(n_bytes) < shift
            arr[mask] = 0
            return bytes(arr)

        elif fmode == FailureMode.INTERMITTENT:
            # Random spikes of bad entropy
            arr = np.frombuffer(os.urandom(n_bytes), dtype=np.uint8).copy()
            spike_indices = self._rng.choice(n_bytes,
                                             size=n_bytes // 100,
                                             replace=False)
            arr[spike_indices] = 0
            return bytes(arr)

        else:
            raise ValueError(f"Unknown failure mode: {fmode}")

    # ------------------------------------------------------------------
    # Stream interface for continuous generation
    # ------------------------------------------------------------------

    def stream(self, bytes_per_chunk: int = 65536):
        """Yields chunks of entropy continuously."""
        while True:
            yield self.generate(bytes_per_chunk)


# ----------------------------------------------------------------------
# Quick self-test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    print("Entropy Simulator self-test")
    print("=" * 60)

    # Test high quality
    sim = EntropySimulator(SimulatorConfig(mode=EntropyMode.HIGH_QUALITY))
    data = sim.generate(1024)
    print(f"High quality:    {len(data)} bytes,  first 16: {data[:16].hex()}")

    # Test modeled laser
    sim = EntropySimulator(SimulatorConfig(
        mode=EntropyMode.MODELED_LASER,
        target_h_min=0.85
    ))
    data = sim.generate(1024)
    arr = np.frombuffer(data, dtype=np.uint8)
    counts = np.bincount(arr, minlength=256)
    p_max = counts.max() / len(arr)
    h_min = -np.log2(p_max) if p_max > 0 else 0.0
    print(f"Modeled laser:   {len(data)} bytes,  measured H_min: {h_min:.3f}")

    # Test each failure mode
    for fmode in FailureMode:
        if fmode == FailureMode.NONE:
            continue
        sim = EntropySimulator(SimulatorConfig(
            mode=EntropyMode.FAILURE_INJECT, failure_mode=fmode
        ))
        data = sim.generate(1024)
        arr = np.frombuffer(data, dtype=np.uint8)
        counts = np.bincount(arr, minlength=256)
        p_max = counts.max() / len(arr)
        h_min = -np.log2(p_max) if p_max > 0 else 0.0
        unique = (counts > 0).sum()
        print(f"Failure {fmode.value:14s}: H_min={h_min:.3f}, "
              f"unique values: {unique}")
