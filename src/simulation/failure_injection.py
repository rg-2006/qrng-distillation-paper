"""
Failure injection scenarios for benchmarks.

Generates entropy streams that transition from healthy to compromised
at specified points, allowing measurement of health test detection latency.
"""

from __future__ import annotations

import numpy as np

from src.simulation.entropy_sim import (
    EntropySimulator, SimulatorConfig, EntropyMode, FailureMode
)


def generate_transition_stream(
    n_bytes_total:    int,
    n_bytes_healthy:  int,
    failure_mode:     FailureMode,
    target_h_min:     float = 0.85
) -> bytes:
    """Generate a stream that transitions from healthy to failed.
    
    Returns concatenated bytes: [healthy_section || failed_section]
    Used to measure how many bytes of bad data slip through before
    health tests detect the failure.
    """
    healthy_sim = EntropySimulator(SimulatorConfig(
        mode=EntropyMode.MODELED_LASER, target_h_min=target_h_min
    ))
    failed_sim = EntropySimulator(SimulatorConfig(
        mode=EntropyMode.FAILURE_INJECT, failure_mode=failure_mode
    ))

    healthy = healthy_sim.generate(n_bytes_healthy)
    failed  = failed_sim.generate(n_bytes_total - n_bytes_healthy)

    return healthy + failed


def measure_detection_latency(
    health_test_func,
    failure_mode:     FailureMode,
    block_size:       int   = 4096,
    max_blocks:       int   = 100,
    target_h_min:     float = 0.85
) -> dict:
    """Measure how many bytes pass before a health test detects failure.
    
    Args:
        health_test_func: callable(bytes) -> dict with 'rct_failed', 'apt_failed' keys
        failure_mode:     which failure to inject after warm-up
        block_size:       bytes per block
        max_blocks:       maximum blocks to test
    
    Returns:
        dict with detection_block, detection_bytes, detector
    """
    failed_sim = EntropySimulator(SimulatorConfig(
        mode=EntropyMode.FAILURE_INJECT, failure_mode=failure_mode
    ))

    for block_idx in range(max_blocks):
        block = failed_sim.generate(block_size)
        result = health_test_func(block)

        if result.get('rct_failed') or result.get('apt_failed'):
            return {
                'failure_mode':    failure_mode.value,
                'detection_block': block_idx,
                'detection_bytes': (block_idx + 1) * block_size,
                'detector':        'RCT' if result.get('rct_failed') else 'APT',
                'detected':        True,
            }

    return {
        'failure_mode':    failure_mode.value,
        'detection_block': max_blocks,
        'detection_bytes': max_blocks * block_size,
        'detector':        None,
        'detected':        False,
    }
