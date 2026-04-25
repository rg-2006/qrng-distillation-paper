"""
GPU-resident certified entropy pool.

The pool is a circular buffer in HBM that the distillation pipeline writes
to and the channel isolation layer reads from. It is the only point of
contact between entropy production and entropy delivery.

Key design properties:
  - Producer (distillation) holds exclusive write access
  - Consumers (per-customer channels) have independent read heads
  - Pool stores ONLY certified post-extraction entropy
  - Health metadata is stored alongside each block for attestation
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import cupy as cp
import numpy as np


@dataclass
class BlockMetadata:
    """Health and provenance metadata for a single entropy block."""
    sequence:    int
    h_min:       float
    rct_passed:  bool
    apt_passed:  bool
    timestamp_ns: int
    block_size:  int


@dataclass
class PoolStats:
    """Rolling statistics over recent blocks."""
    blocks_written:   int = 0
    blocks_read:      int = 0
    blocks_rejected:  int = 0
    rolling_h_min:    list = field(default_factory=list)
    rct_pass_count:   int = 0
    apt_pass_count:   int = 0


class EntropyPool:
    """A circular buffer of certified entropy blocks in GPU memory."""

    def __init__(
        self,
        pool_size_bytes: int = 1 << 30,   # 1 GB default (small for prototyping)
        block_size:      int = 4096,
    ):
        self.pool_size_bytes = pool_size_bytes
        self.block_size      = block_size
        self.n_blocks        = pool_size_bytes // block_size

        # Allocate the pool buffer in GPU memory
        self.buffer = cp.zeros(pool_size_bytes, dtype=cp.uint8)

        # Metadata stored host-side (Python list, indexed mod n_blocks)
        self.metadata: list[Optional[BlockMetadata]] = [None] * self.n_blocks

        # Write head — owned exclusively by the distillation pipeline
        self._write_head = 0

        # Per-customer read heads (indexed by customer id)
        self._read_heads: dict[str, int] = {}

        # Stats
        self.stats = PoolStats()

        # Lock guards write_head, read_heads, metadata
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Producer interface
    # ------------------------------------------------------------------

    def write_block(
        self,
        block_data:   cp.ndarray,
        h_min:        float,
        rct_passed:   bool,
        apt_passed:   bool,
    ) -> int:
        """Write one certified block to the pool. Returns sequence number.
        
        block_data must be exactly block_size bytes.
        """
        assert block_data.size == self.block_size, (
            f"Block size mismatch: {block_data.size} != {self.block_size}"
        )

        with self._lock:
            slot = self._write_head % self.n_blocks
            offset = slot * self.block_size

            # Copy block into pool (GPU-to-GPU memcpy, no PCIe involvement)
            self.buffer[offset:offset + self.block_size] = block_data

            # Record metadata
            metadata = BlockMetadata(
                sequence     = self._write_head,
                h_min        = h_min,
                rct_passed   = rct_passed,
                apt_passed   = apt_passed,
                timestamp_ns = time.time_ns(),
                block_size   = self.block_size,
            )
            self.metadata[slot] = metadata

            seq = self._write_head
            self._write_head += 1

            # Stats
            self.stats.blocks_written += 1
            self.stats.rolling_h_min.append(h_min)
            if len(self.stats.rolling_h_min) > 1000:
                self.stats.rolling_h_min.pop(0)
            if rct_passed:  self.stats.rct_pass_count += 1
            if apt_passed:  self.stats.apt_pass_count += 1

            return seq

    def reject_block(self) -> None:
        """Increment the rejection counter (block did not pass health tests)."""
        with self._lock:
            self.stats.blocks_rejected += 1

    # ------------------------------------------------------------------
    # Consumer interface
    # ------------------------------------------------------------------

    def register_customer(self, customer_id: str) -> None:
        """Register a customer; their read head starts at the current write head."""
        with self._lock:
            self._read_heads[customer_id] = self._write_head

    def unregister_customer(self, customer_id: str) -> None:
        with self._lock:
            self._read_heads.pop(customer_id, None)

    def read_block(
        self, customer_id: str, timeout_s: float = 5.0
    ) -> Optional[tuple[cp.ndarray, BlockMetadata]]:
        """Read the next block for a customer.
        
        Blocks until data is available or timeout expires.
        Returns (block_data, metadata) or None on timeout.
        """
        deadline = time.time() + timeout_s

        while time.time() < deadline:
            with self._lock:
                if customer_id not in self._read_heads:
                    raise ValueError(f"Customer {customer_id} not registered")

                read_head = self._read_heads[customer_id]

                # Is there a new block to read?
                if read_head < self._write_head:
                    slot = read_head % self.n_blocks
                    offset = slot * self.block_size

                    # Copy block out (GPU-resident — no host transfer here)
                    block = self.buffer[offset:offset + self.block_size].copy()
                    metadata = self.metadata[slot]

                    self._read_heads[customer_id] += 1
                    self.stats.blocks_read += 1

                    return block, metadata

            # No new data; wait briefly
            time.sleep(0.0001)   # 100 µs

        return None

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @property
    def write_head(self) -> int:
        with self._lock:
            return self._write_head

    @property
    def pool_depth_blocks(self) -> int:
        """Number of blocks currently in the pool buffer (capped at n_blocks)."""
        with self._lock:
            return min(self._write_head, self.n_blocks)

    @property
    def pool_depth_bits(self) -> int:
        return self.pool_depth_blocks * self.block_size * 8

    def get_health_summary(self) -> dict:
        with self._lock:
            n = max(1, len(self.stats.rolling_h_min))
            mean_h = float(np.mean(self.stats.rolling_h_min)) if self.stats.rolling_h_min else 0.0
            std_h  = float(np.std(self.stats.rolling_h_min))  if self.stats.rolling_h_min else 0.0

            total = max(1, self.stats.blocks_written)
            return {
                'blocks_written':     self.stats.blocks_written,
                'blocks_read':        self.stats.blocks_read,
                'blocks_rejected':    self.stats.blocks_rejected,
                'current_h_min':      self.stats.rolling_h_min[-1] if self.stats.rolling_h_min else 0.0,
                'rolling_h_min_mean': mean_h,
                'rolling_h_min_std':  std_h,
                'rct_pass_rate':      self.stats.rct_pass_count / total,
                'apt_pass_rate':      self.stats.apt_pass_count / total,
                'pool_depth_gbits':   self.pool_depth_bits / 1e9,
            }
