"""
Per-customer channel isolation.

Each customer's stream is cryptographically isolated from other customers
even though they share the same certified entropy pool. We achieve this by
running an independent Toeplitz re-extraction with a customer-specific seed.

Key claim being proven by the paper:
  Two customers (A and B) with different Toeplitz seeds, reading from the
  same shared pool, produce statistically independent output streams.
  
Mechanism:
  pool_block (shared) → re-extract with seed_A → output_A (delivered to A)
                     → re-extract with seed_B → output_B (delivered to B)
  
  Since seed_A != seed_B, and Toeplitz extraction is injective on uniform
  input, output_A and output_B are uncorrelated bit-for-bit.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

import cupy as cp
import numpy as np

from src.cuda.kernels import toeplitz_extract_gpu
from src.server.entropy_pool import EntropyPool, BlockMetadata


@dataclass
class ChannelConfig:
    customer_id:        str
    pool_block_size:    int = 4096        # Must match pool block size
    output_block_size:  int = 2048        # After re-extraction
    seed_bytes_extra:   int = 8           # Padding for atomic alignment


class IsolatedChannel:
    """A cryptographically isolated entropy stream for a single customer."""

    def __init__(self, pool: EntropyPool, config: ChannelConfig):
        self.pool        = pool
        self.config      = config
        self.customer_id = config.customer_id

        # Generate a customer-specific Toeplitz seed.
        # In production: HSM-derived. Here: derived from customer ID + secret salt.
        # We use a stable seed so re-extraction is deterministic.
        salt = b"qrng-paper-prototype-salt-2026"
        seed_material = hashlib.sha512(salt + config.customer_id.encode()).digest()

        n_input_bits  = config.pool_block_size  * 8
        m_output_bits = config.output_block_size * 8
        seed_bytes    = (n_input_bits + m_output_bits - 1 + 7) // 8 + config.seed_bytes_extra

        # Expand the 64-byte SHA-512 hash to the required seed length.
        # We use the hash as a key for an HKDF-style expansion (here:
        # repeated SHA-256 with counter; deterministic and seed-quality).
        expanded = bytearray()
        counter = 0
        while len(expanded) < seed_bytes:
            block = hashlib.sha256(
                seed_material + counter.to_bytes(4, 'big')
            ).digest()
            expanded.extend(block)
            counter += 1
        expanded = bytes(expanded[:seed_bytes])

        self._seed_cpu       = np.frombuffer(expanded, dtype=np.uint8).copy()
        self.seed_gpu        = cp.asarray(self._seed_cpu)
        self.n_input_bits    = n_input_bits
        self.m_output_bits   = m_output_bits

        # Identifier for the seed (used in attestation)
        self.seed_id = hashlib.sha256(expanded).hexdigest()[:16]

        # Register the customer with the pool
        pool.register_customer(config.customer_id)

    # ------------------------------------------------------------------
    # Reading entropy (returns re-extracted bytes)
    # ------------------------------------------------------------------

    def read_block(
        self, timeout_s: float = 5.0
    ) -> tuple[bytes, BlockMetadata, str] | None:
        """Read and re-extract one block for this customer.
        
        Returns (extracted_bytes, source_metadata, seed_id) or None on timeout.
        """
        result = self.pool.read_block(self.customer_id, timeout_s=timeout_s)
        if result is None:
            return None
        pool_block_gpu, metadata = result

        # Re-extract with customer-specific seed
        extracted_gpu = toeplitz_extract_gpu(
            self.seed_gpu, pool_block_gpu,
            self.n_input_bits, self.m_output_bits,
        )

        # Bring to host for delivery (gRPC operates on host bytes)
        extracted_cpu = bytes(extracted_gpu.get())

        return extracted_cpu, metadata, self.seed_id

    def close(self) -> None:
        self.pool.unregister_customer(self.customer_id)


# ----------------------------------------------------------------------
# Channel manager
# ----------------------------------------------------------------------

class ChannelManager:
    """Manages multiple isolated customer channels on a shared pool."""

    def __init__(self, pool: EntropyPool):
        self.pool       = pool
        self._channels: dict[str, IsolatedChannel] = {}

    def create_channel(self, customer_id: str,
                        output_block_size: int = 2048) -> IsolatedChannel:
        if customer_id in self._channels:
            raise ValueError(f"Customer {customer_id} already has a channel")
        config = ChannelConfig(
            customer_id       = customer_id,
            pool_block_size   = self.pool.block_size,
            output_block_size = output_block_size,
        )
        ch = IsolatedChannel(self.pool, config)
        self._channels[customer_id] = ch
        return ch

    def get_channel(self, customer_id: str) -> IsolatedChannel | None:
        return self._channels.get(customer_id)

    def close_channel(self, customer_id: str) -> None:
        ch = self._channels.pop(customer_id, None)
        if ch is not None:
            ch.close()

    def list_customers(self) -> list[str]:
        return list(self._channels.keys())
