"""
Per-block cryptographic attestation.

Every entropy block delivered to a customer carries a signed attestation
binding the block bytes to its health metrics. In production this signing
would be done by an HSM (Thales Luna or similar). For the paper prototype
we use Ed25519 in software — the cryptographic property is identical,
the keys just don't live in tamper-resistant hardware.

Attestation payload (signed):
  block_sequence || timestamp_ns || h_min || rct_passed || apt_passed
                                || customer_id || sha256(block_bytes)

Customers (or their auditors) can verify each block independently using
the public key. This is the strongest IP claim we identified earlier.
"""

from __future__ import annotations

import hashlib
import struct
import time

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey, Ed25519PublicKey
)
from cryptography.hazmat.primitives import serialization


class AttestationSigner:
    """Produces per-block signatures binding block contents to health metrics."""

    def __init__(self, private_key: Ed25519PrivateKey | None = None):
        if private_key is None:
            private_key = Ed25519PrivateKey.generate()
        self._private = private_key
        self._public  = private_key.public_key()

    @classmethod
    def generate_new(cls) -> "AttestationSigner":
        return cls(Ed25519PrivateKey.generate())

    @property
    def public_key_pem(self) -> bytes:
        return self._public.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )

    @staticmethod
    def _build_payload(
        block_sequence: int,
        timestamp_ns:   int,
        h_min:          float,
        rct_passed:     bool,
        apt_passed:     bool,
        customer_id:    str,
        block_bytes:    bytes,
    ) -> bytes:
        block_hash = hashlib.sha256(block_bytes).digest()
        return (
            struct.pack(">QQd??",
                        block_sequence,
                        timestamp_ns,
                        h_min,
                        rct_passed,
                        apt_passed)
            + customer_id.encode('utf-8').ljust(64, b'\x00')[:64]
            + block_hash
        )

    def sign_block(
        self,
        block_sequence: int,
        timestamp_ns:   int,
        h_min:          float,
        rct_passed:     bool,
        apt_passed:     bool,
        customer_id:    str,
        block_bytes:    bytes,
    ) -> bytes:
        payload = self._build_payload(
            block_sequence, timestamp_ns, h_min,
            rct_passed, apt_passed, customer_id, block_bytes
        )
        return self._private.sign(payload)


class AttestationVerifier:
    """Verifies signatures from the public key (used client-side)."""

    def __init__(self, public_key_pem: bytes):
        self._public = serialization.load_pem_public_key(public_key_pem)

    def verify_block(
        self,
        signature:      bytes,
        block_sequence: int,
        timestamp_ns:   int,
        h_min:          float,
        rct_passed:     bool,
        apt_passed:     bool,
        customer_id:    str,
        block_bytes:    bytes,
    ) -> bool:
        from cryptography.exceptions import InvalidSignature
        payload = AttestationSigner._build_payload(
            block_sequence, timestamp_ns, h_min,
            rct_passed, apt_passed, customer_id, block_bytes
        )
        try:
            self._public.verify(signature, payload)
            return True
        except InvalidSignature:
            return False


# ----------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    print("Attestation self-test")
    print("=" * 60)

    signer = AttestationSigner.generate_new()
    verifier = AttestationVerifier(signer.public_key_pem)

    block = b'\x00' * 4096
    sig = signer.sign_block(
        block_sequence=42,
        timestamp_ns=time.time_ns(),
        h_min=7.99,
        rct_passed=True,
        apt_passed=True,
        customer_id="customer_alpha",
        block_bytes=block,
    )
    print(f"Signature: {sig.hex()[:32]}... ({len(sig)} bytes)")

    # Verify with same parameters
    valid = verifier.verify_block(
        sig, 42, sig and 0, 7.99, True, True, "customer_alpha", block
    )
    # Note: above intentionally wrong timestamp to show verification fails
    print(f"Tampered verification: {valid}  (expect False)")

    # Reproduce original timestamp
    correct_ts = time.time_ns()
    sig = signer.sign_block(42, correct_ts, 7.99, True, True, "customer_alpha", block)
    valid = verifier.verify_block(sig, 42, correct_ts, 7.99, True, True, "customer_alpha", block)
    print(f"Correct verification: {valid}  (expect True)")
