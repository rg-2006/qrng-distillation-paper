"""
gRPC client implementation.

The client connects to the entropy server, opens a streaming RPC, and
fills a local buffer continuously in the background. Application code
reads from the local buffer at memory-read speed (~100ns) — the network
latency is hidden by the buffer.

This is the architectural pattern that makes application latency
independent of network RTT.
"""

from __future__ import annotations

import argparse
import collections
import logging
import threading
import time

import grpc

from src.proto_generated import entropy_service_pb2 as pb2
from src.proto_generated import entropy_service_pb2_grpc as pb2_grpc
from src.server.attestation import AttestationVerifier


# ======================================================================
# Local entropy buffer
# ======================================================================

class LocalBuffer:
    """A thread-safe buffer of entropy bytes filled by a background stream."""

    def __init__(self, max_bytes: int = 64 * 1024 * 1024):
        self.max_bytes = max_bytes
        self._buf      = collections.deque()
        self._size     = 0
        self._lock     = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._not_full  = threading.Condition(self._lock)

    @property
    def size(self) -> int:
        with self._lock:
            return self._size

    def put(self, data: bytes) -> None:
        with self._not_full:
            while self._size + len(data) > self.max_bytes:
                self._not_full.wait()
            self._buf.append(data)
            self._size += len(data)
            self._not_empty.notify()

    def get(self, n_bytes: int, timeout_s: float = 5.0) -> bytes | None:
        """Get exactly n_bytes from the buffer.
        
        Returns None on timeout.
        """
        deadline = time.time() + timeout_s
        with self._not_empty:
            while self._size < n_bytes:
                remaining = deadline - time.time()
                if remaining <= 0:
                    return None
                self._not_empty.wait(timeout=remaining)

            # We have enough. Pop chunks and concatenate.
            collected: list[bytes] = []
            collected_size = 0
            while collected_size < n_bytes and self._buf:
                chunk = self._buf.popleft()
                if collected_size + len(chunk) <= n_bytes:
                    collected.append(chunk)
                    collected_size += len(chunk)
                    self._size -= len(chunk)
                else:
                    take = n_bytes - collected_size
                    collected.append(chunk[:take])
                    leftover = chunk[take:]
                    self._buf.appendleft(leftover)
                    self._size -= take
                    collected_size += take
            self._not_full.notify()
            return b''.join(collected)


# ======================================================================
# Streaming client
# ======================================================================

class StreamingEntropyClient:
    """High-level client that maintains a continuously-filled local buffer."""

    def __init__(
        self,
        server_address:   str,
        customer_id:      str,
        auth_token:       str,
        public_key_pem:   bytes | None = None,
        buffer_size:      int = 64 * 1024 * 1024,
        block_size:       int = 4096,
    ):
        self.server_address = server_address
        self.customer_id    = customer_id
        self.auth_token     = auth_token
        self.block_size     = block_size

        self.buffer    = LocalBuffer(max_bytes=buffer_size)
        self._channel  = grpc.insecure_channel(
            server_address,
            options=[
                ('grpc.max_send_message_length',    16 * 1024 * 1024),
                ('grpc.max_receive_message_length', 16 * 1024 * 1024),
                ('grpc.keepalive_time_ms',          30_000),
            ],
        )
        self._stub     = pb2_grpc.EntropyServiceStub(self._channel)
        self._stop     = threading.Event()
        self._stream_thread: threading.Thread | None = None

        # Block accounting
        self.blocks_received = 0
        self.bytes_received  = 0
        self.attestation_failures = 0

        # Optional verifier
        self._verifier = (
            AttestationVerifier(public_key_pem) if public_key_pem else None
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open the stream and begin filling the buffer in the background."""
        if self._stream_thread is not None:
            return
        self._stop.clear()
        self._stream_thread = threading.Thread(
            target=self._run_stream, name="entropy_stream", daemon=True
        )
        self._stream_thread.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            self._channel.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internal streaming loop
    # ------------------------------------------------------------------

    def _run_stream(self) -> None:
        request = pb2.StreamRequest(
            customer_id = self.customer_id,
            auth_token  = self.auth_token,
            block_size  = self.block_size,
            target_mbps = 1000,
        )
        try:
            for block in self._stub.StreamEntropy(request):
                if self._stop.is_set():
                    break

                # Optional: verify attestation
                if self._verifier is not None:
                    valid = self._verifier.verify_block(
                        signature      = block.attestation,
                        block_sequence = block.block_sequence,
                        timestamp_ns   = block.timestamp_ns,
                        h_min          = block.h_min,
                        rct_passed     = block.rct_passed,
                        apt_passed     = block.apt_passed,
                        customer_id    = self.customer_id,
                        block_bytes    = block.entropy,
                    )
                    if not valid:
                        self.attestation_failures += 1
                        logging.warning(
                            f"Attestation FAILED for block {block.block_sequence}"
                        )
                        continue   # Reject unsigned/tampered block

                self.buffer.put(block.entropy)
                self.blocks_received += 1
                self.bytes_received  += len(block.entropy)

        except grpc.RpcError as e:
            if not self._stop.is_set():
                logging.error(f"gRPC stream error: {e.code()}: {e.details()}")
        except Exception as e:
            if not self._stop.is_set():
                logging.exception(f"Stream loop error: {e}")

    # ------------------------------------------------------------------
    # Application-facing API
    # ------------------------------------------------------------------

    def get_entropy(self, n_bytes: int, timeout_s: float = 5.0) -> bytes | None:
        """Application-facing entropy read.
        
        Reads from the LOCAL buffer — this is where the latency advantage
        comes from. No network call happens here.
        """
        return self.buffer.get(n_bytes, timeout_s=timeout_s)


# ======================================================================
# Demo / entry point
# ======================================================================

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--server",      default="localhost:50051")
    parser.add_argument("--customer",    default="demo_customer")
    parser.add_argument("--token",       default="demo_token_123")
    parser.add_argument("--public-key",  default="/tmp/qrng_public_key.pem")
    parser.add_argument("--n-iterations", type=int, default=10)
    args = parser.parse_args()

    # Load public key for attestation verification
    public_key_pem = None
    try:
        with open(args.public_key, "rb") as f:
            public_key_pem = f.read()
    except FileNotFoundError:
        logging.warning("Public key not found; attestation verification disabled")

    client = StreamingEntropyClient(
        server_address=args.server,
        customer_id=args.customer,
        auth_token=args.token,
        public_key_pem=public_key_pem,
    )
    client.start()
    logging.info("Stream started; waiting for buffer to fill...")

    # Wait for buffer to have some data
    time.sleep(1.0)

    print()
    print("=" * 60)
    print("Demonstration: application-level entropy access")
    print("=" * 60)

    latencies_ns: list[int] = []
    for i in range(args.n_iterations):
        t0 = time.perf_counter_ns()
        data = client.get_entropy(32, timeout_s=2.0)
        t1 = time.perf_counter_ns()
        if data is None:
            print(f"[{i}] timeout")
            continue
        latency_us = (t1 - t0) / 1000.0
        latencies_ns.append(t1 - t0)
        print(f"[{i:3d}]  32 bytes   "
              f"latency: {latency_us:8.2f} µs   "
              f"sample: {data[:8].hex()}")

    print()
    if latencies_ns:
        import statistics
        median = statistics.median(latencies_ns) / 1000.0
        p99    = sorted(latencies_ns)[int(len(latencies_ns) * 0.99)] / 1000.0
        print(f"Median application latency: {median:.2f} µs")
        print(f"P99    application latency: {p99:.2f} µs")
    print(f"Blocks received:    {client.blocks_received}")
    print(f"Bytes received:     {client.bytes_received:,}")
    print(f"Attestation fails:  {client.attestation_failures}")

    client.stop()


if __name__ == "__main__":
    main()
