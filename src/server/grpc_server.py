"""
gRPC server implementation.

Implements the EntropyService defined in proto/entropy_service.proto.
The server holds:
  - An EntropyPool (shared GPU-resident certified entropy buffer)
  - A DistillationPipeline (running continuously in a background thread)
  - A ChannelManager (creates per-customer isolated streams)
  - An AttestationSigner (signs every delivered block)

The streaming RPC is the architectural innovation: each customer call
to StreamEntropy holds a long-lived gRPC stream over which the server
continuously pushes EntropyBlock messages without further requests.
"""

from __future__ import annotations

import argparse
import logging
import time
from concurrent import futures

import grpc

from src.proto_generated import entropy_service_pb2 as pb2
from src.proto_generated import entropy_service_pb2_grpc as pb2_grpc

from src.server.entropy_pool   import EntropyPool
from src.server.distillation   import DistillationPipeline, DistillationConfig
from src.server.channels       import ChannelManager
from src.server.attestation    import AttestationSigner
from src.simulation.entropy_sim import (
    EntropySimulator, SimulatorConfig, EntropyMode
)


# ======================================================================
# Service implementation
# ======================================================================

class EntropyServicer(pb2_grpc.EntropyServiceServicer):

    def __init__(
        self,
        pool:        EntropyPool,
        manager:     ChannelManager,
        signer:      AttestationSigner,
        pipeline:    DistillationPipeline,
    ):
        self.pool      = pool
        self.manager   = manager
        self.signer    = signer
        self.pipeline  = pipeline
        self._start_ns = time.time_ns()

    # ------------------------------------------------------------------
    # Tier 1 — unary GetEntropy
    # ------------------------------------------------------------------

    def GetEntropy(self, request: pb2.EntropyRequest,
                    context: grpc.ServicerContext) -> pb2.EntropyResponse:
        # Simple authentication for the prototype
        if not request.api_key:
            context.abort(grpc.StatusCode.UNAUTHENTICATED, "API key required")

        bytes_requested = min(request.bytes_requested, 1024 * 1024)
        if bytes_requested == 0:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "bytes_requested must be > 0")

        # Use a generic shared customer for tier 1
        customer_id = f"tier1_{request.api_key[:16]}"
        ch = self.manager.get_channel(customer_id)
        if ch is None:
            ch = self.manager.create_channel(customer_id)

        # Read enough blocks to satisfy the request
        chunks: list[bytes] = []
        accumulated = 0
        h_min_sum = 0.0
        h_min_n = 0
        last_meta = None

        while accumulated < bytes_requested:
            result = ch.read_block(timeout_s=5.0)
            if result is None:
                context.abort(grpc.StatusCode.DEADLINE_EXCEEDED,
                              "Entropy pool not producing fast enough")
            block_bytes, metadata, seed_id = result
            chunks.append(block_bytes)
            accumulated += len(block_bytes)
            h_min_sum += metadata.h_min
            h_min_n += 1
            last_meta = metadata

        full_bytes = b''.join(chunks)[:bytes_requested]
        avg_h_min = h_min_sum / max(1, h_min_n)
        ts_ns = time.time_ns()

        # Sign the response
        sig = self.signer.sign_block(
            block_sequence=last_meta.sequence if last_meta else 0,
            timestamp_ns=ts_ns,
            h_min=avg_h_min,
            rct_passed=True,
            apt_passed=True,
            customer_id=customer_id,
            block_bytes=full_bytes,
        )

        health = self.pool.get_health_summary()
        return pb2.EntropyResponse(
            entropy=full_bytes,
            block_sequence=last_meta.sequence if last_meta else 0,
            h_min=avg_h_min,
            rct_pass_rate=health['rct_pass_rate'],
            apt_pass_rate=health['apt_pass_rate'],
            certification="NIST-SP-800-90B-PROTOTYPE",
            timestamp_ns=ts_ns,
            block_signature=sig,
        )

    # ------------------------------------------------------------------
    # Tier 2 — server-side streaming
    # ------------------------------------------------------------------

    def StreamEntropy(self, request: pb2.StreamRequest,
                       context: grpc.ServicerContext):
        """Server-side streaming RPC.

        This is the core architectural pattern. The client makes one
        StreamEntropy call; the server pushes EntropyBlock messages
        continuously until the client cancels or disconnects.
        """
        if not request.auth_token:
            context.abort(grpc.StatusCode.UNAUTHENTICATED, "auth_token required")

        customer_id = request.customer_id or f"customer_{request.auth_token[:16]}"

        # Create a per-customer isolated channel
        ch = self.manager.get_channel(customer_id)
        if ch is None:
            ch = self.manager.create_channel(
                customer_id=customer_id,
                output_block_size=request.block_size or 2048,
            )

        logging.info(f"Stream opened for customer {customer_id}")

        try:
            while context.is_active():
                result = ch.read_block(timeout_s=5.0)
                if result is None:
                    continue   # No data yet; keep stream alive
                block_bytes, metadata, seed_id = result

                # Sign the block
                ts_ns = time.time_ns()
                sig = self.signer.sign_block(
                    block_sequence=metadata.sequence,
                    timestamp_ns=ts_ns,
                    h_min=metadata.h_min,
                    rct_passed=metadata.rct_passed,
                    apt_passed=metadata.apt_passed,
                    customer_id=customer_id,
                    block_bytes=block_bytes,
                )

                yield pb2.EntropyBlock(
                    entropy=block_bytes,
                    block_sequence=metadata.sequence,
                    h_min=metadata.h_min,
                    rct_passed=metadata.rct_passed,
                    apt_passed=metadata.apt_passed,
                    timestamp_ns=ts_ns,
                    attestation=sig,
                    toeplitz_seed_id=seed_id,
                )

        except Exception as e:
            logging.warning(f"Stream error for {customer_id}: {e}")
        finally:
            logging.info(f"Stream closed for customer {customer_id}")
            # Channel persists for reconnection; only close on explicit unregister

    # ------------------------------------------------------------------
    # Health monitoring
    # ------------------------------------------------------------------

    def GetHealthStatus(self, request: pb2.HealthRequest,
                         context: grpc.ServicerContext) -> pb2.HealthResponse:
        h = self.pool.get_health_summary()
        uptime = (time.time_ns() - self._start_ns) // 1_000_000_000
        status = "NOMINAL"
        if h['rct_pass_rate'] < 0.99 or h['apt_pass_rate'] < 0.99:
            status = "DEGRADED"
        if h['current_h_min'] < 0.3:
            status = "FAILED"

        return pb2.HealthResponse(
            current_h_min       = h['current_h_min'],
            rolling_h_min_mean  = h['rolling_h_min_mean'],
            rolling_h_min_std   = h['rolling_h_min_std'],
            rct_pass_rate       = h['rct_pass_rate'],
            apt_pass_rate       = h['apt_pass_rate'],
            blocks_delivered    = h['blocks_read'],
            blocks_rejected     = h['blocks_rejected'],
            pool_depth_gbits    = h['pool_depth_gbits'],
            source_status       = status,
            uptime_seconds      = uptime,
        )

    def StreamHealth(self, request: pb2.HealthRequest,
                      context: grpc.ServicerContext):
        while context.is_active():
            h = self.pool.get_health_summary()
            yield pb2.HealthMetrics(
                h_min            = h['current_h_min'],
                rct_passed       = h['rct_pass_rate'] > 0.99,
                apt_passed       = h['apt_pass_rate'] > 0.99,
                timestamp_ns     = time.time_ns(),
                pool_depth_gbits = h['pool_depth_gbits'],
            )
            time.sleep(1.0)


# ======================================================================
# Server entry point
# ======================================================================

def serve(host: str = "0.0.0.0", port: int = 50051,
          n_workers: int = 32) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Create infrastructure components
    pool = EntropyPool(pool_size_bytes=256 * 1024 * 1024,   # 256 MB
                        block_size=4096)
    simulator = EntropySimulator(SimulatorConfig(
        mode=EntropyMode.HIGH_QUALITY
    ))
    pipeline = DistillationPipeline(pool, simulator,
                                     DistillationConfig())
    pipeline.start()

    manager = ChannelManager(pool)
    signer  = AttestationSigner.generate_new()

    # Save public key so clients can verify
    with open("/tmp/qrng_public_key.pem", "wb") as f:
        f.write(signer.public_key_pem)
    logging.info("Public key saved to /tmp/qrng_public_key.pem")

    # Start gRPC server
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=n_workers),
        options=[
            ('grpc.max_send_message_length',    16 * 1024 * 1024),
            ('grpc.max_receive_message_length', 16 * 1024 * 1024),
            ('grpc.keepalive_time_ms',          30_000),
        ],
    )
    pb2_grpc.add_EntropyServiceServicer_to_server(
        EntropyServicer(pool, manager, signer, pipeline),
        server,
    )
    server.add_insecure_port(f"{host}:{port}")
    server.start()
    logging.info(f"gRPC server listening on {host}:{port}")

    try:
        # Print throughput stats every 5 seconds
        while True:
            time.sleep(5.0)
            stats = pipeline.stats()
            health = pool.get_health_summary()
            logging.info(
                f"Throughput: {stats['throughput_gbps']:.3f} Gbps  "
                f"H_min: {health['current_h_min']:.3f}  "
                f"Pool depth: {health['pool_depth_gbits']:.2f} Gbits  "
                f"Customers: {len(manager.list_customers())}"
            )
    except KeyboardInterrupt:
        logging.info("Shutting down...")
        pipeline.stop()
        server.stop(grace=2.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    serve(args.host, args.port, args.workers)
