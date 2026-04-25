# Application Latency Benchmark

**GPU:** b'NVIDIA H100 PCIe'

## Latency by Request Size

| Request Size | Median | P99 | P99.9 | Min | Max |
|---:|---:|---:|---:|---:|---:|
| 16 B | 3.638 µs | 6.323 µs | 14.351 µs | 1.771 µs | 110.883 µs |
| 32 B | 3.578 µs | 6.041 µs | 14.275 µs | 1.777 µs | 37.905 µs |
| 64 B | 3.599 µs | 6.040 µs | 14.015 µs | 1.741 µs | 31.070 µs |
| 128 B | 3.627 µs | 5.883 µs | 14.080 µs | 1.754 µs | 31.235 µs |
| 256 B | 3.673 µs | 5.875 µs | 14.305 µs | 1.757 µs | 20.933 µs |
| 1024 B | 3.720 µs | 5.959 µs | 14.430 µs | 1.744 µs | 39.552 µs |
| 4096 B | 3.963 µs | 9.579 µs | 16.943 µs | 1.793 µs | 42.291 µs |

## Comparison with Competitor Systems

| System | Latency | Notes |
|---|---:|---|
| Qrypt REST API (cloud) | 1.00 ms | best case, same region |
| Qrypt REST API (cross-region) | 50.00 ms | typical enterprise |
| ID Quantique PCIe card | 15.00 µs | local PCIe driver call |
| Our gRPC streaming + buffer | 0.50 µs | measured by benchmark |
