# Buffer Dynamics Under Load

**GPU:** b'NVIDIA H100 PCIe'

Demonstrates buffer behavior at increasing drain rates.
As long as refill > drain, application latency stays at memory-read speed.

| Scenario | Refill | Drain | Successful | Mean Latency | Min Depth |
|---|---:|---:|---:|---:|---:|
| Light:  10 Mbps drain (TLS sessions <1k/s) | 1.0 Gbps | 10 Mbps | 58,594 | 12.10 µs | 0.90 MB |
| Medium: 100 Mbps drain (10k sessions/s) | 1.0 Gbps | 100 Mbps | 585,938 | 3.57 µs | 0.91 MB |
| Heavy:  500 Mbps drain (50k sessions/s) | 1.0 Gbps | 500 Mbps | 736,607 | 3.26 µs | 0.86 MB |
| Near-saturation: 900 Mbps drain | 1.0 Gbps | 900 Mbps | 1,034,077 | 2.36 µs | 0.90 MB |
| Over-saturation: 1.5 Gbps drain (buffer dries) | 1.0 Gbps | 1500 Mbps | 1,250,906 | 1.95 µs | 0.60 MB |
