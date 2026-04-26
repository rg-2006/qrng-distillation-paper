# Health Test Sensitivity

**GPU:** b'NVIDIA H100 80GB HBM3'

## Detection Latency by Failure Mode

| Failure Mode | Detection Rate | Mean Blocks | Mean Bytes | Primary Detector |
|---|---:|---:|---:|---|
| stuck_at | 100.0% | 1.0 | 4.0 KB | BOTH |
| biased | 100.0% | 1.0 | 4.0 KB | BOTH |
| periodic | 100.0% | 1.0 | 4.0 KB | APT |
| gradual | 100.0% | 21.2 | 84.7 KB | APT |
| intermittent | 0.0% | — | — | NONE |

## False Positive Rate (healthy input)

- Tested over 5000 healthy blocks
- RCT FPR: **0.000000** (0 false positives)
- APT FPR: **0.000000** (0 false positives)
- NIST target: 2^-20 = 0.000001
