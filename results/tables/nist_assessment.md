# NIST SP 800-90B Entropy Assessment

**GPU:** b'NVIDIA H100 80GB HBM3'

**Assessment Tool:** builtin_mcv

## Min-Entropy Before and After Extraction

| Scenario | Raw H_min | Extracted H_min | Ratio |
|---|---:|---:|---:|
| high_quality | 7.9397 | 7.9345 | 0.999× |
| modeled_laser | 2.7073 | 7.9256 | 2.928× |

Ideal extracted output: H_min ≈ 7.99 bits/byte (near uniform)
