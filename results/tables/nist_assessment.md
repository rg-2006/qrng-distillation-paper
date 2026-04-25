# NIST SP 800-90B Entropy Assessment

**GPU:** b'NVIDIA H100 PCIe'

**Assessment Tool:** builtin_mcv

## Min-Entropy Before and After Extraction

| Scenario | Raw H_min | Extracted H_min | Ratio |
|---|---:|---:|---:|
| high_quality | 7.9386 | 7.9350 | 1.000× |
| modeled_laser | 7.9340 | 7.9406 | 1.001× |

Ideal extracted output: H_min ≈ 7.99 bits/byte (near uniform)
