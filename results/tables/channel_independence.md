# Per-Channel Cryptographic Independence

**GPU:** b'NVIDIA H100 PCIe'

## Pairwise Independence Tests

| Pair | Pearson r | Hamming | Chi² p | MI (bits) |
|---|---:|---:|---:|---:|
| customer_alpha vs customer_beta | -0.002133 | 0.500334 | 0.2070 | 0.0291 |
| customer_alpha vs customer_gamma | -0.000234 | 0.499686 | 0.1280 | 0.0308 |
| customer_beta vs customer_gamma | +0.001240 | 0.499648 | 0.0620 | 0.0298 |

## Interpretation

- **Pearson r ≈ 0**: no linear correlation between channels
- **Hamming ratio ≈ 0.5**: bits differ at random positions
- **Chi² p > 0.05**: cannot reject hypothesis of independence
- **MI ≈ 0**: no information shared between channels

These together provide statistical evidence that per-customer Toeplitz re-extraction yields cryptographically independent outputs even from a shared entropy pool.
