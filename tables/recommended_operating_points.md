| Objective | 12B setting | 12B throughput | 26B-A4B setting | 26B-A4B throughput | Interpretation |
|---|---|---:|---|---:|---|
| Maximum measured throughput | N=6, p-min=0 | 226.39 tok/s | N=6, p-min=0 | 322.53 tok/s | Fastest tested primary condition |
| Quality-constrained maximum | N=6, p-min=0 | 226.39 tok/s | N=6, p-min=0 | 322.53 tok/s | Paired pilot clears frozen -5-point margin |
| Lower-variance alternative | N=3, p-min=0 | 203.91 tok/s | N=4, p-min=0 | 300.92 tok/s | Exploratory fallback; strict conservative rule returned no eligible point |
| Exact-text reproducibility | OFF | 89.20 tok/s | OFF | 169.58 tok/s | Use when byte/token identity is operationally required |
| Future p-min replication | N=12, p-min=.75 | 230.25 tok/s | N=6, p-min=.50 | 330.50 tok/s | Exploratory only; quality not rerun |

