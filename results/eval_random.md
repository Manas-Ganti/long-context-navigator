policy: `random-reader`  (trained on n_hops <= 3)

| split | n | accuracy | ceiling_violation_rate | steps_over_min | compress_usage | fact_retention |
|---|---|---|---|---|---|---|
| overall | 600 | 0.003 | 0.000 | 0.945 | 0.000 | — |
| in_distribution | 300 | 0.000 | 0.000 | 1.210 | 0.000 | — |
| ood | 300 | 0.007 | 0.000 | 0.679 | 0.000 | — |

| n_hops | n | accuracy | ceiling_violation_rate | steps_over_min | compress_usage | fact_retention |
|---|---|---|---|---|---|---|
| 2 | 126 | 0.000 | 0.000 | 1.500 | 0.000 | — |
| 3 | 174 | 0.000 | 0.000 | 1.000 | 0.000 | — |
| 4 (OOD) | 159 | 0.006 | 0.000 | 0.750 | 0.000 | — |
| 5 (OOD) | 141 | 0.007 | 0.000 | 0.600 | 0.000 | — |
