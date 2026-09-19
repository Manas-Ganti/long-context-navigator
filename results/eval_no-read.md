policy: `no-read`  (trained on n_hops <= 3)

| split | n | accuracy | ceiling_violation_rate | steps_over_min | compress_usage | fact_retention |
|---|---|---|---|---|---|---|
| overall | 600 | 0.000 | 0.000 | 0.157 | 0.000 | — |
| in_distribution | 300 | 0.000 | 0.000 | 0.202 | 0.000 | — |
| ood | 300 | 0.000 | 0.000 | 0.113 | 0.000 | — |

| n_hops | n | accuracy | ceiling_violation_rate | steps_over_min | compress_usage | fact_retention |
|---|---|---|---|---|---|---|
| 2 | 126 | 0.000 | 0.000 | 0.250 | 0.000 | — |
| 3 | 174 | 0.000 | 0.000 | 0.167 | 0.000 | — |
| 4 (OOD) | 159 | 0.000 | 0.000 | 0.125 | 0.000 | — |
| 5 (OOD) | 141 | 0.000 | 0.000 | 0.100 | 0.000 | — |
