policy: `oracle`  (trained on n_hops <= 3)

| split | n | accuracy | ceiling_violation_rate | steps_over_min | compress_usage | fact_retention |
|---|---|---|---|---|---|---|
| overall | 600 | 1.000 | 0.000 | 1.187 | 1.000 | 1.000 |
| in_distribution | 300 | 1.000 | 0.000 | 1.183 | 1.000 | 1.000 |
| ood | 300 | 1.000 | 0.000 | 1.190 | 1.000 | 1.000 |

| n_hops | n | accuracy | ceiling_violation_rate | steps_over_min | compress_usage | fact_retention |
|---|---|---|---|---|---|---|
| 2 | 126 | 1.000 | 0.000 | 1.190 | 1.000 | 1.000 |
| 3 | 174 | 1.000 | 0.000 | 1.178 | 1.000 | 1.000 |
| 4 (OOD) | 159 | 1.000 | 0.000 | 1.198 | 1.000 | 1.000 |
| 5 (OOD) | 141 | 1.000 | 0.000 | 1.182 | 1.000 | 1.000 |
