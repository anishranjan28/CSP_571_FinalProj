# Model card — `segmentation_pipeline.rds`

**Pipeline name:** `uci-retail-segmentation`
**Version:** 1.0.0
**Type:** Unsupervised customer segmentation (PCA + k-means)
**Frameworks:** R 4.x, base + tidyverse + cluster + fpc

---

## Intended use

The pipeline assigns each customer of the UCI Online Retail merchant to one
of three behavioural segments (`A`, `B`, `C`), ranked by descending median
net spend. It is intended as a **descriptive segmentation tool** to support
marketing prioritisation and reporting. It is *not* a predictive model of
future spend; it does not estimate causal effects of any treatment; it does
not score risk.

## Training data

* **Source:** UCI Online Retail (id 352)
* **Period:** Dec 2010 – Dec 2011 (one year)
* **Geography:** UK gift retailer (~83% of revenue is UK)
* **Rows after cleaning:** ~391,150 positive transactions across ~4,334
  customers
* **Cancellations:** kept in a separate file (`online_retail_cancelled.csv`)
  and used to compute net-of-returns features

## Features used by the model

| Feature                | Definition (per customer i)                                              |
|------------------------|--------------------------------------------------------------------------|
| `log_Recency`          | log(1 + days since last positive purchase)                                |
| `log_Frequency`        | log(1 + count of distinct positive invoices)                              |
| `log_NetMonetary`      | log(1 + max(GrossMonetary − ReturnValueAbs, 0))                           |
| `log_UniqueProducts`   | log(1 + count of distinct StockCode purchased)                            |
| `log_AvgOrderValue`    | log(1 + mean invoice total)                                               |
| `AvgUnitPrice_winsor`  | quantity-weighted price-per-item, winsorised at the 99th percentile       |
| `TenureDays`           | LastPurchase − FirstPurchase, in days                                     |

`Country`, `ReturnRate`, and `ReturnValueRate` are deliberately **excluded**
from the modelling matrix and used as profiling variables only.

## Pipeline mechanics

1. `prcomp(center = TRUE, scale. = TRUE)` on the seven scaled features.
2. Retain the first 3 PCs (≥ 80% cumulative variance under the variance-
   threshold and Kaiser rules).
3. `kmeans(centers = 3, nstart = 50)` on the PC scores; `set.seed(1)`.
4. Re-label clusters as `A` / `B` / `C` by descending median NetMonetary
   so labels are stable across re-fits.
5. Persist the entire pipeline — center, scale, rotation, centroids, label
   map, personas — as `segmentation_pipeline.rds`.

## Validation diagnostics (representative)

| Metric                                      | Value          | Threshold / interpretation                |
|---------------------------------------------|----------------|-------------------------------------------|
| Mean silhouette (k = 3)                     | ~0.33          | moderate separation                       |
| Bootstrap Jaccard (k-means, 50 reps)        | ~0.88, 0.93, 0.92 | all > 0.75 → all clusters stable      |
| Bootstrap Jaccard (Ward, 50 reps)           | ~0.81, 0.59, 0.67 | one borderline cluster                |
| Adjusted Rand (k-means vs Ward)             | ~0.5–0.7       | partial agreement                         |
| Adjusted Rand (raw vs scaled PC scores)     | check at run   | > 0.7 → segmentation robust to scaling    |

Values are illustrative; the pipeline's `diagnostics` slot carries the
exact numbers from the most recent run.

## Limitations

* **Gross vs. net monetary.** The pipeline defaults to `NetMonetary`. If
  cancellations are mis-coded in raw data, `NetMonetary` is unreliable and
  the operator should switch `monetary_source` in `config.yml` and re-fit.
* **High-value tail.** A small number of customers in the top 1% of spend
  carry disproportionate weight in cluster centroids, particularly cluster
  A. The `SinglePurchase` flag and `ReturnValueAbs` profile help diagnose
  whether their inclusion is appropriate.
* **One-year window.** Recency and tenure are bounded by the observation
  window; segments that look "recent" may be customers who have not had
  time to lapse.
* **Geographic skew.** UK ≈ 83% of revenue. Country was intentionally
  excluded; conclusions about non-UK customers are based on too little
  data to be reliable per-segment.
* **Descriptive, not causal.** Cluster membership is a label assigned
  unsupervised. It does not predict future behaviour without further
  modelling.

## Ethical and operational notes

* Cluster labels are not customer outcomes. Acting on them — e.g.
  withholding marketing from cluster C — should be evaluated with respect
  to the operator's business objectives and any applicable regulations.
* The pipeline does not use protected demographic features.
* Personas in the pipeline object are short text strings intended for
  internal use; they are not external-facing copy.

## Reproducing this card

```bash
make all          # data + model + reports + tests
```

The `segmentation_pipeline.rds` written by `scripts/run_pipeline.R` carries
the version, timestamp, full diagnostic numbers, and free-text notes
inside the object so this card can be regenerated programmatically if
needed.
