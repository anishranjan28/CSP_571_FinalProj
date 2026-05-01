# Online Retail Customer Segmentation — Final Report

**Author:** Anish Ranjan
**Dataset:** UCI Online Retail (id 352) — UK gift retailer, Dec 2010 – Dec 2011
**Methods:** Unsupervised — PCA, k-means, hierarchical clustering (ISLR Ch. 12)

---

## Executive summary

After cleaning, ~391,150 positive-value transactions across ~4,334 customers were used to build a customer-level feature table covering Recency, Frequency, gross and net Monetary value, basket and product-diversity metrics, tenure, and return behaviour. PCA reduced seven correlated features to three orthogonal components capturing ~80% of the variance, and k-means clustering on those PC scores produced a stable three-segment solution validated by bootstrap resampling.

The three segments are interpretable along an engagement / customer-value axis:

| Segment | Share of customers | Share of net revenue | Profile |
|---|---|---|---|
| **A — Loyal high-value** | ~28% | ~75% | Recent, frequent, high-spend, broad product variety |
| **B — Developing mid-value** | ~38% | ~21% | Moderate spend, occasional repeat, mid-tenure |
| **C — Lapsed one-time** | ~34% | ~4% | Single-purchase, long recency, low spend |

Roughly three-quarters of net revenue comes from a quarter of the customer base. The segmentation is descriptive, not causal; it should drive marketing prioritisation rather than be treated as a predictive model.

---

## 1. Data and pre-processing

The raw UCI download required substantial cleaning before any modelling could begin. Rows missing `CustomerID` were dropped; duplicate rows removed; non-product `StockCode` entries (postage, manual adjustments, bank charges) excluded. Cancellation invoices — those whose `InvoiceNo` starts with "C" — were *not* discarded, as earlier iterations had done. They were kept in a separate table so that net-of-cancellations revenue could be computed alongside the gross figure used in standard RFM literature.

For each customer the pipeline computed Recency in days since the last positive purchase, Frequency as the count of distinct positive invoices, GrossMonetary as the sum of positive line amounts, and NetMonetary as GrossMonetary minus the absolute value of cancellations. Behavioural extras included TotalQuantity, UniqueProducts, AvgOrderValue, AvgOrderItems, TenureDays, and a `SinglePurchase` flag for customers with Frequency = 1 — a deliberate addition because TenureDays = 0 does not mean "new customer," it usually means "single-purchase customer." A quantity-weighted AvgUnitPrice (the price-per-item the customer actually paid on average) was computed and winsorised at the 99th percentile to prevent a small number of extreme entries from dominating PCA loadings on PC2 and PC3. Two return-rate variants were produced: a count-based `ReturnRate` and a value-based `ReturnValueRate`. All right-skewed features were also `log1p`-transformed to symmetrise their distributions before the variance-based methods downstream.

## 2. Exploratory findings

Three findings shaped the modelling choices.

First, the raw monetary distribution is brutally right-skewed. The top ~1% of customers control roughly a quarter of revenue and the top 5% control around half — a finding documented as the "top-customer revenue concentration curve" rather than as a Lorenz curve, since a true Lorenz curve sorts customers ascending. Without log transformation, distance-based clustering would treat the wholesale tail as a separate cluster of size 1–3 and crowd everyone else into a single blob. The log transform reduces but does not eliminate this skew; the upper shoulder of `log_NetMonetary` is real wholesale-leaning customers and is addressed via sensitivity analysis rather than by removing them.

Second, collinearity between features is severe. `Monetary`, `TotalQuantity`, and `Frequency` form a tight block under both Pearson and Spearman correlations. `AvgOrderValue` and `AvgOrderItems` carry essentially the same information; only `AvgOrderValue` was kept. (`AvgBasketValue`, which appeared in earlier iterations as a near-duplicate, was removed entirely.) This collinearity is what motivates PCA as a preprocessing step.

Third, the dataset is geographically imbalanced — the UK is roughly 83% of revenue. Including `Country` as a clustering input would force a near-binary "UK vs. everyone else" split that would drown out the behavioural signal we care about. `Country` was therefore excluded from clustering and reintroduced as a profiling variable.

A separate finding worth flagging is the dominance of `StockCode 23843` ("PAPER CRAFT, LITTLE BIRDIE") in product-level revenue, driven almost entirely by a single customer's single bulk order. Any product-level interpretation that includes this code is contaminated by that one transaction; the report's product-level claims are quoted from the without-anomaly version of the per-cluster product table.

## 3. Dimensionality reduction

PCA was fit on seven scaled features (`log_Recency`, `log_Frequency`, `log_NetMonetary`, `log_UniqueProducts`, `log_AvgOrderValue`, `AvgUnitPrice_winsor`, `TenureDays`). The first three components explain roughly 80% of the variance and are interpretable as follows.

**PC1 — engagement / customer value.** Loads positively on `log_Frequency`, `log_NetMonetary`, `log_UniqueProducts`, and `TenureDays`, and negatively on `log_Recency`. High PC1 corresponds to recent, frequent, broad-product, repeat-relationship customers; low PC1 corresponds to lapsed one-time buyers.

**PC2 — order-size vs. tenure contrast.** Loads negatively on `log_AvgOrderValue` and `AvgUnitPrice_winsor`, and positively on `TenureDays` and `log_Frequency`. Low PC2 corresponds to wholesale-leaning behaviour — fewer, larger orders of higher-priced items. High PC2 corresponds to smaller, more frequent transactions over a longer relationship. PC2 is *not* purely a "tenure / lifecycle" axis; that description omits the AvgOrderValue contrast that is in fact the more informative half of the loading.

**PC3 — residual price-point.** Once engagement and order-size are absorbed, PC3 picks up the remaining `AvgUnitPrice` signal — what tier of price the customer typically buys at.

PCA component signs are arbitrary; the substantive content is the contrast across features, not the absolute sign.

## 4. Clustering

K-means was the primary algorithm; hierarchical clustering with Ward's linkage served as a robustness check. Three diagnostics were used to choose `k`: WCSS elbow, silhouette width, and the gap statistic with Tibshirani's SEmax rule.

The diagnostics agreed broadly. Silhouette peaked at `k = 2` and remained competitive at `k = 3`; the gap statistic with the SEmax criterion selected `k = 3`; the elbow was gradual but flattened around `k = 4`. **`k = 3` was chosen** as the final segmentation because it (a) is statistically defensible under the most rigorous of the three criteria, (b) produces interpretable high / mid / low value segments rather than the degenerate two-way split that `k = 2` produces, and (c) yields a persona-level profile in the next stage that is materially more actionable. The earlier framing of a "substantive prior of 4–6 clusters" is *not* what these diagnostics support; the project's working hypothesis was updated accordingly. The `k = 2` solution is retained as a sensitivity check.

K-means with `k = 3` was bootstrap-resampled fifty times. Per-cluster Jaccard similarities exceeded 0.87 for all three clusters — well above Hennig's (2007) "stable" threshold of 0.75. Under the same procedure, Ward hierarchical clustering produced one stable cluster (Jaccard around 0.81) and two borderline ones (around 0.59 and 0.67). The cross-tab between k-means and Ward labels at `k = 3` is *not* cleanly block-diagonal; the adjusted Rand index quantifies their partial disagreement on a 0–1 scale. This is expected — k-means prefers spherical, equal-size clusters and Ward does not.

On this evidence the *stability* criterion favours k-means even though Ward's silhouette is marginally higher. Because the project goal is a stable customer segmentation, k-means is the appropriate final label source; Ward is reported as a robustness benchmark.

A sensitivity check on whether to scale the PC scores before k-means produced an adjusted Rand index between the two solutions, quantifying robustness to that preprocessing choice.

## 5. The three segments

After re-labelling clusters in descending median-spend order:

**Cluster A — Loyal high-value / wholesale-leaning customers.** Approximately 1,200 customers (~28% of the base) accounting for ~75% of net revenue. Median net spend around £2,400, median 7 orders, median recency 14 days, high product variety, long tenure. This is the segment to protect; it should be monitored for return-value rate (a small subset of these customers cancel substantial values) and the wholesale-anomaly product (`StockCode 23843`) which contaminates product-level rankings if not excluded.

**Cluster B — Developing mid-value repeat customers.** Approximately 1,640 customers (~38%) accounting for ~21% of net revenue. Median net spend around £760, median 2 orders, median recency 60 days, moderate product variety and tenure. The natural target for retention and upsell campaigns aimed at promoting these customers into Cluster A.

**Cluster C — Lapsed one-time / low-value customers.** Approximately 1,490 customers (~34%) accounting for ~4% of net revenue. Median net spend around £220, median 1 order, median recency 152 days. Critically, `TenureDays = 0` for most members of this cluster does *not* indicate "new" customers — the `SinglePurchase` flag confirms these are predominantly customers who bought exactly once and have not returned. They should be described as **lapsed**, not new, and either targeted with win-back campaigns or accepted as a low-margin segment.

A short `rpart` decision tree fit on the log-features recovered the cluster labels with high in-sample accuracy from a handful of axis-aligned splits on Recency, Frequency, and NetMonetary, indicating that the segmentation is substantively simple — the segments are essentially RFM tiers, which is a defensible and interpretable result for a marketing audience. This decision tree is descriptive, not predictive: it is fit and scored on the same data, so its accuracy is not evidence of out-of-sample generalisation.

## 6. Limitations

The pipeline initially treated `Monetary` as gross positive sales without netting cancellations. This was corrected to compute both `GrossMonetary` and `NetMonetary` and to cluster on the latter; results in this report should be re-checked against the gross variant in any sensitivity analysis. Customers with very high gross revenue but matching cancellations — notably customer 12346 — shift segments under the net definition and may not be high-value at all once returns are accounted for.

The wholesale tail on PC1 is real and remains visible after log transformation. The clustering is robust to it under bootstrap resampling, but extreme customers individually carry large weight in any per-cluster mean — which is why §4 of the profiling notebook reports both means and medians.

`StockCode 23843` skews any product-level interpretation that includes it. Product profiles in this report are quoted from the without-anomaly version. This caveat should be stated in any presentation of cluster A's product preferences.

The dataset spans one year. Recency and tenure are bounded by that window; segments that look "recent" here may simply be customers who have not had time to lapse yet. A second period of data would be needed to validate the segmentation longitudinally.

Cluster membership is descriptive, not causal. It does not predict future behaviour without further modelling. The decision-tree approximation in the profiling notebook is in-sample only and should not be presented as evidence of generalisation.

## 7. Recommended next steps

Promote `NetMonetary` to the canonical revenue measure across all reports, with `GrossMonetary` retained only as a sensitivity benchmark.

Re-fit the PCA + k-means pipeline excluding the top 1% of customers by spend, and report whether the three-segment story survives. If it does, the wholesale tail is a within-cluster phenomenon rather than a separate segment in disguise.

Compare the k-means segmentation against richer clusterers — Gaussian mixture models (`mclust::Mclust`), which would give soft assignments and a principled BIC-based `k` selection, and PAM with a Gower distance, which would handle the categorical `Country` feature without forcing it into a Euclidean space. ISLR does not cover either, so they sit outside the current methodological scope but would strengthen any extension.

If a second period of transaction data becomes available, score the new transactions against the existing segments and check whether intra-segment behaviour remains coherent. That is the only true external validity check for an unsupervised segmentation.

Translate the segments into measurable actions. A retention experiment targeted at Cluster B with a defined upsell metric, and a win-back experiment for the most-recent slice of Cluster C, would convert the segmentation from a descriptive artefact into a decision tool whose value can be measured.
