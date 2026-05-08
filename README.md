# UCI Online Retail — Customer Segmentation Pipeline

End-to-end unsupervised customer segmentation on the UCI Online Retail
dataset (id 352): UK gift retailer, Dec 2010 – Dec 2011, ~4,300 customers.

The repository is structured as a small reproducible pipeline rather than a
collection of notebooks. **One command (`./setup.sh`) takes a clean checkout
to fully reproduced data, fitted models, knitted reports, and passing
tests.**

## Quick start

```bash
./setup.sh
```

functions:

1. Detect your OS (Linux / macOS / Git Bash on Windows).
2. Create a Python venv at `./.venv` (skipped if it already exists).
3. Install Python dependencies from `requirements.txt`.
4. Install only the R packages that aren't already present.
5. Sanity-check `make`, `pandoc`, and a LaTeX engine.
6. Run `make all`.

Useful flags:

```bash
./setup.sh --no-build   # env setup only, skip the build
./setup.sh --no-r       # skip R package install (Python-only setups)
./setup.sh --help
```

If you'd rather drive the build yourself, install the deps once and use the
Makefile directly:

```bash
pip install -r requirements.txt
R -e 'install.packages(c("tidyverse","scales","yaml","here","cluster",
  "factoextra","fpc","mclust","dendextend","rpart","rpart.plot",
  "corrplot","GGally","gridExtra","testthat","knitr","rmarkdown"))'
make all
```

`make all` runs four sub-targets:

| Target          | Does                                                                |
|-----------------|---------------------------------------------------------------------|
| `make data`     | Fetches UCI data, cleans, builds customer features (Python).        |
| `make model`    | PCA + k-means + diagnostics + saves `segmentation_pipeline.rds`.    |
| `make reports`  | Knits every `.Rmd` in `reports/` to HTML/PDF.                       |
| `make test`     | Runs all `testthat` tests in `tests/`.                              |

## Repository layout

```
uci-retail-segmentation/
├── README.md
├── setup.sh                   # one-command bootstrap (idempotent)
├── Makefile
├── config.yml                 # single source of truth for all paths / params
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/                   # raw UCI fetch (cache lands here)
│   ├── interim/               # online_retail_clean.csv, ..._cancelled.csv
│   ├── processed/             # customer_features.csv, PCA scores, clusters
│   └── outputs/               # data_quality_summary.txt, scoring outputs
│
├── src/
│   ├── python/
│   │   └── data_prep.py       # fetch, clean, build customer feature table
│   └── R/
│       ├── 00_setup.R         # package loads + global seed
│       ├── 01_validate_data.R # data-integrity checks
│       ├── 02_features.R      # load + validate the customer feature table
│       ├── 03_pca.R           # fit_pca()
│       ├── 04_clustering.R    # kmeans_diagnostics(), fit_kmeans(), bootstrap, ARI
│       ├── 05_profiles.R      # cluster_headline(), feature_profile(), top products
│       ├── 06_score_new_customers.R  # score_customers() against persisted pipeline
│       └── utils.R            # config loader, path helpers, log_info, with_seed
│
├── scripts/
│   ├── run_pipeline.R         # orchestrates 02 -> 05, saves segmentation_pipeline.rds
│   └── render_reports.R       # knits every .Rmd in reports/
│
├── reports/
│   ├── 01_EDA.Rmd
│   ├── 02_PCA.Rmd
│   ├── 03_clustering.Rmd
│   ├── 04_profiles.Rmd
│   └── final_report.Rmd
│
├── models/
│   ├── pca_model.rds
│   ├── kmeans_model.rds
│   ├── segmentation_pipeline.rds   # the deployable artefact
│   └── model_card.md               # methodology + limitations
│
├── tests/
│   ├── test_data_integrity.R
│   ├── test_pca_outputs.R
│   ├── test_cluster_outputs.R
│   └── test_scoring.R
│
└── figures/                   # generated PNG/PDF figures
```

## Pipeline summary

1. **Cleaning.** Drop missing-CustomerID rows, drop duplicates, drop
   non-product StockCodes, separate cancellations from positive sales.
2. **Customer features.** R, F, M (gross + net), TotalQuantity,
   UniqueProducts, AvgOrderValue, quantity-weighted AvgUnitPrice
   (winsorised at p99), TenureDays, SinglePurchase flag, ReturnRate /
   ReturnValueRate. Right-skewed features are `log1p`-transformed.
3. **PCA** on the seven-feature scaled matrix; first 3 PCs retained
   (~80% variance).
4. **K-means** on PC scores with `k = 3`, `nstart = 50`, fixed seed.
   `k = 2` carried as a sensitivity model.
5. **Validation.** Bootstrap Jaccard stability (Hennig 2007), adjusted
   Rand index against Ward hierarchical, scaling-sensitivity ARI between
   raw and scaled PC scores.
6. **Profiles.** Headline revenue/customer share, median + mean feature
   profiles, top products with and without StockCode 23843 anomaly.
7. **Persisted pipeline.** `segmentation_pipeline.rds` carries the
   prcomp object, k-means centroids, label map, personas, version,
   timestamp, and full diagnostics — so new customers can be scored
   without rerunning anything.

## Mathematical specification (short form)

Standardise the modelling matrix `X ∈ R^{n × p}` column-wise to obtain
`Z`. Compute correlation `S = (1/(n−1)) Z^T Z`. PCA returns eigenvectors
`v_1, …, v_p` with eigenvalues `λ_1 ≥ … ≥ λ_p`. PC scores
`PC_{ik} = z_i^T v_k`; proportion of variance `PVE_k = λ_k / Σ λ_m`. We
keep the smallest `K` such that `Σ_{k≤K} PVE_k ≥ 0.80`.

K-means minimises `Σ_k Σ_{i∈C_k} ||y_i − μ_k||²` with
`μ_k = (1/|C_k|) Σ_{i∈C_k} y_i`, where `y_i` is the PC-score vector for
customer `i`. Choice of `K = 3` defended by elbow, mean silhouette,
gap-statistic SE-max rule, and bootstrap Jaccard ≥ 0.75 across all three
clusters.

Scoring a new customer with feature vector `x_new`:
`z_new = (x_new − μ_train) / σ_train`,
`y_new = z_new V_{1:K}`,
`cluster(y_new) = argmin_k ||y_new − μ_k||²`.

## Reproducibility

* `./setup.sh` is the canonical entry point. Idempotent.
* Fixed seed (`config.yml: seed: 1`) is applied before every random step.
* Each module asserts data-shape invariants on entry; tests in `tests/`
  re-check them after the pipeline writes outputs.
* `data/raw/`, `data/interim/`, `data/processed/`, `data/outputs/`,
  `models/*.rds`, and knitted `reports/` are ignored by git so the only
  thing that ships is the source.

## Caveats

The segmentation is descriptive, not causal. It depends on choices
documented in `models/model_card.md`: monetary source (net vs gross),
winsorisation, log transforms, PC scaling, and `k`. Re-fitting under a
different choice may move customers between segments — that's the
intended use of the sensitivity diagnostics built into the pipeline.

## License

MIT (see `LICENSE`).
