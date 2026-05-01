# scripts/run_pipeline.R — orchestrate the modelling pipeline.
#
# Assumes `python src/python/data_prep.py` has already produced the cleaned
# data files. Loads features, fits PCA, fits k-means, builds a stable
# cluster label mapping, computes profiles, and persists every artefact —
# including a self-contained `segmentation_pipeline.rds` that can score
# new customers without re-running anything else.

source(file.path("src", "R", "00_setup.R"))
source(file.path("src", "R", "02_features.R"))
source(file.path("src", "R", "03_pca.R"))
source(file.path("src", "R", "04_clustering.R"))
source(file.path("src", "R", "05_profiles.R"))


# ---------------------------------------------------------------------------
# 1. Load + validate
# ---------------------------------------------------------------------------
log_info("Loading customer features ...")
customer <- load_customer_features()
log_info(sprintf("Loaded %d customers, %d columns.",
                 nrow(customer), ncol(customer)))


# ---------------------------------------------------------------------------
# 2. PCA
# ---------------------------------------------------------------------------
log_info("Fitting PCA ...")
pca <- fit_pca(customer)


# ---------------------------------------------------------------------------
# 3. K-means + diagnostics + stability
# ---------------------------------------------------------------------------
pc <- load_pc_scores(n_components = pca$n_components)
Xpc <- pc |> dplyr::select(dplyr::starts_with("PC")) |> as.matrix()
rownames(Xpc) <- pc$CustomerID

log_info("Computing k-means diagnostics ...")
diag <- kmeans_diagnostics(Xpc)
log_info(sprintf("Mean silhouette at k=%d: %.3f",
                 CONFIG$clustering$k_star,
                 diag$silhouette$silhouette[
                   diag$silhouette$k == CONFIG$clustering$k_star]))

log_info("Fitting final k-means + sensitivity (k = 2) ...")
km <- fit_kmeans(Xpc)

log_info("Bootstrap stability ...")
boot <- bootstrap_stability(Xpc)
log_info(sprintf("Bootstrap Jaccard (k-means): %s",
                 paste(sprintf("%.2f", boot$kmeans$bootmean), collapse = ", ")))

log_info("Scaling sensitivity ...")
scale_sens <- scale_sensitivity(Xpc)
log_info(sprintf("Adjusted Rand (raw vs scaled PCs): %.3f", scale_sens$ari))

log_info("Hierarchical clustering (Ward / complete / average) ...")
hc <- fit_hclust(diag$distance)
hc_labels <- cutree(hc$ward, k = CONFIG$clustering$k_star)
ari_km_hc <- ari(km$main$cluster, hc_labels)
log_info(sprintf("Adjusted Rand (k-means vs Ward): %.3f", ari_km_hc))


# ---------------------------------------------------------------------------
# 4. Stable cluster labels (A > B > C by median NetMonetary)
# ---------------------------------------------------------------------------
label_map <- build_label_map(km$main$cluster, pc$CustomerID, customer)
clusters <- save_clusters(pc$CustomerID, km$main$cluster, label_map)
log_info("Saved customer_clusters.csv with stable A/B/C labels.")


# ---------------------------------------------------------------------------
# 5. Profiles + labelled customer table
# ---------------------------------------------------------------------------
cust <- customer |>
  dplyr::inner_join(clusters |> dplyr::select(CustomerID, cluster),
                    by = "CustomerID")

headline <- cluster_headline(cust)
profile  <- feature_profile(cust)
labelled <- save_labeled_customers(customer, clusters)
log_info("Saved customer_features_labeled.csv.")


# ---------------------------------------------------------------------------
# 6. Personas (drawn from feature_profile())
# ---------------------------------------------------------------------------
personas <- list(
  A = "Loyal high-value / core wholesale-leaning customers. Recent, frequent, broad product variety, long tenure. Largest share of net revenue.",
  B = "Developing mid-value repeat customers. Moderate spend and frequency. Retention / upsell target.",
  C = "Lapsed one-time / low-value customers. Single purchase predominant; long recency. Win-back or accept as low-margin."
)


# ---------------------------------------------------------------------------
# 7. Persist segmentation pipeline (the deployable artefact)
# ---------------------------------------------------------------------------
segmentation_pipeline <- list(
  version       = CONFIG$project$version,
  created_at    = Sys.time(),
  feature_cols  = pca$feature_cols,
  preprocessing = list(
    log_transform   = TRUE,
    pca_center      = CONFIG$pca$center,
    pca_scale       = CONFIG$pca$scale,
    avg_unit_price  = "quantity-weighted, winsorised at p99",
    monetary_source = CONFIG$features$monetary_source
  ),
  pca_model         = pca$pca_model,
  n_components      = pca$n_components,
  kmeans_model      = km$main,
  cluster_label_map = label_map,
  personas          = personas,
  training_rows     = nrow(customer),
  diagnostics = list(
    silhouette_k_star = diag$silhouette$silhouette[
      diag$silhouette$k == CONFIG$clustering$k_star],
    bootstrap_jaccard_kmeans = boot$kmeans$bootmean,
    bootstrap_jaccard_ward   = boot$hclust$bootmean,
    ari_kmeans_vs_ward       = ari_km_hc,
    ari_raw_vs_scaled_pc     = scale_sens$ari
  ),
  notes = c(
    "Trained on UCI Online Retail (id 352), Dec 2010 - Dec 2011.",
    "Cluster labels are descriptive groupings, not causal categories.",
    "Sensitive to monetary-source choice (net vs gross), winsorisation,",
    "and PC scaling. See model_card.md for full caveats."
  )
)

dir.create(dirname(abs_path(CONFIG$models$segmentation_pipeline)),
           recursive = TRUE, showWarnings = FALSE)
saveRDS(segmentation_pipeline, abs_path(CONFIG$models$segmentation_pipeline))
saveRDS(km$main, abs_path(CONFIG$models$kmeans_model))
log_info(sprintf("Saved segmentation_pipeline.rds (version %s).",
                 CONFIG$project$version))


# ---------------------------------------------------------------------------
# 8. Console summary
# ---------------------------------------------------------------------------
cat("\n=========== PIPELINE SUMMARY ===========\n")
print(headline)
cat("\nBootstrap Jaccard (k-means):  ", round(boot$kmeans$bootmean, 3), "\n")
cat("Bootstrap Jaccard (Ward):     ", round(boot$hclust$bootmean, 3), "\n")
cat("ARI k-means vs Ward:          ", round(ari_km_hc, 3), "\n")
cat("ARI raw vs scaled PC scores:  ", round(scale_sens$ari, 3), "\n")
cat("========================================\n")
