# 03_pca.R — fit and persist PCA on the customer feature matrix.
#
# Mathematical specification:
#   Z      = standardised feature matrix (zero mean, unit variance per column)
#   S      = (1 / (n - 1)) Z^T Z
#   v_k    = k-th eigenvector of S, eigenvalues lambda_1 >= ... >= lambda_p
#   PC_ik  = z_i^T v_k                                    (score for customer i)
#   PVE_k  = lambda_k / sum_m lambda_m                    (proportion variance)
#
# We retain the first n_components PCs (default 3, configured in config.yml).

source(here::here("src", "R", "00_setup.R"))
source(here::here("src", "R", "02_features.R"))


#' Fit PCA, save the model + score CSV, return summary metadata.
fit_pca <- function(customer = NULL,
                    feature_cols = CONFIG$features$feature_cols,
                    n_components = CONFIG$pca$n_components,
                    pca_model_path = abs_path(CONFIG$models$pca_model),
                    scores_path = abs_path(CONFIG$data$customer_pc_scores),
                    seed = GLOBAL_SEED) {

  if (is.null(customer)) customer <- load_customer_features()
  mm <- build_modelling_matrix(customer, feature_cols)

  with_seed(seed, {
    pr.out <- prcomp(mm$X,
                     center = CONFIG$pca$center,
                     scale. = CONFIG$pca$scale)
  })

  pve <- pr.out$sdev^2 / sum(pr.out$sdev^2)
  cum_pve <- cumsum(pve)

  # Persist model object (for scoring new customers later).
  dir.create(dirname(pca_model_path), recursive = TRUE, showWarnings = FALSE)
  saveRDS(pr.out, pca_model_path)

  # Save first n_components PC scores keyed by CustomerID.
  k_keep <- max(n_components,
                which(cum_pve >= CONFIG$pca$variance_target)[1])
  k_keep <- max(k_keep, n_components, na.rm = TRUE)

  scores_df <- as.data.frame(pr.out$x[, seq_len(k_keep), drop = FALSE])
  names(scores_df) <- paste0("PC", seq_len(k_keep))
  scores_df <- cbind(CustomerID = mm$customer_id, scores_df)

  validate_pc_scores(scores_df, n_components = k_keep)
  dir.create(dirname(scores_path), recursive = TRUE, showWarnings = FALSE)
  readr::write_csv(scores_df, scores_path)

  log_info(sprintf("PCA: %d components retained (%.1f%% variance).",
                   k_keep, 100 * cum_pve[k_keep]))

  invisible(list(
    pca_model     = pr.out,
    n_components  = k_keep,
    pve           = pve,
    cum_pve       = cum_pve,
    feature_cols  = feature_cols,
    customer_id   = mm$customer_id
  ))
}
