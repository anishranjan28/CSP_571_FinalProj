# 06_score_new_customers.R — score new customers against the trained
# segmentation pipeline.
#
# Mathematical specification:
#   For a new customer with feature vector x_new:
#     z_new = (x_new - mu_train) / sd_train          (apply training scaling)
#     y_new = z_new V[, 1:n_components]              (project to PC space)
#     k_new = argmin_k || y_new - mu_k ||^2          (nearest training centroid)
#
# The trained `segmentation_pipeline.rds` carries everything needed for
# this — feature columns, prcomp object (with center/scale vectors and
# rotation matrix), kmeans centroids, and the raw->letter label map.

source(here::here("src", "R", "00_setup.R"))


#' Load the persisted segmentation pipeline.
load_pipeline <- function(path = abs_path(CONFIG$models$segmentation_pipeline)) {
  require_file(path, "segmentation_pipeline.rds")
  readRDS(path)
}


#' Validate that a new customer feature table has the columns we need.
validate_new_customers <- function(new_customers, feature_cols) {
  require_columns(new_customers,
                  c("CustomerID", feature_cols),
                  label = "new_customers")
  require_unique(new_customers, "CustomerID", label = "new_customers")
  na_in_features <- colSums(is.na(new_customers[, feature_cols, drop = FALSE]))
  if (any(na_in_features > 0)) {
    stop("Missing values in new-customer features:\n",
         paste(names(na_in_features[na_in_features > 0]),
               na_in_features[na_in_features > 0],
               sep = " = ", collapse = "\n"))
  }
  invisible(new_customers)
}


#' Assign each row of a PC matrix to its nearest k-means centroid.
nearest_centroid <- function(Y_new, centers) {
  # squared Euclidean distance from each point to each centroid
  d2 <- apply(centers, 1, function(mu)
              rowSums((Y_new - matrix(mu, nrow = nrow(Y_new),
                                      ncol = length(mu), byrow = TRUE))^2))
  if (is.null(dim(d2))) d2 <- matrix(d2, ncol = nrow(centers))
  apply(d2, 1, which.min)
}


#' Score a new customer feature table.
#'
#' @param new_customers data.frame with CustomerID + feature columns
#' @param pipeline      output of load_pipeline()
#' @return tibble with CustomerID, PC1..PCk, raw_cluster, cluster (letter), persona
score_customers <- function(new_customers, pipeline = NULL) {
  if (is.null(pipeline)) pipeline <- load_pipeline()

  validate_new_customers(new_customers, pipeline$feature_cols)

  X_new <- as.matrix(new_customers[, pipeline$feature_cols, drop = FALSE])

  # prcomp's predict() applies the saved center/scale + rotation.
  Y_new_full <- predict(pipeline$pca_model, newdata = X_new)
  Y_new <- Y_new_full[, seq_len(pipeline$n_components), drop = FALSE]

  raw_cluster <- nearest_centroid(Y_new, pipeline$kmeans_model$centers)

  raw_to_label <- setNames(pipeline$cluster_label_map$label,
                           as.character(pipeline$cluster_label_map$raw_cluster))
  letter_label <- raw_to_label[as.character(raw_cluster)]

  persona <- pipeline$personas[letter_label]

  out <- tibble::tibble(
    CustomerID  = new_customers$CustomerID,
    raw_cluster = raw_cluster,
    cluster     = factor(letter_label,
                         levels = pipeline$cluster_label_map$label),
    persona     = unname(persona)
  )

  pc_df <- as.data.frame(Y_new)
  names(pc_df) <- paste0("PC", seq_len(ncol(pc_df)))

  dplyr::bind_cols(out, pc_df)
}
