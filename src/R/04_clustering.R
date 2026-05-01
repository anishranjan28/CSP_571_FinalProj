# 04_clustering.R — k-means clustering on the PC scores, with Ward as a
# robustness check, bootstrap stability via Hennig (2007), and a
# scaling-sensitivity check via the adjusted Rand index.
#
# Mathematical specification:
#   y_i in R^p           PC score vector for customer i
#   minimise over C_1..C_K of   sum_k sum_{i in C_k} || y_i - mu_k ||^2
#                              where  mu_k = (1 / |C_k|) sum_{i in C_k} y_i
#
# Diagnostics for choosing K:
#   * total within-cluster sum of squares (elbow)
#   * mean silhouette width:    s(i) = (b(i) - a(i)) / max(a(i), b(i))
#   * Tibshirani (2001) gap statistic with the SE-max rule
#   * bootstrap Jaccard stability (Hennig 2007); >= 0.75 stable, < 0.6 noise
#
# K_star is configured in config.yml; default 3.

source(file.path("src", "R", "00_setup.R"))
source(file.path("src", "R", "02_features.R"))


#' Load PC scores produced by fit_pca().
load_pc_scores <- function(path = abs_path(CONFIG$data$customer_pc_scores),
                           n_components = CONFIG$pca$n_components) {
  require_file(path, "customer_pc_scores.csv")
  pc <- readr::read_csv(path, show_col_types = FALSE)
  validate_pc_scores(pc, n_components = n_components)
  pc
}


#' Sweep k-means across k_grid and return diagnostic tibbles.
kmeans_diagnostics <- function(Xpc, k_grid = CONFIG$clustering$k_grid,
                               nstart = CONFIG$clustering$nstart,
                               iter_max = CONFIG$clustering$iter_max,
                               seed = GLOBAL_SEED) {
  d <- dist(Xpc)

  with_seed(seed, {
    wcss <- vapply(k_grid, function(k) {
      kmeans(Xpc, centers = k, nstart = nstart, iter.max = iter_max)$tot.withinss
    }, numeric(1))

    sil <- vapply(k_grid, function(k) {
      cl <- kmeans(Xpc, centers = k, nstart = nstart, iter.max = iter_max)$cluster
      mean(cluster::silhouette(cl, d)[, "sil_width"])
    }, numeric(1))

    gap <- cluster::clusGap(Xpc, FUN = kmeans, nstart = nstart,
                            K.max = max(k_grid), B = 50)
  })

  list(
    elbow      = tibble::tibble(k = k_grid, wcss = wcss),
    silhouette = tibble::tibble(k = k_grid, silhouette = sil),
    gap        = gap,
    distance   = d
  )
}


#' Fit final k-means at k_star and a sensitivity model at k = 2.
fit_kmeans <- function(Xpc, k_star = CONFIG$clustering$k_star,
                       seed = GLOBAL_SEED) {
  with_seed(seed, {
    km_main <- kmeans(Xpc, centers = k_star,
                      nstart = CONFIG$clustering$nstart,
                      iter.max = CONFIG$clustering$iter_max)
    km_k2 <- kmeans(Xpc, centers = 2,
                    nstart = CONFIG$clustering$nstart,
                    iter.max = CONFIG$clustering$iter_max)
  })
  list(main = km_main, k2 = km_k2)
}


#' Hierarchical clustering at k_star with Ward, complete, and average linkage.
fit_hclust <- function(d, k_star = CONFIG$clustering$k_star) {
  list(
    ward     = hclust(d, method = "ward.D2"),
    complete = hclust(d, method = "complete"),
    average  = hclust(d, method = "average"),
    k_star   = k_star
  )
}


#' Bootstrap Jaccard stability (Hennig 2007) for k-means and Ward.
bootstrap_stability <- function(Xpc, k_star = CONFIG$clustering$k_star,
                                B = CONFIG$clustering$bootstrap_B,
                                seed = GLOBAL_SEED) {
  with_seed(seed, {
    boot_km <- fpc::clusterboot(
      Xpc, B = B, bootmethod = "boot",
      clustermethod = fpc::kmeansCBI,
      krange = k_star, seed = seed, count = FALSE
    )
    boot_hc <- fpc::clusterboot(
      Xpc, B = B, bootmethod = "boot",
      clustermethod = fpc::hclustCBI, method = "ward.D2",
      k = k_star, seed = seed, count = FALSE
    )
  })
  stopifnot(length(boot_km$bootmean) == k_star,
            length(boot_hc$bootmean) == k_star)
  list(kmeans = boot_km, hclust = boot_hc)
}


#' Adjusted Rand index between two cluster vectors (mclust::adjustedRandIndex).
ari <- function(cl_a, cl_b) {
  mclust::adjustedRandIndex(cl_a, cl_b)
}


#' Sensitivity: k-means on raw PC scores vs. scaled PC scores.
scale_sensitivity <- function(Xpc, k_star = CONFIG$clustering$k_star,
                              seed = GLOBAL_SEED) {
  with_seed(seed, {
    km_raw    <- kmeans(Xpc,        centers = k_star,
                        nstart = CONFIG$clustering$nstart,
                        iter.max = CONFIG$clustering$iter_max)
    km_scaled <- kmeans(scale(Xpc), centers = k_star,
                        nstart = CONFIG$clustering$nstart,
                        iter.max = CONFIG$clustering$iter_max)
  })
  list(km_raw = km_raw, km_scaled = km_scaled,
       ari = ari(km_raw$cluster, km_scaled$cluster))
}


#' Re-label cluster IDs as A > B > C ... by descending median NetMonetary.
#' Returns a tibble: raw_cluster -> stable letter label.
build_label_map <- function(km_cluster, customer_id, customer_features) {
  monetary <- customer_features |>
    dplyr::filter(CustomerID %in% customer_id) |>
    dplyr::arrange(match(CustomerID, customer_id)) |>
    dplyr::pull(NetMonetary)

  med_by_raw <- tibble::tibble(raw_cluster = km_cluster, value = monetary) |>
    dplyr::group_by(raw_cluster) |>
    dplyr::summarise(med = median(value), .groups = "drop") |>
    dplyr::arrange(dplyr::desc(med))

  med_by_raw$label <- LETTERS[seq_len(nrow(med_by_raw))]
  med_by_raw |> dplyr::select(raw_cluster, label)
}


#' Save the clusters CSV with both raw and re-labelled IDs.
save_clusters <- function(customer_id, km_cluster, label_map,
                          k_star = CONFIG$clustering$k_star,
                          source = CONFIG$clustering$final_label_source,
                          path = abs_path(CONFIG$data$customer_clusters)) {

  raw_to_label <- setNames(label_map$label, as.character(label_map$raw_cluster))
  letter_label <- raw_to_label[as.character(km_cluster)]

  out <- tibble::tibble(
    CustomerID     = customer_id,
    cluster        = factor(letter_label, levels = LETTERS[seq_len(k_star)]),
    raw_cluster    = km_cluster,
    cluster_source = source,
    k              = k_star
  )

  validate_clusters(out, k_star = k_star)
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  readr::write_csv(out, path)
  invisible(out)
}
