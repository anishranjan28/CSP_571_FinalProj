# tests/test_scoring.R
#
# End-to-end smoke test for the scoring path: load the persisted
# segmentation pipeline, score the training customers through it, and
# check that the output shape and labels are consistent.

library(testthat)
source(here::here("src", "R", "00_setup.R"))
source(here::here("src", "R", "06_score_new_customers.R"))


test_that("segmentation_pipeline.rds loads and has required slots", {
  skip_if(!file.exists(abs_path(CONFIG$models$segmentation_pipeline)),
          "segmentation_pipeline.rds not present")
  pipe <- load_pipeline()
  expect_true(all(c("pca_model", "kmeans_model", "feature_cols",
                    "n_components", "cluster_label_map", "personas",
                    "version", "created_at") %in% names(pipe)))
  expect_s3_class(pipe$pca_model,    "prcomp")
  expect_s3_class(pipe$kmeans_model, "kmeans")
})


test_that("score_customers() returns the expected shape", {
  skip_if(!file.exists(abs_path(CONFIG$models$segmentation_pipeline)) ||
          !file.exists(abs_path(CONFIG$data$customer_features)),
          "required artefacts missing")

  pipe     <- load_pipeline()
  customer <- readr::read_csv(abs_path(CONFIG$data$customer_features),
                              show_col_types = FALSE)
  scored   <- score_customers(customer, pipe)

  expect_equal(nrow(scored), nrow(customer))
  expect_true(all(c("CustomerID", "raw_cluster", "cluster", "persona") %in%
                  names(scored)))
  expect_setequal(scored$CustomerID, customer$CustomerID)
  expect_false(any(is.na(scored$cluster)))
  expect_setequal(levels(scored$cluster),
                  pipe$cluster_label_map$label)
})


test_that("scoring agrees with training labels for in-sample customers", {
  # Sanity check: applying predict() to the training set should reproduce
  # the assignments saved to customer_clusters.csv (modulo nearest-centroid
  # tie-breaking, which should be vanishingly rare here).
  skip_if(!file.exists(abs_path(CONFIG$models$segmentation_pipeline)) ||
          !file.exists(abs_path(CONFIG$data$customer_clusters))   ||
          !file.exists(abs_path(CONFIG$data$customer_features)),
          "required artefacts missing")

  pipe     <- load_pipeline()
  customer <- readr::read_csv(abs_path(CONFIG$data$customer_features),
                              show_col_types = FALSE)
  clusters <- readr::read_csv(abs_path(CONFIG$data$customer_clusters),
                              show_col_types = FALSE)

  scored <- score_customers(customer, pipe) |>
    dplyr::select(CustomerID, scored_cluster = cluster)
  joined <- dplyr::inner_join(clusters, scored, by = "CustomerID")
  agreement <- mean(as.character(joined$cluster) ==
                    as.character(joined$scored_cluster))
  expect_gt(agreement, 0.99)
})
