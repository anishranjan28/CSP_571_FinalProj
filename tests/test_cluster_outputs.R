# tests/test_cluster_outputs.R

library(testthat)
source(here::here("src", "R", "00_setup.R"))


test_that("customer_clusters.csv exists and has expected structure", {
  skip_if(!file.exists(abs_path(CONFIG$data$customer_clusters)),
          "customer_clusters.csv not present (run pipeline first)")
  cl <- readr::read_csv(abs_path(CONFIG$data$customer_clusters),
                        show_col_types = FALSE)

  expect_true(all(c("CustomerID", "cluster", "raw_cluster",
                    "cluster_source", "k") %in% names(cl)))
  expect_false(anyDuplicated(cl$CustomerID) > 0)
  expect_false(any(is.na(cl$cluster)))
  expect_equal(unique(cl$k), CONFIG$clustering$k_star)
  expect_equal(length(unique(cl$cluster)), CONFIG$clustering$k_star)
})


test_that("kmeans_model.rds is a kmeans object with the configured K", {
  skip_if(!file.exists(abs_path(CONFIG$models$kmeans_model)),
          "kmeans_model.rds not present")
  km <- readRDS(abs_path(CONFIG$models$kmeans_model))
  expect_s3_class(km, "kmeans")
  expect_equal(nrow(km$centers), CONFIG$clustering$k_star)
})
