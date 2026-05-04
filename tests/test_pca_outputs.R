# tests/test_pca_outputs.R

library(testthat)
source(here::here("src", "R", "00_setup.R"))
source(here::here("src", "R", "02_features.R"))


test_that("pca_model.rds exists and is a prcomp object", {
  skip_if(!file.exists(abs_path(CONFIG$models$pca_model)),
          "pca_model.rds not present (run pipeline first)")
  pr <- readRDS(abs_path(CONFIG$models$pca_model))
  expect_s3_class(pr, "prcomp")
  expect_true(length(pr$sdev) >= CONFIG$pca$n_components)
})


test_that("customer_pc_scores.csv has expected shape", {
  skip_if(!file.exists(abs_path(CONFIG$data$customer_pc_scores)),
          "customer_pc_scores.csv not present")
  pc <- readr::read_csv(abs_path(CONFIG$data$customer_pc_scores),
                        show_col_types = FALSE)
  expect_true("CustomerID" %in% names(pc))
  expect_true(!anyDuplicated(pc$CustomerID))
  expect_true(all(complete.cases(pc)))
  for (k in seq_len(CONFIG$pca$n_components)) {
    expect_true(paste0("PC", k) %in% names(pc))
  }
})


test_that("PC scores row count equals customer feature row count", {
  skip_if(!file.exists(abs_path(CONFIG$data$customer_pc_scores)) ||
          !file.exists(abs_path(CONFIG$data$customer_features)),
          "required CSVs not present")
  pc       <- readr::read_csv(abs_path(CONFIG$data$customer_pc_scores),
                              show_col_types = FALSE)
  customer <- readr::read_csv(abs_path(CONFIG$data$customer_features),
                              show_col_types = FALSE)
  expect_equal(nrow(pc), nrow(customer))
  expect_setequal(pc$CustomerID, customer$CustomerID)
})
