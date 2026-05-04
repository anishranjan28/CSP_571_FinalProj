# tests/test_data_integrity.R
#
# Light-weight integrity checks on the cleaned data and engineered features.
# Run with `make test` or `Rscript -e 'testthat::test_dir("tests")'`.

library(testthat)
# Use here::here() so paths resolve to project root regardless of the
# working directory testthat::test_dir() chooses. The .here marker at
# the project root anchors this.
source(here::here("src", "R", "00_setup.R"))
source(here::here("src", "R", "02_features.R"))


test_that("transactions table loads and validates", {
  skip_if(!file.exists(abs_path(CONFIG$data$online_retail_clean)),
          "transactions CSV not present")
  tx <- load_transactions()
  expect_true(nrow(tx) > 0)
  expect_true("CustomerID" %in% names(tx))
  expect_true(all(tx$Quantity  > 0))
  expect_true(all(tx$UnitPrice > 0))
})


test_that("cancellations table loads and validates", {
  skip_if(!file.exists(abs_path(CONFIG$data$online_retail_cancelled)),
          "cancellations CSV not present")
  cancelled <- load_cancellations()
  expect_true(nrow(cancelled) >= 0)
  expect_true(all(grepl("^C", cancelled$InvoiceNo)))
})


test_that("customer feature table validates", {
  skip_if(!file.exists(abs_path(CONFIG$data$customer_features)),
          "customer_features CSV not present")
  customer <- load_customer_features()
  expect_true(nrow(customer) > 0)
  expect_true(!anyDuplicated(customer$CustomerID))
  expect_true(all(customer$Frequency >= 1))
  expect_true(all(customer$Recency  >= 0))
  expect_true(all(is.finite(customer$GrossMonetary)))
  expect_true(all(is.finite(customer$NetMonetary)))
  expect_true(all(CONFIG$features$feature_cols %in% names(customer)))
})


test_that("modelling matrix has no NAs and matches customer count", {
  skip_if(!file.exists(abs_path(CONFIG$data$customer_features)),
          "customer_features CSV not present")
  customer <- load_customer_features()
  mm <- build_modelling_matrix(customer)
  expect_equal(nrow(mm$X), nrow(customer))
  expect_true(all(complete.cases(mm$X)))
  expect_setequal(colnames(mm$X), CONFIG$features$feature_cols)
})
