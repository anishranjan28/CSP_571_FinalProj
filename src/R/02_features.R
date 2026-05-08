# 02_features.R — load the customer feature table produced by Python.
#
# All heavy feature engineering happens in src/python/data_prep.py. This
# module is a thin loader + validator + helpers that downstream R steps
# can import without re-implementing schema knowledge.

source(here::here("src", "R", "00_setup.R"))
source(here::here("src", "R", "01_validate_data.R"))


#' Load and validate the customer-level feature table.
load_customer_features <- function(path = abs_path(CONFIG$data$customer_features)) {
  require_file(path, "customer_features.csv")
  customer <- readr::read_csv(path, show_col_types = FALSE)
  validate_customer_features(customer)
  customer
}


#' Build the modelling matrix X used by PCA + k-means.
#'
#' @return list(X = matrix, customer_id = integer vector)
build_modelling_matrix <- function(customer = NULL,
                                   feature_cols = CONFIG$features$feature_cols) {
  if (is.null(customer)) customer <- load_customer_features()

  X_df <- customer |>
    dplyr::select(CustomerID, dplyr::all_of(feature_cols)) |>
    tidyr::drop_na()

  if (nrow(X_df) != nrow(customer))
    stop("drop_na removed rows from modelling matrix; investigate NA features.")

  list(
    X           = as.matrix(X_df[, feature_cols]),
    customer_id = X_df$CustomerID,
    feature_cols = feature_cols
  )
}


#' Load the cleaned transaction table.
load_transactions <- function(path = abs_path(CONFIG$data$online_retail_clean)) {
  require_file(path, "online_retail_clean.csv")
  tx <- readr::read_csv(path, show_col_types = FALSE)
  tx$InvoiceDate <- as.POSIXct(tx$InvoiceDate, tz = "UTC")
  tx$Date        <- as.Date(tx$Date)
  validate_transactions(tx)
  tx
}


#' Load the cancellations table.
load_cancellations <- function(path = abs_path(CONFIG$data$online_retail_cancelled)) {
  require_file(path, "online_retail_cancelled.csv")
  cancelled <- readr::read_csv(path, show_col_types = FALSE)
  cancelled$InvoiceDate <- as.POSIXct(cancelled$InvoiceDate, tz = "UTC")
  validate_cancellations(cancelled)
  cancelled
}
