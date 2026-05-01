# 01_validate_data.R — runtime data-integrity checks.
#
# Every function below returns the data frame on success and stops with a
# clear error on failure. Used by both the pipeline orchestration and the
# testthat tests.

source(file.path("src", "R", "00_setup.R"))


#' Validate the customer feature table produced by Python.
#'
#' Checks: CustomerID exists, is unique, is integer; required modelling
#' features present and non-NA; Frequency > 0; Recency >= 0; monetary
#' values finite.
validate_customer_features <- function(customer, feature_cols = CONFIG$features$feature_cols) {

  require_columns(customer, c("CustomerID", "Recency", "Frequency",
                              "GrossMonetary", "NetMonetary",
                              feature_cols),
                  label = "customer_features")
  require_unique(customer, "CustomerID", label = "customer_features")

  if (!is.numeric(customer$CustomerID))
    stop("CustomerID must be numeric / integer.")

  if (any(customer$Frequency < 1, na.rm = TRUE))
    stop("Found Frequency < 1; positive-only invoices should give Frequency >= 1.")

  if (any(customer$Recency < 0, na.rm = TRUE))
    stop("Found Recency < 0; should be days since last purchase.")

  na_in_features <- colSums(is.na(customer[, feature_cols, drop = FALSE]))
  if (any(na_in_features > 0)) {
    stop("Missing values in modelling features:\n",
         paste(names(na_in_features[na_in_features > 0]),
               na_in_features[na_in_features > 0],
               sep = " = ", collapse = "\n"))
  }

  if (any(!is.finite(customer$GrossMonetary)))
    stop("GrossMonetary contains non-finite values.")
  if (any(!is.finite(customer$NetMonetary)))
    stop("NetMonetary contains non-finite values.")

  invisible(customer)
}


#' Validate the cleaned transaction table.
validate_transactions <- function(tx) {
  require_columns(tx, c("CustomerID", "InvoiceNo", "InvoiceDate",
                        "Quantity", "UnitPrice", "Amount"),
                  label = "transactions")
  if (any(tx$Quantity <= 0, na.rm = TRUE))
    stop("Transactions table should contain only positive quantities.")
  if (any(tx$UnitPrice <= 0, na.rm = TRUE))
    stop("Transactions table should contain only positive unit prices.")
  invisible(tx)
}


#' Validate the cancellations table.
validate_cancellations <- function(cancelled) {
  require_columns(cancelled, c("CustomerID", "InvoiceNo", "Quantity",
                               "UnitPrice", "Amount"),
                  label = "cancellations")
  if (!all(grepl("^C", cancelled$InvoiceNo)))
    stop("Cancellations should have InvoiceNo starting with 'C'.")
  invisible(cancelled)
}


#' Validate the PC scores table.
validate_pc_scores <- function(pc, n_components = CONFIG$pca$n_components) {
  required <- c("CustomerID", paste0("PC", seq_len(n_components)))
  require_columns(pc, required, label = "customer_pc_scores")
  require_unique(pc, "CustomerID", label = "customer_pc_scores")
  if (any(!complete.cases(pc)))
    stop("PC scores contain missing values.")
  invisible(pc)
}


#' Validate the cluster assignments table.
validate_clusters <- function(clusters, k_star = CONFIG$clustering$k_star) {
  require_columns(clusters,
                  c("CustomerID", "cluster", "cluster_source", "k"),
                  label = "customer_clusters")
  require_unique(clusters, "CustomerID", label = "customer_clusters")
  if (any(is.na(clusters$cluster)))
    stop("Some customers have NA cluster assignments.")
  n_levels <- length(unique(clusters$cluster))
  if (n_levels != k_star)
    stop("Expected ", k_star, " unique clusters, found ", n_levels)
  invisible(clusters)
}
