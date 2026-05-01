# 05_profiles.R — descriptive cluster profiles for the report.
#
# Produces:
#   * a per-cluster headline table (n, revenue, share)
#   * mean + median feature profiles
#   * top products per cluster (with and without anomaly StockCode 23843)
#   * the labelled customer table for downstream scoring / SVM extension

source(file.path("src", "R", "00_setup.R"))
source(file.path("src", "R", "02_features.R"))


#' Headline summary: customers, revenue, share per cluster.
cluster_headline <- function(cust) {
  total_rev_net <- sum(cust$NetMonetary)
  cust |>
    dplyr::group_by(cluster) |>
    dplyr::summarise(
      customers        = dplyr::n(),
      revenue_net      = sum(NetMonetary),
      revenue_gross    = sum(GrossMonetary),
      median_net_spend = median(NetMonetary),
      median_orders    = median(Frequency),
      median_recency_d = median(Recency),
      .groups = "drop"
    ) |>
    dplyr::mutate(
      pct_customers = customers   / sum(customers),
      pct_revenue   = revenue_net / total_rev_net
    ) |>
    dplyr::arrange(cluster)
}


#' Per-feature median + mean profile (heavy-tailed RFM needs both).
feature_profile <- function(cust) {
  cust |>
    dplyr::group_by(cluster) |>
    dplyr::summarise(
      n                    = dplyr::n(),
      Recency_med          = median(Recency),
      Recency_mean         = mean(Recency),
      Frequency_med        = median(Frequency),
      Frequency_mean       = mean(Frequency),
      NetMonetary_med      = median(NetMonetary),
      NetMonetary_mean     = mean(NetMonetary),
      AvgOrderValue_med    = median(AvgOrderValue),
      AvgOrderValue_mean   = mean(AvgOrderValue),
      UniqueProducts_med   = median(UniqueProducts),
      UniqueProducts_mean  = mean(UniqueProducts),
      AvgUnitPrice_med     = median(AvgUnitPrice_winsor),
      TenureDays_med       = median(TenureDays),
      SinglePurchase_share = mean(SinglePurchase),
      ReturnRate_med       = median(ReturnRate),
      ReturnRate_mean      = mean(ReturnRate),
      ReturnValueRate_med  = median(ReturnValueRate),
      ReturnValueRate_mean = mean(ReturnValueRate),
      .groups = "drop"
    )
}


#' Top N products per cluster, with optional anomaly exclusion.
top_products_by_cluster <- function(tx, n = 5, exclude_codes = NULL) {
  if (length(exclude_codes)) tx <- dplyr::filter(tx, !StockCode %in% exclude_codes)
  tx |>
    dplyr::group_by(cluster, StockCode, Description) |>
    dplyr::summarise(revenue = sum(Amount), .groups = "drop") |>
    dplyr::group_by(cluster) |>
    dplyr::slice_max(revenue, n = n) |>
    dplyr::ungroup()
}


#' Save the labelled customer table (for SVM / scoring downstream).
save_labeled_customers <- function(customer, clusters,
                                   path = abs_path(CONFIG$data$customer_features_labeled)) {
  out <- customer |>
    dplyr::inner_join(clusters |> dplyr::select(CustomerID, cluster),
                      by = "CustomerID")
  stopifnot(nrow(out) == nrow(customer))
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  readr::write_csv(out, path)
  invisible(out)
}
