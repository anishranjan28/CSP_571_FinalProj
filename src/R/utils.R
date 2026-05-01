# utils.R — shared helpers loaded by every R module / report.
#
# Responsibilities:
#   * locate the project root reliably regardless of caller location
#   * load config.yml once and expose it as a list
#   * resolve relative paths from config to absolute paths
#   * provide standard logging + a deterministic-seed wrapper

suppressPackageStartupMessages({
  library(yaml)
  library(here)
})


# ---------------------------------------------------------------------------
# Project root + config loader
# ---------------------------------------------------------------------------

#' Locate the project root.
#'
#' Uses `here::here()` which walks up from the calling file looking for
#' anchor files (`.Rproj`, `DESCRIPTION`, `.git`, etc.). Falls back to the
#' working directory.
project_root <- function() {
  tryCatch(here::here(), error = function(e) getwd())
}

#' Load config.yml once and cache it.
#'
#' @param path optional override path to config.yml
load_config <- function(path = NULL) {
  if (is.null(path)) {
    path <- file.path(project_root(), "config.yml")
  }
  if (!file.exists(path)) {
    stop("config.yml not found at: ", path)
  }
  yaml::read_yaml(path)
}

#' Resolve a config-relative path to an absolute path.
#'
#' @param rel relative path string from config
abs_path <- function(rel) {
  file.path(project_root(), rel)
}


# ---------------------------------------------------------------------------
# Logging + reproducibility
# ---------------------------------------------------------------------------

log_info <- function(...) {
  msg <- paste0("[", format(Sys.time(), "%H:%M:%S"), "] ", ...)
  message(msg)
}

#' Run an expression with a fixed seed and restore the previous RNG state.
with_seed <- function(seed, expr) {
  old_state <- if (exists(".Random.seed", envir = .GlobalEnv))
    get(".Random.seed", envir = .GlobalEnv) else NULL
  on.exit({
    if (!is.null(old_state)) {
      assign(".Random.seed", old_state, envir = .GlobalEnv)
    }
  }, add = TRUE)
  set.seed(seed)
  force(expr)
}


# ---------------------------------------------------------------------------
# Defensive checks shared across modules
# ---------------------------------------------------------------------------

#' Stop with a clear message if a CSV is missing.
require_file <- function(path, label = "file") {
  if (!file.exists(path)) {
    stop(label, " not found at ", path,
         "\nRun the upstream pipeline step first.")
  }
  invisible(path)
}

#' Stop if a data frame is missing required columns.
require_columns <- function(df, cols, label = "data frame") {
  missing <- setdiff(cols, names(df))
  if (length(missing)) {
    stop(label, " is missing required columns: ",
         paste(missing, collapse = ", "))
  }
  invisible(df)
}

#' Stop if a column has duplicates.
require_unique <- function(df, col, label = "data frame") {
  if (anyDuplicated(df[[col]])) {
    stop(label, " has duplicate values in column ", col)
  }
  invisible(df)
}
