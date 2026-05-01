# scripts/render_reports.R — knit every .Rmd in reports/ into HTML + PDF
# (whichever the YAML allows). Skips the file if rendering fails so one
# broken report doesn't stop the rest.

source(file.path("src", "R", "00_setup.R"))

reports_dir <- abs_path(CONFIG$reports_dir)
rmd_files <- list.files(reports_dir, pattern = "\\.Rmd$", full.names = TRUE)

if (!length(rmd_files)) {
  log_info("No .Rmd files found in reports/. Nothing to render.")
  quit(save = "no", status = 0)
}

results <- vapply(rmd_files, function(f) {
  log_info(sprintf("Rendering %s ...", basename(f)))
  ok <- tryCatch({
    rmarkdown::render(f, quiet = TRUE,
                      envir = new.env(parent = globalenv()))
    TRUE
  }, error = function(e) {
    message("  -> failed: ", conditionMessage(e))
    FALSE
  })
  ok
}, logical(1))

cat("\n=========== RENDER SUMMARY ===========\n")
print(tibble::tibble(report = basename(rmd_files), ok = results))
cat("======================================\n")

if (!all(results)) quit(save = "no", status = 1)
