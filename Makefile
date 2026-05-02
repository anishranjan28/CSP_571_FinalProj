# UCI Online Retail customer segmentation pipeline.
# Run `make all` from the repo root to reproduce everything.

PY      := python3
R       := Rscript
CONFIG  := config.yml

.PHONY: all data model reports test clean help

help:
	@echo "Targets:"
	@echo "  make data     -  fetch + clean + engineer features (Python)"
	@echo "  make model    -  fit PCA, k-means, save segmentation_pipeline.rds"
	@echo "  make reports  -  knit all .Rmd files in reports/ to HTML/PDF"
	@echo "  make test     -  run testthat tests in tests/"
	@echo "  make all      -  data -> model -> reports -> test"
	@echo "  make clean    -  delete generated data, models, and report outputs"

all: data model reports test

# ---------------------------------------------------------------------------
# 1. Data: fetch UCI, clean, build customer feature table.
# ---------------------------------------------------------------------------
data:
	$(PY) src/python/data_prep.py

# ---------------------------------------------------------------------------
# 2. Model: PCA -> k-means -> profiles -> segmentation_pipeline.rds
# ---------------------------------------------------------------------------
model: data
	$(R) scripts/run_pipeline.R

# ---------------------------------------------------------------------------
# 3. Reports: knit every .Rmd in reports/.
# ---------------------------------------------------------------------------
reports: model
	$(R) scripts/render_reports.R

# ---------------------------------------------------------------------------
# 4. Tests
# ---------------------------------------------------------------------------
test:
	$(R) -e 'testthat::test_dir("tests", reporter = "summary")'

# ---------------------------------------------------------------------------
# 5. Clean
# ---------------------------------------------------------------------------
clean:
	rm -rf data/interim/* data/processed/* data/outputs/*
	rm -rf models/*.rds
	rm -rf reports/*_cache reports/*_files
	rm -rf reports/*.html reports/*.pdf reports/*.tex reports/*.log
