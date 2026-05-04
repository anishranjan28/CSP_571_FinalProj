# 00_setup.R — load all packages used across the pipeline.
#
# Sourced by every other src/R/ module. Keeping the package list in one
# place means a missing dependency fails fast at the top of any run, not
# halfway through a long bootstrap.

suppressPackageStartupMessages({
  # Tidy data + I/O
  library(tidyverse)    # dplyr, readr, ggplot2, tidyr, purrr, ...
  library(scales)
  library(yaml)
  library(here)

  # Modelling
  library(cluster)      # silhouette, clusGap, pam
  library(factoextra)   # fviz_*
  library(fpc)          # clusterboot
  library(mclust)       # adjustedRandIndex
  library(dendextend)
  library(rpart)
  library(rpart.plot)
})

source(here::here("src", "R", "utils.R"))

# Single source of truth for the seed used everywhere.
CONFIG <- load_config()
GLOBAL_SEED <- CONFIG$seed
set.seed(GLOBAL_SEED)
