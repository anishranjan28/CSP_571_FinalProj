#!/usr/bin/env bash
# setup.sh — one-command bootstrap for the UCI Online Retail segmentation
# pipeline.
#
# What it does, in order:
#   1. Detects the OS (Linux / macOS / Git Bash on Windows).
#   2. Creates a Python virtual environment at ./.venv (idempotent).
#   3. Installs Python dependencies from requirements.txt.
#   4. Installs any missing R packages (idempotent — checks first).
#   5. Sanity-checks make, pandoc, and a LaTeX engine for PDF rendering.
#   6. Runs `make all` to fetch data, fit the model, knit reports, and run tests.
#
# Re-runnable. Anything already installed is skipped, so calling
# `./setup.sh` again after a code change is safe and fast.
#
# Usage:
#   ./setup.sh              # everything
#   ./setup.sh --no-build   # env setup only, skip `make all`
#   ./setup.sh --no-r       # skip R package install (e.g. on a CI Python-only run)
#   ./setup.sh --help

set -euo pipefail

# ---------------------------------------------------------------------------
# 0. Flags
# ---------------------------------------------------------------------------
RUN_BUILD=1
INSTALL_R=1

for arg in "$@"; do
  case "$arg" in
    --no-build) RUN_BUILD=0 ;;
    --no-r)     INSTALL_R=0 ;;
    --help|-h)
      sed -n '2,21p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown flag: $arg" >&2
      echo "Use --help for usage." >&2
      exit 2
      ;;
  esac
done

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
log()  { printf '\033[1;34m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m  %s\n' "$*" >&2; }
err()  { printf '\033[1;31m[err]\033[0m   %s\n' "$*" >&2; }
die()  { err "$*"; exit 1; }

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# ---------------------------------------------------------------------------
# 1. Detect OS
# ---------------------------------------------------------------------------
case "$(uname -s)" in
  Linux*)               OS="linux" ;;
  Darwin*)              OS="macos" ;;
  MINGW*|MSYS*|CYGWIN*) OS="windows" ;;
  *)                    OS="unknown" ;;
esac
log "OS detected: $OS"

# Windows venvs put binaries under Scripts/, POSIX puts them under bin/.
if [[ "$OS" == "windows" ]]; then
  VENV_BIN=".venv/Scripts"
  PY_CMDS=(python python3 py)
else
  VENV_BIN=".venv/bin"
  PY_CMDS=(python3 python)
fi

# ---------------------------------------------------------------------------
# 2. Locate a usable Python
# ---------------------------------------------------------------------------
PY=""
for cand in "${PY_CMDS[@]}"; do
  if command -v "$cand" >/dev/null 2>&1; then
    PY="$cand"; break
  fi
done
[[ -n "$PY" ]] || die "No Python interpreter found. Install Python 3.10+ and retry."

PY_VERSION="$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
log "Python: $PY ($PY_VERSION)"

# ---------------------------------------------------------------------------
# 3. Create / reuse the venv
# ---------------------------------------------------------------------------
if [[ ! -d ".venv" ]]; then
  log "Creating virtualenv at ./.venv"
  "$PY" -m venv .venv
else
  log "Reusing existing virtualenv at ./.venv"
fi

# Resolve venv pip / python explicitly rather than activating the venv —
# `source activate` doesn't propagate cleanly across all bash flavours.
VENV_PY="$VENV_BIN/python"
VENV_PIP="$VENV_BIN/pip"
[[ -x "$VENV_PY" || -f "${VENV_PY}.exe" ]] || die "venv python not found at $VENV_PY"

log "Upgrading pip in the venv"
"$VENV_PY" -m pip install --quiet --upgrade pip

# ---------------------------------------------------------------------------
# 4. Python dependencies
# ---------------------------------------------------------------------------
if [[ -f "requirements.txt" ]]; then
  log "Installing Python deps from requirements.txt"
  "$VENV_PIP" install --quiet -r requirements.txt
else
  warn "No requirements.txt at project root — skipping Python dep install."
fi

# ---------------------------------------------------------------------------
# 5. R dependencies (idempotent — only installs what's missing)
# ---------------------------------------------------------------------------
if [[ "$INSTALL_R" -eq 1 ]]; then
  if command -v Rscript >/dev/null 2>&1; then
    log "Installing missing R packages"
    Rscript --no-save --no-restore - <<'R_EOF'
required <- c(
  # Tidy + I/O
  "tidyverse", "scales", "yaml", "here",
  # Modelling
  "cluster", "factoextra", "fpc", "mclust", "dendextend",
  "rpart", "rpart.plot",
  # EDA
  "corrplot", "GGally", "gridExtra",
  # Reports + tests
  "knitr", "rmarkdown", "testthat"
)
installed <- rownames(installed.packages())
missing   <- setdiff(required, installed)

if (length(missing) == 0L) {
  cat("[setup] All R packages already installed.\n")
} else {
  cat(sprintf("[setup] Installing %d R package(s): %s\n",
              length(missing), paste(missing, collapse = ", ")))
  install.packages(missing,
                   repos = "https://cloud.r-project.org",
                   quiet = TRUE)
  still_missing <- setdiff(required, rownames(installed.packages()))
  if (length(still_missing) > 0L) {
    stop(sprintf("Failed to install: %s",
                 paste(still_missing, collapse = ", ")))
  }
}
R_EOF
  else
    warn "Rscript not on PATH — skipping R package install."
    warn "Install R from https://cloud.r-project.org and rerun, or pass --no-r."
    INSTALL_R=0
  fi
else
  log "Skipping R package install (--no-r)"
fi

# ---------------------------------------------------------------------------
# 6. Build prerequisites — sanity-check, don't install
# ---------------------------------------------------------------------------
if [[ "$RUN_BUILD" -eq 1 ]]; then
  command -v make >/dev/null 2>&1 || die \
    "GNU make is required to run \`make all\`. Install it and retry, or pass --no-build."

  command -v pandoc >/dev/null 2>&1 || warn \
    "pandoc not found — rmarkdown::render will fail to knit reports. \
Install via: pacman -S pandoc (Git Bash) / apt install pandoc / brew install pandoc."

  if ! command -v xelatex >/dev/null 2>&1 && ! command -v pdflatex >/dev/null 2>&1; then
    warn "No LaTeX engine found (xelatex/pdflatex). PDF report rendering will fail."
    warn "On R, install via: install.packages('tinytex'); tinytex::install_tinytex()"
  fi
fi

# ---------------------------------------------------------------------------
# 7. Run the build
# ---------------------------------------------------------------------------
if [[ "$RUN_BUILD" -eq 1 ]]; then
  log "Running \`make all\` — fetch data, fit model, knit reports, run tests"
  # Make data_prep.py find the venv's Python via PATH; the Makefile uses
  # `python` rather than the venv binary directly.
  export PATH="$PROJECT_ROOT/$VENV_BIN:$PATH"
  make all
  log "Done. Knitted reports are in reports/, the deployable model is in models/segmentation_pipeline.rds."
else
  log "Environment ready. Run \`make all\` when you want to build."
fi
