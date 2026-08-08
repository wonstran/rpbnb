#!/usr/bin/env Rscript
# Reproducible test-tier runner for rpbnb. Run from the package root:
#
#   Rscript tools/test-tiers.R fast          # nominal fast tier (slow gates skipped)
#   Rscript tools/test-tiers.R slow-predict  # RP prediction end-to-end fits
#   Rscript tools/test-tiers.R slow-copula   # test-rpbnb-copula.R (slowest Rcpp file)
#   Rscript tools/test-tiers.R slow-tmb      # TMB profiling/inference (~7 min)
#   Rscript tools/test-tiers.R all           # all of the above, in order
#
# Exit status is non-zero if any selected tier has a failure or error, so the
# script is CI-friendly. Prediction *logic* is covered fast by
# tests/testthat/test-predict-unit.R (no optimization); the slow-predict tier is
# the end-to-end fit-through-predict integration.

args <- commandArgs(trailingOnly = TRUE)
tier <- if (length(args)) args[[1]] else "fast"
valid <- c("fast", "slow-predict", "slow-copula", "slow-tmb", "all")
if (!tier %in% valid) {
  stop("unknown tier '", tier, "'. Use one of: ", paste(valid, collapse = ", "),
       call. = FALSE)
}

Sys.setenv(NOT_CRAN = "true")
suppressMessages({
  if (requireNamespace("pkgload", quietly = TRUE)) {
    tryCatch(pkgload::load_all(quiet = TRUE), error = function(e) library(rpbnb))
  } else {
    library(rpbnb)
  }
})

run_files <- function(files) {
  tot <- c(pass = 0L, fail = 0L, warn = 0L, skip = 0L)
  for (f in files) {
    r  <- testthat::test_file(file.path("tests/testthat", f), reporter = "silent")
    df <- as.data.frame(r)
    tot <- tot + c(sum(df$passed), sum(df$failed), sum(df$warning), sum(df$skipped))
  }
  tot
}

all_files  <- list.files("tests/testthat", pattern = "^test-.*\\.R$")
# Excluded from the fast tier because they dominate wall clock even with their
# internal skip_slow() gates honoured: test-rpbnb-copula.R is the slowest Rcpp
# file, and the two TMB files below refit the model along a profile grid.
tmb_slow_files <- c("test-dependence-profile.R", "test-inference-memory.R")
fast_files <- setdiff(all_files, c("test-rpbnb-copula.R", tmb_slow_files))
report <- function(label, t) {
  cat(sprintf("[%-13s] pass=%d fail=%d warn=%d skip=%d\n",
              label, t[["pass"]], t[["fail"]], t[["warn"]], t[["skip"]]))
  t
}

failures <- 0L
if (tier %in% c("fast", "all")) {
  Sys.unsetenv("RPBNB_RUN_SLOW")                    # ensure slow gates skip
  t <- report("fast", run_files(fast_files));        failures <- failures + t[["fail"]]
}
if (tier %in% c("slow-predict", "all")) {
  Sys.setenv(RPBNB_RUN_SLOW = "1")
  t <- report("slow-predict", run_files("test-predict-dist.R")); failures <- failures + t[["fail"]]
  Sys.unsetenv("RPBNB_RUN_SLOW")
}
if (tier %in% c("slow-copula", "all")) {
  t <- report("slow-copula", run_files("test-rpbnb-copula.R"));  failures <- failures + t[["fail"]]
}
if (tier %in% c("slow-tmb", "all")) {
  Sys.setenv(RPBNB_RUN_SLOW = "1")
  t <- report("slow-tmb", run_files(tmb_slow_files));            failures <- failures + t[["fail"]]
  Sys.unsetenv("RPBNB_RUN_SLOW")
}

if (failures > 0L) quit(status = 1L)
