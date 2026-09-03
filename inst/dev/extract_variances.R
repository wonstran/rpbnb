#!/usr/bin/env Rscript
# Minimal: print per-config dependence + dispersion (report-side) and sd scales.
devtools::load_all("C:\\Users\\zwang9\\repos\\rpbnb")
setwd("C:\\Users\\zwang9\\repos\\rpbnb")

fits <- c(
  sml_frank         = "results/fit_benchmark_open_v2_sml_frank_2026-08-21-003228.rds",
  sml_kimeldorf     = "results/fit_benchmark_open_v2_sml_kimeldorf_2026-08-21-013111.rds",
  sml_normal        = "results/fit_benchmark_open_v2_sml_normal_2026-08-21-134835.rds",
  laplace_frank     = "results/fit_benchmark_open_v2_laplace_frank_2026-08-21-150128.rds",
  laplace_kimeldorf = "results/fit_benchmark_open_v2_laplace_kimeldorf_2026-08-21-152832.rds",
  laplace_normal    = "results/fit_benchmark_open_v2_laplace_normal_2026-08-22-015347.rds"
)
sc <- readRDS(fits[1])$scaling
cn <- names(fits)

for (nm in cn) {
  f  <- readRDS(fits[[nm]])
  fx <- summary(f$sdreport, "fixed")
  rep <- summary(f$sdreport, "report")
  depnm <- if (f$dependence$family=="normal")"rho" else "theta"
  sd <- function(nl){ lv<-as.numeric(fx[nl,"Estimate"]); var<-strsplit(nl,":",2)[[2]];
                      if(var %in% names(sc)) exp(lv)/as.numeric(sc[[var]]["scale"]) else exp(lv) }
  cat(sprintf("%-16s dep_%s=%.4f m1=%.4f m2=%.4f sd1SR40=%.4g sd1MPD=%.4g sd2SR40=%.4g\n",
      nm, depnm, as.numeric(rep[depnm,"Estimate"]), as.numeric(rep["m1","Estimate"]),
      as.numeric(rep["m2","Estimate"]), sd_est("log_sd1:SR40_MI3"),
      sd_est("log_sd1:MPD_ME"), sd_est("log_sd2:SR40_MI3")))
  if (depnm=="rho") { cat("   (note rho shown; no theta)\n") }
}