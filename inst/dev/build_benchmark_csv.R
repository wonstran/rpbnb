#!/usr/bin/env Rscript
# Build a tidy results table (original covariate units) for the comparison report.
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
elapsed <- c(5172.8, 3522.4, 21543.1, 4370.7, 1622.5, 37512.9)
cn <- names(fits)
n  <- length(cn)

sc <- readRDS(fits[1])$scaling
cont <- c("SR40_MI3","MPD_ME","LNAADT_3","IRI_ME","ACCPNTS","CS_MINAB","DP10_ME")

rows <- vector("list", n)
for (i in seq_len(n)) {
  f     <- readRDS(fits[[cn[i]]])
  c     <- f$coef
  fx    <- summary(f$sdreport, "fixed")
  rep   <- summary(f$sdreport, "report")
  depnm <- if (f$dependence$family == "normal") "rho" else "theta"

  row <- list(config = cn[i])
  row$method    <- if (grepl("^laplace", cn[i])) "laplace" else "sml"
  row$copula    <- if (grepl("_normal$", cn[i])) "normal"
                   else if (grepl("_frank$", cn[i])) "frank" else "kimeldorf"
  row$elapsed_s <- elapsed[i]
  row$logLik    <- as.numeric(f$logLik)
  row$AIC       <- AIC(f)
  row$BIC       <- BIC(f)
  row$npar      <- length(c)
  row$conv      <- f$optimizer$convergence
  row$max_abs_grad <- if (is.null(f$optimizer$max_abs_gradient)) NA else f$optimizer$max_abs_gradient

  # back-transform betas to original units
  bstd <- c
  b0   <- bstd
  for (p in rownames(bstd)) {
    if (!grepl("^b[12]:", p)) next
    var <- sub("^b[12]:", "", p)
    if (var %in% cont && var %in% names(sc)) b0[p, 1] <- bstd[p, 1] / as.numeric(sc[[var]]["scale"])
  }
  for (eq in 1:2) {
    lab  <- paste0("b", eq, ":")
    intp <- paste0(lab, "(Intercept)")
    adj  <- 0
    for (var in cont) {
      nm <- paste0(lab, var)
      if (nm %in% rownames(b0) && var %in% names(sc)) adj <- adj + b0[nm, 1] * as.numeric(sc[[var]]["center"])
    }
    b0[intp, 1] <- b0[intp, 1] - adj
  }
  for (p in rownames(b0)) row[[sub(":", "_", p)]] <- b0[p, 1]

  sd_est <- function(nmlog) {
    lv <- as.numeric(fx[nmlog, "Estimate"])
    var <- strsplit(nmlog, ":", 2)[[2]]
    if (var %in% names(sc)) exp(lv) / as.numeric(sc[[var]]["scale"]) else exp(lv)
  }
  row$sd1_SR40 <- sd_est("log_sd1:SR40_MI3")
  row$sd1_MPD  <- sd_est("log_sd1:MPD_ME")
  row$sd2_SR40 <- sd_est("log_sd2:SR40_MI3")
  row$m1       <- as.numeric(rep["m1","Estimate"])
  row$m2       <- as.numeric(rep["m2","Estimate"])
  row$m1_se    <- as.numeric(rep["m1","Std. Error"])
  row$m2_se    <- as.numeric(rep["m2","Std. Error"])
  row$dep      <- as.numeric(rep[depnm,"Estimate"])
  row$dep_se   <- as.numeric(rep[depnm,"Std. Error"])
  row$tau      <- as.numeric(rep["tau","Estimate"])

  rows[[i]] <- row
}

# tidy long table
long <- do.call(rbind, lapply(rows, function(r) {
  tib <- do.call(data.frame, r)
  tib$param <- names(r)[-(length(r) - 4)]  # placeholder; simpler tibble below
  tib
}))