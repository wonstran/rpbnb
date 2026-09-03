# Assemble benchmark comparison data for the report (original covariate units).
# Reads the 6 saved fits and produces results/benchmark_open_v2_report_data.rds
devtools::load_all()
setwd("C:\\Users\\zwang9\\repos\\rpbnb")

fits <- c(
  sml_frank         = "results/fit_benchmark_open_v2_sml_frank_2026-08-21-003228.rds",
  sml_kimeldorf     = "results/fit_benchmark_open_v2_sml_kimeldorf_2026-08-21-013111.rds",
  sml_normal        = "results/fit_benchmark_open_v2_sml_normal_2026-08-21-134835.rds",
  laplace_frank     = "results/fit_benchmark_open_v2_laplace_frank_2026-08-21-150128.rds",
  laplace_kimeldorf = "results/fit_benchmark_open_v2_laplace_kimeldorf_2026-08-21-152832.rds",
  laplace_normal    = "results/fit_benchmark_open_v2_laplace_normal_2026-08-22-015347.rds"
)
elapsed <- c(sml_fr=5172.8, sml_kimeldorf=3522.4, sml_normal=21543.1,
             laplace_fr=4370.7, laplace_kimeldorf=1622.5, laplace_normal=37512.9)
cn <- names(fits)
n  <- length(cn)

gof <- data.frame(
  config = cn,
  method = ifelse(grepl("^laplace", cn), "laplace", "sml"),
  copula = ifelse(grepl("_normal$", cn), "normal", ifelse(grepl("_fr$", cn), "frank", "kimeldorf")),
  elapsed_s = elapsed,
  logLik = 0, AIC = 0, BIC = 0, npar = 30L, conv = 0,
  max_abs_grad = 0,
  row.names = NULL
)

# Effects tables (original units): betas + variance + dependence
sc <- readRDS(fits[1])$scaling
cont <- c("SR40_MI3","MPD_ME","LNAADT_3","IRI_ME","ACCPNTS","CS_MINAB","DP10_ME")
all_betas <- c("(Intercept)","SR40_MI3","MPD_ME","LNAADT_3","IRI_ME","G_ABG2","DP50LE","SP50LE","ACCPNTS","SIGNAL1","NEAR_SIG","CS_MINAB","DP10_ME","RUT_L")
w <- "b1:"

effects <- list()
for (i in seq_len(n)) {
  f  <- readRDS(fits[[cn[i]]])
  c  <- f$coef
  fx <- summary(f$sdreport, "fixed")
  rep <- summary(f$sdreport, "report")
  dep_name <- if (f$dependence$family=="normal") "rho" else "theta"
  # back-transform betas to original units
  grow <- list(config = cn[i])
  for (eq in 1:2) {
    lab <- paste0("b", eq, ":")
    for (par in rownames(c)) {
      if (!grepl(paste0("^", lab), par)) next
      nm <- sub(paste0("^", lab), "", par)
      grow[[paste0("b", eq, "_", nm)]] <- c[par, 1]
    }
  }
  # sd scales original units
  sd_est <- function(nmlog){
    lv <- as.numeric(fx[nmlog, "Estimate"])
    var <- strsplit(nmlog, ":", 2)[[2]]
    if (var %in% namesfitness(sc)) exp(lv)/as.sc[[var]]$scale else exp(lv)
  }
  grow$sd1_SR40 <- sd_est("log_sd1:SR40_MI3")
  grow$sd1_MPD  <- sd_est("log_sd1:MPD_ME")
  grow$sd2_SR40 <- sd_est("log_sd2:SR40_MI3")
  grow$m1 <- as.numeric(rep["m1","Estimate"]); grow$m2 <- as.numeric(rep["m2","Estimate"])
  grow$m1_se <- as.numeric(rep["m1","Std. Error"]); grow$m2_se <- as.numeric(rep["m2","Std. Error"])
  grow$dep <- as.numeric(rep[dep_name,"Estimate"]); grow$dep_se <- as.numeric(rep[dep_name,"Std. Error"])
  grow$tau <- as.numeric(rep["tau","Estimate"])
  effects[[cn[i]]] <- grow
  # GOF + timing
  gof[i,"logLik"] <- as.numeric(f$logLik)
  gof[i,"AIC"] <- AIC(f); gof[i,"BIC"] <- BIC(f)
  gof[i,"npar"] <- length(c); gof[i,"conv"] <- f$optimizer$convergence
  gof[i,"max_abs_grad"] <- if (is.null(f$optimizer$max_abs_gradient)) NA else f$optimizer$max_abs_gradient
}

saveRDS(list(gof=gof, effects=effects, fits=fits, scaling=sc), 
        "results/benchmark_open_v2_report.rds")
cat("Saved results/benchmark_open_v2_report.rds\n")
print(gof)