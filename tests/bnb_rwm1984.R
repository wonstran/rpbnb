# =====================================================================
# Bivariate negative binomial (Famoye/Sarmanov) demonstration on the
# German health-survey data rwm1984_clean.csv.
#
# Outcomes:
#   docvis  - number of doctor visits
#   hospvis - number of hospital visits
#
# Run with:  Rscript tests/bnb_rwm1984.R   (from the package root)
# Also executed by R CMD check, so it must run end-to-end without error.
# =====================================================================

library(rpbnb)

# ---- Load the bundled sample data (works both installed and in-source) ------
csv <- system.file("extdata", "rwm1984_clean.csv", package = "rpbnb")
if (!nzchar(csv) || !file.exists(csv)) {
  # fall back to the repository copy when running from a source checkout
  for (p in c("inst/extdata/rwm1984_clean.csv",
              "data/rwm1984_clean.csv",
              "../inst/extdata/rwm1984_clean.csv")) {
    if (file.exists(p)) { csv <- p; break }
  }
}
stopifnot(nzchar(csv), file.exists(csv))
d <- read.csv(csv)

cat("Data:", basename(csv), "-", nrow(d), "observations\n\n")

# ---- Descriptive statistics for the two count outcomes ----------------------
desc <- function(y) c(mean = mean(y), var = var(y), max = max(y),
                      pct_zero = mean(y == 0))
cat("--- Outcome descriptives ---\n")
print(round(rbind(docvis = desc(d$docvis), hospvis = desc(d$hospvis)), 3))
cat("Both outcomes are overdispersed (var >> mean), motivating an NB model.\n\n")

# ---- Model formulas ---------------------------------------------------------
f1 <- docvis  ~ age + outwork + female + married + kids
f2 <- hospvis ~ age + outwork + female

# ---- 1. Independence baseline: two univariate NB2 margins -------------------
fit_ind <- fit_bnb(f1, f2, data = d, dependence = "independence")

# ---- 2. Famoye/Sarmanov bivariate NB with a dependence parameter ------------
fit_fam <- fit_bnb(f1, f2, data = d, dependence = "famoye")

cat("======================================================================\n")
cat(" Famoye bivariate NB fit\n")
cat("======================================================================\n")
print(summary(fit_fam))

# ---- 3. Does modelling dependence help? -------------------------------------
ll_ind <- as.numeric(logLik(fit_ind))
ll_fam <- as.numeric(logLik(fit_fam))
lr_stat <- 2 * (ll_fam - ll_ind)          # 1 extra parameter (lambda)
cat("\n--- Independence vs. Famoye ---\n")
cat(sprintf("logLik  independence = %.3f   famoye = %.3f\n", ll_ind, ll_fam))
cat(sprintf("AIC     independence = %.3f   famoye = %.3f\n",
            AIC(fit_ind), AIC(fit_fam)))
cat(sprintf("LR statistic (1 df)  = %.3f   p = %.4g\n",
            lr_stat, stats::pchisq(lr_stat, df = 1, lower.tail = FALSE)))

# ---- 4. Goodness of fit -----------------------------------------------------
cat("\n")
gof <- bnb_gof(fit_fam, print_output = TRUE)

# ---- 5. Average marginal effects (both equations) ---------------------------
cat("\n--- Average marginal effects (docvis + hospvis) ---\n")
me <- bnb_marginal_effects(fit_fam, which = "both", type = "AME",
                           print_output = TRUE)

# ---- 6. Predicted conditional means -----------------------------------------
pred <- predict(fit_fam)
cat("\n--- Predicted means (first 5 observations) ---\n")
print(round(head(pred, 5), 3))

cat("\nDone.\n")
