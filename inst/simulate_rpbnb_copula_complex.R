#!/usr/bin/env Rscript
# =============================================================================
# Generate a RICH copula random-parameter bivariate NB (RP-BNB) sample dataset
# -- the copula analogue of inst/simulate_rpbnb_complex.R / _dependent.R.
#
#   * 5000 observations
#   * 8 independent variables: 3 continuous + 5 dummy (0/1)
#   * a mix of FIXED and RANDOM coefficients in each equation
#   * GENUINE copula dependence between the two NB margins (Gaussian copula,
#     rho = 0.5) -- unlike the Famoye path, the dependence is a copula on the
#     two NB CDFs, and the fitted native parameter here is rho (Kendall's tau
#     is reported too).
#
# WHY THE RANDOM COEFFICIENTS SIT ON CONTINUOUS REGRESSORS ONLY
# ------------------------------------------------------------
# The copula RP estimator is WEAKLY IDENTIFIED when a random coefficient sits on
# a 0/1 dummy: the NB dispersion m_t and the random-coefficient scale sd both
# generate overdispersion, and a dummy's discrete mean-structure lets the
# simulated-ML objective drift to a higher-likelihood mode far from truth
# (m_t -> 0 / sign-flipped dummy). That is a finite-sample identification
# pathology, not a code bug -- the FIXED-coefficient copula MLE recovers the same
# dummy design cleanly. So, exactly as in simulate_rpbnb_dependent.R, the random
# coefficients here are placed on CONTINUOUS regressors (well identified); the
# five dummies enter as ordinary FIXED effects. See the copula-recovery memo.
# =============================================================================

devtools::load_all(quiet = TRUE)

n <- 5000L

# ---- 1. Covariates: 3 continuous + 5 dummy ---------------------------------
set.seed(606)
covariates <- data.frame(
  # continuous (standardized scales keep the linear predictor well behaved)
  x_age    = rnorm(n, 0, 1),
  x_income = rnorm(n, 0, 1),
  x_score  = rnorm(n, 0, 1),
  # dummy variables (0/1) with varying prevalence
  d_female  = rbinom(n, 1, 0.50),
  d_urban   = rbinom(n, 1, 0.60),
  d_married = rbinom(n, 1, 0.55),
  d_college = rbinom(n, 1, 0.40),
  d_smoker  = rbinom(n, 1, 0.25)
)

# ---- 2. True coefficient MEANS (fixed + random-coefficient locations) -------
beta1 <- c("(Intercept)" = 0.50,
           x_age = 0.20, x_income = 0.15, x_score = -0.10,   # x_age RANDOM
           d_female = 0.30, d_urban = -0.25, d_married = 0.10,
           d_college = 0.20, d_smoker = 0.35)                # all dummies FIXED

beta2 <- c("(Intercept)" = 0.30,
           x_age = -0.10, x_income = 0.25, x_score = 0.15,   # x_income RANDOM
           d_female = -0.20, d_urban = 0.30, d_married = -0.15,
           d_college = 0.25, d_smoker = 0.10)                # all dummies FIXED

# ---- 3. RANDOM coefficients (one per equation, CONTINUOUS -> identified) -----
# Normal random coefficients: realized beta_i = mean + sd * z, z ~ N(0, 1). One
# well-identified random slope per equation (as in simulate_rpbnb_dependent.R);
# the copula RP model is weakly identified with more random-coefficient variances
# competing against the NB dispersions, so keep it lean for a clean reference.
random_1 <- list(x_age    = list(dist = "normal", sd = 0.30))
random_2 <- list(x_income = list(dist = "normal", sd = 0.25))

# ---- 4. NB2 dispersions & copula dependence ---------------------------------
dispersion <- c(m1 = 0.50, m2 = 0.60)
cop <- copula("normal", par = 0.5)   # Gaussian copula, rho = 0.5 (positive)

# ---- 5. Simulate ------------------------------------------------------------
sim <- simulate_rpbnb_copula(
  n          = n,
  beta1      = beta1,
  beta2      = beta2,
  random_1   = random_1,
  random_2   = random_2,
  dispersion = dispersion,
  copula     = cop,
  covariates = covariates,
  seed       = 707
)

data <- sim$data

# ---- 6. Report --------------------------------------------------------------
cat("=== Complex copula RP-BNB dataset ===\n")
cat("Observations :", nrow(data), "\n")
cat("Variables    :", paste(setdiff(names(data), c("y1", "y2")), collapse = ", "), "\n")
cat(sprintf("Copula       : %s  (rho = %.2f, Kendall tau = %.3f)\n",
            sim$true$copula, sim$true$theta, sim$true$tau))
cat("Random coefs : eq1 {x_age}, eq2 {x_income}  (continuous -> identified)\n\n")

cat("First 6 rows:\n"); print(head(data))
cat("\nOutcome summaries:\n"); print(summary(data[, c("y1", "y2")]))
cat(sprintf("\ny1: mean=%.3f var=%.3f (overdispersed: var>mean)\n",
            mean(data$y1), var(data$y1)))
cat(sprintf("y2: mean=%.3f var=%.3f\n", mean(data$y2), var(data$y2)))
cat(sprintf("Pearson(y1, y2)  = %.4f\n", cor(data$y1, data$y2)))
cat(sprintf("Spearman(y1, y2) = %.4f   (POSITIVE copula dependence built in)\n",
            cor(data$y1, data$y2, method = "spearman")))

# ---- 7. Persist data + ground-truth parameters ------------------------------
dir.create("data", showWarnings = FALSE)
out_csv <- file.path("data", "simulated_rpbnb_copula_complex.csv")
write.csv(data, out_csv, row.names = FALSE)

truth <- list(
  beta1 = beta1, beta2 = beta2,
  random_1 = random_1, random_2 = random_2,
  dispersion = dispersion,
  copula = sim$true$copula, theta = sim$true$theta, tau = sim$true$tau,
  random_names_1 = names(random_1),
  random_names_2 = names(random_2)
)
out_rds <- file.path("data", "simulated_rpbnb_copula_complex_truth.rds")
saveRDS(truth, out_rds)

cat("\nSaved data  ->", out_csv, "\n")
cat("Saved truth ->", out_rds, "\n")
