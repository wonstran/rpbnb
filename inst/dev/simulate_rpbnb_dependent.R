#!/usr/bin/env Rscript
# =============================================================================
# Generate an RP-BNB sample dataset with GENUINELY CORRELATED (y1, y2), i.e. a
# non-zero Famoye/Sarmanov dependence parameter lambda -- unlike the Phase-1
# simulate_rpbnb() which is limited to independent margins (lambda = 0).
#
# Joint pmf (Famoye 2010, Eq. 4):
#   P(y1,y2) = P_NB(y1; mu1,m1) * P_NB(y2; mu2,m2)
#              * [1 + lambda (e^{-y1} - c1)(e^{-y2} - c2)],
#   c_t = (1 + d m_t mu_t)^{-1/m_t},  d = 1 - e^{-1}.
#
# Sampling: the Famoye marginals are exactly NB, so draw y1 ~ NB(mu1,m1), then
# y2 from the tilted conditional P(y2|y1) proportional to P_NB(y2)*[1+lambda*...].
# lambda is clamped inside the validity bounds (multiplicative factor >= 0).
# =============================================================================

devtools::load_all(quiet = TRUE)

n <- 5000L
d <- 1 - exp(-1)

# ---- 1. Covariates: 3 continuous + 5 dummy ---------------------------------
set.seed(303)
covariates <- data.frame(
  x_age    = rnorm(n), x_income = rnorm(n), x_score = rnorm(n),
  d_female  = rbinom(n, 1, 0.50), d_urban  = rbinom(n, 1, 0.60),
  d_married = rbinom(n, 1, 0.55), d_college = rbinom(n, 1, 0.40),
  d_smoker  = rbinom(n, 1, 0.25)
)

# ---- 2. True coefficient means ---------------------------------------------
beta1 <- c("(Intercept)" = 0.50, x_age = 0.20, x_income = 0.15, x_score = -0.10,
           d_female = 0.30, d_urban = -0.25, d_married = 0.10,
           d_college = 0.20, d_smoker = 0.35)
beta2 <- c("(Intercept)" = 0.30, x_age = -0.10, x_income = 0.25, x_score = 0.15,
           d_female = -0.20, d_urban = 0.30, d_married = -0.15,
           d_college = 0.25, d_smoker = 0.10)

# ---- 3. Random coefficients (continuous only -> well identified) -----------
random_1 <- list(x_age    = list(dist = "normal", sd = 0.30))
random_2 <- list(x_income = list(dist = "normal", sd = 0.25))
dispersion <- c(m1 = 0.50, m2 = 0.60)

# ---- 4. Per-individual realized means (draw the random coefficients) --------
set.seed(404)
b1 <- matrix(rep(beta1, each = n), n, dimnames = list(NULL, names(beta1)))
b2 <- matrix(rep(beta2, each = n), n, dimnames = list(NULL, names(beta2)))
b1[, "x_age"]    <- beta1[["x_age"]]    + 0.30 * rnorm(n)   # random slope eq1
b2[, "x_income"] <- beta2[["x_income"]] + 0.25 * rnorm(n)   # random slope eq2

Xc <- cbind(`(Intercept)` = 1, as.matrix(covariates))
mu1 <- exp(rowSums(Xc[, names(beta1)] * b1))
mu2 <- exp(rowSums(Xc[, names(beta2)] * b2))
m1 <- dispersion[["m1"]]; m2 <- dispersion[["m2"]]
c1 <- c_val(mu1, m1); c2 <- c_val(mu2, m2)

# ---- 5. Choose a valid lambda (positive dependence) -------------------------
# Global validity bounds: intersect per-observation lambda bounds.
bnds <- vapply(seq_len(n), function(i) lambda_bounds_vec(c1[i], c2[i]), numeric(2))
lam_lo <- max(bnds[1, ]); lam_hi <- min(bnds[2, ])
lambda <- 0.7 * lam_hi                    # safely inside; positive correlation
cat(sprintf("Lambda validity bounds: [%.3f, %.3f]  -> using lambda = %.4f\n",
            lam_lo, lam_hi, lambda))

# ---- 6. Sample (y1, y2) from the Famoye BNB conditional ---------------------
set.seed(505)
y1 <- rnbinom(n, size = 1 / m1, mu = mu1)

ymax <- 400L
ygrid <- 0:ymax
# P_NB(y2) for each obs: n x (ymax+1)
P2 <- outer(seq_len(n), ygrid, function(i, k) dnbinom(k, size = 1 / m2, mu = mu2[i]))
# tilt factor 1 + lambda (e^{-y1} - c1)(e^{-y2} - c2)
tilt <- 1 + lambda * (exp(-y1) - c1) * outer(rep(1, n), exp(-ygrid)) -
        lambda * (exp(-y1) - c1) * c2
cond <- P2 * tilt
cond[cond < 0] <- 0                        # guard tiny numerical negatives
cond <- cond / rowSums(cond)
cdf  <- t(apply(cond, 1, cumsum))
u    <- runif(n)
y2   <- max.col(u <= cdf, ties.method = "first") - 1L   # inverse-CDF sample

data <- data.frame(y1 = y1, y2 = y2, covariates)

# ---- 7. Report --------------------------------------------------------------
cat("\n=== Correlated RP-BNB dataset ===\n")
cat("Observations:", nrow(data), "\n")
cat(sprintf("y1: mean=%.3f var=%.3f\n", mean(y1), var(y1)))
cat(sprintf("y2: mean=%.3f var=%.3f\n", mean(y2), var(y2)))
cat(sprintf("cor(y1, y2)          = %.4f   (POSITIVE dependence built in)\n",
            cor(y1, y2)))
cat(sprintf("Spearman(y1, y2)     = %.4f\n", cor(y1, y2, method = "spearman")))
cat("First rows:\n"); print(head(data))

# ---- 8. Persist data + ground truth ----------------------------------------
dir.create("data", showWarnings = FALSE)
write.csv(data, file.path("data", "simulated_rpbnb_dependent.csv"), row.names = FALSE)
truth <- list(beta1 = beta1, beta2 = beta2,
              random_1 = random_1, random_2 = random_2,
              dispersion = dispersion, lambda = lambda,
              random_names_1 = names(random_1), random_names_2 = names(random_2))
saveRDS(truth, file.path("data", "simulated_rpbnb_dependent_truth.rds"))
cat("\nSaved data  -> data/simulated_rpbnb_dependent.csv\n")
cat("Saved truth -> data/simulated_rpbnb_dependent_truth.rds\n")
cat(sprintf("TRUE lambda = %.4f\n", lambda))
