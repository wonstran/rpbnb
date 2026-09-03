# install.packages("cmdstanr", repos = c("https://mc-stan.org/r-packages/", getOption("repos")))
library(cmdstanr)
cmdstanr::check_cmdstan_toolchain()
# cmdstanr::install_cmdstan(overwrite = TRUE) # only if you need to rebuild CmdStan

# ---- Stan model ----
stan_code <- "
functions {
  real nb_logpmf_y_mu_r(int y, real mu, real r) {
    real p = r / (r + mu);
    return lgamma(y + r) - lgamma(r) - lgamma(y + 1)
           + r * log(p) + y * log1m(p);
  }
}
data {
  int<lower=1> N;
  int<lower=1> K1;
  int<lower=1> K2;
  matrix[N, K1] X1;
  matrix[N, K2] X2;
  array[N] int<lower=0> y1;
  array[N] int<lower=0> y2;
}
parameters {
  vector[K1] beta1;
  vector[K2] beta2;
  real log_m1;
  real log_m2;
  real z_lambda;
}
transformed parameters {
  real m1 = exp(log_m1);
  real m2 = exp(log_m2);
  real r1 = 1.0 / m1;
  real r2 = 1.0 / m2;
  vector[N] mu1 = exp(X1 * beta1);
  vector[N] mu2 = exp(X2 * beta2);
  real d_const = 1.0 - exp(-1.0);
  vector[N] c1;
  vector[N] c2;
  for (n in 1:N) {
    c1[n] = pow(1 + d_const * m1 * mu1[n], -1.0 / m1);
    c2[n] = pow(1 + d_const * m2 * mu2[n], -1.0 / m2);
  }
  real lam_lo;
  real lam_hi;
  {
    real lam_min_n;
    real lam_max_n;
    lam_lo = negative_infinity();
    lam_hi = positive_infinity();
    for (n in 1:N) {
      lam_min_n = -1.0 / ((1.0 - c1[n]) * (1.0 - c2[n]));
      lam_lo = fmax(lam_lo, lam_min_n);
      lam_max_n = 1.0 / fmax(c1[n] * (1.0 - c2[n]), c2[n] * (1.0 - c1[n]));
      lam_hi = fmin(lam_hi, lam_max_n);
    }
  }
  real eps = 1e-6;
  real lambda_hat = lam_lo + (lam_hi - lam_lo) * (eps + (1 - 2 * eps) * inv_logit(z_lambda));
}
model {
  beta1 ~ normal(0, 5);
  beta2 ~ normal(0, 5);
  log_m1 ~ normal(0, 1);
  log_m2 ~ normal(0, 1);
  z_lambda ~ normal(0, 1);
  for (n in 1:N) {
    real dep = fmax(1e-300, 1
                    + lambda_hat * (exp(-y1[n]) - c1[n]) * (exp(-y2[n]) - c2[n]));
    target += nb_logpmf_y_mu_r(y1[n], mu1[n], r1)
           +  nb_logpmf_y_mu_r(y2[n], mu2[n], r2)
           +  log(dep);
  }
}
generated quantities {
  vector[N] log_lik;
  for (n in 1:N) {
    real dep = fmax(1e-300, 1
                    + lambda_hat * (exp(-y1[n]) - c1[n]) * (exp(-y2[n]) - c2[n]));
    log_lik[n] = nb_logpmf_y_mu_r(y1[n], mu1[n], 1.0 / m1)
               + nb_logpmf_y_mu_r(y2[n], mu2[n], 1.0 / m2)
               + log(dep);
  }
}
"
stan_file <- write_stan_file(stan_code, dir = tempdir(), basename = "bnbr_famoye")
mod <- cmdstan_model(stan_file)

# ---- R helpers for simulation + R-side log_lik fallback ----
d_const <- 1 - exp(-1)
c_val   <- function(mu, m) (1 + d_const * m * mu)^(-1/m)
rnb_mu_r <- function(n, mu, r) stats::rpois(n, stats::rgamma(n, shape = r, scale = mu/r))

rY2_given_Y1 <- function(y1, mu2, m2, lambda, c1) {
  r2 <- 1/m2; c2 <- c_val(mu2, m2); a <- lambda * (exp(-y1) - c1)
  M  <- if (a >= 0) 1 + a*(1 - c2) else 1 - a*c2
  repeat {
    y2 <- rnb_mu_r(1, mu2, r2)
    w  <- 1 + a * (exp(-y2) - c2); if (w < 0 && w > -1e-15) w <- 0
    if (stats::runif(1) <= max(0, w)/M) return(y2)
  }
}
lambda_bounds_vec <- function(c1, c2) {
  lam_min <- -1 / ((1 - c1) * (1 - c2))
  lam_max <-  1 / pmax(c1 * (1 - c2), c2 * (1 - c1))
  c(max(lam_min), min(lam_max))
}
nb_logpmf_y_mu_r_R <- function(y, mu, r) {
  p <- r / (r + mu)
  lgamma(y + r) - lgamma(r) - lgamma(y + 1) + r*log(p) + y*log1p(-p)
}

# Build a named list of optimized params across cmdstanr versions
extract_opt_params <- function(opt) {
  # try modern
  out <- try(opt$optimized_params(), silent = TRUE)
  if (!inherits(out, "try-error")) {
    if (is.data.frame(out)) out <- as.list(out[1, ])
    return(out)
  }
  # legacy
  out <- try(opt$mle(), silent = TRUE)
  if (!inherits(out, "try-error")) {
    v <- out
    if (is.matrix(v)) v <- v[1, ]
    v <- as.numeric(v)
    names(v) <- colnames(out)
    return(as.list(v))
  }
  stop("Couldn't extract optimized parameters. Please update cmdstanr.")
}

# R fallback for log_lik (matches Stan’s transformed params and mapping)
log_lik_from_opt_R <- function(opt_list, X1, X2, y1, y2) {
  K1 <- ncol(X1); K2 <- ncol(X2)
  
  # Rebuild beta vectors from names like beta1[1], beta2[3], etc.
  b1_idx <- order(as.integer(sub("beta1\\[|\\]", "", names(opt_list)[grepl("^beta1\\[", names(opt_list))])))
  b2_idx <- order(as.integer(sub("beta2\\[|\\]", "", names(opt_list)[grepl("^beta2\\[", names(opt_list))])))
  beta1  <- unlist(opt_list[grepl("^beta1\\[", names(opt_list))])[b1_idx]; names(beta1) <- NULL
  beta2  <- unlist(opt_list[grepl("^beta2\\[", names(opt_list))])[b2_idx]; names(beta2) <- NULL
  stopifnot(length(beta1) == K1, length(beta2) == K2)
  
  log_m1 <- as.numeric(opt_list[["log_m1"]])
  log_m2 <- as.numeric(opt_list[["log_m2"]])
  z_lambda <- as.numeric(opt_list[["z_lambda"]])
  
  m1 <- exp(log_m1); m2 <- exp(log_m2)
  r1 <- 1 / m1;     r2 <- 1 / m2
  
  mu1 <- as.vector(exp(X1 %*% beta1))
  mu2 <- as.vector(exp(X2 %*% beta2))
  c1  <- c_val(mu1, m1)
  c2  <- c_val(mu2, m2)
  
  bnds <- lambda_bounds_vec(c1, c2)
  lamLo <- bnds[1]; lamHi <- bnds[2]
  eps <- 1e-6
  lambda_hat <- lamLo + (lamHi - lamLo) * (eps + (1 - 2*eps) * plogis(z_lambda))
  
  dep <- pmax(1e-300, 1 + lambda_hat * (exp(-y1) - c1) * (exp(-y2) - c2))
  ll1 <- nb_logpmf_y_mu_r_R(y1, mu1, r1)
  ll2 <- nb_logpmf_y_mu_r_R(y2, mu2, r2)
  
  ll1 + ll2 + log(dep)
}

# ---- Simulate data ----
set.seed(123)
n  <- 2000
df <- data.frame(
  x1_norm = stats::rnorm(n),
  x1_uni  = stats::runif(n, -1, 1),
  x2_norm = stats::rnorm(n),
  x2_bin  = stats::rbinom(n, 1, 0.4)
)
X1 <- stats::model.matrix(~ x1_norm + x1_uni, df)
X2 <- stats::model.matrix(~ x2_norm + x2_bin, df)

beta1_true <- c(log(2.0), 0.30, -0.20)
beta2_true <- c(log(4.0), -0.15, 0.50)
m1_true    <- 0.60
m2_true    <- 0.40

mu1_true <- as.vector(exp(X1 %*% beta1_true))
mu2_true <- as.vector(exp(X2 %*% beta2_true))
c1_true  <- c_val(mu1_true, m1_true)
c2_true  <- c_val(mu2_true, m2_true)
b_sim    <- lambda_bounds_vec(c1_true, c2_true)
lambda_true <- 0.5 * b_sim[2]

rBNBR <- function(n, X1, X2, beta1, beta2, m1, m2, lambda, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  mu1 <- as.vector(exp(X1 %*% beta1)); mu2 <- as.vector(exp(X2 %*% beta2))
  r1  <- 1/m1; c1 <- c_val(mu1, m1)
  y1  <- rnb_mu_r(n, mu1, r1)
  y2  <- integer(n)
  for (i in seq_len(n)) y2[i] <- rY2_given_Y1(y1[i], mu2[i], m2, lambda, c1[i])
  data.frame(Y1 = y1, Y2 = y2)
}

sim <- rBNBR(n, X1, X2, beta1_true, beta2_true, m1_true, m2_true, lambda_true, seed = 99)
y1 <- as.integer(sim$Y1)
y2 <- as.integer(sim$Y2)

stan_data <- list(
  N = n,
  K1 = ncol(X1),
  K2 = ncol(X2),
  X1 = X1,
  X2 = X2,
  y1 = y1,
  y2 = y2
)

# ---- Optimize (BFGS) ----
K1 <- ncol(X1); K2 <- ncol(X2)
opt <- mod$optimize(
  data = stan_data,
  algorithm = "bfgs",
  init = list(list(
    beta1    = rep(0, K1),
    beta2    = rep(0, K2),
    log_m1   = log(0.5),
    log_m2   = log(0.5),
    z_lambda = 0
  )),
  seed = 123
)

# ---- Get optimized params (cross-version safe) ----
opt_list <- extract_opt_params(opt)
print(opt_list)  # quick look

# ---- Try generate_quantities in 2 ways; else R fallback ----
gq <- try(mod$generate_quantities(data = stan_data, fitted_params = opt), silent = TRUE)
if (inherits(gq, "try-error")) {
  # try with CSV path(s) from optimize
  gq <- try(mod$generate_quantities(data = stan_data, fitted_params = opt$output_files()), silent = TRUE)
}

if (!inherits(gq, "try-error")) {
  # success path
  log_lik_draws <- as.data.frame(gq$draws("log_lik"))
  print(head(log_lik_draws))
} else {
  # FINAL FALLBACK: compute log_lik in R
  message("cmdstanr generate_quantities not available for optimize output; using R fallback.")
  log_lik_vec <- log_lik_from_opt_R(opt_list, X1, X2, y1, y2)
  print(head(log_lik_vec))
  cat("sum log_lik (R fallback):", sum(log_lik_vec), "\n")
}
