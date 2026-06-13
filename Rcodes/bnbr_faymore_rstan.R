# install.packages("rstan")
# install.packages("loo")
library(rstan)
library(loo)

options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

# ---------- Load data ----------
dat_path <- "C:/Users/wonst/OneDrive/Project/MVNB/data/rwm1984_clean.csv"
raw <- read.csv(dat_path, stringsAsFactors = FALSE)

# Formulas
f1 <- docvis ~ MarM + SinM + SinF + kids + outwork + postHS
f2 <- hospvis ~ MarM + SinM + SinF + kids + outwork + postHS

# Design matrices (with intercepts) + responses
X1 <- model.matrix(f1, raw)
X2 <- model.matrix(f2, raw)
y1 <- as.integer(raw$docvis)
y2 <- as.integer(raw$hospvis)
stopifnot(all(y1 >= 0), all(y2 >= 0))

N  <- nrow(X1); K1 <- ncol(X1); K2 <- ncol(X2)

# ---------- Stan model ----------
stan_code <- "
functions {
  // NB log PMF with mean mu and size r (r = 1/m)
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
  array[N] int<lower=0> y1;   // Stan new array syntax
  array[N] int<lower=0> y2;
}
parameters {
  vector[K1] beta1;
  vector[K2] beta2;
  real log_m1;
  real log_m2;
  real z_lambda; // unconstrained, mapped into (lam_lo, lam_hi)
}
transformed parameters {
  real m1 = exp(log_m1);
  real m2 = exp(log_m2);
  real r1 = 1.0 / m1;
  real r2 = 1.0 / m2;

  vector[N] mu1 = exp(X1 * beta1);
  vector[N] mu2 = exp(X2 * beta2);

  // Famoye c(mu,m) = (1 + d*m*mu)^(-1/m), d = 1 - exp(-1)
  real d_const = 1.0 - exp(-1.0);
  vector[N] c1;
  vector[N] c2;
  for (n in 1:N) {
    c1[n] = pow(1 + d_const * m1 * mu1[n], -1.0 / m1);
    c2[n] = pow(1 + d_const * m2 * mu2[n], -1.0 / m2);
  }

  // Global lambda bounds across all observations
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

  // Map z -> lambda strictly inside (lam_lo, lam_hi)
  real eps = 1e-6;
  real lambda_hat = lam_lo + (lam_hi - lam_lo) * (eps + (1 - 2 * eps) * inv_logit(z_lambda));
}
model {
  // Priors (tune as needed)
  beta1 ~ normal(0, 5);
  beta2 ~ normal(0, 5);
  log_m1 ~ normal(0, 1);
  log_m2 ~ normal(0, 1);
  z_lambda ~ normal(0, 1);

  // Likelihood (NB marginals + dependence factor)
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

sm <- stan_model(model_code = stan_code)

stan_data <- list(
  N = N, K1 = K1, K2 = K2,
  X1 = X1, X2 = X2,
  y1 = y1, y2 = y2
)

# ---------- Option A: Optimization (BFGS MLE) ----------
opt <- optimizing(
  sm,
  data = stan_data,
  init = list(
    beta1    = rep(0, K1),
    beta2    = rep(0, K2),
    log_m1   = log(0.5),
    log_m2   = log(0.5),
    z_lambda = 0
  ),
  as_vector = FALSE,
  verbose = TRUE
)
cat("\n--- Optimized (MLE-style) parameters ---\n")
print(opt$par[names(opt$par) %in% c("beta1","beta2","m1","m2","lambda_hat")])

# ---------- Option B: Full Bayesian sampling ----------
fit <- sampling(
  sm,
  data = stan_data,
  seed = 123,
  chains = 4, iter = 2000, warmup = 1000, thin = 1,
  control = list(adapt_delta = 0.9, max_treedepth = 12)
)

cat("\n--- Posterior summaries ---\n")
print(fit, pars = c("beta1","beta2","m1","m2","lambda_hat"), probs = c(.025,.5,.975))

# Labelled beta tables
summ <- summary(fit, pars = c("beta1","beta2","m1","m2","lambda_hat"))$summary
cn1 <- colnames(X1); cn2 <- colnames(X2)
beta1_idx <- grep("^beta1\\[", rownames(summ))
beta2_idx <- grep("^beta2\\[", rownames(summ))
beta1_tab <- summ[beta1_idx, c("mean","sd","2.5%","97.5%")]; rownames(beta1_tab) <- cn1
beta2_tab <- summ[beta2_idx, c("mean","sd","2.5%","97.5%")]; rownames(beta2_tab) <- cn2
cat("\n--- Beta1 (docvis) ---\n"); print(beta1_tab)
cat("\n--- Beta2 (hospvis) ---\n"); print(beta2_tab)

# ---------- LOO / WAIC (preferred over DIC in Stan world) ----------
log_lik_array <- rstan::extract(fit, pars = "log_lik")$log_lik  # draws x N
loo_res  <- loo(log_lik_array)
waic_res <- waic(log_lik_array)
cat("\n--- LOO ---\n"); print(loo_res)
cat("\n--- WAIC ---\n"); print(waic_res)
