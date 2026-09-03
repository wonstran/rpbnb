functions {
  // Log PMF of NB parameterized by (mu, r) with r = 1/m
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
  int<lower=0> y1[N];
  int<lower=0> y2[N];
}

parameters {
  vector[K1] beta1;
  vector[K2] beta2;
  real log_m1;
  real log_m2;
  real z_lambda; // unconstrained, mapped to lambda via global bounds
}

transformed parameters {
  // convenience
  real m1 = exp(log_m1);
  real m2 = exp(log_m2);
  real r1 = 1.0 / m1;
  real r2 = 1.0 / m2;

  vector[N] mu1 = exp(X1 * beta1);
  vector[N] mu2 = exp(X2 * beta2);

  // c(mu, m) = (1 + d*m*mu)^(-1/m), d = 1 - exp(-1)
  real d_const = 1.0 - exp(-1.0);
  vector[N] c1;
  vector[N] c2;
  for (n in 1:N) {
    c1[n] = pow(1 + d_const * m1 * mu1[n], -1.0 / m1);
    c2[n] = pow(1 + d_const * m2 * mu2[n], -1.0 / m2);
  }

  // Global lambda bounds across all i (sufficient for positivity)
  real lam_lo; // = max_i  -1 / ((1 - c1_i)(1 - c2_i))
  real lam_hi; // = min_i   1 / max( c1_i*(1 - c2_i), c2_i*(1 - c1_i) )
  {
    real lam_min_n;
    real lam_max_n;
    lam_lo = negative_infinity();
    lam_hi = positive_infinity();
    for (n in 1:N) {
      lam_min_n = -1.0 / ((1.0 - c1[n]) * (1.0 - c2[n]));
      lam_lo = fmax(lam_lo, lam_min_n);

      // max() of two positive terms
      lam_max_n = 1.0 / fmax(c1[n] * (1.0 - c2[n]), c2[n] * (1.0 - c1[n]));
      lam_hi = fmin(lam_hi, lam_max_n);
    }
  }

  // Map z -> lambda strictly inside (lam_lo, lam_hi)
  real eps = 1e-6;
  real lambda_hat = lam_lo + (lam_hi - lam_lo) * (eps + (1 - 2 * eps) * inv_logit(z_lambda));
}

model {
  // Weakly-informative priors (adjust as you like)
  beta1 ~ normal(0, 5);
  beta2 ~ normal(0, 5);
  log_m1 ~ normal(0, 1);
  log_m2 ~ normal(0, 1);
  z_lambda ~ normal(0, 1);

  // Likelihood
  for (n in 1:N) {
    real dep = fmax(1e-300, 1
                    + lambda_hat * (exp(-y1[n]) - c1[n]) * (exp(-y2[n]) - c2[n]));
    target += nb_logpmf_y_mu_r(y1[n], mu1[n], r1)
           +  nb_logpmf_y_mu_r(y2[n], mu2[n], r2)
           +  log(dep);
  }
}

generated quantities {
  // Per-observation log-lik for loo/waic
  vector[N] log_lik;
  for (n in 1:N) {
    real dep = fmax(1e-300, 1
                    + lambda_hat * (exp(-y1[n]) - c1[n]) * (exp(-y2[n]) - c2[n]));
    log_lik[n] = nb_logpmf_y_mu_r(y1[n], mu1[n], 1.0 / m1)
               + nb_logpmf_y_mu_r(y2[n], mu2[n], 1.0 / m2)
               + log(dep);
  }
}
