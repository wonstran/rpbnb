# -------------------- packages --------------------
if (!requireNamespace("runjags", quietly = TRUE)) install.packages("runjags")
library(runjags)

# -------------------- helpers & simulation --------------------
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

rBNBR <- function(n, X1, X2, beta1, beta2, m1, m2, lambda, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  mu1 <- as.vector(exp(X1 %*% beta1)); mu2 <- as.vector(exp(X2 %*% beta2))
  r1  <- 1/m1; c1 <- c_val(mu1, m1)
  y1  <- rnb_mu_r(n, mu1, r1)
  y2  <- integer(n)
  for (i in seq_len(n)) y2[i] <- rY2_given_Y1(y1[i], mu2[i], m2, lambda, c1[i])
  data.frame(Y1 = y1, Y2 = y2)
}

# simulate data
set.seed(123)
n  <- 2000
df <- data.frame(
  x1_norm = rnorm(n),
  x1_uni  = runif(n, -1, 1),
  x2_norm = rnorm(n),
  x2_bin  = rbinom(n, 1, 0.4)
)
X1 <- model.matrix(~ x1_norm + x1_uni, df)
X2 <- model.matrix(~ x2_norm + x2_bin, df)

beta1_true <- c(log(2.0),  0.30, -0.20)
beta2_true <- c(log(4.0), -0.15,  0.50)
m1_true    <- 0.60
m2_true    <- 0.40

mu1_true <- as.vector(exp(X1 %*% beta1_true))
mu2_true <- as.vector(exp(X2 %*% beta2_true))
c1_true  <- c_val(mu1_true, m1_true)
c2_true  <- c_val(mu2_true, m2_true)
b_sim    <- lambda_bounds_vec(c1_true, c2_true)
lambda_true <- 0.5 * b_sim[2]

sim <- rBNBR(n, X1, X2, beta1_true, beta2_true, m1_true, m2_true, lambda_true, seed = 99)
y1 <- as.integer(sim$Y1); y2 <- as.integer(sim$Y2)

# -------------------- JAGS model --------------------
bnbr_model_string <- "
model{
  d_const <- 1 - exp(-1)
  eps  <- 1.0E-6
  tiny <- 1.0E-12
  Cphi <- 30

  for (k in 1:K1) { beta1[k] ~ dnorm(0, 0.04) }
  for (k in 1:K2) { beta2[k] ~ dnorm(0, 0.04) }
  log_m1 ~ dnorm(0, 1)
  log_m2 ~ dnorm(0, 1)
  z_lambda ~ dnorm(0, 1)

  m1 <- exp(log_m1)
  m2 <- exp(log_m2)
  r1 <- 1 / m1
  r2 <- 1 / m2

  for (n in 1:N){
    mu1[n] <- exp( inprod(X1[n,], beta1) )
    mu2[n] <- exp( inprod(X2[n,], beta2) )
    c1[n]  <- pow(1 + d_const * m1 * mu1[n], -1/m1)
    c2[n]  <- pow(1 + d_const * m2 * mu2[n], -1/m2)
  }

  # running bounds WITHOUT temporary scalars (avoids redefinition)
  lam_lo_cum[1] <- -1/((1 - c1[1]) * (1 - c2[1]))
  lam_hi_cum[1] <-  1 / max( c1[1]*(1 - c2[1]), c2[1]*(1 - c1[1]) )
  for (n in 2:N){
    lam_lo_cum[n] <- max(lam_lo_cum[n-1], -1/((1 - c1[n]) * (1 - c2[n])))
    lam_hi_cum[n] <- min(lam_hi_cum[n-1],  1 / max( c1[n]*(1 - c2[n]), c2[n]*(1 - c1[n]) ))
  }
  lam_lo <- lam_lo_cum[N]
  lam_hi <- lam_hi_cum[N]

  lambda_hat <- lam_lo + (lam_hi - lam_lo) * (eps + (1 - 2*eps) * ilogit(z_lambda))

  for (n in 1:N){
    p1[n] <- r1 / (r1 + mu1[n])
    p2[n] <- r2 / (r2 + mu2[n])
    y1[n] ~ dnegbin(p1[n], r1)
    y2[n] ~ dnegbin(p2[n], r2)

    dep[n]  <- 1 + lambda_hat * (exp(-y1[n]) - c1[n]) * (exp(-y2[n]) - c2[n])
    depc[n] <- max(dep[n], tiny)

    zeros[n] ~ dpois(phi[n])
    phi[n] <- max(1.0E-9, Cphi - log(depc[n]))   # ensure non-negative mean
  }

  m1_out <- m1
  m2_out <- m2
  lambda_out <- lambda_hat
}
"

# -------------------- data, inits, run --------------------
data_jags <- list(
  N = n, K1 = ncol(X1), K2 = ncol(X2),
  X1 = X1, X2 = X2, y1 = y1, y2 = y2,
  zeros = rep(0L, n)
)
K1 <- ncol(X1); K2 <- ncol(X2)
inits_fun <- function(){
  list(beta1 = rnorm(K1, 0, 0.2),
       beta2 = rnorm(K2, 0, 0.2),
       log_m1 = log(0.5), log_m2 = log(0.5), z_lambda = 0)
}

runjags::runjags.options(method = "rjags")
jm <- runjags::run.jags(
  model     = bnbr_model_string,
  monitor   = c("beta1","beta2","m1_out","m2_out","lambda_out"),
  data      = data_jags,
  inits     = inits_fun,
  n.chains  = 3,
  adapt     = 1000,
  burnin    = 2000,
  sample    = 4000
)

print(jm)
summ <- summary(jm)
print(summ)
