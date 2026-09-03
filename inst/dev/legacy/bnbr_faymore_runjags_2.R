# =========================================================
# BNBR (Famoye) via runjags — prints DIC (with fallback)
# =========================================================
rm(list=ls())

# --- packages ---
if (!requireNamespace("runjags", quietly = TRUE)) install.packages("runjags")
suppressPackageStartupMessages(library(runjags))

# --- load your data ---
dat_path <- "C:\\Users\\wonst\\OneDrive\\Project\\MVNB\\data\\rwm1984_clean.csv"
raw <- read.csv(dat_path, stringsAsFactors = FALSE)

# Formulas
f1 <- docvis ~ MarM + SinM + SinF + kids + outwork + postHS
f2 <- hospvis ~ MarM + SinM + SinF + kids + outwork + postHS

# Model matrices + responses
X1 <- model.matrix(f1, raw)
X2 <- model.matrix(f2, raw)
y1 <- as.integer(raw$docvis)
y2 <- as.integer(raw$hospvis)
if (any(y1 < 0 | y2 < 0)) stop("Counts must be non-negative")

N  <- nrow(X1); K1 <- ncol(X1); K2 <- ncol(X2)
zeros <- rep(0L, N)

# Constants
d_const <- 1 - exp(-1)    # Famoye c(mu,m)
L_const <- 0.99           # keeps dep term strictly positive
C_const <- 50.0           # big constant for zeros-trick (ensures Poisson mean >= 0)

# --- JAGS model (zeros-trick for dependence) ---
model_string <- "
model {

  # Priors
  for (k in 1:K1) { beta1[k] ~ dnorm(0, 0.01) }
  for (k in 1:K2) { beta2[k] ~ dnorm(0, 0.01) }
  log_m1 ~ dnorm(0, 1)
  log_m2 ~ dnorm(0, 1)
  z_lambda ~ dnorm(0, 1)

  # Transforms
  m1 <- exp(log_m1)
  m2 <- exp(log_m2)
  r1 <- 1.0 / m1
  r2 <- 1.0 / m2

  # safe lambda in (-L_const, L_const) via tanh
  ex2 <- exp(2*z_lambda)
  tanh_z <- (ex2 - 1) / (ex2 + 1)
  lambda <- L_const * tanh_z

  for (n in 1:N) {
    # means
    eta1[n] <- inprod(X1[n,], beta1)
    eta2[n] <- inprod(X2[n,], beta2)
    mu1[n] <- exp(eta1[n])
    mu2[n] <- exp(eta2[n])

    # Famoye c(mu,m)
    c1[n] <- pow(1 + d_const * m1 * mu1[n], -1/m1)
    c2[n] <- pow(1 + d_const * m2 * mu2[n], -1/m2)

    # NB marginals: p = r/(r+mu)
    p1[n] <- r1 / (r1 + mu1[n])
    p2[n] <- r2 / (r2 + mu2[n])

    y1[n] ~ dnegbin(p1[n], r1)
    y2[n] ~ dnegbin(p2[n], r2)

    # dependence factor via zeros-trick
    dep_raw[n] <- 1 + lambda * (exp(-y1[n]) - c1[n]) * (exp(-y2[n]) - c2[n])
    dep[n] <- max(dep_raw[n], 1.0E-12)
    phi[n] <- C_const - log(dep[n])     # Poisson mean >= 0
    zeros[n] ~ dpois(phi[n])
  }

  # monitor 'deviance' so dic module can compute DIC
}
"

# --- data/inits for JAGS ---
jags_data <- list(
  N = N, K1 = K1, K2 = K2,
  X1 = X1, X2 = X2,
  y1 = y1, y2 = y2,
  zeros = zeros,
  d_const = d_const,
  L_const = L_const,
  C_const = C_const
)

make_inits <- function() list(
  beta1 = rep(0, K1),
  beta2 = rep(0, K2),
  log_m1 = log(0.5),
  log_m2 = log(0.5),
  z_lambda = 0
)
inits_list <- list(make_inits(), make_inits(), make_inits())

# --- run MCMC (no DIC= arg; load dic module & monitor deviance) ---
set.seed(123)
jm <- run.jags(
  model     = model_string,
  data      = jags_data,
  inits     = inits_list,
  n.chains  = 3,
  burnin    = 2000,
  sample    = 4000,
  thin      = 2,
  monitor   = c("beta1","beta2","m1","m2","lambda","deviance"),
  modules   = c("glm","dic"),   # <- important for DIC
  method    = "parallel",
  summarise = FALSE
)

# --- print summary ---
print(jm)

# --- extract tidy summaries (means/SD/95% CI) ---
summ <- summary(jm)
keep <- grepl("^(beta1\\[|beta2\\[|m1$|m2$|lambda$)", rownames(summ))
cat("\n--- Posterior summary (means, SD, 95% CI) ---\n")
print(summ[keep, c("Mean","SD","Lower95","Upper95")])

# label betas by column names
cn1 <- colnames(X1); cn2 <- colnames(X2)
b1_rows <- grep("^beta1\\[", rownames(summ))
b2_rows <- grep("^beta2\\[", rownames(summ))
beta1_tab <- summ[b1_rows, c("Mean","SD","Lower95","Upper95")]; rownames(beta1_tab) <- cn1
beta2_tab <- summ[b2_rows, c("Mean","SD","Lower95","Upper95")]; rownames(beta2_tab) <- cn2
cat("\n--- Beta1 (docvis) ---\n"); print(beta1_tab)
cat("\n--- Beta2 (hospvis) ---\n"); print(beta2_tab)

# --- DIC: try from runjags; fall back if not present ---
cat("\n--- DIC ---\n")
dic_val <- jm$DIC
if (is.null(dic_val)) {
  # fallback from deviance draws: Dbar + pD (with pD ≈ var(dev)/2)
  # note: this approximation differs slightly from JAGS dic module's Dhat method
  dev_draws <- extract(jm, what = "deviance")  # matrix (iter x chains) or vector
  dev_vec <- as.numeric(dev_draws)
  Dbar <- mean(dev_vec, na.rm = TRUE)
  pD   <- var(dev_vec, na.rm = TRUE) / 2
  DIC  <- Dbar + pD
  cat(sprintf("DIC (fallback): %.3f  [Dbar=%.3f, pD≈%.3f]\n", DIC, Dbar, pD))
} else {
  print(dic_val)
}

