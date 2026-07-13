#!/usr/bin/env Rscript
# Fit the Gaussian-copula RP-BNB dataset and compare to truth.
devtools::load_all(quiet = TRUE)
data  <- read.csv("data/simulated_rpbnb_copula.csv")
truth <- readRDS("data/simulated_rpbnb_copula_truth.rds")
fit <- fit_rpbnb(y1 ~ x_age + x_income, y2 ~ x_age + x_income, data = data,
                 random_1 = "x_age", random_2 = "x_income",
                 dependence = copula("normal"),
                 draws = 200, seed = 20240712,
                 # compute_se = FALSE for speed; set compute_se = TRUE for standard errors (numeric Hessian, slower).
                 control = rpbnb_control(print_level = 1, compute_se = FALSE))
print(fit)
rho_hat <- tanh(fit$coef[["z_theta"]])
cat(sprintf("\nCopula rho: true %.3f  estimated %.3f\n", truth$theta, rho_hat))
