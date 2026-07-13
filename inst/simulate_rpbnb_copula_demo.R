#!/usr/bin/env Rscript
# Generate a copula-dependent RP-BNB dataset (Gaussian copula, rho = 0.6).
devtools::load_all(quiet = TRUE)
sim <- simulate_rpbnb_copula(
  n = 3000,
  beta1 = c("(Intercept)" = 0.4, x_age = 0.20, x_income = 0.15),
  beta2 = c("(Intercept)" = 0.3, x_age = -0.10, x_income = 0.25),
  random_1 = list(x_age = list(sd = 0.30)),
  random_2 = list(x_income = list(sd = 0.25)),
  dispersion = c(m1 = 0.5, m2 = 0.6),
  copula = copula("normal", par = 0.6), seed = 707)
cat(sprintf("n=%d  true rho=%.2f  Kendall tau=%.3f  Spearman(y1,y2)=%.3f\n",
            nrow(sim$data), sim$true$theta, sim$true$tau,
            cor(sim$data$y1, sim$data$y2, method = "spearman")))
dir.create("data", showWarnings = FALSE)
write.csv(sim$data, "data/simulated_rpbnb_copula.csv", row.names = FALSE)
saveRDS(sim$true, "data/simulated_rpbnb_copula_truth.rds")
cat("Saved data/simulated_rpbnb_copula.csv + truth rds\n")
