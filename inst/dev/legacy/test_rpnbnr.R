rm(list=ls())

source("C:\\Users\\wonst\\OneDrive\\Project\\MVNB\\Rcodes\\rpbnbr_faymore.R", 
       chdir = TRUE, 
       encoding = "UTF-8")

csv_path = "C:\\Users\\wonst\\OneDrive\\Project\\MVNB\\data\\rwm1984_clean.csv"

# =========================================================
# Example 1: Your rwm1984_clean.csv (edit path & variable names)
# =========================================================
raw <- read.csv(csv_path)
f1 <- docvis ~ MarM + SinM + SinF + kids + outwork + postHS
f2 <- hospvis ~ MarM + SinM + SinF + kids + outwork + postHS

# choose random slopes by NAME (must be column names in the model matrix, incl. "(Intercept)" if desired)
rand1 <- c("kids")     # example: random slope for 'kids' in eq1
rand2 <- c("outwork")  # example: random slope for 'outwork' in eq2

fit <- rpbnbr_bfgs(raw, f1, f2,
                   rand_names1 = rand1, rand_names2 = rand2,
                   n_draws = 300, 
                   n_cores = max(1L, parallel::detectCores()-1L),
                   control = list(iterlim = 300, reltol = 1e-8, printLevel = 2))

plot_ll_evals(fit)
plot_ll_iterations(fit)