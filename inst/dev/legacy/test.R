rm(list=ls())

source("C:\\Users\\wonst\\OneDrive\\Project\\MVNB\\Rcodes\\bnbr_faymore.R", 
       chdir = TRUE, 
       encoding = "UTF-8")

data = read.csv("C:\\Users\\wonst\\OneDrive\\Project\\MVNB\\data\\rwm1984_clean.csv")

f1 <- docvis ~ MarM + SinM + SinF + kids + outwork + postHS
f2 <- hospvis ~ MarM + SinM + SinF + kids + outwork + postHS

fit <- bnbr_faymore(
  data = data,
  f1   = f1,   # NB for Y1
  f2   = f2,   # NB for Y2
  start = NULL, method = "BFGS",
  use_analytic_grad = TRUE,
  control = list(iterlim = 200, reltol = 1e-8, printLevel = 2)
)

plot_ll_by_iteration(fit)
out <- bnbr_gof(fit)