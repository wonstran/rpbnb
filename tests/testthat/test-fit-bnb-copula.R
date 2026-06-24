# Tests for copula dependence in fit_bnb()

# ---- copula() constructor ----

test_that("copula() returns an rpbnb_copula object with the correct family", {
  cop <- copula("frank")
  expect_s3_class(cop, "rpbnb_copula")
  expect_equal(cop$family, "frank")
})

test_that("copula() accepts normal and kimeldorf families", {
  expect_equal(copula("normal")$family, "normal")
  expect_equal(copula("kimeldorf")$family, "kimeldorf")
})

test_that("copula() rejects unknown families", {
  expect_error(copula("gumbel"), "frank")
})

test_that("copula() uses match.arg partial matching", {
  expect_equal(copula("fra")$family, "frank")
  expect_equal(copula("nor")$family, "normal")
})

# ---- Discrete-copula PMF sums to 1 ----

test_that("frank copula discrete PMF sums to approximately 1", {
  mu1 <- 2; m1 <- 0.5; r1 <- 1/m1
  mu2 <- 3; m2 <- 0.8; r2 <- 1/m2
  theta <- 2
  ymax <- 30
  ys <- 0:ymax
  total <- 0
  for (y1 in ys) for (y2 in ys) {
    a  <- pnbinom(y1, size=r1, mu=mu1); am <- if (y1>0) pnbinom(y1-1, size=r1, mu=mu1) else 0
    b  <- pnbinom(y2, size=r2, mu=mu2); bm <- if (y2>0) pnbinom(y2-1, size=r2, mu=mu2) else 0
    p <- rpbnb:::frank_cdf(a,b,theta) - rpbnb:::frank_cdf(am,b,theta) -
         rpbnb:::frank_cdf(a,bm,theta) + rpbnb:::frank_cdf(am,bm,theta)
    total <- total + p
  }
  expect_gt(total, 0.999)
  expect_lt(total, 1.001)
})

test_that("normal copula discrete PMF sums to approximately 1", {
  mu1 <- 2; m1 <- 0.5; r1 <- 1/m1
  mu2 <- 3; m2 <- 0.8; r2 <- 1/m2
  rho <- 0.4
  ymax <- 30
  ys <- 0:ymax
  total <- 0
  for (y1 in ys) for (y2 in ys) {
    a  <- pnbinom(y1, size=r1, mu=mu1); am <- if (y1>0) pnbinom(y1-1, size=r1, mu=mu1) else 0
    b  <- pnbinom(y2, size=r2, mu=mu2); bm <- if (y2>0) pnbinom(y2-1, size=r2, mu=mu2) else 0
    p <- rpbnb:::normal_cdf(a,b,rho) - rpbnb:::normal_cdf(am,b,rho) -
         rpbnb:::normal_cdf(a,bm,rho) + rpbnb:::normal_cdf(am,bm,rho)
    total <- total + p
  }
  expect_gt(total, 0.999)
  expect_lt(total, 1.001)
})

test_that("kimeldorf copula discrete PMF sums to approximately 1", {
  mu1 <- 2; m1 <- 0.5; r1 <- 1/m1
  mu2 <- 3; m2 <- 0.8; r2 <- 1/m2
  theta <- 1
  ymax <- 30
  ys <- 0:ymax
  total <- 0
  for (y1 in ys) for (y2 in ys) {
    a  <- pnbinom(y1, size=r1, mu=mu1); am <- if (y1>0) pnbinom(y1-1, size=r1, mu=mu1) else 0
    b  <- pnbinom(y2, size=r2, mu=mu2); bm <- if (y2>0) pnbinom(y2-1, size=r2, mu=mu2) else 0
    p <- rpbnb:::kimeldorf_cdf(a,b,theta) - rpbnb:::kimeldorf_cdf(am,b,theta) -
         rpbnb:::kimeldorf_cdf(a,bm,theta) + rpbnb:::kimeldorf_cdf(am,bm,theta)
    total <- total + p
  }
  expect_gt(total, 0.999)
  expect_lt(total, 1.001)
})

# ---- Kendall's tau ----

test_that("normal_tau matches known formula at rho = 0.5", {
  expect_equal(rpbnb:::normal_tau(0.5), 1/3, tolerance = 1e-10)
})

test_that("kimeldorf_tau matches known formula", {
  expect_equal(rpbnb:::kimeldorf_tau(2), 0.5, tolerance = 1e-10)
  expect_equal(rpbnb:::kimeldorf_tau(0), 0, tolerance = 1e-10)
})

test_that("frank_tau is 0 at theta=0, positive for theta>0, negative for theta<0", {
  expect_equal(rpbnb:::frank_tau(0), 0, tolerance = 1e-10)
  expect_gt(rpbnb:::frank_tau(2), 0)
  expect_lt(rpbnb:::frank_tau(-2), 0)
})

test_that("copula_tau_and_deriv returns finite values for all families", {
  r <- rpbnb:::copula_tau_and_deriv("frank", 1)
  expect_true(is.finite(r$tau)); expect_true(is.finite(r$dtau_dz))
  r2 <- rpbnb:::copula_tau_and_deriv("normal", 0.5)
  expect_true(is.finite(r2$tau)); expect_true(is.finite(r2$dtau_dz))
  r3 <- rpbnb:::copula_tau_and_deriv("kimeldorf", 0)
  expect_true(is.finite(r3$tau)); expect_true(is.finite(r3$dtau_dz))
})

# ---- Analytic gradient matches numDeriv ----

.make_grad_setup <- function(seed = 42) {
  set.seed(seed)
  n  <- 40
  X1 <- cbind(1, rnorm(n)); X2 <- cbind(1, rnorm(n))
  y1 <- rpois(n, 2);        y2 <- rpois(n, 3)
  list(X1=X1, X2=X2, y1=y1, y2=y2,
       par = c(0.3, 0.1, 0.2, -0.1, log(0.5), log(0.6), 0.5))
}

test_that("frank copula analytic gradient matches numDeriv", {
  s  <- .make_grad_setup()
  ll <- function(p) sum(rpbnb:::copula_loglik_vec(p, s$y1, s$y2, s$X1, s$X2, "frank"))
  num <- numDeriv::grad(ll, s$par)
  ana <- rpbnb:::copula_grad_vec(s$par, s$y1, s$y2, s$X1, s$X2, "frank")
  expect_equal(unname(ana), num, tolerance = 1e-4)
})

test_that("normal copula analytic gradient matches numDeriv", {
  s  <- .make_grad_setup(43)
  # z_theta=0.5 -> rho = tanh(0.5) ~ 0.46, safely interior
  ll <- function(p) sum(rpbnb:::copula_loglik_vec(p, s$y1, s$y2, s$X1, s$X2, "normal"))
  num <- numDeriv::grad(ll, s$par)
  ana <- rpbnb:::copula_grad_vec(s$par, s$y1, s$y2, s$X1, s$X2, "normal")
  expect_equal(unname(ana), num, tolerance = 1e-4)
})

test_that("kimeldorf copula analytic gradient matches numDeriv", {
  s  <- .make_grad_setup(44)
  # z_theta=0.5 -> theta=exp(0.5)~1.65, positive
  ll <- function(p) sum(rpbnb:::copula_loglik_vec(p, s$y1, s$y2, s$X1, s$X2, "kimeldorf"))
  num <- numDeriv::grad(ll, s$par)
  ana <- rpbnb:::copula_grad_vec(s$par, s$y1, s$y2, s$X1, s$X2, "kimeldorf")
  expect_equal(unname(ana), num, tolerance = 1e-4)
})

# ---- fit_bnb with copula() dispatches correctly ----

test_that("fit_bnb with copula() returns a bnb_fit with the copula family", {
  skip_if_not(file.exists(system.file("extdata", "rwm1984_clean.csv", package = "rpbnb")))
  d <- read.csv(system.file("extdata", "rwm1984_clean.csv", package = "rpbnb"))
  fit <- fit_bnb(docvis ~ outwork, hospvis ~ outwork, data = d,
                 dependence = copula("frank"))
  expect_s3_class(fit, "bnb_fit")
  expect_equal(fit$dependence, "frank")
  expect_true(is.finite(as.numeric(logLik(fit))))
  expect_true(all(is.finite(fit$se)))
  expect_true(!is.null(fit$cop_tau))
})

test_that("fit_bnb frank copula logLik > independence logLik on rwm1984", {
  skip_if_not(file.exists(system.file("extdata", "rwm1984_clean.csv", package = "rpbnb")))
  d <- read.csv(system.file("extdata", "rwm1984_clean.csv", package = "rpbnb"))
  fi <- fit_bnb(docvis ~ outwork, hospvis ~ outwork, data = d, dependence = "independence")
  ff <- fit_bnb(docvis ~ outwork, hospvis ~ outwork, data = d, dependence = copula("frank"))
  expect_gte(as.numeric(logLik(ff)) + 1e-6, as.numeric(logLik(fi)))
})

test_that("fit_bnb normal copula converges with finite SEs on rwm1984", {
  skip_if_not(file.exists(system.file("extdata", "rwm1984_clean.csv", package = "rpbnb")))
  d <- read.csv(system.file("extdata", "rwm1984_clean.csv", package = "rpbnb"))
  fit <- fit_bnb(docvis ~ outwork, hospvis ~ outwork, data = d,
                 dependence = copula("normal"))
  expect_true(fit$convergence$converged)
  expect_true(all(is.finite(fit$se)))
  expect_true(!is.null(fit$cop_tau))
})

test_that("fit_bnb kimeldorf copula converges with finite SEs on rwm1984", {
  skip_if_not(file.exists(system.file("extdata", "rwm1984_clean.csv", package = "rpbnb")))
  d <- read.csv(system.file("extdata", "rwm1984_clean.csv", package = "rpbnb"))
  fit <- fit_bnb(docvis ~ outwork, hospvis ~ outwork, data = d,
                 dependence = copula("kimeldorf"))
  expect_true(fit$convergence$converged)
  expect_true(all(is.finite(fit$se)))
  expect_true(!is.null(fit$cop_tau))
})
