test_that("RP-BNB simulated-LL analytic gradient matches numDeriv", {
  set.seed(11)
  n  <- 30
  X1 <- cbind(1, rnorm(n)); X2 <- cbind(1, rnorm(n))
  y1 <- rpois(n, 2); y2 <- rpois(n, 2)
  XR1 <- X1[, 2, drop = FALSE]; XR2 <- X2[, 2, drop = FALSE]
  rand_idx1 <- 2L; rand_idx2 <- 2L
  Z <- rpbnb:::halton_normal(60, 2, burn = 50)
  Z1 <- Z[, 1, drop = FALSE]; Z2 <- Z[, 2, drop = FALSE]
  par <- c(0.2, 0.1, 0.15, -0.1, log(0.3), log(0.3), log(0.4), log(0.4), 0.1)

  # The analytic gradient treats the global lambda bounds (max/min of the
  # per-draw Sarmanov bounds) as FROZEN -- exactly as the legacy estimator and
  # the frozen-bounds Hessian objective do. Differentiating the moving-bounds
  # objective therefore differs by the omitted d(bound)/d(theta) term. The
  # correct reference for this gradient is the frozen-bounds objective, with
  # bounds pinned at `par`.
  bnds <- local({
    m1 <- exp(par[7]); m2 <- exp(par[8])
    sd1 <- exp(par[5]); sd2 <- exp(par[6])
    xb1 <- as.vector(X1 %*% par[1:2]); xb2 <- as.vector(X2 %*% par[3:4])
    Z1sd <- sweep(Z1, 2, sd1, `*`); Z2sd <- sweep(Z2, 2, sd2, `*`)
    lo <- -Inf; hi <- Inf
    for (r in seq_len(nrow(Z1))) {
      mu1 <- exp(xb1 + as.vector(XR1 %*% Z1sd[r, ]))
      mu2 <- exp(xb2 + as.vector(XR2 %*% Z2sd[r, ]))
      b <- rpbnb:::lambda_bounds_vec(rpbnb:::c_val(mu1, m1),
                                     rpbnb:::c_val(mu2, m2))
      lo <- max(lo, b[1]); hi <- min(hi, b[2])
    }
    c(lo, hi)
  })

  obj <- function(p) rpbnb:::bnbr_rp_ll_fixed_bounds(
    p, y1, y2, X1, X2, XR1, XR2, rand_idx1, rand_idx2, Z1, Z2, bnds[1], bnds[2])
  v   <- rpbnb:::bnbr_rp_ll_and_grad(par, y1, y2, X1, X2, XR1, XR2,
                                     rand_idx1, rand_idx2, Z1, Z2)
  ana <- attr(v, "gradient")
  num <- numDeriv::grad(obj, par)
  expect_equal(ana, num, tolerance = 1e-3)
})

test_that("fit_rpbnb recovers simulated parameters (small case)", {
  skip_on_cran()
  sim <- simulate_rpbnb(
    n = 1500,
    beta1 = c("(Intercept)" = 0.2, x1 = 0.4),
    beta2 = c("(Intercept)" = 0.1, x1 = -0.3),
    random_1 = list(x1 = list(sd = 0.5)),
    random_2 = NULL,
    dispersion = c(m1 = 0.4, m2 = 0.5),
    lambda = 0,
    seed = 99
  )
  fit <- fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim$data,
                   random_1 = "x1", draws = 200, seed = 123,
                   control = rpbnb_control(compute_se = FALSE))
  expect_s3_class(fit, "rpbnb_fit")
  cf <- coef(fit)
  # Absolute closeness (the small-magnitude intercept makes relative
  # tolerance misleading; the estimate is ~0.034 from truth).
  expect_lt(abs(unname(cf[["b1:(Intercept)"]]) - 0.2), 0.15)
  expect_lt(abs(unname(cf[["b1:x1"]])          - 0.4), 0.15)
  # coef is estimation-scale: sd recovered via exp(log_sd)
  sd_hat <- exp(unname(cf[["log_sd1:x1"]]))
  expect_lt(abs(sd_hat - 0.5), 0.25)
  # natural-scale m stored as a field
  expect_lt(abs(unname(fit$m1) - 0.4), 0.2)
})

test_that("fit_rpbnb with sd->0 approaches fixed-parameter BNB", {
  skip_on_cran()
  sim <- simulate_rpbnb(
    n = 800,
    beta1 = c("(Intercept)" = 0.3, x1 = 0.2),
    beta2 = c("(Intercept)" = 0.2, x1 = 0.1),
    random_1 = list(x1 = list(sd = 1e-4)),
    dispersion = c(m1 = 0.4, m2 = 0.4), lambda = 0, seed = 5
  )
  rp <- fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim$data, random_1 = "x1",
                  draws = 150, seed = 1, control = rpbnb_control(compute_se = FALSE))
  bn <- fit_bnb(y1 ~ x1, y2 ~ x1, data = sim$data, dependence = "famoye")
  expect_equal(unname(coef(rp)[["b1:x1"]]),
               unname(coef(bn)[["b1:x1"]]), tolerance = 0.1)
})

test_that("fit_rpbnb errors on unknown random name", {
  sim <- simulate_rpbnb(n = 100, beta1 = c("(Intercept)" = 0.2, x1 = 0.3),
                        beta2 = c("(Intercept)" = 0.1, x1 = 0.1),
                        dispersion = c(m1 = 0.4, m2 = 0.4), seed = 1)
  expect_error(
    fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim$data, random_1 = "nope",
              draws = 50, control = rpbnb_control(compute_se = FALSE)),
    "nope"
  )
})

test_that("fit_rpbnb compute_se=TRUE yields a finite covariance and SEs", {
  skip_on_cran()
  sim <- simulate_rpbnb(n = 400,
    beta1 = c("(Intercept)" = 0.2, x1 = 0.3),
    beta2 = c("(Intercept)" = 0.1, x1 = -0.2),
    random_1 = list(x1 = list(sd = 0.4)),
    dispersion = c(m1 = 0.4, m2 = 0.4), lambda = 0, seed = 7)
  fit <- fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim$data, random_1 = "x1",
                   draws = 80, seed = 3,
                   control = rpbnb_control(compute_se = TRUE, draws_hessian = 40))
  V <- vcov(fit)
  expect_true(is.matrix(V))
  expect_equal(nrow(V), length(coef(fit)))
  expect_true(all(is.finite(fit$se)))
  expect_true(all(diag(V) >= 0))
})

test_that("fit_rpbnb parallel path matches sequential and respects workers", {
  skip_on_cran()
  skip_if_not_installed("parallel")
  sim <- simulate_rpbnb(n = 300,
    beta1 = c("(Intercept)" = 0.2, x1 = 0.3),
    beta2 = c("(Intercept)" = 0.1, x1 = -0.2),
    random_1 = list(x1 = list(sd = 0.4)),
    dispersion = c(m1 = 0.4, m2 = 0.4), lambda = 0, seed = 8)
  seq_fit <- fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim$data, random_1 = "x1",
                       draws = 60, seed = 2,
                       control = rpbnb_control(compute_se = FALSE, n_cores = 1))
  par_fit <- fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim$data, random_1 = "x1",
                       draws = 60, seed = 2,
                       control = rpbnb_control(compute_se = FALSE, n_cores = 2))
  # Same draws + seed => statistically identical estimates (parallel only splits the draw loop)
  expect_equal(unname(coef(seq_fit)), unname(coef(par_fit)), tolerance = 1e-6)
})
