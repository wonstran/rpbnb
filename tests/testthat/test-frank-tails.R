# Frank's cell probability in the far tail. The template returns it as a LOG
# (frank_log_cell_prob()) built from the margins' log masses, rather than as a
# linear probability the caller floors at 1e-300 before logging.
#
# The floor was not a guard against underflow noise: on the truck data's `m1`
# boundary LR test -- margin 1 forced Poisson -- observation 2230 (y = 125
# against mu = 0.193) carries a cell probability of 1.03e-300, three ulp above
# it. One inner-Newton step crosses it, and on the far side the objective is the
# constant -log(1e-300) = 690.776 rather than the likelihood. Laplace
# differentiates that kink twice: the second difference of -log p in log(mu1)
# came back -12,181 at h = 1e-2 and -94,836 at h = 1e-3 where the true curvature
# is mu1 = 0.193, which cost the inner Hessian its positive definiteness
# ("PD hess?: FALSE" -> "Newton drop out: Too many failed attempts" -> a NaN
# outer gradient).

test_that("a Frank cell probability far below 1e-300 is not clipped", {
  skip_on_cran()
  # One observation deep in a Poisson margin's tail (y = 300 against mu = 1,
  # log mass -1415.9) among nine ordinary ones. The whole point is the size of
  # the objective: with the old 1e-300 floor that tail cell could contribute at
  # most 690.776 nats no matter how improbable it really was.
  d <- data.frame(y1 = c(rep(1L, 9L), 300L), y2 = rep(0L, 10L))
  fit <- fit_rpbnb_tmb(y1 ~ 1, y2 ~ 1, data = d, dependence = copula("frank"),
                       poisson_1 = TRUE, draws = 2, seed = 1, keep = "full",
                       inference = "none", method = "sml",
                       control = rpbnb_tmb_control(iterlim = 1L,
                                                   print_level = 0L))
  obj <- fit$obj
  expect_identical(names(obj$par), c("beta1", "beta2", "log_m2", "z_dep"))
  par <- c(beta1 = 0, beta2 = 0, log_m2 = 0, z_dep = 2)  # mu1 = mu2 = m2 = 1

  # Reference: the same log-space identity, evaluated independently in R.
  th <- 35 * tanh(2 / 35)
  frank_M <- function(u, v) {
    (exp(-th * u) * (-expm1(-th * v)) +
       exp(-th * v) * (-expm1(-th * (1 - v)))) / (-expm1(-th))
  }
  log_cell <- function(y1, y2, mu1 = 1, mu2 = 1, r2 = 1) {
    a  <- ppois(y1, mu1); am <- if (y1 > 0) ppois(y1 - 1, mu1) else 0
    b  <- pnbinom(y2, size = r2, mu = mu2)
    bm <- if (y2 > 0) pnbinom(y2 - 1, size = r2, mu = mu2) else 0
    log_delta <- function(um, log_pmf) {
      log_x <- log(th) + log_pmf
      -th * um + if (exp(log_x) < 1e-8) log_x else log(-expm1(-exp(log_x)))
    }
    L <- log_delta(am, dpois(y1, mu1, log = TRUE)) +
      log_delta(bm, dnbinom(y2, size = r2, mu = mu2, log = TRUE)) -
      log(-expm1(-th)) - log(frank_M(am, b)) - log(frank_M(a, bm))
    (if (L < -30) L else log(-log1p(-exp(L)))) - log(th)
  }
  reference <- -(9 * log_cell(1L, 0L) + log_cell(300L, 0L))

  expect_equal(obj$fn(par), reference, tolerance = 1e-10)
  # And it is nowhere near what the floor would have allowed: the tail cell
  # contributes about 1424 nats, not the 690.776 the clip capped it at.
  floored <- -(9 * log_cell(1L, 0L)) + 690.7755
  expect_gt(obj$fn(par), floored + 700)
})

test_that("Frank agrees with the linear telescoped form where that form is sound", {
  skip_on_cran()
  # The rewrite must not move ordinary values. These counts are small enough
  # that the linear form the template used before -- dA built from the mass
  # itself, p returned and then logged -- is accurate to full precision, so it
  # is a valid reference here and only here.
  d <- data.frame(y1 = c(0L, 1L, 2L, 3L, 5L, 8L), y2 = c(0L, 0L, 1L, 2L, 1L, 4L))
  fit <- fit_rpbnb_tmb(y1 ~ 1, y2 ~ 1, data = d, dependence = copula("frank"),
                       draws = 2, seed = 1, keep = "full", inference = "none",
                       method = "sml",
                       control = rpbnb_tmb_control(iterlim = 1L,
                                                   print_level = 0L))
  obj <- fit$obj
  par <- c(beta1 = 0.4, beta2 = 0.1, log_m1 = -0.3, log_m2 = 0.2, z_dep = 1.5)
  expect_identical(names(obj$par), names(par))

  th <- 35 * tanh(1.5 / 35)
  r1 <- exp(0.3); r2 <- exp(-0.2)
  mu1 <- exp(0.4); mu2 <- exp(0.1)
  linear_cell <- function(y1, y2) {
    a  <- pnbinom(y1, size = r1, mu = mu1)
    am <- if (y1 > 0) pnbinom(y1 - 1, size = r1, mu = mu1) else 0
    pa <- dnbinom(y1, size = r1, mu = mu1)
    b  <- pnbinom(y2, size = r2, mu = mu2)
    bm <- if (y2 > 0) pnbinom(y2 - 1, size = r2, mu = mu2) else 0
    pb <- dnbinom(y2, size = r2, mu = mu2)
    D <- expm1(-th)
    M <- (1 + expm1(-th * am) * expm1(-th * b) / D) *
      (1 + expm1(-th * a) * expm1(-th * bm) / D)
    dA <- exp(-th * am) * expm1(-th * pa)
    dB <- exp(-th * bm) * expm1(-th * pb)
    -log1p(dA * dB / (D * M)) / th
  }
  reference <- -sum(mapply(function(u, v) log(linear_cell(u, v)), d$y1, d$y2))
  expect_equal(obj$fn(par), reference, tolerance = 1e-8)
})

test_that("a Laplace inner solve survives a Frank cell far below the floor", {
  skip_on_cran()
  # The failure this file exists for: method = "laplace", a Poisson margin, and
  # one observation whose cell probability sits under the old floor. The inner
  # Newton used to return NaN there, taking the outer gradient with it.
  set.seed(4)
  n <- 60
  x1 <- rnorm(n)
  d <- data.frame(y1 = rpois(n, exp(0.3 + 0.4 * x1)),
                  y2 = rpois(n, exp(0.2 - 0.2 * x1)), x1 = x1)
  d$y1[1L] <- 250L   # far into the tail of its own Poisson mean

  fit <- fit_rpbnb_tmb(y1 ~ x1, y2 ~ x1, data = d, random_1 = "x1",
                       dependence = copula("frank"), poisson_1 = TRUE,
                       method = "laplace", seed = 2, keep = "full",
                       inference = "none",
                       control = rpbnb_tmb_control(iterlim = 5L,
                                                   print_level = 0L,
                                                   n_cores = 1L))
  obj <- fit$obj
  # A strong Frank dependence is where the old clip bit; evaluate there rather
  # than wherever five nlminb iterations happened to stop.
  par <- fit$optimizer$par
  par[["z_dep"]] <- 3
  value <- obj$fn(par)
  expect_true(is.finite(value))
  expect_true(all(is.finite(obj$gr(par))))
  # What actually broke was the inner Hessian, not the value: a clipped cell
  # put curvature of the wrong sign and five orders of magnitude too large on
  # that observation's latent rows, so the matrix was neither finite nor
  # positive definite and TMB's inner Newton dropped out.
  hess <- obj$env$spHess(obj$env$last.par, random = TRUE)
  expect_false(anyNA(hess@x))
  expect_false(inherits(try(Matrix::Cholesky(hess), silent = TRUE),
                        "try-error"))

  # The latent is free to move mu1 toward the tail count, so the objective
  # here need not exceed the old floor. That the cell is not clipped is what
  # the two SML tests above pin; this one pins that Laplace can differentiate
  # it twice.
  expect_true(is.finite(obj$env$f(obj$env$last.par, order = 0)))
})
