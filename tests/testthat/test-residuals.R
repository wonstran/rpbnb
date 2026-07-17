# Residual-diagnostics tests. Fast tier builds synthetic fits (no optimization);
# a slow-gated tier exercises real fits. See helper-slow.R::skip_slow.

# ---- fixtures --------------------------------------------------------------
make_bnb_resid_fixture <- function() {
  set.seed(11)
  n  <- 200
  x  <- rnorm(n)
  X  <- stats::model.matrix(~ x)
  b1 <- c(0.3, 0.2); b2 <- c(0.1, -0.1)
  mu1 <- as.vector(exp(X %*% b1)); mu2 <- as.vector(exp(X %*% b2))
  m1 <- 0.4; m2 <- 0.5
  y1 <- stats::rnbinom(n, size = 1 / m1, mu = mu1)
  y2 <- stats::rnbinom(n, size = 1 / m2, mu = mu2)
  coef <- c(b1, b2, log(m1), log(m2))
  names(coef) <- c("b1:(Intercept)", "b1:x", "b2:(Intercept)", "b2:x",
                   "log_m1", "log_m2")
  structure(list(coef = coef, mu1 = mu1, mu2 = mu2, Y1 = y1, Y2 = y2,
                 X1 = X, X2 = X, dependence = "famoye"), class = "bnb_fit")
}

test_that("bnb Pearson residual matches (y-mu)/sqrt(mu + m*mu^2)", {
  f  <- make_bnb_resid_fixture()
  pr <- residuals(f, type = "pearson", margin = "y1")
  m1 <- exp(f$coef[["log_m1"]])
  expect_equal(pr, (f$Y1 - f$mu1) / sqrt(f$mu1 + m1 * f$mu1^2), tolerance = 1e-12)
})

test_that("bnb deviance residual: sign matches y-mu, ~0 when y==mu, finite at y=0", {
  f  <- make_bnb_resid_fixture()
  dv <- residuals(f, type = "deviance", margin = "y1")
  expect_equal(sign(dv), sign(f$Y1 - f$mu1))
  expect_true(all(is.finite(dv)))
  # a point with y exactly equal to mu has ~0 deviance residual
  m1 <- exp(f$coef[["log_m1"]]); r1 <- 1 / m1
  y0 <- 3; mu0 <- 3
  d0 <- rpbnb:::.nb2_deviance_resid(y0, mu0, m1)
  expect_equal(d0, 0, tolerance = 1e-8)
})

test_that("bnb RQR lies in (qnorm(F(y-1)), qnorm(F(y))) and is seed-reproducible", {
  f   <- make_bnb_resid_fixture()
  rq1 <- residuals(f, type = "quantile", margin = "y1", seed = 42)
  rq2 <- residuals(f, type = "quantile", margin = "y1", seed = 42)
  expect_equal(rq1, rq2)                         # reproducible
  m1 <- exp(f$coef[["log_m1"]]); r1 <- 1 / m1
  Fhi <- stats::pnbinom(f$Y1,     size = r1, mu = f$mu1)
  Flo <- ifelse(f$Y1 > 0, stats::pnbinom(f$Y1 - 1, size = r1, mu = f$mu1), 0)
  expect_true(all(rq1 >= stats::qnorm(Flo) - 1e-9 & rq1 <= stats::qnorm(Fhi) + 1e-9))
  rq3 <- residuals(f, type = "quantile", margin = "y1", seed = 7)
  expect_false(isTRUE(all.equal(rq1, rq3)))      # different seed differs
})

test_that("bnb residuals(margin='both') returns a two-column data frame", {
  f <- make_bnb_resid_fixture()
  d <- residuals(f, type = "pearson", margin = "both")
  expect_s3_class(d, "data.frame")
  expect_named(d, c("y1", "y2"))
  expect_equal(nrow(d), length(f$Y1))
})

test_that("residuals(seed=) does not disturb the caller's RNG stream", {
  f <- make_bnb_resid_fixture()
  set.seed(99); a <- runif(1)
  set.seed(99); invisible(residuals(f, type = "quantile", margin = "both", seed = 5)); b <- runif(1)
  expect_equal(a, b)
})

make_rp_resid_fixture <- function(dist1 = "normal", R = 128L) {
  set.seed(21)
  n    <- 60
  slab <- rpbnb:::rand_dist_registry[[dist1]]$scale_label
  x    <- seq(-1, 1, length.out = n)
  X    <- stats::model.matrix(~ x)               # cols: (Intercept), x
  b1   <- c(0.2, 0.4); b2 <- c(0.1, -0.3)
  m1   <- 0.4; m2 <- 0.5; sd1 <- 0.5
  coef <- c(b1, b2, log(sd1), log(m1), log(m2), 0)
  names(coef) <- c("b1:(Intercept)", "b1:x", "b2:(Intercept)", "b2:x",
                   paste0(slab, "1:x"), "log_m1", "log_m2", "z_lambda")
  Z1  <- matrix((seq_len(R) - 0.5) / R, ncol = 1L) # uniform grid for the eq-1 random coef
  Z2  <- matrix(0, R, 0)
  xb1 <- as.vector(X %*% b1); xb2 <- as.vector(X %*% b2)
  dev1 <- rpbnb:::rand_realize(Z1, dist1, 1, b1[2], sd1)$dev
  mu1_mat <- vapply(seq_len(R),
                    function(r) pmin(exp(xb1 + X[, 2] * dev1[r, 1]), 1e15),
                    numeric(n))
  mu1 <- rowMeans(mu1_mat)
  mu2 <- exp(xb2)
  y1  <- stats::rnbinom(n, size = 1 / m1, mu = mu1)
  y2  <- stats::rnbinom(n, size = 1 / m2, mu = mu2)
  structure(list(
    coef = coef, rand_idx1 = 2L, rand_idx2 = integer(0),
    rp_meta = list(dist1 = dist1, dist2 = character(0),
                   sign1 = 1, sign2 = numeric(0), Z1 = Z1, Z2 = Z2),
    X1 = X, X2 = X, Y1 = y1, Y2 = y2, m1 = m1, m2 = m2, mu1 = mu1, mu2 = mu2,
    formula_1 = y1 ~ x, formula_2 = y2 ~ x), class = "rpbnb_fit")
}

test_that("rp mixture CDF corners match a brute-force draw average", {
  f  <- make_rp_resid_fixture("normal")
  cc <- rpbnb:::.rp_mixture_cdf(f, 1L)
  # brute force from the same fixture draws
  X <- f$X1; b1 <- f$coef[c("b1:(Intercept)", "b1:x")]; r1 <- 1 / f$m1
  Z1 <- f$rp_meta$Z1; sd1 <- exp(f$coef[["log_sd1:x"]])
  dev1 <- rpbnb:::rand_realize(Z1, "normal", 1, b1[2], sd1)$dev
  xb1 <- as.vector(X %*% b1)
  Fhi <- rowMeans(vapply(seq_len(nrow(Z1)),
    function(r) stats::pnbinom(f$Y1, size = r1, mu = pmin(exp(xb1 + X[, 2] * dev1[r, 1]), 1e15)),
    numeric(nrow(X))))
  expect_equal(cc$Fhi, Fhi, tolerance = 1e-12)
})

test_that("rp fully-fixed equation reduces to plain NB2 RQR", {
  f  <- make_rp_resid_fixture("normal")
  # equation 2 is fixed: mixture RQR must equal the NB2 RQR at mu2
  rq_rp <- residuals(f, type = "quantile", margin = "y2", seed = 3)
  r2 <- 1 / f$m2
  Fhi <- stats::pnbinom(f$Y2,     size = r2, mu = f$mu2)
  Flo <- ifelse(f$Y2 > 0, stats::pnbinom(f$Y2 - 1, size = r2, mu = f$mu2), 0)
  set.seed(3); rq_ref <- stats::qnorm(Flo + stats::runif(length(f$Y2)) * (Fhi - Flo))
  expect_equal(rq_rp, rq_ref, tolerance = 1e-12)
})

test_that("rp Pearson uses the mixture marginal variance (>= NB2-at-mean var)", {
  f  <- make_rp_resid_fixture("normal")
  pr <- residuals(f, type = "pearson", margin = "y1")
  v  <- rpbnb:::.rp_mixture_var(f, 1L)
  expect_equal(pr, (f$Y1 - f$mu1) / sqrt(v), tolerance = 1e-12)
  # mixture variance strictly exceeds NB2-at-the-mean where the coef is random
  expect_true(all(v >= f$mu1 + f$m1 * f$mu1^2 - 1e-9))
})

test_that("rp deviance residuals error with an explanatory message", {
  f <- make_rp_resid_fixture("normal")
  expect_error(residuals(f, type = "deviance", margin = "y1"), "not defined")
})

test_that("rp lognormal analytic-Inf rows give NA residuals with a warning", {
  f <- make_rp_resid_fixture("lognormal")   # eq-1 random coef lognormal, sign +1
  inf <- rpbnb:::.rp_inf_rows(f$X1, f$rand_idx1, f$rp_meta$dist1, f$rp_meta$sign1)
  skip_if(!any(inf), "fixture produced no analytic-Inf rows")
  rq <- suppressWarnings(residuals(f, type = "quantile", margin = "y1", seed = 1))
  expect_true(all(is.na(rq[inf])))
  expect_warning(residuals(f, type = "quantile", margin = "y1", seed = 1), "infinite")
})

test_that("rp lognormal analytic-Inf rows give NA pearson residuals with a warning", {
  f <- make_rp_resid_fixture("lognormal")   # eq-1 random coef lognormal, sign +1
  inf <- rpbnb:::.rp_inf_rows(f$X1, f$rand_idx1, f$rp_meta$dist1, f$rp_meta$sign1)
  skip_if(!any(inf), "fixture produced no analytic-Inf rows")
  pr <- suppressWarnings(residuals(f, type = "pearson", margin = "y1"))
  expect_true(all(is.na(pr[inf])))
  expect_warning(residuals(f, type = "pearson", margin = "y1"), "infinite")
})

test_that("plot.bnb_fit runs to a null device and restores par()", {
  f  <- make_bnb_resid_fixture()
  op <- par(no.readonly = TRUE)
  grDevices::pdf(NULL)
  on.exit({ grDevices::dev.off() }, add = TRUE)
  expect_error(plot(f, margin = "both", seed = 1), NA)   # NA = expect no error
  expect_equal(par("mfrow"), op$mfrow)                   # par restored
})

test_that("plot.rpbnb_fit runs to a null device without error", {
  f <- make_rp_resid_fixture("normal")
  grDevices::pdf(NULL)
  on.exit({ grDevices::dev.off() }, add = TRUE)
  expect_error(plot(f, margin = "y1", which = c(1, 2), seed = 1), NA)
})

test_that("plot 'which' subsets panels (single panel, no par change needed)", {
  f <- make_bnb_resid_fixture()
  grDevices::pdf(NULL)
  on.exit({ grDevices::dev.off() }, add = TRUE)
  expect_error(plot(f, margin = "y1", which = 2, seed = 1), NA)
})

test_that("plot.rpbnb_fit degrades gracefully when a margin's residuals are all NA", {
  f <- make_rp_resid_fixture("lognormal")
  # force every row into the analytic-Inf branch (strictly positive x, sign +1)
  f$X1[, "x"] <- seq(0.1, 2, length.out = nrow(f$X1))
  inf <- rpbnb:::.rp_inf_rows(f$X1, f$rand_idx1, f$rp_meta$dist1, f$rp_meta$sign1)
  expect_true(all(inf))   # self-check: this fixture really is all-NA for margin y1
  grDevices::pdf(NULL)
  on.exit({ grDevices::dev.off() }, add = TRUE)
  expect_error(suppressWarnings(plot(f, margin = "y1", seed = 1)), NA)
})
