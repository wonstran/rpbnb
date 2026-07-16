make_diag_fit <- function() {
  set.seed(31)
  d <- data.frame(x = rnorm(500), z = rbinom(500, 1, 0.4))
  d$y1 <- rnbinom(500, size = 2, mu = exp(0.3 + 0.2 * d$x + 0.1 * d$z))
  d$y2 <- rnbinom(500, size = 2, mu = exp(0.1 - 0.1 * d$x))
  fit_bnb(y1 ~ x + z, y2 ~ x, data = d, dependence = "famoye")
}

test_that("bnb_gof returns finite AIC/BIC and pseudo-R2 in [0,1]", {
  g <- bnb_gof(make_diag_fit(), print_output = FALSE)
  expect_true(is.finite(g$AIC) && is.finite(g$BIC))
  expect_true(all(g$pseudoR2 >= 0 & g$pseudoR2 <= 1, na.rm = TRUE))
})

test_that("bnb_gof returns NA pseudo-R2 with a warning when the null model fails", {
  # A degenerate (all-zero) response makes the intercept-only glm.nb null fail;
  # bnb_gof should warn and return NA pseudo-R2 rather than aborting.
  fake <- structure(
    list(logLik = -50, npar = 4L, nobs = 30L, AIC = 108, BIC = 114,
         dependence = "independence",
         Y1 = rep(0L, 30), Y2 = rpois(30, 2)),
    class = "bnb_fit")
  expect_warning(g <- bnb_gof(fake, print_output = FALSE), "[Nn]ull model")
  expect_true(all(is.na(g$pseudoR2)))
  expect_equal(g$AIC, 108)              # full-model metrics still returned
})

test_that("bnb_gof fits the intercept-only null for a copula fit", {
  set.seed(7)
  d <- data.frame(x = rnorm(300))
  d$y1 <- rnbinom(300, size = 2, mu = exp(0.3 + 0.2 * d$x))
  d$y2 <- rnbinom(300, size = 2, mu = exp(0.1 - 0.1 * d$x))
  fit <- fit_bnb(y1 ~ x, y2 ~ x, data = d, dependence = copula("normal"))
  g <- bnb_gof(fit, print_output = FALSE)
  # Pre-fix, fit$dependence = "normal" was passed to fit_bnb() and match.arg
  # rejected it, so the null failed and every pseudo-R^2 was NA.
  expect_false(is.null(g$null_fit))
  expect_true(is.finite(g$logLik_null))
  expect_true(any(is.finite(g$pseudoR2)))
})

test_that(".null_model_loglik requires a finite log-likelihood AND convergence", {
  ok  <- structure(list(logLik = -123.4,
                        convergence = list(converged = TRUE)), class = "bnb_fit")
  expect_equal(rpbnb:::.null_model_loglik(ok), -123.4)

  # finite logLik but the optimizer did not converge -> NA + warning
  bad <- structure(list(logLik = -123.4,
                        convergence = list(converged = FALSE)), class = "bnb_fit")
  expect_warning(v <- rpbnb:::.null_model_loglik(bad), "converge")
  expect_true(is.na(v))

  # NULL (failed) fit -> NA, no error
  expect_true(is.na(rpbnb:::.null_model_loglik(NULL)))
})

test_that("bnb_gof returns raw (unclamped) pseudo-R2 when the full model is worse than null", {
  set.seed(11)
  fake <- structure(
    list(logLik = -1e5, npar = 4L, nobs = 200L, AIC = 2e5, BIC = 2e5,
         dependence = "independence", cop_family = NULL,
         Y1 = rpois(200, 2), Y2 = rpois(200, 2)),
    class = "bnb_fit")
  # suppressWarnings: the synthetic intercept-only null trips glm.nb's iteration
  # limit on this fake data; irrelevant to the clamp behavior under test.
  g <- suppressWarnings(bnb_gof(fake, print_output = FALSE))
  # A deliberately terrible full logLik must yield a negative McFadden R^2,
  # not a value clamped to 0.
  expect_true(is.finite(g$pseudoR2[["McFadden"]]))
  expect_lt(g$pseudoR2[["McFadden"]], 0)
})

test_that("bnb_marginal_effects returns a row per requested variable", {
  me <- bnb_marginal_effects(make_diag_fit(), which = "y1", type = "AME",
                             print_output = FALSE)
  expect_true(all(c("x", "z") %in% me$Name))
  expect_true(all(is.finite(me$Estimate)))
})

test_that("bnb_elasticities runs for both margins", {
  el <- bnb_elasticities(make_diag_fit(), which = "both", type = "AME",
                         print_output = FALSE)
  expect_true(is.list(el) && all(c("y1", "y2") %in% names(el)))
})

test_that("bnb_marginal_effects AME and SE match an independent delta-method computation", {
  fit <- make_diag_fit()
  me <- bnb_marginal_effects(fit, which = "y1", type = "AME", print_output = FALSE)

  # Independent recomputation for the continuous variable 'x'
  X1 <- fit$X1
  p1 <- ncol(X1)
  beta1 <- fit$coef[grep("^b1:", names(fit$coef))]
  names(beta1) <- sub("^b1:", "", names(beta1))
  Vb1 <- fit$vcov[1:p1, 1:p1, drop = FALSE]
  mu  <- as.vector(exp(X1 %*% beta1))
  n   <- nrow(X1)
  j   <- which(colnames(X1) == "x")

  ame_x <- mean(beta1[["x"]] * mu)
  Xmubar <- as.numeric(t(X1) %*% (mu / n))   # E[mu * x_k]
  g <- beta1[["x"]] * Xmubar
  g[j] <- g[j] + mean(mu)
  se_x <- sqrt(as.numeric(t(g) %*% Vb1 %*% g))

  row_x <- me[me$Name == "x", ]
  expect_equal(row_x$Estimate, ame_x, tolerance = 1e-8)
  expect_equal(row_x$StdErr,   se_x,  tolerance = 1e-8)
})
