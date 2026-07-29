# Tests for offset() support across the fitting paths and for the newdata
# prediction contract predict(fit) == predict(fit, training_data). Also covers
# the RP prediction column-remap (random coefficients located by name after a
# factor level is absent in newdata).

# ---- .prepare_bnb_data extracts equation-specific offsets ----

test_that(".prepare_bnb_data extracts per-equation offsets (0 when absent)", {
  d <- data.frame(y1 = c(1, 2, 3, 4), y2 = c(0, 1, 2, 3),
                  x = c(0.1, 0.2, 0.3, 0.4), e = c(1, 2, 5, 10))
  prep <- rpbnb:::.prepare_bnb_data(y1 ~ x + offset(log(e)), y2 ~ x, d)
  expect_equal(prep$off1, log(d$e))
  expect_equal(prep$off2, numeric(4))          # equation 2 has no offset -> zeros
})

# ---- independence path (the review's reproduced inconsistency) ----

test_that("independence offset: predict(fit) equals predict(fit, training_data)", {
  set.seed(1); n <- 200
  e <- sample(c(1, 10), n, replace = TRUE)
  d <- data.frame(y1 = rpois(n, 2 * e), y2 = rpois(n, 1.5 * e),
                  x = rnorm(n), e = e)
  fit <- suppressWarnings(fit_bnb(y1 ~ x + offset(log(e)),
                                  y2 ~ x + offset(log(e)),
                                  data = d, dependence = "independence"))
  p0 <- predict(fit)                 # cached fitted means (include offset)
  p1 <- predict(fit, newdata = d)    # explicit newdata -> must add offset too
  expect_equal(p0$mu1, p1$mu1, tolerance = 1e-8)
  expect_equal(p0$mu2, p1$mu2, tolerance = 1e-8)
  # And the ten-exposure rows are ~10x the one-exposure rows (offset is used).
  hi <- which(e == 10)[1]; lo <- which(e == 1)[1]
  expect_gt(p0$mu1[hi] / p0$mu1[lo], 5)
})

test_that("independence intercept-only offset recovers exp(b0)*exposure", {
  set.seed(2); n <- 4000
  e <- sample(c(1, 4), n, replace = TRUE)
  d <- data.frame(y1 = rpois(n, 3 * e), y2 = rpois(n, 3 * e), e = e)
  fit <- suppressWarnings(fit_bnb(y1 ~ 1 + offset(log(e)),
                                  y2 ~ 1 + offset(log(e)),
                                  data = d, dependence = "independence"))
  b0 <- coef(fit)[["b1:(Intercept)"]]
  # E[y1 | e] = exp(b0) * e; with true rate 3 per unit exposure, exp(b0) ~ 3.
  expect_equal(exp(b0), 3, tolerance = 0.1)
  p <- predict(fit)
  expect_equal(p$mu1, exp(b0) * e, tolerance = 1e-8)
})

# ---- famoye path ----

test_that("famoye offset enters the fit and prediction is newdata-consistent", {
  set.seed(3); n <- 400
  e <- sample(c(1, 3), n, replace = TRUE)
  d <- data.frame(y1 = rpois(n, 2 * e), y2 = rpois(n, 1.5 * e),
                  x = rnorm(n), e = e)
  fit_off <- suppressWarnings(fit_bnb(y1 ~ x + offset(log(e)),
                                      y2 ~ x + offset(log(e)),
                                      data = d, dependence = "famoye"))
  fit_no  <- suppressWarnings(fit_bnb(y1 ~ x, y2 ~ x, data = d,
                                      dependence = "famoye"))
  # The offset changes the model: intercepts must differ from the no-offset fit.
  expect_false(isTRUE(all.equal(unname(coef(fit_off)["b1:(Intercept)"]),
                                unname(coef(fit_no)["b1:(Intercept)"]))))
  # And it recovers the true per-exposure rate: E[y1|e] = exp(b0)*e with rate 2.
  expect_equal(exp(coef(fit_off)[["b1:(Intercept)"]]), 2, tolerance = 0.15)
  p0 <- predict(fit_off)
  p1 <- predict(fit_off, newdata = d)
  expect_equal(p0$mu1, p1$mu1, tolerance = 1e-8)
  expect_equal(p0$mu2, p1$mu2, tolerance = 1e-8)
})

# ---- fixed copula path ----

test_that("fixed copula offset: predict(fit) equals predict(fit, training_data)", {
  set.seed(6); n <- 300
  e <- sample(c(1, 4), n, replace = TRUE)
  d <- data.frame(y1 = rpois(n, 2 * e), y2 = rpois(n, 1.5 * e),
                  x = rnorm(n), e = e)
  fit <- suppressWarnings(fit_bnb(y1 ~ x + offset(log(e)),
                                  y2 ~ x + offset(log(e)),
                                  data = d, dependence = copula("frank")))
  p0 <- predict(fit)
  p1 <- predict(fit, newdata = d)
  expect_equal(p0$mu1, p1$mu1, tolerance = 1e-8)
  expect_equal(p0$mu2, p1$mu2, tolerance = 1e-8)
})

# ---- RP copula path (slow) ----

test_that("RP copula offset: predict(fit) equals predict(fit, training_data)", {
  skip_slow()
  set.seed(7); n <- 300
  e <- sample(c(1, 4), n, replace = TRUE)
  d <- data.frame(y1 = rpois(n, 2 * e), y2 = rpois(n, 1.5 * e),
                  x = rnorm(n), e = e)
  fit <- suppressWarnings(fit_rpbnb(y1 ~ x + offset(log(e)),
                                    y2 ~ x + offset(log(e)),
                                    data = d, random_1 = "x", draws = 40,
                                    dependence = copula("frank"),
                                    control = rpbnb_control(compute_se = FALSE)))
  p0 <- suppressWarnings(predict(fit))
  p1 <- suppressWarnings(predict(fit, newdata = d))
  expect_equal(p0$mu1, p1$mu1, tolerance = 1e-8)
  expect_equal(p0$mu2, p1$mu2, tolerance = 1e-8)
})

# ---- RP path (fast: small draws, no skip) ----

test_that("RP offset: predict(fit) equals predict(fit, training_data)", {
  set.seed(4); n <- 200
  e <- sample(c(1, 5), n, replace = TRUE)
  d <- data.frame(y1 = rpois(n, 2 * e), y2 = rpois(n, 1.5 * e),
                  x = rnorm(n), e = e)
  fit <- suppressWarnings(fit_rpbnb(y1 ~ x + offset(log(e)),
                                    y2 ~ x + offset(log(e)),
                                    data = d, random_1 = "x", draws = 40,
                                    control = rpbnb_control(compute_se = FALSE)))
  p0 <- suppressWarnings(predict(fit))
  p1 <- suppressWarnings(predict(fit, newdata = d))
  expect_equal(p0$mu1, p1$mu1, tolerance = 1e-8)
  expect_equal(p0$mu2, p1$mu2, tolerance = 1e-8)
})

# ---- RP prediction column remap by name (Finding P2) ----

test_that("RP predict works when a training factor level is absent in newdata", {
  set.seed(5); n <- 300
  g <- factor(sample(c("a", "b", "c"), n, replace = TRUE))
  x <- rnorm(n)
  d <- data.frame(y1 = rpois(n, exp(0.2 + 0.3 * x)),
                  y2 = rpois(n, exp(0.1 - 0.2 * x)),
                  g = g, x = x)
  # Training design columns: (Intercept), gb, gc, x -- the random coefficient is
  # on x, which sits AFTER the factor dummies (training position 4).
  fit <- suppressWarnings(fit_rpbnb(y1 ~ g + x, y2 ~ x, data = d, random_1 = "x",
                                    draws = 40,
                                    control = rpbnb_control(compute_se = FALSE)))
  # Newdata contains only levels a and b: a naive position-based lookup would
  # shift x from column 4 to column 3 and crash / mispredict. Name-based remap
  # plus the stored terms/xlevels rebuild the training columns.
  nd <- data.frame(g = factor(c("a", "b"), levels = c("a", "b")),
                   x = c(0.5, -0.5))
  pr <- suppressWarnings(predict(fit, newdata = nd))
  expect_equal(nrow(pr), 2L)
  expect_true(all(is.finite(pr$mu1)))
})

# ============================================================================
# Follow-up review (2026-07-21 17:01): offset must reach residuals, marginal
# effects, and the GOF null; the row mask must reject transformed infinities.
# ============================================================================

# ---- transformed infinities are rejected by the common-row mask ----

test_that("a transformation that yields -Inf (log(0)) drops that row from both eqns", {
  d <- data.frame(y1 = c(1, 2, 3, 4), y2 = c(2, 3, 4, 5),
                  x = c(1, 0, 2, 3), z = c(0.1, 0.2, 0.3, 0.4))
  prep <- rpbnb:::.prepare_bnb_data(y1 ~ log(x), y2 ~ z, d)
  expect_equal(length(prep$Y1), length(prep$Y2))
  expect_equal(unname(prep$Y1), c(1L, 3L, 4L))    # row 2 (log(0) = -Inf) dropped
  expect_true(all(is.finite(prep$X1)))            # no -Inf leaks into the design
  expect_true(all(is.finite(prep$X2)))
})

test_that("an overflow transformation (exp of a huge value = Inf) drops that row", {
  d <- data.frame(y1 = c(1, 2, 3), y2 = c(0, 1, 2),
                  z = c(0.1, 0.2, 0.3), x = c(1, 800, 2))
  prep <- rpbnb:::.prepare_bnb_data(y1 ~ z, y2 ~ exp(x), d)   # exp(800) = Inf
  expect_equal(unname(prep$Y2), c(0L, 2L))        # row 2 dropped from both eqns
  expect_equal(unname(prep$Y1), c(1L, 3L))
  expect_true(all(is.finite(prep$X2)))
})

test_that("a non-finite Inf response still errors (not silently dropped)", {
  d <- data.frame(y1 = c(1, Inf, 3), y2 = c(0, 1, 2), x = c(0.1, 0.2, 0.3))
  expect_error(fit_bnb(y1 ~ x, y2 ~ x, data = d, dependence = "famoye"),
               "non-finite")
})

# ---- RP residuals carry the fitted offset ----

test_that("RP residual machinery uses the fitted offset (means match predict)", {
  set.seed(11); n <- 200
  e <- sample(c(1, 5), n, replace = TRUE)
  d <- data.frame(y1 = rpois(n, 2 * e), y2 = rpois(n, 1.5 * e),
                  x = rnorm(n), e = e)
  fit <- suppressWarnings(fit_rpbnb(y1 ~ x + offset(log(e)),
                                    y2 ~ x + offset(log(e)),
                                    data = d, random_1 = "x", draws = 40,
                                    control = rpbnb_control(compute_se = FALSE)))
  # The per-draw mean matrix used by residuals must integrate to predict()'s mean.
  parts <- rpbnb:::.rp_margin_parts(fit, 1L)
  mu_resid <- rowMeans(rpbnb:::.rp_margin_mu_draws(parts))
  expect_equal(mu_resid, suppressWarnings(predict(fit))$mu1, tolerance = 1e-8)
  # Response residual is consistent with the offset-aware predict() mean.
  rr <- residuals(fit, type = "response", margin = "y1")
  expect_equal(rr, fit$Y1 - suppressWarnings(predict(fit))$mu1, tolerance = 1e-8)
  # Pearson residuals are finite and offset-aware (variance from mixture over
  # offset-inclusive per-draw means).
  pr <- residuals(fit, type = "pearson", margin = "y1")
  expect_true(all(is.finite(pr)))
})

test_that("RP copula residual reconstruction (NULL cached mean) is offset-aware", {
  skip_slow()
  set.seed(12); n <- 250
  e <- sample(c(1, 4), n, replace = TRUE)
  d <- data.frame(y1 = rpois(n, 2 * e), y2 = rpois(n, 1.5 * e),
                  x = rnorm(n), e = e)
  fit <- suppressWarnings(fit_rpbnb(y1 ~ x + offset(log(e)),
                                    y2 ~ x + offset(log(e)),
                                    data = d, random_1 = "x", draws = 40,
                                    dependence = copula("frank"),
                                    control = rpbnb_control(compute_se = FALSE)))
  # Copula RP stores mu1 = NULL, so .rp_fitted_mean reconstructs from the draws;
  # it must equal predict()'s offset-aware integrated mean.
  fm <- rpbnb:::.rp_fitted_mean(fit, 1L)
  expect_equal(fm, suppressWarnings(predict(fit))$mu1, tolerance = 1e-8)
})

# ---- marginal effects use the offset ----

test_that("fixed AME continuous effect uses the offset (matches beta * predict mean)", {
  set.seed(13); n <- 400
  e <- sample(c(1, 5), n, replace = TRUE)
  d <- data.frame(y1 = rpois(n, exp(0.3 + 0.2 * rnorm(n)) * e),
                  y2 = rpois(n, 1.5 * e), x = rnorm(n), e = e)
  d$y1 <- rpois(n, exp(0.3 + 0.2 * d$x) * e)     # ensure y1 depends on x
  fit <- suppressWarnings(fit_bnb(y1 ~ x + offset(log(e)), y2 ~ x + offset(log(e)),
                                  data = d, dependence = "independence"))
  me <- bnb_marginal_effects(fit, which = "y1", type = "AME", vars = "x",
                             print_output = FALSE)
  beta_x <- coef(fit)[["b1:x"]]
  # Documented continuous AME: mean(beta_x * mu), mu offset-aware from predict().
  expect_equal(me$Estimate[me$Name == "x"],
               mean(beta_x * predict(fit)$mu1), tolerance = 1e-6)
  # And it differs materially from the offset-free value the old code returned.
  mu_no_off <- as.vector(exp(fit$X1 %*% coef(fit)[grep("^b1:", names(coef(fit)))]))
  expect_false(isTRUE(all.equal(me$Estimate[me$Name == "x"],
                                mean(beta_x * mu_no_off))))
})

test_that("fixed MEM continuous effect uses the mean offset", {
  set.seed(14); n <- 400
  e <- sample(c(1, 5), n, replace = TRUE)
  d <- data.frame(x = rnorm(n), e = e)
  d$y1 <- rpois(n, exp(0.3 + 0.2 * d$x) * e); d$y2 <- rpois(n, 1.5 * e)
  fit <- suppressWarnings(fit_bnb(y1 ~ x + offset(log(e)), y2 ~ x + offset(log(e)),
                                  data = d, dependence = "independence"))
  me <- bnb_marginal_effects(fit, which = "y1", type = "MEM", vars = "x",
                             print_output = FALSE)
  beta <- coef(fit)[grep("^b1:", names(coef(fit)))]
  names(beta) <- sub("^b1:", "", names(beta))
  Xbar <- colMeans(fit$X1)
  mu_bar <- exp(sum(Xbar * beta) + mean(fit$predict_meta$off1))   # MEM at mean offset
  expect_equal(me$Estimate[me$Name == "x"], beta[["x"]] * mu_bar, tolerance = 1e-6)
})

test_that("RP marginal effects run and use the offset (smoke + magnitude)", {
  set.seed(15); n <- 200
  e <- sample(c(1, 5), n, replace = TRUE)
  d <- data.frame(x = rnorm(n), e = e)
  d$y1 <- rpois(n, exp(0.2 + 0.3 * d$x) * e); d$y2 <- rpois(n, 1.5 * e)
  fit <- suppressWarnings(fit_rpbnb(y1 ~ x + offset(log(e)), y2 ~ x + offset(log(e)),
                                    data = d, random_1 = "x", draws = 40,
                                    control = rpbnb_control(compute_se = FALSE)))
  me <- suppressWarnings(rpbnb_marginal_effects(fit, which = "y1", type = "AME",
                                                vars = "x", print_output = FALSE))
  # The absolute AME scales with the (offset-inflated) mean, so it exceeds the
  # bare coefficient in magnitude for these exposures.
  expect_true(is.finite(me$Estimate[me$Name == "x"]))
  expect_gt(abs(me$Estimate[me$Name == "x"]), abs(coef(fit)[["b1:x"]]))
})

# ---- bnb_gof() null retains the offset ----

test_that("bnb_gof null model carries the training offset", {
  set.seed(16); n <- 300
  e <- sample(c(1, 5), n, replace = TRUE)
  d <- data.frame(y1 = rpois(n, 2 * e), y2 = rpois(n, 1.5 * e),
                  x = rnorm(n), e = e)
  fit <- suppressWarnings(fit_bnb(y1 ~ x + offset(log(e)), y2 ~ x + offset(log(e)),
                                  data = d, dependence = "independence"))
  g <- suppressWarnings(bnb_gof(fit, print_output = FALSE))
  # The null must retain each margin's offset (previously it stored all zeros).
  expect_equal(g$null_fit$predict_meta$off1, fit$predict_meta$off1)
  expect_equal(g$null_fit$predict_meta$off2, fit$predict_meta$off2)
  # Null logLik matches direct intercept+offset univariate NB nulls. glm.nb's
  # own iteration/theta warnings on this reference fit are irrelevant here.
  o1 <- fit$predict_meta$off1; o2 <- fit$predict_meta$off2
  ll_ref <- suppressWarnings(
    as.numeric(logLik(MASS::glm.nb(fit$Y1 ~ 1 + offset(o1)))) +
    as.numeric(logLik(MASS::glm.nb(fit$Y2 ~ 1 + offset(o2)))))
  expect_equal(g$logLik_null, ll_ref, tolerance = 1e-4)
})

# ---- parallel RP diagnostic SEs use current (offset-aware) worker code -------

test_that("parallel RP diagnostic SEs match sequential on an offset fit", {
  skip_slow()
  # Alternating exposures so the offset materially changes the mean (and thus the
  # absolute marginal effect and its delta-method SE). If a worker ran a stale,
  # offset-free estimand, its SE would differ from the sequential value.
  set.seed(21); n <- 200
  e1 <- ifelse(seq_len(n) %% 2 == 0, 5, 1)
  e2 <- ifelse(seq_len(n) %% 2 == 0, 3, 1)
  d <- data.frame(x = rnorm(n), e1 = e1, e2 = e2)
  d$y1 <- rpois(n, exp(0.2 + 0.3 * d$x) * e1)
  d$y2 <- rpois(n, exp(0.1) * e2)
  fit <- suppressWarnings(fit_rpbnb(y1 ~ x + offset(log(e1)),
                                    y2 ~ x + offset(log(e2)),
                                    data = d, random_1 = "x", draws = 40))
  skip_if(any(!is.finite(fit$vcov)), "fit has no finite vcov")

  me1 <- suppressWarnings(rpbnb_marginal_effects(fit, which = "y1", type = "AME",
                            vars = "x", print_output = FALSE, n_cores = 1))
  me2 <- suppressWarnings(rpbnb_marginal_effects(fit, which = "y1", type = "AME",
                            vars = "x", print_output = FALSE, n_cores = 2))
  expect_equal(me1$Estimate, me2$Estimate, tolerance = 1e-8)
  expect_equal(me1$StdErr,   me2$StdErr,   tolerance = 1e-6)

  el1 <- suppressWarnings(rpbnb_elasticities(fit, which = "y1", type = "AME",
                            vars = "x", print_output = FALSE, n_cores = 1))
  el2 <- suppressWarnings(rpbnb_elasticities(fit, which = "y1", type = "AME",
                            vars = "x", print_output = FALSE, n_cores = 2))
  expect_equal(el1$Estimate, el2$Estimate, tolerance = 1e-8)
  expect_equal(el1$StdErr,   el2$StdErr,   tolerance = 1e-6)
})
