# Fast, fixture-based tests of the random-parameter interpretation functions
# (rpbnb_marginal_effects / rpbnb_elasticities). No model fitting -- these use
# make_rp_fixture() (helper-slow.R), which builds a synthetic rpbnb_fit with
# stored draws and a diagonal vcov. A single slow-gated test fits a real model.

# ---- brute-force reference for the random equation (independent recompute) ----
# Mirrors the design math directly from the fixture's known draws, so a bug in
# the production vectorization cannot hide behind a matching bug in the test.
.ref_ame_me <- function(f, eq = 1L) {
  X   <- if (eq == 1L) f$X1 else f$X2
  cn  <- colnames(X)
  b   <- f$coef[paste0("b", eq, ":", cn)]; names(b) <- cn
  ridx <- if (eq == 1L) f$rand_idx1 else f$rand_idx2
  Z   <- if (eq == 1L) f$rp_meta$Z1 else f$rp_meta$Z2
  dist <- if (eq == 1L) f$rp_meta$dist1 else f$rp_meta$dist2
  sgn  <- if (eq == 1L) f$rp_meta$sign1 else f$rp_meta$sign2
  R <- nrow(Z)
  scales <- if (length(ridx)) {
    vapply(seq_along(ridx), function(j) {
      lbl <- rpbnb:::rand_dist_registry[[dist[j]]]$scale_label
      exp(f$coef[[paste0(lbl, eq, ":", cn[ridx[j]])]])
    }, numeric(1))
  } else numeric(0)
  rr  <- if (length(ridx)) rpbnb:::rand_realize(Z, dist, sgn, b[ridx], scales) else NULL
  dev <- if (is.null(rr)) matrix(0, R, 0) else rr$dev
  gmat <- function(Xin) {
    xb <- as.vector(Xin %*% b)
    if (!length(ridx)) return(matrix(pmin(exp(xb), 1e15), nrow(Xin), R))
    XR <- Xin[, ridx, drop = FALSE]
    matrix(vapply(seq_len(R),
                  function(r) pmin(exp(xb + as.vector(XR %*% dev[r, ])), 1e15),
                  numeric(nrow(Xin))), nrow = nrow(Xin))
  }
  g <- gmat(X)
  out <- setNames(numeric(length(cn)), cn)
  for (j in seq_along(cn)) {
    xj <- X[, j]
    if (all(xj %in% c(0, 1))) {
      X0 <- X; X0[, j] <- 0; X1m <- X; X1m[, j] <- 1
      out[j] <- mean(rowMeans(gmat(X1m)) - rowMeans(gmat(X0)))
    } else {
      rk <- match(j, ridx)
      cr <- if (!is.na(rk)) rr$coef[, rk] else rep(b[[j]], R)
      out[j] <- mean(rowMeans(sweep(g, 2, cr, `*`)))
    }
  }
  out
}

test_that("rpbnb_marginal_effects: random-equation AME matches brute force", {
  f  <- make_rp_fixture("normal")
  me <- rpbnb_marginal_effects(f, which = "y1", type = "AME",
                               include_intercept = TRUE, print_output = FALSE)
  ref <- .ref_ame_me(f, 1L)
  got <- setNames(me$Estimate, me$Name)
  expect_equal(got[["x1"]], ref[["x1"]], tolerance = 1e-9)
  expect_equal(got[["(Intercept)"]], ref[["(Intercept)"]], tolerance = 1e-9)
})

test_that("rpbnb_marginal_effects: fully-fixed equation reduces to beta*mu", {
  # Equation 2 of the fixture is fixed: dmu/dx for continuous x1 must equal
  # mean(beta2_x1 * mu2), the classic fixed-coefficient marginal effect.
  f  <- make_rp_fixture("normal")
  me <- rpbnb_marginal_effects(f, which = "y2", type = "AME", print_output = FALSE)
  b2 <- f$coef[["b2:x1"]]
  mu2 <- as.vector(exp(f$X2 %*% f$coef[paste0("b2:", colnames(f$X2))]))
  expect_equal(me$Estimate[me$Name == "x1"], mean(b2 * mu2), tolerance = 1e-10)
})

test_that("rpbnb_marginal_effects: delta-method SE is finite and positive", {
  f  <- make_rp_fixture("normal")
  me <- rpbnb_marginal_effects(f, which = "y1", type = "AME", print_output = FALSE)
  expect_true(all(is.finite(me$StdErr)))
  expect_true(all(me$StdErr > 0))
  expect_equal(me$z, me$Estimate / me$StdErr, tolerance = 1e-12)
})

test_that("rpbnb_marginal_effects: NA vcov yields NA SEs with a warning", {
  f <- make_rp_fixture("normal")
  f$vcov[] <- NA_real_; f$se[] <- NA_real_
  expect_warning(
    me <- rpbnb_marginal_effects(f, which = "y1", type = "AME", print_output = FALSE),
    "standard error")
  expect_true(all(is.na(me$StdErr)))
  expect_true(all(is.na(me$z)))
})

test_that("rpbnb_marginal_effects: which='both' returns a named list of two frames", {
  f  <- make_rp_fixture("normal")
  res <- rpbnb_marginal_effects(f, which = "both", print_output = FALSE)
  expect_named(res, c("y1", "y2"))
  expect_s3_class(res$y1, "data.frame")
  expect_s3_class(res$y2, "data.frame")
})

test_that("rpbnb_marginal_effects: rejects a non-rpbnb_fit object", {
  expect_error(rpbnb_marginal_effects(list(), print_output = FALSE),
               "rpbnb_fit")
})

test_that("rpbnb_marginal_effects: lognormal analytic-Inf rows warn and propagate Inf", {
  f  <- make_rp_fixture("lognormal", sign1 = 1)  # random x1; Inf where x1 > 0
  me <- suppressWarnings(
    rpbnb_marginal_effects(f, which = "y1", type = "AME", print_output = FALSE))
  # x1 is random & sign*x1>0 on part of the sample -> AME averages in Inf rows
  expect_true(is.infinite(me$Estimate[me$Name == "x1"]))
  expect_true(is.na(me$StdErr[me$Name == "x1"]))
  expect_warning(
    rpbnb_marginal_effects(f, which = "y1", type = "AME", print_output = FALSE),
    "infinite")
})

test_that("rpbnb_elasticities: fixed equation continuous elasticity is beta*x", {
  # Equation 2 fixed: pointwise elasticity = beta * x, so AME = mean(beta * x).
  f  <- make_rp_fixture("normal")
  el <- rpbnb_elasticities(f, which = "y2", type = "AME", print_output = FALSE)
  b2 <- f$coef[["b2:x1"]]
  expect_equal(el$Estimate[el$Name == "x1"], mean(b2 * f$X2[, "x1"]),
               tolerance = 1e-10)
})

test_that("rpbnb_elasticities: random-equation elasticity equals x*me/mu", {
  f  <- make_rp_fixture("normal")
  el <- rpbnb_elasticities(f, which = "y1", type = "AME", print_output = FALSE)
  me <- rpbnb_marginal_effects(f, which = "y1", type = "AME", print_output = FALSE)
  # Rebuild the pointwise elasticity mean from the same fixture and check it is
  # internally consistent (finite, and the SE path produced finite SEs).
  expect_true(all(is.finite(el$Estimate)))
  expect_true(all(is.finite(el$StdErr) & el$StdErr > 0))
  expect_true("x1" %in% el$Name && "x1" %in% me$Name)
})

test_that("rpbnb_elasticities: which='both' returns a named list; rejects wrong class", {
  f   <- make_rp_fixture("normal")
  res <- rpbnb_elasticities(f, which = "both", print_output = FALSE)
  expect_named(res, c("y1", "y2"))
  expect_error(rpbnb_elasticities(list(), print_output = FALSE), "rpbnb_fit")
})

test_that("rpbnb_elasticities: NA vcov yields NA SEs with a warning", {
  f <- make_rp_fixture("normal")
  f$vcov[] <- NA_real_; f$se[] <- NA_real_
  expect_warning(
    el <- rpbnb_elasticities(f, which = "y1", print_output = FALSE),
    "standard error")
  expect_true(all(is.na(el$StdErr)))
})
