# Gate for expensive end-to-end tests (real model fits). The nominal fast suite
# skips these; the slow tier runs them with RPBNB_RUN_SLOW=1. Prediction *logic*
# is covered fast by fixture-based unit tests (test-predict-unit.R) that build a
# synthetic rpbnb_fit and never optimize.
skip_slow <- function() {
  # Only recognized truthy tokens enable the slow tier; "0", "false", "", etc.
  # skip, so a conventional false value does not accidentally launch a long run.
  run <- tolower(Sys.getenv("RPBNB_RUN_SLOW")) %in% c("1", "true", "yes", "on")
  testthat::skip_if(!run, "slow end-to-end test (set RPBNB_RUN_SLOW=1 to run)")
}

# Build a synthetic rpbnb_fit with stored draws, for testing predict() semantics
# without any optimization. Equation 1 carries one random coefficient on `x1`;
# equation 2 is fixed. `dist1`/`sign1` select the random-coefficient family.
make_rp_fixture <- function(dist1 = "uniform", sign1 = 1, R = 64L) {
  slab <- rpbnb:::rand_dist_registry[[dist1]]$scale_label
  coef <- c(0.3, 0.4, 0.1, -0.2, log(0.4), log(0.5), log(0.5), 0)
  names(coef) <- c("b1:(Intercept)", "b1:x1", "b2:(Intercept)", "b2:x1",
                   paste0(slab, "1:x1"), "log_m1", "log_m2", "z_lambda")
  Z1 <- matrix((seq_len(R) - 0.5) / R, ncol = 1L)   # deterministic uniform grid
  xtrain <- data.frame(x1 = seq(-1, 1, length.out = 10))
  X1 <- stats::model.matrix(~ x1, xtrain)
  structure(list(
    coef = coef,
    rand_idx1 = 2L, rand_idx2 = integer(0),
    rp_meta = list(dist1 = dist1, dist2 = character(0),
                   sign1 = sign1, sign2 = numeric(0),
                   Z1 = Z1, Z2 = matrix(0, R, 0)),
    formula_1 = y1 ~ x1, formula_2 = y2 ~ x1,
    X1 = X1, X2 = X1, mu1 = NULL, mu2 = NULL, draws = R,
    xtrain = xtrain
  ), class = "rpbnb_fit")
}
