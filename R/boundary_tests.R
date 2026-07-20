# Merged boundary likelihood-ratio tests for an rpbnb_fit: the random-coefficient
# SDs (sd1:*, sd2:*) and the NB2 dispersions (m1, m2). Each is a positive
# parameter whose null sits on the boundary of the parameter space, so the
# natural-scale summary reports no Wald z/p; the valid test is a boundary-
# corrected LR test against a nested restricted fit (lr_test(boundary = TRUE)).
# This helper builds and runs every such restricted fit and merges the results.

# Reconstruct a fit_rpbnb random spec (named list of list(dist, sign)) from the
# per-equation names/dist/sign stored on a fit. Returns NULL for an all-fixed
# equation. The result round-trips through parse_rand_spec().
.build_rand_spec <- function(names, dist, sign) {
  if (!length(names)) return(NULL)
  stats::setNames(
    lapply(seq_along(names), function(j) {
      # parse_rand_spec() only accepts `sign` for lognormal; omit it otherwise
      # (it defaults to 1) so the reconstructed spec round-trips cleanly.
      if (identical(dist[j], "lognormal")) list(dist = dist[j], sign = sign[j])
      else list(dist = dist[j])
    }),
    names)
}

# Natural-scale SD label for one random coefficient, matching the summary table
# (log_sd -> sd, log_w -> w, log_s -> s), e.g. .sd_label("normal", 1, "hhninc")
# = "sd1:hhninc".
.sd_label <- function(dist, eq, name) {
  pfx <- sub("^log_", "", rand_dist_registry[[dist]]$scale_label)
  paste0(pfx, eq, ":", name)
}

#' Boundary-corrected LR tests for all boundary parameters of an rpbnb_fit
#'
#' Runs a likelihood-ratio test for every random-coefficient standard deviation
#' (`sd1:*`, `sd2:*`) and NB2 dispersion (`m1`, `m2`) of a fitted
#' random-parameter bivariate NB model, and merges them into one table. These
#' are the parameters whose null lies on the boundary of the parameter space
#' (SD = 0, or dispersion `m = 0` = Poisson), for which the natural-scale summary
#' reports no Wald `z`/`p`; each test uses the 50:50 chi-square boundary
#' correction of [lr_test()] (`boundary = TRUE`).
#'
#' Each test refits a properly nested restricted model. With **multiple random
#' coefficients in an equation, each SD is tested individually** (a 1-df
#' restriction). The restricted fit keeps the full random specification but sets
#' the tested coefficient's draw column to the distribution median (`u = 0.5`),
#' which zeroes that coefficient's per-draw *deviation* exactly for every
#' supported distribution (`u = 0.5` maps to base 0 for normal/lognormal/
#' triangular, and to the centered value `2 * 0.5 - 1 = 0` for uniform). The
#' coefficient therefore collapses to its SD-zero null -- the ordinary fixed
#' coefficient `b` for normal/uniform/triangular, and `sign * exp(b)` for
#' lognormal -- on every draw, independent of the covariate scale. Its (now
#' inert) log-scale is pinned only to drop it from the free-parameter count. Every
#' other draw column is the full model's exact stored draw, so the two simulated
#' log-likelihoods are compared on common random numbers, and each restricted fit
#' is **warm-started from the full fit's coefficients** so the start-sensitive
#' simulated objective does not settle at an inferior optimum. A restricted fit
#' that fails to converge yields `NA` inference (with a warning) rather than a
#' p-value, and a non-converged full `fit` is rejected.
#'
#' @param fit A famoye `rpbnb_fit` (the full model). Copula fits are not
#'   supported (Poisson-limit margins are unavailable there).
#' @param data The data frame the model was fit on. Required -- the fit object
#'   does not store it, and every restricted model is refit on the same data.
#' @param control An [rpbnb_control()] for the restricted refits. Defaults to
#'   `compute_se = FALSE` (the LR test needs only `logLik` and the degrees of
#'   freedom). `draws`/`draw_type`/`seed` are taken from `fit`, not `control`.
#' @param which Which boundary parameters to test: `"sd"`, `"dispersion"`, or
#'   both (the default).
#' @return An object of class `rpbnb_boundary_tests`: a data frame with columns
#'   `Parameter`, `LR`, `df`, `p.value`, `Signif` (one row per boundary
#'   parameter), and a `print` method.
#' @seealso [lr_test()], [fit_rpbnb()]
#' @export
#' @examples
#' \donttest{
#' sim <- simulate_rpbnb(n = 600,
#'   beta1 = c("(Intercept)" = 0.2, x1 = 0.4),
#'   beta2 = c("(Intercept)" = 0.1, x1 = -0.3),
#'   random_1 = list(x1 = list(sd = 0.5)),
#'   dispersion = c(m1 = 0.4, m2 = 0.5), seed = 1)
#' fit <- fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim$data, random_1 = "x1",
#'                  draws = 200, seed = 1)
#' rpbnb_boundary_tests(fit, sim$data)
#' }
rpbnb_boundary_tests <- function(fit, data,
                                 control = rpbnb_control(compute_se = FALSE),
                                 which = c("sd", "dispersion")) {
  if (!inherits(fit, "rpbnb_fit")) {
    stop("`fit` must be an rpbnb_fit (from fit_rpbnb()).", call. = FALSE)
  }
  if (!is.null(fit$cop_family)) {
    stop("rpbnb_boundary_tests() supports famoye fits only; copula fits are ",
         "not supported (Poisson-limit margins are unavailable there).",
         call. = FALSE)
  }
  if (!is.data.frame(data)) stop("`data` must be a data frame.", call. = FALSE)
  which <- match.arg(which, c("sd", "dispersion"), several.ok = TRUE)

  # Convergence is a hard precondition: an LR test compares two maximized
  # log-likelihoods, so a non-converged full fit invalidates every comparison.
  if (!isTRUE(fit$convergence$converged)) {
    stop("`fit` (the full model) did not converge (maxLik code ",
         fit$convergence$code, ": ", fit$convergence$message,
         "). Refit it to convergence before running boundary tests.",
         call. = FALSE)
  }

  # Full random specs, reconstructed from the fit.
  names1 <- colnames(fit$X1)[fit$rand_idx1]
  names2 <- colnames(fit$X2)[fit$rand_idx2]
  dist1  <- fit$rp_meta$dist1; sign1 <- fit$rp_meta$sign1
  dist2  <- fit$rp_meta$dist2; sign2 <- fit$rp_meta$sign2
  full1  <- .build_rand_spec(names1, dist1, sign1)
  full2  <- .build_rand_spec(names2, dist2, sign2)

  # Every restricted fit keeps the full spec, reuses the full model's stored
  # optimization draws (common random numbers), and warm-starts from the full
  # fit's coefficients so the start-sensitive Famoye objective does not settle at
  # an inferior local optimum (which could inflate or clamp the LR statistic).
  full_draws <- list(Z1 = fit$rp_meta$Z1, Z2 = fit$rp_meta$Z2)
  refit <- function(poisson_1 = FALSE, poisson_2 = FALSE, fixed = NULL,
                    opt_draws = full_draws) {
    fit_rpbnb(fit$formula_1, fit$formula_2, data = data,
              random_1 = full1, random_2 = full2,
              draws = fit$draws, draw_type = fit$draw_type, seed = fit$seed,
              start = fit$coef, control = control, dependence = "famoye",
              poisson_1 = poisson_1, poisson_2 = poisson_2,
              .fixed = fixed, .opt_draws = opt_draws)
  }
  # Optimization-parameterization name of a random coefficient's log-scale, e.g.
  # .scale_par("normal", 1, "x1") = "log_sd1:x1" (matches fit_rpbnb par_names).
  .scale_par <- function(dist, eq, name) {
    paste0(rand_dist_registry[[dist]]$scale_label, eq, ":", name)
  }
  # Exact scale-zero null for one random column: copy the full draw matrices and
  # set the tested column to the distribution median (u = 0.5). This gives a zero
  # deviation on every draw for every supported distribution -- normal/lognormal/
  # triangular map u = 0.5 to base 0, and uniform to the centered value
  # (2 * 0.5 - 1 = 0) -- so the coefficient collapses to its SD-zero null (the
  # fixed b for normal/uniform/triangular; sign * exp(b) for lognormal),
  # independent of the covariate values, while all other columns keep the full
  # model's exact draws.
  zeroed <- function(eq, k) {
    z <- full_draws
    if (eq == 1L) z$Z1[, k] <- 0.5 else z$Z2[, k] <- 0.5
    z
  }
  # Pin the tested log-scale (its value is inert once the column is zeroed) so it
  # is held fixed and excluded from npar, giving the 1-df restriction.
  pin_scale <- function(dist, eq, name) {
    nm <- .scale_par(dist, eq, name)
    stats::setNames(fit$coef[[nm]], nm)
  }
  # A restricted refit that did not converge cannot supply a valid maximized
  # log-likelihood; report NA inference (with a warning) rather than a
  # misleading p-value from an unfinished optimization.
  test_row <- function(param, rest) {
    if (!isTRUE(rest$convergence$converged)) {
      warning("Restricted fit for '", param, "' did not converge (maxLik code ",
              rest$convergence$code, ": ", rest$convergence$message,
              "); reporting NA for this parameter.", call. = FALSE)
      return(data.frame(Parameter = param, LR = NA_real_, df = NA_integer_,
                        p.value = NA_real_, Signif = NA_character_,
                        stringsAsFactors = FALSE))
    }
    lr <- lr_test(rest, fit, boundary = TRUE)
    data.frame(Parameter = param, LR = lr$statistic, df = lr$df,
               p.value = lr$p.value, Signif = signif_stars(lr$p.value),
               stringsAsFactors = FALSE)
  }

  rows <- list()

  if ("sd" %in% which) {
    # Equation 1: zero each coefficient's draw column in turn (exact SD-zero null).
    for (k in seq_along(names1)) {
      rows[[length(rows) + 1L]] <-
        test_row(.sd_label(dist1[k], 1, names1[k]),
                 refit(fixed = pin_scale(dist1[k], 1, names1[k]),
                       opt_draws = zeroed(1L, k)))
    }
    # Equation 2.
    for (k in seq_along(names2)) {
      rows[[length(rows) + 1L]] <-
        test_row(.sd_label(dist2[k], 2, names2[k]),
                 refit(fixed = pin_scale(dist2[k], 2, names2[k]),
                       opt_draws = zeroed(2L, k)))
    }
  }

  if ("dispersion" %in% which) {
    rows[[length(rows) + 1L]] <-
      test_row("m1", refit(poisson_1 = TRUE))
    rows[[length(rows) + 1L]] <-
      test_row("m2", refit(poisson_2 = TRUE))
  }

  out <- do.call(rbind, rows)
  rownames(out) <- NULL
  structure(out, class = c("rpbnb_boundary_tests", "data.frame"))
}

#' @export
print.rpbnb_boundary_tests <- function(x, digits = 4, ...) {
  cat("Boundary-parameter LR tests (boundary-corrected, 50:50 chi-square mixture)\n")
  cat("H0: parameter = 0 (random SD absent, or margin Poisson)\n\n")
  tab <- as.data.frame(x)
  tab$LR      <- formatC(tab$LR, format = "f", digits = digits)
  tab$p.value <- formatC(tab$p.value, format = "f", digits = digits)
  print(tab, row.names = FALSE, right = TRUE)
  cat("\nSignif: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1\n")
  invisible(x)
}
