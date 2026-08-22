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

# ---------------------------------------------------------------------------
# Which groups of parameters a boundary-test run covers. Both engines take the
# same three names, so one switch reads the same way whichever engine ran.
.BOUNDARY_TEST_GROUPS <- c("sd", "dispersion", "dependence")

# rpbnb()'s `boundary_tests` argument accepts a logical (the historical form) or
# a character subset of .BOUNDARY_TEST_GROUPS. TRUE keeps meaning exactly what
# it always did -- scales and dispersions -- rather than silently growing a
# third refit; "dependence" and "all" are how you ask for the association test.
.normalize_boundary_tests <- function(x, arg = "boundary_tests") {
  if (is.null(x)) return(character(0))
  if (is.logical(x)) {
    if (length(x) != 1L || is.na(x)) {
      stop("`", arg, "` must be TRUE/FALSE or a character vector naming test ",
           "groups (", paste(.BOUNDARY_TEST_GROUPS, collapse = ", "),
           ", all).", call. = FALSE)
    }
    return(if (x) c("sd", "dispersion") else character(0))
  }
  if (!is.character(x)) {
    stop("`", arg, "` must be TRUE/FALSE or a character vector naming test ",
         "groups (", paste(.BOUNDARY_TEST_GROUPS, collapse = ", "),
         ", all).", call. = FALSE)
  }
  if (!length(x)) return(character(0))
  if ("none" %in% x) return(character(0))
  if ("all" %in% x) return(.BOUNDARY_TEST_GROUPS)
  bad <- setdiff(x, .BOUNDARY_TEST_GROUPS)
  if (length(bad)) {
    stop("`", arg, "` has unknown group(s): ", paste(bad, collapse = ", "),
         ". Valid: ", paste(.BOUNDARY_TEST_GROUPS, collapse = ", "),
         ", all, none.", call. = FALSE)
  }
  intersect(.BOUNDARY_TEST_GROUPS, x)      # canonical order
}

# The dependence family of a fit, as one of "famoye", "independence", "frank",
# "normal", "kimeldorf". Reads the classic engine's `cop_family` and the TMB
# engine's `dependence` so the boundary-test code below does not branch on the
# fit class.
.fit_dep_family <- function(fit) {
  if (!is.null(fit$cop_family)) return(fit$cop_family)
  dep <- fit$dependence
  if (inherits(dep, "rpbnb_copula")) return(dep$family)
  if (is.character(dep) && length(dep) == 1L) return(dep)
  "famoye"
}

# Canonical `Parameter` value for a family's dependence parameter in the
# boundary-test table. These match the names summary.rpbnb_tmb_fit() already
# prints for the dependence block, so one table serves both summaries.
.dep_boundary_param <- function(family) {
  switch(family,
         famoye = "lam", frank = "theta", normal = "rho", kimeldorf = "theta",
         NULL)
}

# Is the null "no association" on the BOUNDARY of this family's parameter
# space? Frank's theta ranges over R, the Gaussian rho over (-1, 1), and the
# Famoye lambda over an interval containing 0 -- all interior, so the LR
# statistic is an ordinary chi-square(1). Clayton/Kimeldorf constrains
# theta > 0, so its independence null sits on the boundary and needs the 50:50
# mixture. Getting this wrong halves (or doubles) the p-value.
.dep_null_is_boundary <- function(family) identical(family, "kimeldorf")

# Clayton theta at which the classic engine's copula code takes its exact
# independence branch. Both the R reference implementation (copula_core.R) and
# the C++ kernel (src/copula_parallel.cpp) short-circuit the CDF, its partials,
# and dC/dtheta to the product copula below theta = 1e-10; exp(-30) = 9.4e-14
# clears that with room to spare. Pinning nearer the threshold (say exp(-20) =
# 2.1e-9) would NOT take the branch, and the four-corner cell probability would
# then be a difference of numbers agreeing to ~1e-9 -- noise, not a likelihood.
.CLAYTON_INDEP_Z <- -30

# Working-scale value of z_lambda that maps to lambda = 0 (independence) under
# the admissible interval the RESTRICTED refit will freeze -- which is the
# interval implied by `fit$coef`, since that is what warm-starts the refit.
# fit_rpbnb() maps z through lo + (hi - lo) * (eps + (1 - 2 eps) * plogis(z)),
# and the interval always brackets 0 (independence is always admissible), so
# this inversion is well defined; NA_real_ signals a degenerate interval, which
# the caller turns into an NA row rather than a wrong pin.
.famoye_indep_z <- function(fit) {
  b <- .rp_support_bounds(
    fit$coef, fit$X1, fit$X2, fit$rand_idx1, fit$rand_idx2,
    fit$rp_meta$dist1, fit$rp_meta$dist2, fit$rp_meta$sign1, fit$rp_meta$sign2,
    pois1 = isTRUE(fit$poisson_1), pois2 = isTRUE(fit$poisson_2),
    off1 = .fit_offset(fit, 1L), off2 = .fit_offset(fit, 2L))
  lo <- unname(b[["lower"]]); hi <- unname(b[["upper"]])
  if (!(is.finite(lo) && is.finite(hi) && lo < 0 && hi > 0)) return(NA_real_)
  eps <- 1e-6
  p <- ((0 - lo) / (hi - lo) - eps) / (1 - 2 * eps)
  if (!(p > 0 && p < 1)) return(NA_real_)
  stats::qlogis(p)
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
#' @param fit A converged `rpbnb_fit` (the full model), from a Famoye or a
#'   [copula()] dependence. Both paths use the exact `m = 0` branch for the
#'   dispersion tests.
#' @param data The data frame the model was fit on. Required -- the fit object
#'   does not store it, and every restricted model is refit on the same data.
#' @param control An [rpbnb_control()] for the restricted refits. Defaults to
#'   `compute_se = FALSE` (the LR test needs only `logLik` and the degrees of
#'   freedom). `draws`/`draw_type`/`seed` are taken from `fit`, not `control`.
#' @param which Which parameter groups to test, any subset of `"sd"` (the
#'   random-coefficient scales), `"dispersion"` (the NB2 dispersions `m1`,
#'   `m2`), and `"dependence"` (the association parameter). The default is
#'   `c("sd", "dispersion")` -- the two boundary-null groups. `"dependence"`
#'   is opt-in because it costs another full refit and, for three of the four
#'   families, tests an *interior* null whose ordinary Wald `z` in `summary()`
#'   is already valid.
#'
#'   The dependence test restricts the model to independence and compares it to
#'   `fit`: one row labelled `lam` (Famoye), `theta` (Frank / Clayton), or `rho`
#'   (Gaussian). The restriction pins the working-scale dependence parameter at
#'   its family's independence value rather than refitting a different family,
#'   so both fits keep the same draws and the same parameter block and the
#'   comparison is a clean 1-df restriction. Only Clayton's null (`theta > 0`,
#'   so `theta = 0` is a boundary) takes the 50:50 mixture; Famoye, Frank, and
#'   Gaussian nulls are interior and get an ordinary chi-square(1).
#' @return An object of class `rpbnb_boundary_tests`: a data frame with columns
#'   `Parameter`, `LR`, `df`, `p.value`, `Signif` (one row per boundary
#'   parameter), and a `print` method.
#'
#'   Under Famoye dependence with uniform or triangular random coefficients
#'   (or a single varying margin), the full and restricted fits' admissible
#'   lambda intervals are frozen at different starting values, so each LR
#'   statistic compares maxima over slightly different lambda ranges; see the
#'   "Famoye caveat" section of [lr_test()]. Normal/lognormal coefficients in
#'   both margins are unaffected (the interval is the constant `c(-1, 1)`).
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
  if (!is.data.frame(data)) stop("`data` must be a data frame.", call. = FALSE)
  which <- match.arg(which, .BOUNDARY_TEST_GROUPS, several.ok = TRUE)

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
  full_dep <- if (!is.null(fit$cop_family)) copula(fit$cop_family) else "famoye"
  # The Poisson flags default to the FULL fit's own, not to FALSE: with
  # poisson_1 = TRUE on `fit`, a refit at poisson_1 = FALSE would be a LARGER
  # model, not a restriction of it, and every scale test would silently compare
  # non-nested fits.
  refit <- function(poisson_1 = isTRUE(fit$poisson_1),
                    poisson_2 = isTRUE(fit$poisson_2), fixed = NULL,
                    opt_draws = full_draws) {
    fit_rpbnb(fit$formula_1, fit$formula_2, data = data,
              random_1 = full1, random_2 = full2,
              draws = fit$draws, draw_type = fit$draw_type, seed = fit$seed,
              start = fit$coef, control = control, dependence = full_dep,
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
  na_row <- function(param) {
    data.frame(Parameter = param, LR = NA_real_, df = NA_integer_,
               p.value = NA_real_, Signif = NA_character_,
               stringsAsFactors = FALSE)
  }
  # A restricted refit that did not converge cannot supply a valid maximized
  # log-likelihood; report NA inference (with a warning) rather than a
  # misleading p-value from an unfinished optimization.
  #
  # `boundary` is per-test, not a constant: the scale and dispersion nulls sit
  # on the boundary of the parameter space, but the dependence null does not
  # for every family (see .dep_null_is_boundary()), and applying the 50:50
  # mixture where the null is interior halves the p-value for no reason.
  test_row <- function(param, rest, boundary = TRUE) {
    if (!isTRUE(rest$convergence$converged)) {
      warning("Restricted fit for '", param, "' did not converge (maxLik code ",
              rest$convergence$code, ": ", rest$convergence$message,
              "); reporting NA for this parameter.", call. = FALSE)
      return(na_row(param))
    }
    lr <- lr_test(rest, fit, boundary = boundary)
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
    # A margin already pinned to its Poisson limit has no free dispersion to
    # restrict: refitting it would produce the same model, and lr_test() would
    # (correctly) refuse a 0-df comparison. Skip it, as the TMB engine does.
    if (!isTRUE(fit$poisson_1)) {
      rows[[length(rows) + 1L]] <- test_row("m1", refit(poisson_1 = TRUE))
    }
    if (!isTRUE(fit$poisson_2)) {
      rows[[length(rows) + 1L]] <- test_row("m2", refit(poisson_2 = TRUE))
    }
  }

  # Dependence parameter: H0 is "no association", i.e. the independence copula.
  # The classic engine has no `dependence = "independence"` fitter, so the
  # restriction is imposed by pinning the working-scale dependence parameter at
  # the value its own family maps to independence -- z_theta = 0 for Frank
  # (theta = 0) and the Gaussian copula (rho = tanh(0) = 0), z_theta far below
  # the product-copula cutoff for Clayton, and the z that maps to lambda = 0 for
  # Famoye. Pinning (rather than refitting a different family) keeps the two
  # models on the same draws, the same parameter block, and a clean 1-df
  # restriction.
  if ("dependence" %in% which) {
    fam <- .fit_dep_family(fit)
    param <- .dep_boundary_param(fam)
    if (is.null(param)) {
      warning("No dependence parameter to test for dependence = \"", fam,
              "\"; skipping the dependence boundary test.", call. = FALSE)
    } else {
      pin <- if (identical(fam, "famoye")) {
        stats::setNames(.famoye_indep_z(fit), "z_lambda")
      } else if (identical(fam, "kimeldorf")) {
        stats::setNames(.CLAYTON_INDEP_Z, "z_theta")
      } else {
        stats::setNames(0, "z_theta")
      }
      if (!is.finite(pin)) {
        warning("Could not construct the independence restriction for '",
                param, "' (the fit's admissible lambda interval does not ",
                "bracket 0); reporting NA for this parameter.", call. = FALSE)
        rows[[length(rows) + 1L]] <- na_row(param)
      } else {
        rows[[length(rows) + 1L]] <-
          test_row(param, refit(fixed = pin),
                   boundary = .dep_null_is_boundary(fam))
      }
    }
  }

  if (!length(rows)) {
    stop("No boundary parameters to test for which = c(",
         paste0("\"", which, "\"", collapse = ", "),
         "): the model has nothing restrictable in the requested group(s).",
         call. = FALSE)
  }
  out <- do.call(rbind, rows)
  rownames(out) <- NULL
  structure(out, class = c("rpbnb_boundary_tests", "data.frame"))
}

#' @export
print.rpbnb_boundary_tests <- function(x, digits = 4, ...) {
  # The header describes the tests actually in the table. A dependence row can
  # be an ORDINARY chi-square (its null is interior for every family except
  # Clayton/Kimeldorf), so announcing the 50:50 mixture unconditionally would
  # misdescribe it -- and this table is the standalone report people read.
  dep_row <- intersect(x$Parameter, c("lam", "theta", "rho"))
  if (length(dep_row)) {
    cat("Parameter LR tests\n")
    cat("H0: parameter = 0 (random scale absent, margin Poisson, or\n")
    cat("    no association for the ", paste(dep_row, collapse = "/"),
        " row)\n", sep = "")
    cat("Scale and dispersion rows use the boundary-corrected 50:50\n")
    cat("chi-square mixture; a dependence row uses it only for the\n")
    cat("Clayton/Kimeldorf copula, whose theta > 0 makes 0 a boundary.\n\n")
  } else {
    cat("Boundary-parameter LR tests (boundary-corrected, 50:50 chi-square mixture)\n")
    cat("H0: parameter = 0 (random SD absent, or margin Poisson)\n\n")
  }
  tab <- as.data.frame(x)
  tab$LR      <- formatC(tab$LR, format = "f", digits = digits)
  tab$p.value <- formatC(tab$p.value, format = "f", digits = digits)
  print(tab, row.names = FALSE, right = TRUE)
  cat("\nSignif: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1\n")
  # TMB-engine fits estimated by Laplace can hand individual tests to an SML
  # pair when the restricted model has no Laplace optimum (see
  # ?rpbnb_tmb_boundary_tests, `sml_fallback`). The classic engine never sets
  # this attribute, so the footnote never prints there.
  fb <- attr(x, "sml_fallback")
  if (length(fb)) {
    cat("Estimated by an SML pair (Laplace restricted fit did not converge):",
        paste(fb, collapse = ", "), "\n")
  }
  invisible(x)
}
