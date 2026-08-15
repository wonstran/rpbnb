# Boundary-corrected LR tests for an rpbnb_tmb_fit's random-coefficient
# scales and NB2 dispersions. The TMB-engine counterpart of
# R/boundary_tests.R's rpbnb_boundary_tests().

# The template computes sd = exp(clamp_ad(log_sd, -20, 20)) (src/rpbnb_tmb.cpp),
# and the R-side Famoye bound uses exp(pmax(log_sd, -20)) to match, so -20 is
# this parameterization's canonical "scale is zero": exp(-20) = 2.1e-9, which
# multiplies every draw's deviation to numerically nothing. Pinning lower would
# be clamped back to -20 by both, so it would not be a different restriction.
.TMB_SCALE_ZERO <- -20

#' Boundary-corrected LR tests for an rpbnb_tmb_fit's boundary parameters
#'
#' Runs a likelihood-ratio test for every random-coefficient scale
#' (`sd1:*`, `sd2:*`) and NB2 dispersion (`m1`, `m2`) of a fitted TMB-engine
#' random-parameter bivariate NB model, and merges them into one table. These
#' are the parameters whose null lies on the boundary of the parameter space
#' (scale = 0, or dispersion `m = 0` = Poisson), for which
#' \code{summary(fit)} reports no Wald \code{z}/\code{p}; each test uses the
#' 50:50 chi-square boundary correction of [lr_test()] (`boundary = TRUE`).
#'
#' Each test refits a properly nested restricted model, otherwise identical
#' to `fit` (same formulas, random-coefficient specification, dependence,
#' seed, and estimator, and by default the same `draws`), warm-started
#' from `fit$coef` and run with `inference = "none"` since the LR test
#' needs only `logLik` and the parameter count.
#'
#' **Dispersions** are restricted with `poisson_1`/`poisson_2 = TRUE`, the
#' template's exact `m = 0` branch.
#'
#' **Scales** are restricted by pinning that coefficient's `log_sd` at the
#' parameterization's zero (`-20`; the template clamps `log_sd` to
#' `[-20, 20]` and computes `sd = exp(log_sd)`, so this is `sd = 2.1e-9` --
#' numerically zero on every draw) and holding it out of the free-parameter
#' count, giving a 1-df restriction. With **multiple random coefficients in
#' an equation, each scale is tested individually.**
#'
#' Pinning the scale rather than dropping the coefficient from
#' `random_1`/`random_2` is what preserves **common random numbers**: the
#' Halton draw matrix keeps the same width, so every *other* random
#' coefficient draws from exactly the dimensions it did in the full fit, and
#' the two simulated log-likelihoods differ only by the restriction under
#' test. Dropping the name instead would renumber the remaining coefficients'
#' Halton dimensions, and the LR statistic would absorb that reshuffling as
#' extra simulation noise.
#'
#' A [message()] (`"Boundary LR test: <parameter>..."`) reports each
#' restricted refit right before it starts, unless `control$print_level` is
#' `0`; suppress it with [suppressMessages()] if needed.
#'
#' @param fit A converged `rpbnb_tmb_fit` (from [fit_rpbnb_tmb()] or
#'   `rpbnb(engine = "tmb")`).
#' @param data The data frame the model was fit on. Required -- the fit
#'   object does not store it, and every restricted model is refit on the
#'   same data. With `standardize = TRUE`, pass the standardized data
#'   (`rpbnb(engine = "tmb", boundary_tests = TRUE)` does this automatically;
#'   a manual call needs `rpbnb:::.apply_scaling(data, fit$scaling)`).
#' @param control An [rpbnb_tmb_control()] for the restricted refits.
#'   Defaults to `rpbnb_tmb_control(print_level = 1, n_cores =
#'   fit$parallel$requested, max_workload = Inf)` -- the same `n_cores` the
#'   original fit was called with (1 if `fit` predates the stored
#'   `$parallel` field). `seed`/`method` are taken from `fit`, not `control`.
#' @param which Which boundary parameters to test: `"sd"`, `"dispersion"`, or
#'   both (the default).
#' @param draws Number of Halton simulation draws for the restricted refits.
#'   Defaults to `fit$draws` -- the same number `fit` itself used. Raising
#'   this trades refit cost for precision without needing to refit `fit` at
#'   a higher `draws`; lowering it is cheaper but noisier. A value other
#'   than `fit$draws` still shares `fit$seed` (so the Halton sequence's
#'   *prefix* is identical) but no longer draws the exact same simulated
#'   log-likelihood surface as `fit`, so the LR statistic picks up simulation
#'   noise beyond the restriction under test -- prefer the default unless you
#'   have a specific reason to diverge.
#' @param force_parallel_gaussian Opt-in override of the Gaussian-copula
#'   single-thread safety cap (see `?fit_rpbnb_tmb`), forwarded to every
#'   restricted refit. Default `FALSE`. This is intentionally a separate
#'   argument rather than something read off `fit`: `fit` does not record
#'   whether the original fit used the override, so passing
#'   `force_parallel_gaussian = TRUE` here is required even when the
#'   original `fit_rpbnb_tmb()`/`rpbnb()` call also passed it -- otherwise
#'   every refit silently falls back to one thread regardless of
#'   `control$n_cores`.
#' @return An object of class `rpbnb_boundary_tests` -- the same class
#'   [rpbnb_boundary_tests()] returns, with columns `Parameter`, `LR`, `df`,
#'   `p.value`, `Signif` (one row per boundary parameter) and the same
#'   `print()` method -- so the two engines' results are interchangeable
#'   wherever that class is consumed (e.g. `summary.rpbnb_tmb_fit()`'s scale
#'   and dispersion blocks, once `$boundary_tests` is attached to the fit).
#'
#'   Scale rows are labelled by the distribution's own scale parameter
#'   (`sd` for normal/lognormal, `w` for uniform/triangular half-width, `s`
#'   for a lognormal log-scale), matching `summary()`'s row names.
#' @seealso [lr_test()], [fit_rpbnb_tmb()], [rpbnb_boundary_tests()] (the
#'   classic-engine counterpart)
#' @export
#' @examples
#' \donttest{
#' sim <- simulate_rpbnb(n = 600,
#'   beta1 = c("(Intercept)" = 0.2, x1 = 0.4),
#'   beta2 = c("(Intercept)" = 0.1, x1 = -0.3),
#'   random_1 = list(x1 = list(sd = 0.5)),
#'   dispersion = c(m1 = 0.4, m2 = 0.5), seed = 1)
#' fit <- fit_rpbnb_tmb(y1 ~ x1, y2 ~ x1, data = sim$data, random_1 = "x1",
#'                      draws = 100)
#' rpbnb_tmb_boundary_tests(fit, sim$data)
#' }
rpbnb_tmb_boundary_tests <- function(fit, data, control = NULL,
                                     which = c("sd", "dispersion"),
                                     draws = fit$draws,
                                     force_parallel_gaussian = FALSE) {
  if (!inherits(fit, "rpbnb_tmb_fit")) {
    stop("`fit` must be an rpbnb_tmb_fit (from fit_rpbnb_tmb() or ",
         "rpbnb(engine = \"tmb\")).", call. = FALSE)
  }
  if (!is.data.frame(data)) stop("`data` must be a data frame.", call. = FALSE)
  if (is.null(fit$formula_1) || is.null(fit$formula_2)) {
    stop("`fit` does not carry its fitting formulas -- refit with the ",
         "current package version (older fit_rpbnb_tmb() builds did not ",
         "store formula_1/formula_2/draws/seed/poisson_1/poisson_2 on the ",
         "returned object).", call. = FALSE)
  }
  # nlminb's convergence code: 0 is success, matching optim()'s convention.
  if (!identical(fit$optimizer$convergence, 0L)) {
    stop("`fit` (the full model) did not converge (nlminb code ",
         fit$optimizer$convergence, ": ", fit$optimizer$message,
         "). Refit it to convergence before running boundary tests.",
         call. = FALSE)
  }
  which <- match.arg(which, c("sd", "dispersion"), several.ok = TRUE)
  n_disp_free <- sum(!isTRUE(fit$poisson_1), !isTRUE(fit$poisson_2))
  n_scales <- length(fit$rand_idx1) + length(fit$rand_idx2)
  testable <- ("dispersion" %in% which && n_disp_free > 0L) ||
    ("sd" %in% which && n_scales > 0L)
  if (!testable) {
    stop("No boundary parameters to test for which = c(",
         paste0("\"", which, "\"", collapse = ", "), "): the model has ",
         n_scales, " random-coefficient scale(s) and ", n_disp_free,
         " unrestricted dispersion(s).", call. = FALSE)
  }
  if (is.null(control)) {
    # Same n_cores the original fit was called with (before any Gaussian-
    # copula safety cap; .resolve_gaussian_threads() re-applies that cap on
    # each restricted refit exactly as it did on the original fit), not a
    # hardcoded 1 -- older fits without a stored $parallel fall back to 1.
    default_cores <- if (!is.null(fit$parallel$requested)) {
      fit$parallel$requested
    } else {
      1L
    }
    control <- rpbnb_tmb_control(print_level = 1L, n_cores = default_cores,
                                 max_workload = Inf)
  }
  if (!inherits(control, "rpbnb_tmb_control")) {
    stop("`control` must be an `rpbnb_tmb_control` object from ",
         "rpbnb_tmb_control(); got `", class(control)[1L], "`.", call. = FALSE)
  }

  # Full random specs, reconstructed from the fit -- shared with the classic
  # engine's rpbnb_boundary_tests() (R/boundary_tests.R), which is why this
  # file doesn't redefine it.
  names1 <- colnames(fit$X1)[fit$rand_idx1]
  names2 <- colnames(fit$X2)[fit$rand_idx2]
  dist1  <- fit$rp_meta$dist1; sign1 <- fit$rp_meta$sign1
  dist2  <- fit$rp_meta$dist2; sign2 <- fit$rp_meta$sign2
  if (is.null(fit$X1) || is.null(fit$X2) || is.null(fit$rp_meta)) {
    stop("`fit` does not retain its design/draws (keep = \"compact\"?); ",
         "refit with keep = \"postfit\" or \"full\" to run boundary tests.",
         call. = FALSE)
  }
  full1 <- .build_rand_spec(names1, dist1, sign1)
  full2 <- .build_rand_spec(names2, dist2, sign2)

  # Same seed => same Halton draws (TMB regenerates them from `seed` rather
  # than storing/reusing a draw matrix the way the classic engine's
  # .opt_draws does), so the restricted and full fits share common random
  # numbers as long as random_1/random_2 -- and so the draw dimensionality --
  # are unchanged, AND draws == fit$draws (the `draws` argument's default).
  # That is exactly why a scale restriction PINS log_sd via .fixed instead of
  # dropping the name from random_1/random_2: dropping it would renumber the
  # remaining coefficients' Halton dimensions and break the comparison.
  # Warm-started from the full fit's coefficients; both a pinned log_m
  # (Poisson limit) and a pinned log_sd (.fixed) are overwritten to their
  # restricted values inside fit_rpbnb_tmb() regardless of what `start`
  # carries.
  refit <- function(poisson_1 = isTRUE(fit$poisson_1),
                    poisson_2 = isTRUE(fit$poisson_2),
                    fixed = NULL) {
    # A restricted refit's warm start pins one parameter (a dispersion to the
    # Poisson limit, or a scale to the parameterization's zero) while leaving
    # every other coordinate at the FULL fit's optimum -- a combination that
    # optimum was never fit under and that fit_rpbnb_tmb()'s own nlminb
    # recovery (obj$env$last.par.best) can do nothing with if the objective
    # is non-finite at every point it tries, including this warm start
    # itself (observed on the truck data's m1 test under a Kimeldorf
    # copula: the very first outer gradient evaluation came back NaN, so no
    # finite point was ever recorded to recover to, and fit_rpbnb_tmb()
    # correctly re-raised rather than fabricate one). That is one bad
    # restriction among several independent ones, not a reason to lose the
    # whole boundary-test batch (or, folded into rpbnb(), the whole fit) --
    # test_row() below already has a well-defined "did not converge" path
    # for exactly this outcome, so THIS SPECIFIC error is funneled into it as
    # a stub with convergence = NA rather than propagating. Anything else
    # (a real bug, a bad argument) is deliberately left to propagate as an
    # ordinary error rather than being relabelled "did not converge" -- that
    # would just as often mask a genuine defect as report a numerical one.
    tryCatch(
      fit_rpbnb_tmb(
        fit$formula_1, fit$formula_2, data = data,
        random_1 = full1, random_2 = full2,
        dependence = fit$dependence,
        seed = fit$seed, draws = draws, start = fit$coef,
        control = control, inference = "none",
        poisson_1 = poisson_1, poisson_2 = poisson_2,
        method = if (is.null(fit$method)) "sml" else fit$method,
        .fixed = fixed, force_parallel_gaussian = force_parallel_gaussian
      ),
      error = function(e) {
        if (!grepl("NA/NaN", conditionMessage(e), fixed = TRUE)) stop(e)
        list(optimizer = list(convergence = NA_integer_,
                              message = conditionMessage(e)))
      }
    )
  }
  # Optimization-parameterization name of a random coefficient's log-scale
  # ("log_sd1:x1" for a normal), and the natural-scale label summary() prints
  # for the same row ("sd1:x1") -- both derived from rand_dist_registry's
  # scale_label, so a new distribution follows automatically. .sd_label() is
  # shared with the classic engine (R/boundary_tests.R).
  .scale_par <- function(dist, eq, name) {
    paste0(rand_dist_registry[[dist]]$scale_label, eq, ":", name)
  }

  # A restricted refit that did not converge cannot supply a valid maximized
  # log-likelihood; report NA inference (with a warning) rather than a
  # misleading p-value from an unfinished optimization.
  test_row <- function(param, rest) {
    if (!identical(rest$optimizer$convergence, 0L)) {
      warning("Restricted fit for '", param, "' did not converge (nlminb ",
              "code ", rest$optimizer$convergence, ": ",
              rest$optimizer$message, "); reporting NA for this parameter.",
              call. = FALSE)
      return(data.frame(Parameter = param, LR = NA_real_, df = NA_integer_,
                        p.value = NA_real_, Signif = NA_character_,
                        stringsAsFactors = FALSE))
    }
    lr <- lr_test(rest, fit, boundary = TRUE)
    data.frame(Parameter = param, LR = lr$statistic, df = lr$df,
              p.value = lr$p.value, Signif = signif_stars(lr$p.value),
              stringsAsFactors = FALSE)
  }

  # Announces each restricted refit before it runs (not after -- print_level
  # now defaults to 1, so TMB's own non-silent optimizer output interleaves
  # with these; putting the message first is what makes that output
  # attributable to the right parameter). Silenced the same way rpbnb()
  # silences its own boundary-tests message: control$print_level == 0.
  announce <- function(param) {
    if (is.null(control$print_level) || control$print_level > 0) {
      message("Boundary LR test: ", param, "...")
    }
  }

  rows <- list()

  if ("sd" %in% which) {
    # Equation 1, then equation 2: pin each coefficient's log-scale at the
    # parameterization's zero in turn (exact scale-zero null on every draw).
    for (k in seq_along(names1)) {
      par_nm <- .scale_par(dist1[k], 1L, names1[k])
      label <- .sd_label(dist1[k], 1L, names1[k])
      announce(label)
      rows[[length(rows) + 1L]] <- test_row(
        label, refit(fixed = stats::setNames(.TMB_SCALE_ZERO, par_nm)))
    }
    for (k in seq_along(names2)) {
      par_nm <- .scale_par(dist2[k], 2L, names2[k])
      label <- .sd_label(dist2[k], 2L, names2[k])
      announce(label)
      rows[[length(rows) + 1L]] <- test_row(
        label, refit(fixed = stats::setNames(.TMB_SCALE_ZERO, par_nm)))
    }
  }

  if ("dispersion" %in% which) {
    if (!isTRUE(fit$poisson_1)) {
      announce("m1")
      rows[[length(rows) + 1L]] <- test_row("m1", refit(poisson_1 = TRUE))
    }
    if (!isTRUE(fit$poisson_2)) {
      announce("m2")
      rows[[length(rows) + 1L]] <- test_row("m2", refit(poisson_2 = TRUE))
    }
  }

  out <- do.call(rbind, rows)
  rownames(out) <- NULL
  structure(out, class = c("rpbnb_boundary_tests", "data.frame"))
}
