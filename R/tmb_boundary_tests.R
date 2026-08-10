# Boundary-corrected LR test for the NB2 dispersions of an rpbnb_tmb_fit. The
# TMB-engine counterpart of R/boundary_tests.R's rpbnb_boundary_tests(),
# scoped to m1/m2 (the classic engine also tests the random-coefficient SDs;
# see the "Which which" note in rpbnb_tmb_boundary_tests()'s doc for why that
# is not offered here).

#' Boundary-corrected LR test for the NB2 dispersions of an rpbnb_tmb_fit
#'
#' Runs a likelihood-ratio test for each unrestricted NB2 dispersion (`m1`,
#' `m2`) of a fitted TMB-engine random-parameter bivariate NB model: the
#' parameter whose null (`m = 0`, the Poisson limit) lies on the boundary of
#' the parameter space, for which \code{summary(fit)}'s "Dispersion (m1, m2)"
#' block reports no Wald \code{z}/\code{p}. Each test refits the model with
#' that margin pinned at its Poisson limit (\code{poisson_1}/\code{poisson_2
#' = TRUE}) -- otherwise identical to \code{fit} (same formulas,
#' random-coefficient specification, dependence, draws, and seed, so the two
#' fits share the same simulated draws) -- and applies [lr_test()]'s 50:50
#' chi-square boundary correction.
#'
#' The restricted refit is warm-started from \code{fit$coef} (the pinned
#' margin's own coefficient is overwritten to the Poisson-limit placeholder
#' internally, so this is safe to pass as-is) and run with
#' \code{inference = "none"}, since the LR test needs only \code{logLik} and
#' the parameter count.
#'
#' # Which parameters this tests
#'
#' Only NB2 dispersions. The classic engine's [rpbnb_boundary_tests()] also
#' tests each random-coefficient standard deviation, by zeroing that
#' coefficient's simulation draws so it collapses to its SD-zero null on
#' every draw while every other column keeps the full model's exact draws
#' (see that function's documentation). The TMB engine has no equivalent
#' \code{.opt_draws}-style mechanism to reuse; dropping a name from
#' \code{random_1}/\code{random_2} on a refit would change which Halton
#' dimensions the remaining random coefficients draw from, so the two fits
#' would no longer share common random numbers and the resulting LR
#' statistic would carry extra simulation noise beyond the restriction being
#' tested. Testing a TMB random-coefficient SD this way remains possible by
#' hand -- refit with the name dropped from \code{random_1}/\code{random_2}
#' and call [lr_test()] directly -- just not wrapped here.
#'
#' @param fit A converged `rpbnb_tmb_fit` (from [fit_rpbnb_tmb()] or
#'   `rpbnb(engine = "tmb")`).
#' @param data The data frame the model was fit on. Required -- the fit
#'   object does not store it, and the restricted refit needs it. With
#'   `standardize = TRUE`, pass the standardized data (`rpbnb(engine = "tmb",
#'   boundary_tests = TRUE)` does this automatically; a manual call needs
#'   `rpbnb:::.apply_scaling(data, fit$scaling)`).
#' @param control An [rpbnb_tmb_control()] for the restricted refits.
#'   Defaults to `rpbnb_tmb_control(print_level = 0, n_cores = 1,
#'   max_workload = Inf)`. `draws`/`seed`/`method` are taken from `fit`, not
#'   `control`.
#' @return An object of class `rpbnb_boundary_tests` -- the same class
#'   [rpbnb_boundary_tests()] returns, with columns `Parameter`, `LR`, `df`,
#'   `p.value`, `Signif` (one row per unrestricted dispersion) and the same
#'   `print()` method -- so the two engines' results are interchangeable
#'   wherever that class is consumed (e.g. `summary.rpbnb_tmb_fit()`'s
#'   dispersion block, once `$boundary_tests` is attached to the fit).
#' @seealso [lr_test()], [fit_rpbnb_tmb()], [rpbnb_boundary_tests()] (the
#'   classic-engine counterpart, which also tests random-coefficient SDs)
#' @export
#' @examples
#' \donttest{
#' sim <- simulate_rpbnb(n = 600,
#'   beta1 = c("(Intercept)" = 0.2, x1 = 0.4),
#'   beta2 = c("(Intercept)" = 0.1, x1 = -0.3),
#'   dispersion = c(m1 = 0.4, m2 = 0.5), seed = 1)
#' fit <- fit_rpbnb_tmb(y1 ~ x1, y2 ~ x1, data = sim$data, draws = 100)
#' rpbnb_tmb_boundary_tests(fit, sim$data)
#' }
rpbnb_tmb_boundary_tests <- function(fit, data, control = NULL) {
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
  if (isTRUE(fit$poisson_1) && isTRUE(fit$poisson_2)) {
    stop("Both margins are already Poisson-restricted (poisson_1 = ",
         "poisson_2 = TRUE in `fit`); there is no dispersion parameter left ",
         "to test.", call. = FALSE)
  }
  if (is.null(control)) {
    control <- rpbnb_tmb_control(print_level = 0L, n_cores = 1L,
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
  # are unchanged. Warm-started from the full fit's coefficients; the pinned
  # margin's own log_m entry is overwritten to the Poisson-limit placeholder
  # inside fit_rpbnb_tmb() regardless of what's passed here.
  refit <- function(poisson_1, poisson_2) {
    fit_rpbnb_tmb(
      fit$formula_1, fit$formula_2, data = data,
      random_1 = full1, random_2 = full2,
      dependence = fit$dependence,
      seed = fit$seed, draws = fit$draws, start = fit$coef,
      control = control, inference = "none",
      poisson_1 = poisson_1, poisson_2 = poisson_2,
      method = if (is.null(fit$method)) "sml" else fit$method
    )
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

  rows <- list()
  if (!isTRUE(fit$poisson_1)) {
    rows[[length(rows) + 1L]] <- test_row(
      "m1", refit(poisson_1 = TRUE, poisson_2 = isTRUE(fit$poisson_2)))
  }
  if (!isTRUE(fit$poisson_2)) {
    rows[[length(rows) + 1L]] <- test_row(
      "m2", refit(poisson_1 = isTRUE(fit$poisson_1), poisson_2 = TRUE))
  }

  out <- do.call(rbind, rows)
  rownames(out) <- NULL
  structure(out, class = c("rpbnb_boundary_tests", "data.frame"))
}
