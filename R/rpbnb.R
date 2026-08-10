#' Fit a random-parameter bivariate NB model with either engine
#'
#' A common front end over the package's two estimation engines. `engine =
#' "classic"` calls [fit_rpbnb()] (Rcpp/OpenMP simulated likelihood, `maxLik` BFGS);
#' `engine = "tmb"` calls [fit_rpbnb_tmb()] (TMB automatic differentiation,
#' `nlminb` with restart polish). Both fitters remain exported and can be called
#' directly; with `standardize = FALSE` (the default) this wrapper adds nothing
#' to the fit itself and returns exactly what the chosen fitter returns.
#'
#' What it does add is argument checking across the two APIs. The engines do not
#' take the same arguments, and passing one engine's argument to the other is a
#' mistake that is easy to make and expensive to notice — a standardized
#' coefficient table printed under an "original units" heading looks perfectly
#' plausible. Every extra argument is therefore matched by name against the
#' selected fitter's own formals, and anything that does not belong is an error
#' rather than a silently ignored `...` entry.
#'
#' # Automatic centring and scaling
#'
#' `standardize = TRUE` automates the pattern in `inst/rpbnb_frank_open.R` and
#' `inst/tmb_rpbnb_frank_open.R`: continuous predictors are centred and scaled
#' (mean 0, SD 1) before fitting, which keeps a bounded random-coefficient
#' carrier from acting as a disguised random intercept (see those scripts'
#' headers) and fixes the design matrix's conditioning when regressors span
#' very different ranges. Continuous predictors are identified automatically —
#' numeric, non-factor columns used by either formula with more than two
#' distinct values, so 0/1 (or any two-level numeric) indicators are left
#' alone — or supplied explicitly via `continuous_vars`. Variables that appear
#' only inside an `offset()` are never standardized.
#'
#' The fitted design itself (the stored `X1`/`X2`, `mu1`/`mu2`, simulation
#' draws) stays on the standardized scale, exactly as in the two scripts above,
#' so `predict()`, marginal effects, and boundary/LR tests keep working
#' unchanged. Only the coefficient table `print()` and `summary()` display is
#' affected: it is back-transformed to the covariates' original units by the
#' exact affine chain rule (no refit, no numerical differentiation) and is the
#' *only* coefficient table shown — there is no separate standardized-scale
#' table to reconcile. `coef()`/`vcov()` still return the standardized-scale
#' values that match the stored design; call `.rpbnb_orig_units()`
#' (internal, mirrors what `print()` displays) if the numeric original-units
#' vector is needed directly. The scaling actually used is stored on the fit
#' as `$scaling` (a named list of `c(center=, scale=)`) and `$continuous_vars`.
#'
#' # Boundary LR tests (`engine = "classic"` only)
#'
#' `boundary_tests = TRUE` runs [rpbnb_boundary_tests()] on the fit as soon as
#' it converges (against the same data the fit itself used -- the standardized
#' copy, when `standardize = TRUE`) and attaches the result as `$boundary_tests`.
#' The random-coefficient SDs and the NB2 dispersions (`m1`, `m2`) have a null
#' that sits on the boundary of the parameter space, so an ordinary Wald `z`/`p`
#' does not test it; [rpbnb_boundary_tests()] runs the valid boundary-corrected
#' LR test instead by refitting each restricted model. `summary()`/`print()`
#' then show that test's `LR`/`df`/`p` for those rows in the natural-scale
#' block, in place of the `NA` they would otherwise carry. Only supported for
#' `engine = "classic"` ([rpbnb_boundary_tests()] requires an `rpbnb_fit`); it
#' is an error under `engine = "tmb"`. A [message()] reports how many
#' restricted refits are about to run (one per random-coefficient SD, one per
#' estimated NB2 dispersion) before they start, unless `control$print_level`
#' is `0`; suppress it with [suppressMessages()] if needed.
#'
#' Each restricted refit costs roughly as much as the original fit (more for a
#' [copula()] dependence than for `"famoye"`; see [rpbnb_boundary_tests()]'s
#' timing note), so this defaults to `FALSE`. For finer control -- testing only
#' `"sd"` or only `"dispersion"`, or reusing one boundary-test run across
#' several summaries -- call [rpbnb_boundary_tests()] directly on the fit and
#' assign its result to `fit$boundary_tests` (with `standardize = TRUE`,
#' reconstruct the fitting-scale data first: `rpbnb:::.apply_scaling(data,
#' fit$scaling)`).
#'
#' # Which arguments go with which engine
#'
#' | Argument | `engine = "classic"` | `engine = "tmb"` |
#' | --- | --- | --- |
#' | `draw_type`, `.fixed`, `.opt_draws` | yes | error |
#' | `inference`, `keep`, `method`, `force_parallel_gaussian` | error | yes |
#' | `offset()` in a formula | yes | error |
#' | `dependence = "independence"` | error | yes |
#' | `control` class | `rpbnb_control` | `rpbnb_tmb_control` |
#' | optimizer | `maxLik::maxLik(method = "BFGS")` | `stats::nlminb` + restarts |
#'
#' Both engines freeze the Famoye admissible lambda interval at the starting
#' values (so the analytic gradient is exactly the derivative of the optimized
#' objective) and validate the fitted lambda against the interval admissible at
#' the optimum afterwards — check `lambda_admissible` on the fit. The
#' fixed-parameter [fit_bnb()] takes the opposite trade-off (moving bounds,
#' admissible by construction); see the decision note in `R/bnb_likelihood.R`.
#'
#' @param formula_1,formula_2 Model formulas for the two count responses.
#' @param data A data frame containing the model variables.
#' @param engine Estimation engine: `"classic"` (default) or `"tmb"`.
#' @param random_1,random_2 Random-coefficient specifications for each equation.
#' @param draws Number of simulation draws.
#' @param seed Random seed for the draw sequence.
#' @param start Optional named or unnamed starting values.
#' @param dependence `"famoye"` (default), a [copula()] object, or
#'   `"independence"` (TMB engine only).
#' @param poisson_1,poisson_2 Restrict the corresponding margin to its Poisson
#'   limit.
#' @param standardize Centre and scale continuous predictors before fitting,
#'   and display fitted coefficients back-transformed to their original units
#'   (see "Automatic centring and scaling" above). Default `FALSE`.
#' @param continuous_vars Optional character vector overriding automatic
#'   continuous-predictor detection when `standardize = TRUE`. Must be columns
#'   of `data`; ignored when `standardize = FALSE`.
#' @param boundary_tests Run [rpbnb_boundary_tests()] after fitting and attach
#'   the result as `$boundary_tests`, so `summary()`/`print()` show a
#'   boundary-corrected LR test (rather than `NA`) for the random-coefficient
#'   SDs and NB2 dispersions (see "Boundary LR tests" above). `engine =
#'   "classic"` only; default `FALSE` (each restricted refit costs roughly
#'   another full fit).
#' @param control An `rpbnb_control()` object when `engine = "classic"`, or an
#'   `rpbnb_tmb_control()` object when `engine = "tmb"`. Defaults to the right
#'   one for the chosen engine. The two are not interchangeable and are never
#'   translated into one another.
#' @param ... Further arguments passed to the selected fitter. Names are
#'   validated against that fitter's formals; an argument belonging to the other
#'   engine, or an unrecognised name, is an error.
#'
#' @return The engine-native fit object, identical to what a direct call to the
#'   underlying fitter would return: an object of class `rpbnb_fit` for
#'   `engine = "classic"`, or `rpbnb_tmb_fit` for `engine = "tmb"`. The class
#'   therefore depends on `engine`; test with `inherits(fit, "rpbnb_tmb_fit")`
#'   if you need to branch. No wrapper class is introduced, so every existing S3
#'   method and post-estimation function works unchanged. With
#'   `standardize = TRUE`, two extra fields are attached -- `$scaling` and
#'   `$continuous_vars` -- and `print()`/`summary()` use them to display
#'   original-units coefficients. With `boundary_tests = TRUE`, a third field
#'   `$boundary_tests` (the [rpbnb_boundary_tests()] result) is attached, and
#'   `print()`/`summary()` use it to show the SD/dispersion rows' LR test.
#'   Nothing else on the object changes.
#'
#' @seealso [fit_rpbnb()], [fit_rpbnb_tmb()], [rpbnb_control()],
#'   [rpbnb_tmb_control()], [copula()], [fit_bnb()]
#' @export
#' @examples
#' \donttest{
#' d <- read.csv(system.file("extdata", "rwm1984_bnb.csv", package = "rpbnb"))
#' fit <- rpbnb(docvis ~ outwork, hospvis ~ outwork, data = d,
#'              engine = "tmb", random_1 = "outwork", draws = 50)
#' }
rpbnb <- function(formula_1, formula_2, data,
                  engine = c("classic", "tmb"),
                  random_1 = NULL, random_2 = NULL,
                  draws = 400, seed = 1234, start = NULL,
                  dependence = "famoye",
                  poisson_1 = FALSE, poisson_2 = FALSE,
                  standardize = FALSE, continuous_vars = NULL,
                  boundary_tests = FALSE,
                  control = NULL,
                  ...) {
  engine <- match.arg(engine)

  this_fit  <- if (engine == "classic") fit_rpbnb     else fit_rpbnb_tmb
  other_fit <- if (engine == "classic") fit_rpbnb_tmb else fit_rpbnb
  other_nm  <- if (engine == "classic") "tmb"         else "classic"

  # Validate the dots against the selected fitter's own formals rather than a
  # hard-coded list, so this stays correct as either fitter gains arguments --
  # and so a typo ("drawtype") is an error instead of a silent no-op. Because
  # every surviving name is an exact formal, R's partial matching cannot bite
  # at the do.call() below.
  dots <- list(...)
  if (length(dots)) {
    nm <- names(dots)
    if (is.null(nm) || any(!nzchar(nm))) {
      stop("Every extra argument to rpbnb() must be named. Positional ",
           "pass-through is not supported, because the two engines do not ",
           "take the same arguments in the same order.", call. = FALSE)
    }
    allowed <- setdiff(names(formals(this_fit)),  "...")
    foreign <- setdiff(names(formals(other_fit)), "...")
    bad <- setdiff(nm, allowed)
    if (length(bad)) {
      wrong_engine <- intersect(bad, foreign)
      unknown      <- setdiff(bad, foreign)
      msg <- character()
      if (length(wrong_engine)) {
        msg <- c(msg, sprintf(
          "%s only accepted by engine = \"%s\", but engine = \"%s\" was chosen.",
          paste0("`", wrong_engine, "`", collapse = ", "), other_nm, engine))
      }
      if (length(unknown)) {
        msg <- c(msg, sprintf("%s not an argument of either engine.",
                              paste0("`", unknown, "`", collapse = ", ")))
      }
      stop(paste(msg, collapse = " "),
           "\n  classic-only: draw_type, .fixed, .opt_draws",
           "\n  tmb-only: inference, keep, method, force_parallel_gaussian",
           "\n  See ?rpbnb for the full argument matrix.", call. = FALSE)
    }
  }

  # Engine-typed control. The default must NOT be rpbnb_control() in the
  # signature: a default evaluated at call time would hand the classic control
  # object to the TMB engine.
  if (is.null(control)) {
    control <- if (engine == "classic") rpbnb_control() else rpbnb_tmb_control()
  } else {
    want <- if (engine == "classic") "rpbnb_control"   else "rpbnb_tmb_control"
    ctor <- if (engine == "classic") "rpbnb_control()" else "rpbnb_tmb_control()"
    if (!inherits(control, want)) {
      stop("engine = \"", engine, "\" needs a `", want, "` object; got `",
           class(control)[1L], "`. Build it with ", ctor, ".\n",
           "  The two control objects are not interchangeable and are never ",
           "translated: fields sharing a name mean different things ",
           "(`iterlim` is a maxLik BFGS limit vs. an nlminb limit; `n_cores` ",
           "is worker processes vs. OpenMP threads), and only ",
           "rpbnb_tmb_control() has gradtol/restarts/max_workload/",
           "parallel_tape.", call. = FALSE)
    }
  }

  # rpbnb_boundary_tests() only accepts an rpbnb_fit (from fit_rpbnb()); check
  # this up front rather than after paying for the fit.
  if (isTRUE(boundary_tests) && engine != "classic") {
    stop("boundary_tests = TRUE requires engine = \"classic\": ",
         "rpbnb_boundary_tests() only accepts an rpbnb_fit (from fit_rpbnb()), ",
         "not the TMB engine's rpbnb_tmb_fit.", call. = FALSE)
  }

  # Dependence structures the two engines do not share.
  if (engine == "classic" && identical(dependence, "independence")) {
    stop("engine = \"classic\" does not implement dependence = \"independence\" ",
         "for the random-parameter model. Use engine = \"tmb\", or ",
         "fit_bnb(dependence = \"independence\") for the fixed-parameter ",
         "model.", call. = FALSE)
  }
  # copula(par =) is simulation-only. fit_rpbnb_tmb() already rejects it;
  # fit_rpbnb() accepts and ignores it. Its signature is frozen, so rpbnb()
  # carries the stricter rule for both engines -- deliberately the safer door.
  if (inherits(dependence, "rpbnb_copula") && !is.null(dependence$par)) {
    stop("`copula(par = )` is a simulation-only argument and has no effect on ",
         "fitting: the dependence parameter is always estimated. Drop `par`, ",
         "or use `start` to set its working-scale starting value.",
         call. = FALSE)
  }

  # Automatic centring/scaling (see "Automatic centring and scaling" above).
  # `data` is reassigned to the standardized copy so the fitter below sees it;
  # `scaling` stays NULL (and is never attached to the fit) unless at least
  # one continuous predictor was found.
  scaling <- NULL
  if (isTRUE(standardize)) {
    cvars <- if (is.null(continuous_vars)) {
      .identify_continuous_vars(formula_1, formula_2, data)
    } else {
      miss <- setdiff(continuous_vars, names(data))
      if (length(miss)) {
        stop("`continuous_vars` not found in `data`: ",
             paste(miss, collapse = ", "), call. = FALSE)
      }
      continuous_vars
    }
    if (length(cvars)) {
      scaling <- .compute_scaling(data, cvars)
      data <- .apply_scaling(data, scaling)
    }
  }

  args <- c(list(formula_1 = formula_1, formula_2 = formula_2, data = data,
                 random_1 = random_1, random_2 = random_2,
                 draws = draws, seed = seed, start = start,
                 dependence = dependence, control = control,
                 poisson_1 = poisson_1, poisson_2 = poisson_2),
            dots)
  fit <- do.call(this_fit, args)
  if (!is.null(scaling)) {
    fit$scaling <- scaling
    fit$continuous_vars <- names(scaling)
  }

  # Boundary LR tests (see "Boundary LR tests" above). `data` is already the
  # standardized copy at this point when standardize = TRUE, matching what
  # `fit` was actually estimated on -- rpbnb_boundary_tests() refits restricted
  # models against it and needs that to be the same design. compute_se is
  # forced off for the restricted refits: the LR test needs only logLik and
  # df, not their standard errors.
  if (isTRUE(boundary_tests)) {
    bt_control <- control
    bt_control$compute_se <- FALSE
    if (is.null(control$print_level) || control$print_level > 0) {
      n_sd   <- length(fit$rand_idx1) + length(fit$rand_idx2)
      n_disp <- sum(!isTRUE(fit$poisson_1), !isTRUE(fit$poisson_2))
      n_tot  <- n_sd + n_disp
      message(sprintf(
        "rpbnb(): running boundary LR tests (%d restricted refit%s: %d random-coefficient SD%s, %d NB2 dispersion%s)...",
        n_tot, if (n_tot == 1L) "" else "s",
        n_sd, if (n_sd == 1L) "" else "s",
        n_disp, if (n_disp == 1L) "" else "s"))
    }
    fit$boundary_tests <- rpbnb_boundary_tests(fit, data = data, control = bt_control)
  }
  fit
}
