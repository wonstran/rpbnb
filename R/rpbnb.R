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
#' # Boundary LR tests
#'
#' `boundary_tests` runs a likelihood-ratio test on the fit as soon as it
#' converges (against the same data the fit itself used -- the standardized
#' copy, when `standardize = TRUE`) and attaches the result as
#' `$boundary_tests`. It is a switch over three independent groups, so the
#' cost is paid only for the parameters actually in question:
#'
#' | `boundary_tests` | tests | restricted refits |
#' | --- | --- | --- |
#' | `FALSE` / `"none"` | nothing | 0 |
#' | `TRUE` | `c("sd", "dispersion")` | one per scale + one per free `m` |
#' | `"sd"` | random-coefficient scales | one per scale |
#' | `"dispersion"` | overdispersions `m1`, `m2` | one per free `m` |
#' | `"dependence"` | the association parameter | 1 |
#' | `"all"` | all three groups | all of the above |
#'
#' The random-coefficient SDs and the NB2 dispersions (`m1`, `m2`) have a null
#' that sits on the boundary of the parameter space, so an ordinary Wald
#' `z`/`p` does not test it; the boundary test refits each restricted model
#' instead and `summary()`/`print()` show that test's `LR`/`df`/`p` for those
#' rows in the natural-scale block, in place of the `NA` they would otherwise
#' carry.
#'
#' The **dependence** test is the odd one out and is therefore opt-in even
#' under `boundary_tests = TRUE`. Its null is "no association", i.e. the
#' independence model, which for Famoye (`lam`), Frank (`theta`), and the
#' Gaussian copula (`rho`) sits in the *interior* of the parameter space --
#' the Wald `z` those rows already show is valid, and the LR statistic is an
#' ordinary chi-square(1), not the 50:50 boundary mixture. Only Clayton /
#' Kimeldorf (`theta > 0`) has a genuine boundary null there. Requesting it
#' replaces the dependence row's Wald `z`/`p` with the LR test on both
#' engines, since the two answer the same question.
#'
#' Both engines test the same parameters via [rpbnb_boundary_tests()]
#' (`engine = "classic"`) or [rpbnb_tmb_boundary_tests()]
#' (`engine = "tmb"`). They differ only in how each restricted fit is
#' constructed while preserving common random numbers: for a scale, the
#' classic engine zeroes that coefficient's draw column while the TMB engine
#' pins its `log_sd` and maps it out of the free parameters; for the
#' dependence parameter, the TMB engine refits with its own
#' `dependence = "independence"` family while the classic engine (which has no
#' such fitter) pins the working-scale dependence parameter at its family's
#' independence value.
#'
#' A [message()] reports how many restricted refits are about to run before
#' they start, unless `control$print_level` is `0`; suppress it with
#' [suppressMessages()] if needed.
#'
#' Under `engine = "tmb"` with `method = "laplace"`, a restricted refit that
#' fails to converge is automatically retried with both sides of that one LR
#' estimated by `method = "sml"` rather than reported `NA` -- some
#' restrictions leave Laplace no valid optimum at all (see
#' [rpbnb_tmb_boundary_tests()]'s `sml_fallback` argument, which is where to
#' turn this off).
#'
#' `force_parallel_gaussian` (`engine = "tmb"` only, passed via `...`) is
#' forwarded to every restricted refit, so a Gaussian-copula fit's boundary
#' tests honor `control$n_cores` the same way the original fit did instead
#' of silently re-capping each refit to one thread -- see
#' [rpbnb_tmb_boundary_tests()]'s own `force_parallel_gaussian` argument for
#' why this needs forwarding at all (the fit object does not record whether
#' the override was used).
#'
#' Each restricted refit costs roughly as much as the original fit (more for
#' a [copula()] dependence than for `"famoye"`; see [rpbnb_boundary_tests()]'s
#' timing note), so this defaults to `FALSE`. `boundary_draws` (`engine =
#' "tmb"` only) sets a `draws` for those refits other than the main fit's --
#' e.g. more draws for a more precise boundary test without re-fitting the
#' whole model at that `draws`. It has no classic-engine counterpart:
#' [rpbnb_boundary_tests()] reuses the full fit's exact stored draw matrix
#' (zeroing a column) rather than regenerating draws from a count, so passing
#' `boundary_draws` under `engine = "classic"` is an error. For finer control
#' still -- testing only `"sd"` or only `"dispersion"` (both engines take a
#' `which` argument), or reusing one boundary-test run across several
#' summaries -- call [rpbnb_boundary_tests()]/[rpbnb_tmb_boundary_tests()]
#' directly on the fit and assign its result to `fit$boundary_tests` (with
#' `standardize = TRUE`, reconstruct the fitting-scale data first:
#' `rpbnb:::.apply_scaling(data, fit$scaling)`).
#'
#' # Which arguments go with which engine
#'
#' | Argument | `engine = "classic"` | `engine = "tmb"` |
#' | --- | --- | --- |
#' | `draw_type`, `.fixed`, `.opt_draws` | yes | error |
#' | `inference`, `keep` | error | yes |
#' | `method`, `force_parallel_gaussian` | ignored with a warning | yes |
#' | `offset()` in a formula | yes | error |
#' | `dependence = "independence"` | error | yes |
#' | `boundary_draws` (non-`NULL`) | error | yes |
#' | `control` class | `rpbnb_control` | `rpbnb_control` (same object) |
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
#' @param boundary_tests Which groups of parameters to LR-test after fitting,
#'   via [rpbnb_boundary_tests()] (`engine = "classic"`) or
#'   [rpbnb_tmb_boundary_tests()] (`engine = "tmb"`); the result is attached as
#'   `$boundary_tests` and `summary()`/`print()` show the test (rather than
#'   `NA`, or rather than a Wald `z`) on the corresponding rows. Accepts:
#'
#'   * `FALSE` (default) or `"none"` -- run nothing. Each restricted refit
#'     costs roughly another full fit.
#'   * `TRUE` -- `c("sd", "dispersion")`, the two boundary-null groups. This is
#'     what `TRUE` has always meant and it does not silently grow.
#'   * a character vector, any of `"sd"` (random-coefficient scales),
#'     `"dispersion"` (the NB2 overdispersions `m1`, `m2`), `"dependence"`
#'     (the association parameter), or `"all"` for all three.
#'
#'   So `boundary_tests = "dispersion"` tests overdispersion only,
#'   `boundary_tests = c("dispersion", "dependence")` tests overdispersion and
#'   association without paying for one refit per random-coefficient scale, and
#'   `boundary_tests = "all"` tests everything. See "Boundary LR tests" below.
#' @param boundary_draws Number of Halton simulation draws for the boundary
#'   tests' restricted refits, `engine = "tmb"` only. `NULL` (default) uses
#'   the main fit's `draws`. Ignored when no boundary test was requested
#'   (nothing to apply it to); an error if supplied under `engine =
#'   "classic"` alongside a boundary test (see "Boundary LR tests"
#'   above -- that engine has no `draws` to override).
#' @param control An [rpbnb_control()] object -- one object for both engines,
#'   as of 0.4.1 (`rpbnb_tmb_control()` is a retained alias that returns the
#'   same thing). Settings the chosen engine does not read are ignored and
#'   listed by `print()`/`summary()` of the fit; `iterlim` and `print_level`,
#'   whose defaults differ between the engines, resolve to the chosen engine's
#'   own default unless you set them. Fields sharing a name are still not
#'   translated: `iterlim` is a `maxLik` BFGS limit under `engine = "classic"`
#'   and an `nlminb` limit under `"tmb"`, and `n_cores` is worker processes
#'   versus OpenMP threads.
#' @param ... Further arguments passed to the selected fitter. Names are
#'   validated against that fitter's formals; an argument belonging to the other
#'   engine, or an unrecognised name, is an error. Exception: the TMB tuning
#'   knobs `method` and `force_parallel_gaussian` are dropped with a warning
#'   (not an error) under `engine = "classic"`, so a call can switch engines
#'   without stripping them.
#'
#' @return The engine-native fit object, identical to what a direct call to the
#'   underlying fitter would return: an object of class `rpbnb_fit` for
#'   `engine = "classic"`, or `rpbnb_tmb_fit` for `engine = "tmb"`. The class
#'   therefore depends on `engine`; test with `inherits(fit, "rpbnb_tmb_fit")`
#'   if you need to branch. No wrapper class is introduced, so every existing S3
#'   method and post-estimation function works unchanged. With
#'   `standardize = TRUE`, two extra fields are attached -- `$scaling` and
#'   `$continuous_vars` -- and `print()`/`summary()` use them to display
#'   original-units coefficients. With any `boundary_tests` group requested, a
#'   third field `$boundary_tests` (the [rpbnb_boundary_tests()] result) is
#'   attached, and `print()`/`summary()` use it to show the LR test on the
#'   corresponding rows. Both fitters also attach `$control_ignored` /
#'   `$control_engine`, the control settings the chosen engine did not read.
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
                  boundary_tests = FALSE, boundary_draws = NULL,
                  control = NULL,
                  ...) {
  engine <- match.arg(engine)
  # Validated up front, not at the point of use: a typo in the group name would
  # otherwise surface only after the full fit had already been paid for.
  bt_which <- .normalize_boundary_tests(boundary_tests)

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
    # `method` and `force_parallel_gaussian` are TMB tuning knobs with no
    # classic-engine meaning at all, so a script flipping engine = "tmb" to
    # "classic" need not strip them: drop with a warning instead of erroring.
    # Everything else keeps the hard error -- those names select behaviour the
    # caller presumably wanted.
    if (engine == "classic") {
      ignorable <- intersect(nm, c("method", "force_parallel_gaussian"))
      if (length(ignorable)) {
        warning(paste0("`", ignorable, "`", collapse = ", "),
                " ignored: tmb-only, and engine = \"classic\" was chosen.",
                call. = FALSE)
        dots <- dots[!nm %in% ignorable]
        nm <- names(dots)
      }
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

  # One control object drives both engines (see R/control.R). Resolving it here
  # -- rather than leaving it to the fitter -- is what makes `control` usable in
  # this function too: the boundary-tests message below reads `print_level`, and
  # an unresolved object carries NULL there, which would print under the TMB
  # engine where the default is silence.
  if (is.null(control)) control <- rpbnb_control()
  control <- .resolve_control(control,
                             if (engine == "classic") "classic" else "tmb")

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
  # `fit` was actually estimated on -- both boundary-test functions refit
  # restricted models against it and need that to be the same design.
  # `bt_which` was normalized (and validated) at the top of the function.
  if (length(bt_which)) {
    n_sd   <- if ("sd" %in% bt_which) {
      length(fit$rand_idx1) + length(fit$rand_idx2)
    } else 0L
    n_disp <- if ("dispersion" %in% bt_which) {
      sum(!isTRUE(fit$poisson_1), !isTRUE(fit$poisson_2))
    } else 0L
    n_dep  <- if ("dependence" %in% bt_which &&
                  !identical(dependence, "independence")) 1L else 0L
    n_tot  <- n_sd + n_disp + n_dep
    if (is.null(control$print_level) || control$print_level > 0) {
      message(sprintf(
        "rpbnb(): running boundary LR tests (%d restricted refit%s: %d random-coefficient scale%s, %d NB2 dispersion%s, %d dependence parameter%s)...",
        n_tot, if (n_tot == 1L) "" else "s",
        n_sd, if (n_sd == 1L) "" else "s",
        n_disp, if (n_disp == 1L) "" else "s",
        n_dep, if (n_dep == 1L) "" else "s"))
    }
    if (engine == "classic") {
      # No `draws` knob to honor here: rpbnb_boundary_tests() reuses the full
      # fit's exact stored draw matrix (zeroing a column) rather than
      # regenerating draws from a count, unlike the TMB engine below.
      if (!is.null(boundary_draws)) {
        stop("`boundary_draws` is only supported for engine = \"tmb\": the ",
             "classic engine's rpbnb_boundary_tests() reuses the full fit's ",
             "exact stored draw matrix rather than regenerating draws from a ",
             "count, so there is no `draws` to override.", call. = FALSE)
      }
      # compute_se is forced off for the restricted refits: the LR test needs
      # only logLik and df, not their standard errors.
      bt_control <- control
      bt_control$compute_se <- FALSE
      fit$boundary_tests <- rpbnb_boundary_tests(fit, data = data,
                                                 control = bt_control,
                                                 which = bt_which)
    } else {
      # force_parallel_gaussian is tmb-only and reaches the main fit through
      # `dots` (validated above); the boundary refits need it forwarded
      # explicitly too, since rpbnb_tmb_boundary_tests() has no way to read
      # it back off `fit` -- see its own force_parallel_gaussian argument doc.
      fit$boundary_tests <- rpbnb_tmb_boundary_tests(
        fit, data = data, which = bt_which,
        draws = if (is.null(boundary_draws)) fit$draws else boundary_draws,
        force_parallel_gaussian = isTRUE(dots$force_parallel_gaussian))
    }
  }
  fit
}
