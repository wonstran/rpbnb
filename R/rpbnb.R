#' Fit a random-parameter bivariate NB model with either engine
#'
#' A common front end over the package's two estimation engines. `engine =
#' "cpp"` calls [fit_rpbnb()] (Rcpp/OpenMP simulated likelihood, `maxLik` BFGS);
#' `engine = "tmb"` calls [fit_rpbnb_tmb()] (TMB automatic differentiation,
#' `nlminb` with restart polish). Both fitters remain exported and can be called
#' directly; this wrapper adds nothing to the fit itself and returns exactly
#' what the chosen fitter returns.
#'
#' What it does add is argument checking across the two APIs. The engines do not
#' take the same arguments, and passing one engine's argument to the other is a
#' mistake that is easy to make and expensive to notice — a standardized
#' coefficient table printed under an "original units" heading looks perfectly
#' plausible. Every extra argument is therefore matched by name against the
#' selected fitter's own formals, and anything that does not belong is an error
#' rather than a silently ignored `...` entry.
#'
#' # Which arguments go with which engine
#'
#' | Argument | `engine = "cpp"` | `engine = "tmb"` |
#' | --- | --- | --- |
#' | `draw_type`, `.fixed`, `.opt_draws` | yes | error |
#' | `inference`, `keep`, `method` | error | yes |
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
#' @param engine Estimation engine: `"cpp"` (default) or `"tmb"`.
#' @param random_1,random_2 Random-coefficient specifications for each equation.
#' @param draws Number of simulation draws.
#' @param seed Random seed for the draw sequence.
#' @param start Optional named or unnamed starting values.
#' @param dependence `"famoye"` (default), a [copula()] object, or
#'   `"independence"` (TMB engine only).
#' @param poisson_1,poisson_2 Restrict the corresponding margin to its Poisson
#'   limit.
#' @param control An `rpbnb_control()` object when `engine = "cpp"`, or an
#'   `rpbnb_tmb_control()` object when `engine = "tmb"`. Defaults to the right
#'   one for the chosen engine. The two are not interchangeable and are never
#'   translated into one another.
#' @param ... Further arguments passed to the selected fitter. Names are
#'   validated against that fitter's formals; an argument belonging to the other
#'   engine, or an unrecognised name, is an error.
#'
#' @return The engine-native fit object, identical to what a direct call to the
#'   underlying fitter would return: an object of class `rpbnb_fit` for
#'   `engine = "cpp"`, or `rpbnb_tmb_fit` for `engine = "tmb"`. The class
#'   therefore depends on `engine`; test with `inherits(fit, "rpbnb_tmb_fit")`
#'   if you need to branch. No wrapper class is introduced, so every existing S3
#'   method and post-estimation function works unchanged.
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
                  engine = c("cpp", "tmb"),
                  random_1 = NULL, random_2 = NULL,
                  draws = 400, seed = 1234, start = NULL,
                  dependence = "famoye",
                  poisson_1 = FALSE, poisson_2 = FALSE,
                  control = NULL,
                  ...) {
  engine <- match.arg(engine)

  this_fit  <- if (engine == "cpp") fit_rpbnb     else fit_rpbnb_tmb
  other_fit <- if (engine == "cpp") fit_rpbnb_tmb else fit_rpbnb
  other_nm  <- if (engine == "cpp") "tmb"         else "cpp"

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
           "\n  cpp-only: draw_type, .fixed, .opt_draws",
           "\n  tmb-only: inference, keep, method",
           "\n  See ?rpbnb for the full argument matrix.", call. = FALSE)
    }
  }

  # Engine-typed control. The default must NOT be rpbnb_control() in the
  # signature: a default evaluated at call time would hand the cpp control
  # object to the TMB engine.
  if (is.null(control)) {
    control <- if (engine == "cpp") rpbnb_control() else rpbnb_tmb_control()
  } else {
    want <- if (engine == "cpp") "rpbnb_control"   else "rpbnb_tmb_control"
    ctor <- if (engine == "cpp") "rpbnb_control()" else "rpbnb_tmb_control()"
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

  # Dependence structures the two engines do not share.
  if (engine == "cpp" && identical(dependence, "independence")) {
    stop("engine = \"cpp\" does not implement dependence = \"independence\" ",
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

  args <- c(list(formula_1 = formula_1, formula_2 = formula_2, data = data,
                 random_1 = random_1, random_2 = random_2,
                 draws = draws, seed = seed, start = start,
                 dependence = dependence, control = control,
                 poisson_1 = poisson_1, poisson_2 = poisson_2),
            dots)
  do.call(this_fit, args)
}
