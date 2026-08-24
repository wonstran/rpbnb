# One control object for every estimator in the package.
#
# Until 0.4.x there were two constructors -- rpbnb_control() for the Rcpp/maxLik
# engines and rpbnb_tmb_control() for the TMB engine -- and they were explicitly
# NOT interchangeable: passing one to the other errored. That rule protected a
# real hazard (fields sharing a name mean different things) but it also meant a
# script could not switch `engine =` without rewriting its control call, and the
# two constructors' overlapping fields drifted apart. They are now ONE object:
# rpbnb_control() carries the union of both parameter sets, and every fitter
# takes it. rpbnb_tmb_control() is retained as a thin alias so existing code and
# saved scripts keep working.
#
# The union brings two problems, both handled here rather than at the call sites:
#
# 1. A parameter that means nothing to the estimator you actually called (an
#    `se_method` under the TMB engine, a `gradtol` under maxLik). These are
#    IGNORED rather than rejected -- the whole point of one object is that the
#    same control can be handed to either engine -- and the names that were
#    ignored are recorded on the fit (`$control_ignored`) so print()/summary()
#    can say so out loud. Silence would be the actual hazard; an error would
#    defeat the unification.
#
# 2. Two fields whose historical defaults DISAGREED between the constructors:
#    `iterlim` (300 maxLik / 500 nlminb) and `print_level` (2 maxLik / 0 nlminb).
#    Picking one number would silently change one engine's behaviour, so these
#    default to NULL, meaning "whatever this estimator has always used", and are
#    filled in by .resolve_control() once the estimator is known. An explicitly
#    supplied value is always honored, for both engines.

# Every field the control object carries, in constructor order. Used to build
# the "ignored" report, so a new field must be added here as well as to the
# applicability table below.
.CONTROL_ALL_FIELDS <- c(
  "method", "iterlim", "reltol", "print_level", "draws_hessian", "halton_burn",
  "n_cores", "compute_se", "hessian", "se_method", "hess_eps", "hess_r",
  "gradtol", "restarts", "max_threads", "max_workload", "parallel_tape",
  "tape_chunks"
)

# Which fields each estimator actually READS. Derived by hand from the
# `control$<field>` reads in R/fit_bnb.R, R/fit_rpbnb.R, R/fit_rpbnb_copula.R,
# R/fit_rpbnb_tmb.R and R/tmb_helpers.R -- keep it in step with those, because
# a field listed as applicable but never read reports a setting as honored when
# it is not, which is the failure this table exists to prevent.
#
# `draws_hessian` is deliberately in NO estimator's list: it has been a
# documented no-op since the random-parameter Hessian moved to same-draw
# curvature, so supplying it is always reported as ignored.
.CONTROL_APPLICABLE <- list(
  # fit_rpbnb() and .fit_rpbnb_copula() -- maxLik BFGS, MSL with Halton draws.
  classic = c("method", "iterlim", "reltol", "print_level", "halton_burn",
              "n_cores", "compute_se", "se_method", "hess_eps", "hess_r"),
  # fit_bnb() -- fixed-coefficient BNB; no simulation draws, no cluster path,
  # and `hessian` (not `se_method`) selects its numeric/analytic information.
  bnb     = c("method", "iterlim", "reltol", "print_level", "compute_se",
              "hessian", "hess_eps", "hess_r"),
  # fit_rpbnb_tmb() -- nlminb + restart polish over a TMB tape.
  tmb     = c("iterlim", "reltol", "print_level", "halton_burn", "n_cores",
              "gradtol", "restarts", "max_threads", "max_workload",
              "parallel_tape", "tape_chunks")
)

# The fields whose two historical constructors disagreed (see the header note).
# NULL on the control object means "unset"; these values fill it in.
.CONTROL_ENGINE_DEFAULTS <- list(
  classic = list(iterlim = 300L, print_level = 2L),
  bnb     = list(iterlim = 300L, print_level = 2L),
  tmb     = list(iterlim = 500L, print_level = 0L)
)

# Human-readable estimator names for the ignored-settings note.
.CONTROL_ENGINE_LABEL <- c(
  classic = "the classic (Rcpp/maxLik) random-parameter engine",
  bnb     = "the fixed-coefficient fit_bnb() estimator",
  tmb     = "the TMB engine"
)

#' Control parameters for every rpbnb estimator
#'
#' One control object for all of the package's fitters -- [fit_bnb()],
#' [fit_rpbnb()], [fit_rpbnb_tmb()], and [rpbnb()] with either `engine`. It
#' carries the union of the tuning knobs the estimators use; each fitter reads
#' the ones that apply to it and **ignores the rest**, reporting the ignored
#' names in `print()`/`summary()` of the resulting fit rather than erroring.
#' That is what makes a script able to flip `engine = "classic"` to `"tmb"`
#' without rewriting its control call.
#'
#' @section Which parameters apply to which estimator:
#'
#' | Parameter | [fit_bnb()] | [fit_rpbnb()] | [fit_rpbnb_tmb()] |
#' | --- | --- | --- | --- |
#' | `method`, `compute_se`, `hess_eps`, `hess_r` | yes | yes | ignored |
#' | `iterlim`, `reltol`, `print_level` | yes | yes | yes |
#' | `halton_burn`, `n_cores` | ignored | yes | yes |
#' | `hessian` | yes | ignored | ignored |
#' | `se_method` | ignored | yes | ignored |
#' | `gradtol`, `restarts`, `max_threads`, `max_workload`, `parallel_tape` | ignored | ignored | yes |
#' | `draws_hessian` | ignored | ignored | ignored |
#'
#' Note that `iterlim` and `n_cores` mean different things to different
#' estimators -- a `maxLik` BFGS iteration limit versus an `nlminb` one, worker
#' *processes* versus OpenMP *threads*. They are not translated; each estimator
#' reads the number and applies its own meaning to it.
#'
#' @section Defaults that depend on the estimator:
#'
#' `iterlim` and `print_level` default to `NULL`, which means "this estimator's
#' own long-standing default": `iterlim` is 300 under `maxLik` and 500 under
#' `nlminb`; `print_level` is 2 (progress) under `maxLik` and 0 (silent) under
#' `nlminb`. Supplying either explicitly overrides that for every estimator.
#' `max_threads` defaults to `n_cores` and `max_workload` is computed from
#' available memory by [rpbnb_tmb_max_workload()] the first time a TMB fit
#' needs it (so a non-TMB fit never pays for the memory probe).
#'
#' @param method Optimizer used by the `maxLik` fitters. Only "BFGS" is
#'   implemented and wired through; it is the sole accepted value.
#' @param iterlim Maximum optimizer iterations. `NULL` (default) uses 300 for
#'   the `maxLik` fitters and 500 for the TMB engine's `nlminb`.
#' @param reltol Relative convergence tolerance.
#' @param print_level Optimizer verbosity, and the switch that silences the
#'   boundary-test progress messages. `NULL` (default) uses 2 for the `maxLik`
#'   fitters and 0 (silent) for the TMB engine.
#' @param draws_hessian Retained for backward compatibility but unused by every
#'   estimator: the random-parameter numeric Hessian is taken with the same
#'   optimization draws that produced the estimate (same-draw curvature), so it
#'   no longer resimulates a separate Hessian draw set. Supplying it is always
#'   reported as an ignored setting.
#' @param halton_burn Number of leading Halton points discarded before forming
#'   the simulation draws.
#' @param n_cores Worker processes for [fit_rpbnb()]'s optional cluster path, or
#'   OpenMP threads for [fit_rpbnb_tmb()] (1 = sequential in both cases).
#' @param compute_se If FALSE, skip the Hessian and standard errors. The TMB
#'   engine has its own `inference` argument for this instead.
#' @param hessian How [fit_bnb()] (famoye) computes the Hessian for standard
#'   errors: "numeric" (default, [numDeriv::hessian()]) or "analytic" (the
#'   closed-form Famoye (2010) Appendix Hessian). Both freeze the lambda-bounds
#'   at the optimum and yield the same observed-information SEs.
#' @param se_method Standard-error method for [fit_rpbnb()] (the random-parameter
#'   model): "numeric" (default) uses the [numDeriv::hessian()] observed-
#'   information Hessian; "analytic" uses the closed-form observed-information
#'   Hessian (Famoye (2010) per-draw second derivatives assembled via the Louis
#'   mixture formula) -- exact and much faster than "numeric" for larger models;
#'   "opg" uses the BHHH / outer-product-of-gradients information from the
#'   per-observation scores -- fastest, but relies on the information-matrix
#'   equality so it is unreliable for parameters at a boundary (e.g. a random-
#'   coefficient SD estimated near 0). For copula dependence ([fit_rpbnb()] with
#'   `dependence = copula(...)`), only "opg" (recommended) and "numeric" are
#'   available; "analytic" is not implemented for the copula path and errors.
#' @param hess_eps,hess_r Step and Richardson order for [numDeriv::hessian()]
#'   (used only when the numeric Hessian is selected).
#' @param gradtol Stationarity tolerance for the TMB engine's score, applied
#'   \emph{relative} to the objective: the fit is treated as stationary once
#'   \code{max(abs(gradient))} falls below \code{gradtol * max(1, abs(nll))}.
#'   \code{nlminb()} declares convergence from its relative function test,
#'   which can fire while the score is still far from zero; a Hessian taken
#'   there is not a curvature and its inverse can carry negative variances. The
#'   fit is restarted until the gradient clears this tolerance; missing it is
#'   not warned about on its own, because the copula families legitimately
#'   optimise against a clamp or a probability floor where no step improves the
#'   objective, but it is reported in the warning raised when the Hessian
#'   actually fails to factor, and is kept on the fit as
#'   \code{optimizer$max_abs_gradient}. The scaling matters because the score
#'   is a sum over observations, so an absolute cutoff would tighten with
#'   sample size for no statistical reason.
#' @param restarts Maximum number of times to restart \code{nlminb()} from its
#'   own answer while \code{gradtol} is unmet. Restarting resets the trust
#'   region and step scaling that stalled the first solve. Set to \code{0L} for
#'   the single-call behaviour of earlier versions.
#' @param max_threads Maximum OpenMP threads permitted for one TMB fit.
#'   `NULL` (default) means \code{n_cores}, so threads are not capped unless
#'   set explicitly below \code{n_cores}.
#' @eval .calibration_doc()
#' @param parallel_tape Construct per-thread TMB tapes concurrently. The
#'   default \code{FALSE} constructs them sequentially to reduce peak memory;
#'   objective and gradient evaluation remains parallel.
#' @param tape_chunks TMB engine, SML fits only. Number of draw chunks to
#'   split \code{draws} into (see \code{draws} at [fit_rpbnb_tmb()]).
#'   \code{NULL} (default) auto-selects the smallest sufficient count when
#'   the weighted workload exceeds \code{max_workload}, or \code{1L} (no
#'   chunking) when it does not. Set explicitly to pin a layout regardless of
#'   the auto-threshold; must not exceed \code{draws}. Chunking is exact for
#'   the requested draws (not an approximation) at the cost of somewhat
#'   slower gradient evaluations, and a chunked fit has no taped Hessian --
#'   \code{confint(method = "profile")}/\code{rpbnb_tmb_dependence_profile()}
#'   fall back to a Wald interval with a warning; Wald/optimHess inference
#'   (the default) is unaffected. Ignored for \code{method = "laplace"}
#'   (which has no draw dimension to chunk) and by every non-TMB estimator.
#'
#' @return An object of class `c("rpbnb_control", "rpbnb_tmb_control")` (a named
#'   list). It carries both class names so that every historical
#'   `inherits(control, ...)` check in the package and in user code accepts it.
#' @seealso [rpbnb_tmb_max_workload()], [rpbnb()], [fit_rpbnb()],
#'   [fit_rpbnb_tmb()], [fit_bnb()]
#' @export
#' @examples
#' rpbnb_control(method = "BFGS", iterlim = 200)
#' rpbnb_control(hessian = "analytic")
#' # The same object drives either engine; the TMB-only knobs are simply
#' # ignored by the classic one (and reported as ignored in its summary).
#' rpbnb_control(n_cores = 4, gradtol = 1e-6, se_method = "opg")
rpbnb_control <- function(method = c("BFGS"),
                          iterlim = NULL,
                          reltol = 1e-8,
                          print_level = NULL,
                          draws_hessian = 100L,
                          halton_burn = 300L,
                          n_cores = 1L,
                          compute_se = TRUE,
                          hessian = c("numeric", "analytic"),
                          se_method = c("numeric", "opg", "analytic"),
                          hess_eps = 1e-5,
                          hess_r = 4L,
                          gradtol = 1e-5,
                          restarts = 10L,
                          max_threads = NULL,
                          max_workload = NULL,
                          parallel_tape = FALSE,
                          tape_chunks = NULL) {
  # match.call() names positionally-supplied arguments too (this function has no
  # `...`), so this is the set of names the caller actually wrote -- which is
  # what the ignored-settings report must be based on. Reporting every
  # non-applicable field instead would name knobs the user never touched.
  supplied <- names(as.list(match.call()))[-1L]
  supplied <- intersect(supplied[nzchar(supplied)], .CONTROL_ALL_FIELDS)

  if (length(method) > 1) method <- method[1]
  if (!identical(method, "BFGS")) {
    stop("`method` must be \"BFGS\" (the only implemented optimizer).",
         call. = FALSE)
  }
  if (length(hessian) > 1) hessian <- hessian[1]
  if (!hessian %in% c("numeric", "analytic")) {
    stop("`hessian` must be one of: numeric, analytic", call. = FALSE)
  }
  if (length(se_method) > 1) se_method <- se_method[1]
  if (!se_method %in% c("opg", "numeric", "analytic")) {
    stop("`se_method` must be one of: numeric, analytic, opg", call. = FALSE)
  }

  # Validate BEFORE coercion: as.integer() is what turns a bad value into a
  # silent NA. A negative halton_burn in particular used to break the draw
  # contract outright -- .tmb_halton_uniform() indexes
  # (burn + 1):(burn + n_draws), so a negative burn returned FEWER rows than
  # requested and the likelihood was averaged over a grid of the wrong size.
  .whole_scalar <- function(x, nm, min) {
    if (length(x) != 1L || !is.numeric(x) || is.na(x) || !is.finite(x) ||
        x < min || x != floor(x) || x > .Machine$integer.max) {
      stop(nm, " must be one whole number greater than or equal to ", min, ".",
           call. = FALSE)
    }
  }
  .pos_scalar <- function(x, nm) {
    if (length(x) != 1L || !is.numeric(x) || is.na(x) || !is.finite(x) ||
        x <= 0) {
      stop(nm, " must be one positive finite number.", call. = FALSE)
    }
  }
  if (!is.null(iterlim)) .whole_scalar(iterlim, "iterlim", 1)
  if (!is.null(print_level)) .whole_scalar(print_level, "print_level", 0)
  .whole_scalar(halton_burn, "halton_burn", 0)
  .whole_scalar(n_cores, "n_cores", 1)
  .whole_scalar(draws_hessian, "draws_hessian", 1)
  .whole_scalar(restarts, "restarts", 0)
  .whole_scalar(hess_r, "hess_r", 1)
  if (!is.null(max_threads)) .whole_scalar(max_threads, "max_threads", 1)
  .pos_scalar(reltol, "reltol")
  .pos_scalar(gradtol, "gradtol")
  .pos_scalar(hess_eps, "hess_eps")
  if (!is.null(max_workload) &&
      (length(max_workload) != 1L || !is.numeric(max_workload) ||
       is.na(max_workload) || max_workload <= 0)) {
    stop("max_workload must be one positive number or Inf.", call. = FALSE)
  }
  if (!is.logical(parallel_tape) || length(parallel_tape) != 1L ||
      is.na(parallel_tape)) {
    stop("parallel_tape must be one non-missing logical value.", call. = FALSE)
  }
  if (!is.logical(compute_se) || length(compute_se) != 1L ||
      is.na(compute_se)) {
    stop("compute_se must be one non-missing logical value.", call. = FALSE)
  }
  # NULL or a positive whole number only: a control object is reusable
  # across fits with different draw counts, so the constructor cannot
  # validate tape_chunks <= draws here. That check happens in
  # .resolve_tape_chunks() at fit time, where both values are known.
  if (!is.null(tape_chunks)) .whole_scalar(tape_chunks, "tape_chunks", 1)

  n_cores <- as.integer(n_cores)
  structure(
    list(method = method,
         iterlim = if (is.null(iterlim)) NULL else as.integer(iterlim),
         reltol = as.numeric(reltol),
         print_level = if (is.null(print_level)) NULL else as.integer(print_level),
         draws_hessian = as.integer(draws_hessian),
         halton_burn = as.integer(halton_burn),
         n_cores = n_cores,
         compute_se = isTRUE(compute_se),
         hessian = hessian,
         se_method = se_method,
         hess_eps = as.numeric(hess_eps),
         hess_r = as.integer(hess_r),
         gradtol = as.numeric(gradtol),
         restarts = as.integer(restarts),
         # Not engine-dependent (both constructors defaulted it to n_cores), so
         # it is resolved here rather than in .resolve_control().
         max_threads = if (is.null(max_threads)) n_cores else as.integer(max_threads),
         max_workload = if (is.null(max_workload)) NULL else as.numeric(max_workload),
         parallel_tape = parallel_tape,
         tape_chunks = if (is.null(tape_chunks)) NULL else as.integer(tape_chunks)),
    supplied = supplied,
    class = c("rpbnb_control", "rpbnb_tmb_control")
  )
}

#' Control parameters for the TMB engine (alias of `rpbnb_control()`)
#'
#' Retained so that code written against the pre-unification API keeps working.
#' The two control objects were merged in 0.4.1: this function forwards to
#' [rpbnb_control()] and returns exactly the same object, which every estimator
#' in the package accepts. New code should call [rpbnb_control()] directly.
#'
#' Only the arguments you actually supply are forwarded, so an untouched
#' `iterlim`/`print_level` still resolves to the TMB engine's own defaults (500
#' and 0) when the object is used for a TMB fit -- and to the `maxLik` defaults
#' if the same object is handed to [fit_rpbnb()].
#'
#' @inheritParams rpbnb_control
#' @return The [rpbnb_control()] object.
#' @seealso [rpbnb_control()]
#' @export
#' @examples
#' identical(unclass(rpbnb_tmb_control(n_cores = 2)),
#'           unclass(rpbnb_control(n_cores = 2)))
rpbnb_tmb_control <- function(iterlim = NULL,
                              reltol = 1e-8,
                              gradtol = 1e-5,
                              restarts = 10L,
                              print_level = NULL,
                              n_cores = 1L,
                              max_threads = NULL,
                              max_workload = NULL,
                              parallel_tape = FALSE,
                              halton_burn = 300L,
                              tape_chunks = NULL) {
  # Forward only what the caller wrote, so `supplied` on the returned object
  # records the caller's intent and not this wrapper's own signature -- an
  # untouched field must not be reported as an ignored setting later.
  nm <- names(as.list(match.call())[-1L])
  if (is.null(nm)) nm <- character(0)
  do.call(rpbnb_control, mget(nm[nzchar(nm)], envir = environment()))
}

#' Fill in the estimator-dependent defaults and record ignored settings
#'
#' Called once at the top of each fitter. Returns the control object with
#' `iterlim`/`print_level`/`max_workload` resolved for `engine`, an `"engine"`
#' attribute, and an `"ignored"` attribute naming the fields the caller supplied
#' that this estimator does not read. Idempotent for the same engine.
#' @keywords internal
#' @noRd
.resolve_control <- function(control, engine = c("classic", "tmb", "bnb")) {
  engine <- match.arg(engine)
  if (!inherits(control, "rpbnb_control")) {
    stop("`control` must be an `rpbnb_control` object from rpbnb_control() ",
         "(or its rpbnb_tmb_control() alias); got `", class(control)[1L],
         "`.", call. = FALSE)
  }
  if (identical(attr(control, "engine"), engine)) return(control)

  supplied <- attr(control, "supplied")
  if (is.null(supplied)) supplied <- character(0)
  atts <- attributes(control)

  defs <- .CONTROL_ENGINE_DEFAULTS[[engine]]
  for (f in names(defs)) {
    if (is.null(control[[f]])) control[[f]] <- defs[[f]]
  }
  # Only the TMB engine reads max_workload, and computing it probes system
  # memory (and can warn), so it is deferred to exactly the fits that need it.
  if (engine == "tmb" && is.null(control$max_workload)) {
    control$max_workload <- rpbnb_tmb_max_workload()
  }

  atts$engine <- engine
  atts$ignored <- setdiff(supplied, .CONTROL_APPLICABLE[[engine]])
  attributes(control) <- atts
  control
}

#' Ignored-settings note for print()/summary() of a fit
#'
#' `fit$control_ignored` is the character vector .resolve_control() computed,
#' `fit$control_engine` the estimator it was resolved for. Prints nothing when
#' every supplied setting was used, which is the common case.
#' @keywords internal
#' @noRd
.print_control_ignored <- function(object) {
  ig <- object$control_ignored
  if (!length(ig)) return(invisible(NULL))
  eng <- object$control_engine
  who <- if (!is.null(eng) && eng %in% names(.CONTROL_ENGINE_LABEL)) {
    .CONTROL_ENGINE_LABEL[[eng]]
  } else {
    "this estimator"
  }
  cat("  Control settings ignored (not used by ", who, "): ",
      paste(ig, collapse = ", "), "\n", sep = "")
  invisible(NULL)
}

#' @export
print.rpbnb_control <- function(x, ...) {
  cat("rpbnb control settings\n")
  eng <- attr(x, "engine")
  if (!is.null(eng)) cat("  resolved for:", eng, "\n")
  fmt <- function(v) {
    if (is.null(v)) "<estimator default>" else paste(format(v), collapse = " ")
  }
  for (f in .CONTROL_ALL_FIELDS) {
    cat(sprintf("  %-14s %s\n", f, fmt(x[[f]])))
  }
  supplied <- attr(x, "supplied")
  if (length(supplied)) {
    cat("  supplied by caller:", paste(supplied, collapse = ", "), "\n")
  }
  ignored <- attr(x, "ignored")
  if (length(ignored)) {
    cat("  ignored here:", paste(ignored, collapse = ", "), "\n")
  }
  invisible(x)
}

#' Record on a fit which supplied control settings the estimator ignored
#'
#' Attached to the returned object rather than warned about at call time: an
#' unused setting is not an error (that is the point of one shared control
#' object), but it must not be silent either, or a `se_method = "opg"` that a
#' TMB fit never read would look honored.
#' @keywords internal
#' @noRd
.attach_control_note <- function(fit, control) {
  fit$control_ignored <- attr(control, "ignored")
  fit$control_engine  <- attr(control, "engine")
  fit
}
