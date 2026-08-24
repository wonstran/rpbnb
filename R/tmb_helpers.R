# Resolve the effective TMB thread configuration for a fit's dependence
# family, applying the Gaussian-copula (family_code == 2L) single-thread
# safety cap unless overridden.
#
# Evaluating a Gaussian-copula TMB object built with more than one OpenMP
# thread has reliably crashed the R process (SIGSEGV) on the FIRST objective
# evaluation -- not a rare or data-dependent failure. The trigger is
# REGISTER_ATOMIC(gauss_cell_vec) under OpenMP; the
# `#pragma omp critical(gauss_cell_vec_init)` force-init in src/rpbnb_tmb.cpp
# does not make it re-entrant. Frank and Clayton are unaffected at any thread
# count and never see this cap (this function is a no-op for them).
#
# By default (`force_parallel_gaussian = FALSE`) this caps the request down
# to one thread with a `warning()` rather than attempting the call that would
# crash the process: a slow correct fit is strictly better than a user who
# passed a perfectly ordinary-looking `n_cores = 2L` losing their session and
# any unsaved work. `force_parallel_gaussian = TRUE` is an opt-in escape
# hatch for someone who has read this comment and still wants to try running
# multithreaded anyway (e.g. to test whether a particular TMB/OpenMP build is
# actually affected) -- it honors the request instead of capping it, with a
# louder warning naming the crash risk explicitly. It does not fix the
# underlying defect; test-parallel.R's "serial and parallel copula objectives
# and gradients agree" test remains skip()-ed for exactly this reason, and
# no test in this package exercises real multithreaded Gaussian evaluation.
#
# Deliberately a pure function (no TMB/DLL calls) that only inspects
# `family_code`/`control`/`force_parallel_gaussian` and returns what to
# configure -- so its warning/capping LOGIC is unit-testable without ever
# triggering the crash-prone evaluation itself.
#' @keywords internal
#' @noRd
.resolve_gaussian_threads <- function(family_code, control,
                                      force_parallel_gaussian = FALSE) {
  requested_cores <- control$n_cores
  out <- list(cores = requested_cores, max_threads = control$max_threads,
             parallel_tape = control$parallel_tape)
  if (!identical(family_code, 2L)) return(out)

  wants_parallel <- requested_cores > 1L || isTRUE(control$parallel_tape)
  if (!wants_parallel) return(out)

  if (isTRUE(force_parallel_gaussian)) {
    warning(
      "force_parallel_gaussian = TRUE: running the Gaussian copula with ",
      "n_cores = ", requested_cores, ", parallel_tape = ",
      isTRUE(control$parallel_tape), ". This overrides a safety cap for a ",
      "KNOWN DEFECT: multithreaded evaluation of this family (registered ",
      "Gaussian atomic, not re-entrant under OpenMP) has reliably crashed ",
      "the R process (SIGSEGV) on the first objective evaluation. Save your ",
      "work before proceeding; if R terminates unexpectedly, this override ",
      "is why.", call. = FALSE)
    return(out)
  }

  warning("Gaussian copula fits are restricted to one thread: ",
          "multithreaded evaluation of this family crashes the R process ",
          "(a known defect in the registered Gaussian atomic). Continuing ",
          "with n_cores = 1 and parallel_tape = FALSE; requested n_cores = ",
          requested_cores, ". Pass force_parallel_gaussian = TRUE to fit_rpbnb_tmb() ",
          "to override -- only if you understand and accept the crash risk ",
          "(see ?fit_rpbnb_tmb).", call. = FALSE)
  list(cores = 1L, max_threads = 1L, parallel_tape = FALSE)
}

#' Configure the TMB model DLL's OpenMP thread count
#' @keywords internal
#' @noRd
.configure_tmb_threads <- function(n_cores, max_threads = 4L,
                                   parallel_tape = FALSE,
                                   DLL = "rpbnb") {
  supported <- suppressWarnings(TMB::openmp(max = TRUE, DLL = DLL))
  supported <- as.integer(supported[[1L]])
  if (length(supported) != 1L || is.na(supported) || supported < 1L) {
    supported <- 1L
  }

  requested <- as.integer(n_cores)
  policy_capped <- min(requested, as.integer(max_threads))
  realized <- min(policy_capped, supported)
  TMB::openmp(n = realized, DLL = DLL)
  TMB::config(
    tape.parallel = as.integer(isTRUE(parallel_tape)),
    DLL = DLL
  )

  if (policy_capped < requested) {
    warning(
      sprintf(
        "Requested %d TMB threads; using %d because max_threads = %d.",
        requested, realized, as.integer(max_threads)
      ),
      call. = FALSE
    )
  } else if (realized < requested) {
    warning(
      sprintf("Requested %d TMB threads; using %d supported thread%s.",
              requested, realized, if (realized == 1L) "" else "s"),
      call. = FALSE
    )
  }

  realized
}

#' Construct the RP-BNB TMB objective with an explicit thread setting
#' @keywords internal
#' @noRd
.make_rpbnb_tmb_object <- function(data, parameters, map = NULL,
                                   random = NULL, silent = TRUE,
                                   n_cores = 1L, max_threads = 4L,
                                   parallel_tape = FALSE,
                                   DLL = "rpbnb") {
  realized <- .configure_tmb_threads(
    n_cores, max_threads = max_threads,
    parallel_tape = parallel_tape, DLL = DLL
  )
  obj <- TMB::MakeADFun(
    data = data,
    parameters = parameters,
    map = map,
    random = random,
    DLL = DLL,
    silent = silent
  )
  list(obj = obj, n_cores = realized)
}

#' Reject automatic-differentiation workloads above an explicit budget
#' @keywords internal
#' @noRd
.check_tmb_workload <- function(n, draws, family_code, max_workload,
                                n_threads = 1L, parallel_tape = FALSE) {
  if (is.infinite(max_workload)) return(invisible(0))
  # Weights are measured against PEAK working set, not assumed and not taken
  # from retained tape: see TAPE_CALIBRATION and inst/benchmark_memory.R.
  # Frank peaks at ~3.5x Famoye per unit; Clayton and Gaussian are near 1;
  # independence is cheaper.  Two earlier revisions got this wrong -- one
  # generalised "weight 1" to every family, the other derived the weights from
  # retained tape while the budget scaled peak.  Both under-budgeted Frank.
  family_weight <- unname(TAPE_CALIBRATION$family_weight[[
    match(family_code, c(-1L, 0L, 1L, 2L, 3L))
  ]])
  tape_multiplier <- if (isTRUE(parallel_tape)) as.double(n_threads) else 1
  workload <- as.double(n) * as.double(draws) *
    family_weight * tape_multiplier
  if (!is.finite(workload) || workload > max_workload) {
    stop(
      sprintf(
        paste0(
          "Weighted TMB workload is %s, above max_workload = %s. ",
          "Reduce observations or draws, or explicitly increase ",
          "control$max_workload (Inf disables this guard)."
        ),
        format(workload, scientific = FALSE, trim = TRUE),
        format(max_workload, scientific = FALSE, trim = TRUE)
      ),
      call. = FALSE
    )
  }
  invisible(workload)
}

#' Build the TMB data list for the RP-BNB model
#' @keywords internal
#' @noRd
.build_tmb_data <- function(Y1, Y2, X1, X2, rand_idx1, rand_idx2,
                            Z1, Z2, dist1, dist2, sign1, sign2,
                            family_code, pois1, pois2,
                            lamLo, lamHi, est_method,
                            chunked = 0L, w = NULL, draw_w = NULL) {
  # w/draw_w default to all-ones (no-op weighting/masking) so every existing
  # caller keeps building the ordinary unchunked tape. `chunked` gates the
  # DATA_UPDATE()'d path in src/rpbnb_tmb.cpp entirely, but the template
  # always reads w/draw_w as plain DATA_VECTOR()s, so they must be present
  # (and correctly sized) in the data list regardless of `chunked`.
  if (is.null(w)) w <- rep(1, length(Y1))
  if (is.null(draw_w)) draw_w <- rep(1, max(1L, nrow(as.matrix(Z1))))
  list(
    Y1 = Y1, Y2 = Y2,
    X1 = unname(as.matrix(X1)), X2 = unname(as.matrix(X2)),
    rand_idx1 = as.integer(rand_idx1) - 1L,  # 0-based for C++
    rand_idx2 = as.integer(rand_idx2) - 1L,
    Z1 = unname(as.matrix(Z1)), Z2 = unname(as.matrix(Z2)),
    dist1 = as.integer(dist1), dist2 = as.integer(dist2),
    sign1 = as.integer(sign1), sign2 = as.integer(sign2),
    family = as.integer(family_code),
    pois1 = as.integer(pois1), pois2 = as.integer(pois2),
    lamLo = as.numeric(lamLo), lamHi = as.numeric(lamHi),
    est_method = as.integer(est_method),
    chunked = as.integer(chunked),
    w = as.numeric(w),
    draw_w = as.numeric(draw_w)
  )
}
