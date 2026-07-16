#' Numerically stable per-row log-sum-exp
#' @keywords internal
#' @noRd
row_log_sum_exp <- function(M) {
  m <- apply(M, 1, max)
  m + log(rowSums(exp(M - m)))
}

#' Significance stars for a vector of p-values
#' @keywords internal
#' @noRd
signif_stars <- function(p) {
  as.character(stats::symnum(
    p, corr = FALSE, na = FALSE,
    cutpoints = c(0, .001, .01, .05, .1, 1),
    symbols   = c("***", "**", "*", ".", " ")
  ))
}

#' Marginal NB2 starting values for the two equations.
#'
#' Fits each margin with `MASS::glm.nb` (no intercept term added -- the design
#' matrices already carry their own), returning regression-coefficient and
#' log-dispersion starts. Falls back to zeros / log(0.5) if a margin fails.
#' @return list(b1, b2, log_m1, log_m2).
#' @keywords internal
#' @noRd
.marginal_nb_starts <- function(Y1, X1, Y2, X2) {
  # These are starting-value heuristics only; glm.nb's own convergence/iteration
  # warnings on the marginal fits are irrelevant here, so suppress them.
  g1 <- suppressWarnings(tryCatch(MASS::glm.nb(Y1 ~ X1 - 1), error = function(e) NULL))
  g2 <- suppressWarnings(tryCatch(MASS::glm.nb(Y2 ~ X2 - 1), error = function(e) NULL))
  list(
    b1     = if (!is.null(g1)) unname(stats::coef(g1)) else rep(0, NCOL(X1)),
    b2     = if (!is.null(g2)) unname(stats::coef(g2)) else rep(0, NCOL(X2)),
    log_m1 = if (!is.null(g1)) log(1 / g1$theta) else log(0.5),
    log_m2 = if (!is.null(g2)) log(1 / g2$theta) else log(0.5)
  )
}

#' Resolve a user-supplied start against the canonical parameter names.
#'
#' Returns a finite, canonically-ordered, named start vector:
#' * `NULL` -> the fitter's `default`, named by `par_names`.
#' * unnamed (positional) -> must match `length(par_names)`; named in order.
#' * named -> reordered by name; unknown or duplicate names are rejected, and a
#'   named *partial* start is merged into `default` (missing entries keep their
#'   default). This prevents a fully-named vector in the wrong order from being
#'   silently interpreted positionally.
#' @keywords internal
#' @noRd
.resolve_start <- function(start, default, par_names, label = "start") {
  if (is.null(start)) {
    names(default) <- par_names
    return(default)
  }
  if (!is.numeric(start)) {
    stop("`", label, "` must be a numeric vector.", call. = FALSE)
  }
  nm <- names(start)
  if (is.null(nm) || all(!nzchar(nm))) {                 # positional
    if (length(start) != length(par_names)) {
      stop("`", label, "` must have length ", length(par_names), " (got ",
           length(start), "), or be named.", call. = FALSE)
    }
    if (any(!is.finite(start))) {
      stop("`", label, "` must contain only finite values.", call. = FALSE)
    }
    names(start) <- par_names
    return(start)
  }
  if (any(!nzchar(nm))) {
    stop("`", label, "` mixes named and unnamed elements; name all or none.",
         call. = FALSE)
  }
  if (anyDuplicated(nm)) {
    stop("`", label, "` has duplicate names: ",
         paste(unique(nm[duplicated(nm)]), collapse = ", "), ".", call. = FALSE)
  }
  unknown <- setdiff(nm, par_names)
  if (length(unknown)) {
    stop("`", label, "` has unknown parameter name(s): ",
         paste(unknown, collapse = ", "), ".\nValid names: ",
         paste(par_names, collapse = ", "), ".", call. = FALSE)
  }
  if (any(!is.finite(start))) {
    stop("`", label, "` must contain only finite values.", call. = FALSE)
  }
  out <- default
  names(out) <- par_names
  out[nm] <- start                                       # merge (full or partial)
  out
}

#' Invert an observed-information matrix, recording curvature diagnostics.
#'
#' `info` is the (symmetric) observed information -Hessian (or an OPG S'S). The
#' matrix is checked for positive definiteness; when it is not PD (a boundary,
#' weak identification, a saddle, or numeric-differentiation failure) a ridge is
#' added before inversion and a warning is emitted -- regularized SEs are not
#' observed-information SEs, and the caller stores the returned `diag` on the fit
#' object as `hessian_diag` so the repair is visible rather than silent.
#'
#' @return list(vcov, se, diag) where `diag` records the min/max eigenvalue,
#'   condition number, ridge magnitude, inversion method, and PD/repair flags.
#' @keywords internal
#' @noRd
.observed_info_vcov <- function(info, par_names, label = "model") {
  info <- (info + t(info)) / 2
  ev <- try(eigen(info, symmetric = TRUE, only.values = TRUE), silent = TRUE)
  bad_eig <- inherits(ev, "try-error") || any(!is.finite(ev$values))
  min_eig <- if (bad_eig) NA_real_ else min(ev$values)
  max_eig <- if (bad_eig) NA_real_ else max(ev$values)
  pd <- isTRUE(!bad_eig && min_eig > 0)
  ridge <- 0
  if (!pd) {
    ridge <- if (bad_eig) 1e-2 else 1e-8 - min_eig
    info  <- info + diag(ridge, nrow(info))
    warning("Observed information for the ", label, " is not positive definite",
            if (!bad_eig) paste0(" (min eigenvalue ", signif(min_eig, 3), ")")
            else " (non-finite eigenvalues)",
            "; a ridge of ", signif(ridge, 3), " was added before inversion. The ",
            "resulting standard errors are regularized, not observed-information ",
            "SEs -- inspect fit$hessian_diag.", call. = FALSE)
  }
  vc <- try(solve(info), silent = TRUE)
  method <- "solve"
  if (inherits(vc, "try-error")) { vc <- MASS::ginv(info); method <- "ginv" }
  dimnames(vc) <- list(par_names, par_names)
  se <- sqrt(pmax(diag(vc), 0)); names(se) <- par_names
  cond <- if (pd) max_eig / min_eig else NA_real_
  list(vcov = vc, se = se,
       diag = list(min_eigenvalue = min_eig, max_eigenvalue = max_eig,
                   condition = cond, ridge = ridge, inversion = method,
                   positive_definite = pd, repaired = !pd, label = label))
}

#' Conditional mean bounded away from both overflow (exp() -> Inf feeding
#' downstream NB functions) and underflow (mu -> 0, a 0/0 NaN source in
#' several analytic gradients). Used by every likelihood family so the
#' objective, gradient, and Hessian for a given fit always see the exact
#' same (implicitly capped) function of the parameters.
#' @keywords internal
#' @noRd
.bound_mu <- function(X, beta) {
  pmin(pmax(as.vector(exp(X %*% beta)), 1e-300), 1e15)
}
