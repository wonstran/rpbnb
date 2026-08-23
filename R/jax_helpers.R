# R side of the JAX engine (branch: jax-engine). The Python lives in
# inst/python/rpbnb_jax/ and mirrors src/rpbnb_tmb.cpp with est_method = 0.

#' Locate the package root from the current working directory
#'
#' `.rpbnb_jax_available()` needs the project-local `.venv-jax`, and testthat
#' runs with the working directory set to `tests/testthat/`, so `getwd()`
#' alone finds nothing. Walk up until a directory holding DESCRIPTION appears.
#' Returns `NA_character_` when there is no such ancestor (an installed
#' package, say), which the callers treat as "no project venv".
#' @keywords internal
#' @noRd
.rpbnb_project_root <- function(start = getwd()) {
  path <- normalizePath(start, winslash = "/", mustWork = FALSE)
  repeat {
    if (file.exists(file.path(path, "DESCRIPTION"))) return(path)
    parent <- dirname(path)
    if (identical(parent, path)) return(NA_character_)
    path <- parent
  }
}

#' Is the JAX engine importable?
#' @keywords internal
#' @noRd
.rpbnb_jax_available <- function() {
  if (!requireNamespace("reticulate", quietly = TRUE)) return(FALSE)
  root <- .rpbnb_project_root()
  if (!is.na(root)) {
    venv <- file.path(root, ".venv-jax")
    if (dir.exists(venv)) {
      try(reticulate::use_virtualenv(venv, required = FALSE), silent = TRUE)
    }
  }
  isTRUE(reticulate::py_module_available("jax"))
}

#' Directory holding the rpbnb_jax Python package
#' @keywords internal
#' @noRd
.rpbnb_python_dir <- function() {
  py_dir <- system.file("python", package = "rpbnb")
  if (nzchar(py_dir) && dir.exists(py_dir)) return(py_dir)
  root <- .rpbnb_project_root()
  if (is.na(root)) return("")
  file.path(root, "inst", "python")
}

#' Construct a TMB-shaped objective backed by JAX
#'
#' Returns a list exposing `par`, `fn`, `gr` and `env$last.par.best` -- the
#' surface `fit_rpbnb_tmb()` consumes from `TMB::MakeADFun()`.
#'
#' @param data The list `.build_tmb_data()` produces.
#' @param start Full parameter template, in the declaration order of
#'   `src/rpbnb_tmb.cpp:977-983`: beta1, beta2, log_sd1, log_sd2, log_m1,
#'   log_m2, z_dep.
#' @param free Logical vector over `start`; `FALSE` pins a coordinate, which
#'   is this engine's analogue of a `map = list(x = factor(NA))` entry.
#' @param obs_chunk Observation block size for the copula path. Unused by the
#'   independence and Famoye families, which never build the count grid it
#'   exists to bound.
#' @keywords internal
#' @noRd
.make_rpbnb_jax_object <- function(data, start, free, obs_chunk = 256L) {
  py_dir <- .rpbnb_python_dir()
  if (!nzchar(py_dir)) {
    stop("cannot locate the rpbnb_jax Python sources", call. = FALSE)
  }
  reticulate::py_run_string(sprintf(
    "import sys; p = r'%s'\nif p not in sys.path: sys.path.insert(0, p)",
    py_dir))
  # Submodules are imported by name rather than reached as attributes of
  # rpbnb_jax: `import rpbnb_jax` does not bind them, and having __init__.py
  # pull them in would make the constants it defines a circular import.
  rj_packing <- reticulate::import("rpbnb_jax.packing", delay_load = FALSE)
  rj_objective <- reticulate::import("rpbnb_jax.objective",
                                     delay_load = FALSE)

  k1 <- ncol(data$X1); k2 <- ncol(data$X2)
  q1 <- length(data$rand_idx1); q2 <- length(data$rand_idx2)
  layout <- rj_packing$Layout(
    k1 = as.integer(k1), k2 = as.integer(k2),
    q1 = as.integer(q1), q2 = as.integer(q2),
    template = as.numeric(start),
    free_idx = as.integer(which(free) - 1L)
  )
  fg <- rj_objective$build_objective(data, layout,
                                     obs_chunk = as.integer(obs_chunk))

  env <- new.env(parent = emptyenv())
  env$last.par.best <- NULL
  env$best_value <- Inf
  cache <- new.env(parent = emptyenv())
  cache$par <- NULL

  # fn() and gr() are called back to back at the same point by every
  # optimizer here, and one JAX call already produces both, so the second
  # call is served from the cache rather than re-evaluated.
  evaluate <- function(par) {
    # TMB's obj$fn() with no argument re-evaluates at env$last.par; without
    # this the advertised `par = NULL` default would hand numeric(0) to a
    # jitted function expecting n_free coordinates.
    if (is.null(par)) {
      par <- if (!is.null(cache$par)) cache$par else start[free]
    }
    if (!is.null(cache$par) && identical(cache$par, par)) return(cache$out)
    res <- fg(as.numeric(par))
    out <- list(value = as.numeric(res[[1]]), grad = as.numeric(res[[2]]))
    cache$par <- par
    cache$out <- out
    if (is.finite(out$value) && out$value < env$best_value) {
      env$best_value <- out$value
      env$last.par.best <- par
    }
    out
  }

  list(
    par = start[free],
    fn = function(par = NULL) evaluate(par)$value,
    gr = function(par = NULL) evaluate(par)$grad,
    env = env
  )
}
