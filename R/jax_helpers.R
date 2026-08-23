# R side of the JAX engine (branch: jax-engine). The Python lives in
# inst/python/rpbnb_jax/ and mirrors src/rpbnb_tmb.cpp with est_method = 0.

#' Locate the rpbnb package root from the current working directory
#'
#' `.rpbnb_jax_available()` needs the project-local `.venv-jax`, and testthat
#' runs with the working directory set to `tests/testthat/`, so `getwd()`
#' alone finds nothing. Walk up until a directory holding rpbnb's own
#' DESCRIPTION appears -- the `Package:` field is checked, because any
#' unrelated R project between here and the filesystem root would otherwise
#' claim the search and hand back a `.venv-jax` that is not ours.
#' Returns `NA_character_` when there is no such ancestor (an installed
#' package, say), which the callers treat as "no project venv".
#' @keywords internal
#' @noRd
.rpbnb_project_root <- function(start = getwd()) {
  path <- normalizePath(start, winslash = "/", mustWork = FALSE)
  repeat {
    desc <- file.path(path, "DESCRIPTION")
    if (file.exists(desc)) {
      pkg <- tryCatch(unname(read.dcf(desc, "Package")[1L, 1L]),
                      error = function(e) NA_character_)
      if (identical(pkg, "rpbnb")) return(path)
    }
    parent <- dirname(path)
    if (identical(parent, path)) return(NA_character_)
    path <- parent
  }
}

#' Point reticulate at the project-local .venv-jax
#'
#' Must run before reticulate initialises Python, or it silently keeps
#' whatever interpreter it bootstrapped for itself. Both the availability
#' predicate and the object constructor call this, deliberately: an earlier
#' version selected the venv only in `.rpbnb_jax_available()`, so any caller
#' that skipped the predicate got reticulate's own ephemeral Python and a
#' bare "No module named 'jax'". That coupling is invisible in the test file
#' (skip_if_not() happens to run the predicate first) and would surface only
#' once something wired this engine into a fit.
#'
#' Returns `TRUE` invisibly when a `.venv-jax` was found and handed to
#' `use_virtualenv()`. Selection failure is not itself an error -- whether the
#' engine is usable is decided by a `py_module_available("jax")` check in the
#' caller -- but it is reported, because that check cannot distinguish "no
#' venv on disk" from "venv selected, jax missing from it", and the two want
#' different advice.
#' @keywords internal
#' @noRd
.rpbnb_use_jax_venv <- function() {
  root <- .rpbnb_project_root()
  if (is.na(root)) return(invisible(FALSE))
  venv <- file.path(root, ".venv-jax")
  if (!dir.exists(venv)) return(invisible(FALSE))
  # use_virtualenv() errors once Python is already bound to a different
  # interpreter, which is a legitimate state rather than a fault here.
  ok <- tryCatch({
    reticulate::use_virtualenv(venv, required = FALSE)
    TRUE
  }, error = function(e) FALSE, warning = function(w) FALSE)
  invisible(ok)
}

#' Is the JAX engine importable?
#' @keywords internal
#' @noRd
.rpbnb_jax_available <- function() {
  if (!requireNamespace("reticulate", quietly = TRUE)) return(FALSE)
  .rpbnb_use_jax_venv()
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
#' Two deliberate divergences from that surface, both benign for the callers
#' in this package but worth knowing before wiring anything new to it:
#'
#' * `gr()` returns a length-p numeric vector where `TMB::MakeADFun()` returns
#'   a 1 x p matrix. `stats::nlminb()`, `stats::optimHess()` and
#'   `max(abs(g))` are indifferent; anything that indexes the result as
#'   `g[1, ]` is not.
#' * `env$last.par.best` is updated on `gr()` as well as `fn()`, because one
#'   JAX call produces both and the cache makes the two indistinguishable.
#'   TMB updates it only in `fn()`. The recorded point is the same either way;
#'   only the call that records it differs.
#'
#' @param data The list `.build_tmb_data()` produces.
#' @param start Full parameter template, in the declaration order of
#'   `src/rpbnb_tmb.cpp:977-983`: beta1, beta2, log_sd1, log_sd2, log_m1,
#'   log_m2, z_dep.
#' @param free Logical vector over `start`; `FALSE` pins a coordinate, which
#'   is this engine's analogue of a `map = list(x = factor(NA))` entry.
#' @param obs_chunk Observation block size for the copula path. Accepted and
#'   validated but not yet applied: Frank is the first family to build the
#'   `(n, R, kmax + 1)` count grid this exists to bound, and the grid only
#'   becomes large enough to need blocking at truck scale.
#' @keywords internal
#' @noRd
.make_rpbnb_jax_object <- function(data, start, free, obs_chunk = 256L) {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("the JAX engine needs the reticulate package.", call. = FALSE)
  }
  # Not just in .rpbnb_jax_available() -- see .rpbnb_use_jax_venv(). This
  # constructor must stand on its own, because a fit path has no reason to
  # call the predicate first.
  selected <- .rpbnb_use_jax_venv()
  if (!isTRUE(reticulate::py_module_available("jax"))) {
    stop(sprintf(
      paste0("the JAX engine needs jax in the project-local .venv-jax ",
             "(%s). Run `Rscript tools/jax-setup.R` from the package root ",
             "to create it."),
      if (selected) {
        "the virtualenv was found and selected, but jax is not importable from it"
      } else {
        "no .venv-jax was found above the working directory"
      }
    ), call. = FALSE)
  }
  py_dir <- .rpbnb_python_dir()
  if (!nzchar(py_dir)) {
    stop("cannot locate the rpbnb_jax Python sources", call. = FALSE)
  }
  # import_from_path() rather than py_run_string(sprintf(...)): the latter
  # builds Python source text out of a filesystem path, so an apostrophe in
  # it -- legal on Windows -- is a SyntaxError rather than a bad path.
  rj_packing <- reticulate::import_from_path(
    "rpbnb_jax.packing", path = py_dir, delay_load = FALSE)
  rj_objective <- reticulate::import_from_path(
    "rpbnb_jax.objective", path = py_dir, delay_load = FALSE)

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

  n_free <- sum(free)
  env <- new.env(parent = emptyenv())
  env$last.par.best <- NULL
  env$value.best <- Inf
  cache <- new.env(parent = emptyenv())
  cache$par <- NULL

  # fn() and gr() are called back to back at the same point by every
  # optimizer here, and one JAX call already produces both, so the second
  # call is served from the cache rather than re-evaluated.
  evaluate <- function(par) {
    # TMB's obj$fn() with no argument re-evaluates at the fixed-effect slice
    # of the last parameter vector, env$last.par[env$lfixed()]; here the free
    # vector is already that slice. Without this, the advertised
    # `par = NULL` default would hand numeric(0) to a jitted function
    # expecting n_free coordinates.
    if (is.null(par)) {
      par <- if (!is.null(cache$par)) cache$par else start[free]
    }
    # Checked on this side as well as in Layout.unpack() so the condition
    # carries the R call rather than surfacing as a Python exception. A
    # length-1 par is the dangerous one: reticulate turns an R scalar into a
    # length-1 array, which index assignment would broadcast across every
    # free coordinate.
    if (length(par) != n_free) {
      stop(sprintf(
        "the JAX objective takes %d free parameter(s), got %d",
        n_free, length(par)))
    }
    if (!is.null(cache$par) && identical(cache$par, par)) return(cache$out)
    res <- fg(as.numeric(par))
    out <- list(value = as.numeric(res[[1]]), grad = as.numeric(res[[2]]))
    cache$par <- par
    cache$out <- out
    if (is.finite(out$value) && out$value < env$value.best) {
      env$value.best <- out$value
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
