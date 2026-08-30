#' Report how this installation of rpbnb was compiled
#'
#' Nearly all of this package's running time is compiled likelihood
#' evaluation, so whether its shared object was built with optimization is a
#' performance fact worth roughly a factor of two on every fit -- and nothing
#' in R's own output tells you which build you have. This reports what the
#' compiler actually did.
#'
#' A source install (`install.packages(type = "source")`, `R CMD INSTALL`,
#' `remotes::install_github()`) compiles with R's own `CXXFLAGS`, which is
#' `-O2` on every standard platform, so the optimized build is what you get by
#' default. The package does not -- and by CRAN policy must not -- override
#' those flags. Two things do produce a slow build: a `-O0` (or `-Og`) entry
#' in the user's `~/.R/Makevars`, which applies to every package compiled on
#' that machine, and development helpers that inject debug flags of their own,
#' notably `pkgbuild::compile_dll(debug = TRUE)` -- its default -- which
#' `devtools::load_all()` uses when recompiling changed sources.
#'
#' @return A list, invisibly when printed:
#'   \describe{
#'     \item{`optimized`}{`TRUE` when compiled at `-O1` or above, read from
#'       the compiler's own `__OPTIMIZE__` macro rather than inferred.}
#'     \item{`openmp`}{`TRUE` when OpenMP is available, so `n_cores > 1` can
#'       do anything. A build without it fits single-threaded whatever
#'       `control$n_cores` says.}
#'     \item{`openmp_max_threads`}{The OpenMP runtime's own thread ceiling.}
#'     \item{`assertions_enabled`}{`TRUE` when `NDEBUG` is unset, a further
#'       marker of a debug build.}
#'     \item{`compiler`}{Compiler and version that built the shared object.}
#'   }
#' @examples
#' rpbnb_build_info()
#' @export
rpbnb_build_info <- function() {
  info <- rpbnb_build_flags_cpp()
  structure(info, class = c("rpbnb_build_info", "list"))
}

#' @export
print.rpbnb_build_info <- function(x, ...) {
  cat("rpbnb build\n")
  cat(sprintf("  %-14s %s%s\n", "optimized", x$optimized,
              if (isTRUE(x$optimized)) "" else "   (SLOW -- see ?rpbnb_build_info)"))
  cat(sprintf("  %-14s %s\n", "openmp", x$openmp))
  cat(sprintf("  %-14s %s\n", "max threads", x$openmp_max_threads))
  cat(sprintf("  %-14s %s\n", "assertions", x$assertions_enabled))
  cat(sprintf("  %-14s %s\n", "compiler", x$compiler))
  invisible(x)
}

# Silent on a normal (optimized) install, which is the overwhelmingly common
# case; speaks only when the build on this machine will be about twice as slow
# as it should be. That failure is otherwise invisible -- the package works
# correctly, just slowly -- so a load-time notice is the only place a user
# realistically finds out before paying for it across a long fit.
.onAttach <- function(libname, pkgname) {
  info <- tryCatch(rpbnb_build_flags_cpp(), error = function(e) NULL)
  if (is.null(info) || isTRUE(info$optimized)) return(invisible(NULL))
  packageStartupMessage(
    "rpbnb was compiled WITHOUT optimization (-O0): fits will run roughly ",
    "twice as slow as they should.\n",
    "  Usual causes: a -O0/-Og entry in ~/.R/Makevars, or a package built by ",
    "pkgbuild::compile_dll(debug = TRUE) (its default, used by ",
    "devtools::load_all()).\n",
    "  Reinstall from source for an optimized build; see ?rpbnb_build_info."
  )
  invisible(NULL)
}
