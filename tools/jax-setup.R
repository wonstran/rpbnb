# One-shot setup for the JAX engine experiment (branch: jax-engine).
# Creates a project-local virtualenv so nothing is installed into the
# user's global Python. Run once:  Rscript tools/jax-setup.R

# The venv is sited relative to getwd(), and it is half a gigabyte. Running
# this from the wrong directory would silently create one somewhere
# unexpected, where the root-anchored ^\.venv-jax$ in .Rbuildignore would not
# catch it.
stopifnot("run from the package root (no DESCRIPTION here)" = file.exists("DESCRIPTION"))

if (!requireNamespace("reticulate", quietly = TRUE)) {
  install.packages("reticulate", repos = "https://cloud.r-project.org")
}
venv <- normalizePath(file.path(getwd(), ".venv-jax"), mustWork = FALSE)
if (!dir.exists(venv)) {
  reticulate::virtualenv_create(venv, version = "3.14")
}
# jax is pinned: the deliverable of this branch is a numerical parity result,
# and jax.config handling and dtype promotion have both changed across minor
# releases. numpy/scipy/pytest are left free -- nothing here depends on their
# exact versions.
reticulate::virtualenv_install(venv, packages = c("jax==0.11.1", "numpy", "scipy", "pytest"))
reticulate::use_virtualenv(venv, required = TRUE)
jax <- reticulate::import("jax")
cat("jax", jax$`__version__`, "devices:",
    paste(vapply(jax$devices(), function(d) d$device_kind, character(1)),
          collapse = ", "), "\n")
