# One-shot setup for the JAX engine experiment (branch: jax-engine).
# Creates a project-local virtualenv so nothing is installed into the
# user's global Python. Run once:  Rscript tools/jax-setup.R
if (!requireNamespace("reticulate", quietly = TRUE)) {
  install.packages("reticulate", repos = "https://cloud.r-project.org")
}
venv <- normalizePath(file.path(getwd(), ".venv-jax"), mustWork = FALSE)
if (!dir.exists(venv)) {
  reticulate::virtualenv_create(venv)
}
reticulate::virtualenv_install(venv, packages = c("jax", "numpy", "scipy", "pytest"))
reticulate::use_virtualenv(venv, required = TRUE)
jax <- reticulate::import("jax")
cat("jax", jax$`__version__`, "devices:",
    paste(vapply(jax$devices(), function(d) d$device_kind, character(1)),
          collapse = ", "), "\n")
