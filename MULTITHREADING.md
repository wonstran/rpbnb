# Multithreaded RPBNB (OpenMP)

`fit_rpbnb()` runs its simulated-likelihood objective on a **multithreaded C++
core** (OpenMP) instead of R's process-based `parallel` cluster. The draw loop
is parallelised across threads with shared memory — no per-evaluation
serialization — which is why CPU utilisation and wall-clock scale with cores
where the old cluster path did not.

## What is parallelised

The hot path is the triple loop over the `R` simulation draws × `n`
observations inside `bnbr_rp_ll_and_grad`:

- **Pass 1** — per-draw `mu`, `c`, and Famoye lambda-bounds
- **Pass 2** — the `n × R` log-likelihood matrix
- **Gradient** — per-draw score accumulation (thread-local buffers, reduced once)

All of this now lives in [src/halton_parallel.cpp](src/halton_parallel.cpp)
(`rpbnb_ll_grad_cpp`). The per-draw distribution transforms (`rand_realize`,
i.e. normal / lognormal / uniform / triangular) are still computed in R and
passed in as `dev/dloc/dscale` matrices, so the C++ port carries none of the
distribution-registry logic — it is math-identical to the R reference and is
checked against it in [tests/testthat/test-cpp-likelihood.R](tests/testthat/test-cpp-likelihood.R).

## Usage

Nothing changes in how you call `fit_rpbnb()`. The C++ core is used
automatically when the package is compiled. `n_cores` is the **OpenMP thread
count**:

```r
devtools::load_all()

data <- read.csv("data/simulated_nb_data.csv")

fit <- fit_rpbnb(
  y1 ~ 1 + x1 + x2,
  y2 ~ 1 + x1 + x2,
  data = data,
  random_1 = c("(Intercept)", "x1", "x2"),
  random_2 = c("(Intercept)", "x1", "x2"),
  draws = 500, seed = 42,
  control = rpbnb_control(n_cores = rpbnb_threads())   # all cores
)
```

Helpers:

- `rpbnb_threads()` / `get_num_threads()` — CPU threads OpenMP will use
- `rpbnb_openmp_enabled()` — was the DLL built with OpenMP?
- `set_rcpp_parallel_threads(k)` — set a global thread cap

`n_cores = 1` (the default) runs the C++ core single-threaded. Raise it to
parallelise. If the package is built **without** OpenMP, everything still works
— it just runs single-threaded.

## Build

Requires a C++ compiler with OpenMP (Rtools on Windows ships GCC with libgomp):

```r
setwd("path/to/rpbnb")
pkgbuild::clean_dll()
Rcpp::compileAttributes()
devtools::document()
devtools::load_all()

rpbnb_openmp_enabled()   # TRUE
get_num_threads()        # your core count
```

### Windows link note

On Rtools, `$(SHLIB_OPENMP_CXXLIBS)` can expand to empty, which drops
`-fopenmp` from the **link** step and produces `undefined reference to GOMP_*`.
[src/Makevars.win](src/Makevars.win) therefore uses `$(SHLIB_OPENMP_CXXFLAGS)`
for `PKG_LIBS` as well, so `-fopenmp` is present at link and GCC pulls in
libgomp. Do **not** add `-lomp` (that is the LLVM/clang runtime and does not
exist under Rtools).

## Correctness

The R implementation `bnbr_rp_ll_and_grad` is the oracle. The test suite checks
that the C++ value **and** analytic gradient match it to ~1e-8 across
all-normal, subset-random, and lognormal specifications, and that the
fixed-bounds Hessian objective matches `bnbr_rp_ll_fixed_bounds`. If the port
ever drifts, those tests fail.
