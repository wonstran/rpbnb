# rpbnb

Maximum-likelihood and maximum-simulated-likelihood estimation of bivariate
negative binomial (NB2) regression models, with Famoye/Sarmanov or
discrete-copula (Frank, Gaussian, Clayton) dependence between the two count
outcomes, and a random-parameter (mixed) extension estimated by two
interchangeable engines.

- **Fixed-coefficient models** — `fit_bnb()`: two NB2 (or Poisson-limit)
  margins joined by independence, Famoye/Sarmanov dependence, or a copula.
- **Random-parameter models** — `fit_rpbnb()` / `fit_rpbnb_tmb()` /
  `rpbnb()`: normal, lognormal, uniform, or triangular random coefficients,
  estimated by simulated maximum likelihood (Rcpp/OpenMP or TMB automatic
  differentiation) or, for the TMB engine, a Laplace approximation.
- **Diagnostics and post-estimation** — goodness of fit, marginal effects,
  elasticities, residual checks, boundary-corrected likelihood-ratio tests,
  dependence profiling.
- **Simulators** — generate data from a known bivariate or random-parameter
  NB2 process, for testing, teaching, or power calculations.

## Table of contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Core concepts](#core-concepts)
  - [Fixed vs. random-parameter models](#fixed-vs-random-parameter-models)
  - [Two random-parameter engines](#two-random-parameter-engines)
  - [Dependence structures](#dependence-structures)
- [Function reference](#function-reference)
- [Usage examples](#usage-examples)
- [TMB engine: inference and memory](#tmb-engine-inference-and-memory)
- [Known issue: multithreaded Gaussian copula](#known-issue-multithreaded-gaussian-copula)
- [Example datasets](#example-datasets)
- [Documentation](#documentation)
- [Development](#development)
- [Scope](#scope)

## Installation

**From source (recommended during development):**

```r
# install.packages("devtools")
devtools::install("path/to/rpbnb")          # local checkout
devtools::install_github("wonstran/rpbnb")   # from GitHub, default branch
```

Building from source requires a C++17 toolchain (this package links TMB,
RcppEigen, and Rcpp, and compiles a C++ template): on Windows, install
[Rtools](https://cran.r-project.org/bin/windows/Rtools/); on macOS, install
the Xcode Command Line Tools (`xcode-select --install`) plus a Fortran
compiler (see the [R for macOS "tools" page](https://mac.r-project.org/tools/));
on Linux, `build-essential` (Debian/Ubuntu) or your distribution's equivalent.
The package compiles natively on both Intel and Apple Silicon (arm64) Macs.

**Prebuilt binary / source tarball:** each
[GitHub Release](https://github.com/wonstran/rpbnb/releases) attaches a
Windows x64 `.zip` binary and a `.tar.gz` source package (installs on any
platform, compiling locally):

```r
install.packages("rpbnb_<version>.zip", repos = NULL, type = "win.binary")   # Windows
install.packages("rpbnb_<version>.tar.gz", repos = NULL, type = "source")    # macOS / Linux / any
```

Not on CRAN.

## Quick start

```r
library(rpbnb)
d <- read.csv(system.file("extdata", "rwm1984_clean.csv", package = "rpbnb"))

# Fixed-coefficient bivariate NB (Famoye/Sarmanov dependence)
fit <- fit_bnb(docvis ~ outwork + kids, hospvis ~ outwork + kids,
               data = d, dependence = "famoye")
summary(fit)
bnb_gof(fit)

# Random-parameter BNB: kids has a random coefficient in equation 1
rp <- fit_rpbnb(docvis ~ outwork + kids, hospvis ~ outwork + kids,
                data = d, random_1 = "kids", draws = 400, seed = 1234)
summary(rp)

# Simulate from a known RP-BNB process
sim <- simulate_rpbnb(n = 2000,
  beta1 = c("(Intercept)" = -1.0, x1 = 0.4),
  beta2 = c("(Intercept)" = -0.5, x1 = 0.2),
  random_1 = list(x1 = list(sd = 0.5)),
  dispersion = c(m1 = 0.8, m2 = 0.8), seed = 1234)
```

## Core concepts

### Fixed vs. random-parameter models

`fit_bnb()` estimates a **fixed-coefficient** bivariate NB2 model: every
covariate has a single coefficient shared across observations. The
random-parameter (mixed) models — `fit_rpbnb()`, `fit_rpbnb_tmb()`, and the
`rpbnb()` front end — let selected coefficients vary across observations
according to a chosen distribution (normal, lognormal, uniform, or
triangular), integrated out by simulation (or, under TMB, optionally by a
Laplace approximation).

### Two random-parameter engines

| | `engine = "classic"` (`fit_rpbnb()`) | `engine = "tmb"` (`fit_rpbnb_tmb()`) |
| --- | --- | --- |
| likelihood | Rcpp + OpenMP simulated likelihood | TMB automatic differentiation |
| optimizer | `maxLik::maxLik(method = "BFGS")` | `stats::nlminb` + restart polish |
| control | `rpbnb_control()` | `rpbnb_control()` (one object for both) |
| `offset()` in a formula | yes | no (errors) |
| `dependence = "independence"` | no (use `fit_bnb()`) | yes |
| extras | boundary tests, LR tests, residual plots | Laplace estimator, dependence profiling, memory-aware draw chunking |

`rpbnb(..., engine = )` selects between them; both fitters are also exported
and callable directly. `rpbnb()` checks every extra argument against the
selected engine's own formals, so an argument meant for the other engine — or
a typo — is an error rather than a silently ignored `...` entry.

```r
# Same model, either engine. The return object is the engine's native class.
a <- rpbnb(docvis ~ outwork + kids, hospvis ~ outwork + kids, data = d,
           engine = "classic", random_1 = "kids", draws = 400, seed = 1234)
b <- rpbnb(docvis ~ outwork + kids, hospvis ~ outwork + kids, data = d,
           engine = "tmb", random_1 = "kids", draws = 400, seed = 1234)
```

### Dependence structures

Both margins can be joined by:

- **`"independence"`** — two separate NB2 (or Poisson-limit) margins.
  `fit_bnb()` and `fit_rpbnb_tmb()` only.
- **`"famoye"`** — Famoye/Sarmanov dependence (a bounded association
  parameter, admissible interval recomputed from the fitted means/dispersions).
- **`copula(family, par = NULL)`** — a discrete copula. `family` is one of
  `"frank"`, `"normal"` (Gaussian), or `"kimeldorf"` (Clayton). `par`, the
  copula's native dependence parameter, is used only by the simulators
  (`simulate_rpbnb_copula()`); the fitters always estimate it.

```r
copula("frank")
copula("normal", par = 0.3)   # Gaussian, rho = 0.3 (simulation only)
copula("kimeldorf")           # Clayton
```

## Function reference

### Fitting

| Function | Model |
| --- | --- |
| `fit_bnb()` | Fixed-coefficient bivariate NB2 (independence / Famoye / copula) |
| `fit_rpbnb()` | Random-parameter BNB, classic (Rcpp/OpenMP) engine |
| `fit_rpbnb_tmb()` | Random-parameter BNB, TMB engine (SML or Laplace) |
| `rpbnb()` | Common front end over `fit_rpbnb()`/`fit_rpbnb_tmb()`, plus standardization and boundary-test orchestration |

### Control and utilities

| Function | Purpose |
| --- | --- |
| `rpbnb_control()` | One control object for every fitter above (`rpbnb_tmb_control()` is a retained alias) |
| `rpbnb_tmb_max_workload()` | Convert a memory budget into a TMB `max_workload` value |
| `rpbnb_threads()` | OpenMP thread count available to the classic engine's likelihood |
| `lr_test()` | Likelihood-ratio test between two nested fits, with an optional boundary correction |

### Diagnostics and post-estimation

| Function | Applies to | Purpose |
| --- | --- | --- |
| `bnb_gof()` | `bnb_fit` | Log-likelihood/AIC/BIC and four pseudo-R² measures vs. an intercept-only null |
| `bnb_marginal_effects()` | `bnb_fit` | Average/marginal-at-the-mean effects |
| `bnb_elasticities()` | `bnb_fit` | Elasticities (continuous) / semi-elasticities (binary) |
| `bnb_residual_checks()` | `bnb_fit`, `rpbnb_fit` | Randomized quantile residual diagnostics, outlier flags, misspecification verdict |
| `rpbnb_marginal_effects()` | `rpbnb_fit` | Monte Carlo-integrated marginal effects |
| `rpbnb_elasticities()` | `rpbnb_fit` | Monte Carlo-integrated elasticities |
| `rpbnb_boundary_tests()` | `rpbnb_fit` | Boundary-corrected LR tests for random-coefficient SDs and dispersions (classic engine) |
| `rpbnb_tmb_marginal_effects()` | `rpbnb_tmb_fit` | TMB-engine marginal effects |
| `rpbnb_tmb_elasticities()` | `rpbnb_tmb_fit` | TMB-engine elasticities |
| `rpbnb_tmb_boundary_tests()` | `rpbnb_tmb_fit` | TMB-engine boundary-corrected LR tests, with an SML fallback for non-converging Laplace refits |
| `rpbnb_tmb_dependence_profile()` | `rpbnb_tmb_fit` | Profile-likelihood (or Wald) CI for the dependence parameter |

Every fit class (`bnb_fit`, `rpbnb_fit`, `rpbnb_tmb_fit`) also supports the
standard methods: `coef()`, `vcov()`, `logLik()`, `AIC()`, `BIC()`,
`predict()`, `residuals()`, `summary()`, `print()`; `plot()` is available for
`bnb_fit` and `rpbnb_fit`.

### Simulators

| Function | Generates |
| --- | --- |
| `simulate_bnb()` | Fixed-coefficient Famoye/Sarmanov bivariate NB2 data |
| `simulate_rpbnb()` | Random-parameter Famoye/Sarmanov data (classic-engine format) |
| `simulate_rpbnb_copula()` | Random-parameter copula-dependent data |
| `simulate_rpbnb_tmb()` | Random-parameter data under any TMB-supported dependence (`"famoye"`, `"independence"`, or a copula) via exact conditional-inversion sampling |

## Usage examples

**Fixed-coefficient model with a Poisson-limit boundary test:**

```r
d <- read.csv(system.file("extdata", "rwm1984_clean.csv", package = "rpbnb"))
fit    <- fit_bnb(docvis ~ outwork, hospvis ~ outwork, data = d, dependence = "famoye")
fit_p1 <- fit_bnb(docvis ~ outwork, hospvis ~ outwork, data = d,
                  dependence = "famoye", poisson_1 = TRUE)   # H0: m1 = 0
lr_test(fit_p1, fit, boundary = TRUE)
```

**Copula dependence (fixed-coefficient):**

```r
fit_cop <- fit_bnb(docvis ~ outwork, hospvis ~ outwork, data = d,
                   dependence = copula("normal"))
fit_cop$cop_tau   # estimated Kendall's tau
```

**Random-parameter model with copula dependence (classic engine):**

```r
sim_cop <- simulate_rpbnb_copula(n = 600,
  beta1 = c("(Intercept)" = 0.2, x1 = 0.4),
  beta2 = c("(Intercept)" = 0.1, x1 = -0.3),
  random_1 = list(x1 = list(sd = 0.3)),
  dispersion = c(m1 = 0.4, m2 = 0.5),
  copula = copula("normal", par = 0.5), seed = 1)
fit_cop <- fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim_cop$data, random_1 = "x1",
                     dependence = copula("normal"), draws = 400)
tanh(coef(fit_cop)[["z_theta"]])   # estimated copula rho
```

**Boundary-corrected LR test for a random-coefficient scale:**

```r
sim <- simulate_rpbnb(n = 600,
  beta1 = c("(Intercept)" = 0.2, x1 = 0.4),
  beta2 = c("(Intercept)" = 0.1, x1 = -0.3),
  random_1 = list(x1 = list(sd = 0.5)),
  dispersion = c(m1 = 0.4, m2 = 0.5), seed = 1)
fit <- fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim$data, random_1 = "x1",
                 draws = 400, seed = 1)
rpbnb_boundary_tests(fit, sim$data)          # H0: sd(x1) = 0, and m1 = 0, m2 = 0
```

**`rpbnb()` front end with standardization and boundary tests in one call:**

```r
fit <- rpbnb(docvis ~ outwork + hhninc, hospvis ~ outwork, data = d,
             engine = "tmb", random_1 = "hhninc", draws = 400,
             standardize = TRUE, boundary_tests = "all")
summary(fit)                 # coefficients back-transformed to original units
fit$boundary_tests
```

**Dependence profiling when a boundary leaves `summary()` with `NA` SE:**

```r
fit <- fit_rpbnb_tmb(docvis ~ outwork, hospvis ~ outwork, data = d,
                     dependence = "famoye", keep = "full")
rpbnb_tmb_dependence_profile(fit)
plot(attr(rpbnb_tmb_dependence_profile(fit), "profile"))
```

See `vignette("rpbnb-intro")` for a longer worked example covering the
bivariate model, copula dependence, marginal effects, residual diagnostics,
random parameters, and simulation.

## TMB engine: inference and memory

Use `inference = "diag"` when only coefficient standard errors are required.
This avoids retaining a full covariance matrix; `vcov(fit)` then has the
estimated diagonal and `NA` off-diagonal entries. Use `inference = "none"` to
skip Hessian construction entirely. Marginal-effect and elasticity standard
errors require `inference = "full"`; the lighter modes return their point
estimates with `NA` standard errors and an explicit warning.

Use `keep = "compact"` for the smallest returned object when marginal effects
are not needed. Use `keep = "full"` only for low-level diagnostics that require
`fit$obj` or the response vectors.

TMB tapes are constructed sequentially by default while objective and gradient
evaluation remain parallel, which lowers peak memory without changing fitted
results. Advanced users can opt into concurrent tape construction with
`parallel_tape = TRUE`; peak memory then scales with the thread count, so
`max_threads` caps per-fit concurrency.

`max_workload` rejects oversized observation-by-draw workloads before the
automatic-differentiation tape is built. Every constant behind it is measured
by `inst/tmb_benchmark_memory.R`, whose raw results are committed to
`inst/extdata/memory_calibration.csv`; `TAPE_CALIBRATION` in `R/tmb_utilities.R`
is the single source that the guard, the default, and the generated `?` help all
derive from. The figures restated on this page and in
`dev-docs/tmb-reference/rpbnb.tmb-reference-manual.html` are *copies*, not
derivations — both are checked against `TAPE_CALIBRATION` (by `test-parallel.R`
and by the reference-manual verifier respectively) so a copy that drifts fails
rather than misleads.

The budget is set on **peak** working set rather than retained tape, because
peak is what exhausts memory: TMB records the whole likelihood before pruning
it to each parallel region, so peak runs about 6.1x the retained tape *plus
first-evaluation growth* — roughly 12 kB per weighted observation-draw,
measured directly against peak. The default of `7e5` units therefore targets
about 8 GiB of peak memory for one fit.

### Exact draw chunking (new)

At large `draws` (SML tape size scales with `nrow(data) * draws`), a fit that
would exceed `max_workload` no longer has to be refused. `fit_rpbnb_tmb()`
auto-splits it into several smaller draw chunks replayed over **one** smaller
TMB tape — dropping peak memory to roughly `nrow(data) * ceiling(draws /
chunks)` instead of `nrow(data) * draws`. This is **exact for the requested
`draws`**, not an approximation: the chunk contributions are combined by the
same log-sum-exp identity the unchunked likelihood itself already uses,
verified to agree with an unchunked fit to floating-point precision.

```r
# Auto-chunks silently (a message() names the split unless print_level = 0)
# whenever the estimated workload exceeds max_workload.
fit <- fit_rpbnb_tmb(docvis ~ outwork, hospvis ~ outwork, data = d,
                     random_1 = "outwork", draws = 2000)

# Pin the chunk count explicitly instead of relying on the auto-threshold
# (recommended when max_workload is disabled, or the auto-threshold's
# calibration doesn't fit your data -- see below).
fit <- fit_rpbnb_tmb(docvis ~ outwork, hospvis ~ outwork, data = d,
                     random_1 = "outwork", draws = 2000,
                     control = rpbnb_control(tape_chunks = 10))
```

The auto-chunking threshold is currently derived from the pre-chunking
memory calibration and should be treated as provisional pending a follow-up
measurement of the chunked tape's own memory profile; an explicit
`control$tape_chunks` sidesteps that concern entirely, since the layout is
then stated directly rather than inferred from a workload estimate.
`rpbnb_tmb_boundary_tests()` propagates a chunked fit's memory policy
(including a pinned `tape_chunks`) to its restricted refits automatically,
so a boundary LR test on a large fit stays chunked rather than rebuilding one
full tape.

Chunking trades some gradient-evaluation speed for memory (each outer
gradient step recomputes every chunk's contribution) and gives up the taped
Hessian: `rpbnb_tmb_dependence_profile(method = "profile")` on a chunked fit
falls back to a Wald interval with a warning. Wald/`optimHess` inference (the
default `inference = "full"`/`"diag"`) is unaffected. See
[docs/TMB_SML_large_draws_OOM_guide.md](docs/TMB_SML_large_draws_OOM_guide.md)
for the full design writeup.

All of the above bounds the *simulated* likelihood. `method = "laplace"`
integrates the random coefficients with TMB's Laplace approximation instead,
taping one conditional evaluation per observation and integrating the latents
through a sparse Hessian. Tape size then scales with `nrow(data)` alone and
the draw budget (and chunking) do not apply. That is the option to reach for
when a fit exhausts memory during `MakeADFun()` and chunking is not enough,
rather than raising `max_workload` against RAM you do not have. It supports
normal and lognormal random coefficients, and it is a different approximation
to the same integral — see `?fit_rpbnb_tmb`.

Families differ, so each carries a measured weight: the largest **peak** ratio
to Famoye at matched workload, rounded up to the next tenth. Peak rather than
retained tape, because peak is the quantity being budgeted — the two disagree
enough to matter.

| family | weight | largest peak ratio | largest tape ratio |
|---|---|---|---|
| independence | 0.7 | 0.602 | 0.647 |
| famoye | 1.0 | — | — |
| frank | **3.6** | 3.530 | 2.867 |
| gaussian | 0.9 | 0.844 | 0.947 |
| clayton | 1.1 | 1.079 | 0.921 |

Frank peaks at over three and a half times Famoye per unit, so a Frank fit buys
proportionally fewer draws for the same memory. Raise `max_workload`
deliberately against the memory you actually have —
`inst/tmb_fit_rpbnb_diff_copula.R` shows that opt-in — and `max_workload = Inf`
disables the guard entirely (both the pre-flight refusal and auto-chunking;
pin `tape_chunks` explicitly if you still want chunking with the guard off).

## Known issue: multithreaded Gaussian copula

Evaluating a Gaussian-copula TMB object built with more than one thread
segfaults the R process. Frank and Clayton are unaffected at any thread count,
and Gaussian is fine serially. This predates the 0.4.0 merge — it reproduces
identically in `rpbnb.tmb` 0.3.5 — and the underlying defect, in the registered
Gaussian atomic under OpenMP, is not yet fixed.

**`fit_rpbnb_tmb()` enforces `n_cores = 1L` for the Gaussian family**, with a
warning, so the crash is not reachable from a public call; `parallel_tape` is
forced off for the same reason. Gaussian fits are therefore slower than the
other families until this is repaired. `fit$parallel` records both the
`requested` and `realized` thread counts.

## Example datasets

Available via `system.file("extdata", "<file>", package = "rpbnb")`:

| File | Contents |
| --- | --- |
| `rwm1984.csv` | Raw German health-care utilization panel (docvis, hospvis, and demographic/employment covariates) |
| `rwm1984_clean.csv` | Same data plus derived dummy variables; used throughout the function examples above and the vignette |
| `rwm1984_bnb.csv` | `rwm1984_clean.csv` with generic `y1`/`y2` alias columns |
| `simulated_nb_data.csv` | Small simulated bivariate NB dataset from `simulate_bnb()` |
| `simulated_rpbnb_copula.csv` | Simulated copula-dependent RP-BNB data from `simulate_rpbnb_copula()` |
| `export_dense_all.csv`, `export_open_all.csv` | Highway-segment pavement/safety data used by the benchmark and worked-example scripts under `inst/` |
| `memory_calibration.csv` | Raw TMB memory benchmark measurements behind `TAPE_CALIBRATION` |

## Documentation

- `vignette("rpbnb-intro")` — a worked walkthrough (bivariate model, copula
  dependence, marginal effects, residual diagnostics, random parameters,
  simulation).
- `ref/rpbnb_<version>.pdf` — the CRAN-style PDF reference manual, regenerated
  from the current `man/` pages and attached to each
  [GitHub Release](https://github.com/wonstran/rpbnb/releases).
- `?rpbnb_control` documents every control field and, for the TMB memory
  guard, the full measured-calibration text.
- `NEWS.md` for the change log.

## Development

```r
devtools::load_all()      # load from source for development
devtools::test()          # run the test suite
Rscript tools/test-tiers.R fast   # fast tier only; see the script for slow tiers
devtools::document()      # regenerate man/*.Rd from roxygen comments
```

## Scope

This project ports the mature Famoye BNB MLE and random-parameter BNB SIML
implementations into an installable, tested package.
Implemented: Famoye/Sarmanov and discrete-copula (Frank / Gaussian / Clayton)
dependence for both the fixed and random-parameter models, additional
random-coefficient distributions (normal, lognormal, uniform, triangular), and
a multithreaded Rcpp/OpenMP core for the copula RP likelihood. As of 0.4.0 the
former `rpbnb.tmb` package is merged in as a second engine, adding a TMB
automatic-differentiation likelihood with a Laplace approximation and (0.4.3)
exact draw chunking for large simulated-likelihood fits. Further Monte Carlo
drivers remain planned.
