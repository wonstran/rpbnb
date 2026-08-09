# rpbnb

Bivariate and random-parameter negative binomial regression (NB2), via maximum
likelihood (Famoye/Sarmanov dependence) and maximum simulated likelihood.

The random-parameter model has two estimation engines — an Rcpp/OpenMP
simulated-likelihood engine and a TMB automatic-differentiation engine — behind
a common `rpbnb()` front end.

## Install

```r
# install.packages("devtools")
devtools::install("path/to/rpbnb")
```

## Quick start

```r
library(rpbnb)
d <- read.csv(system.file("extdata", "rwm1984_clean.csv", package = "rpbnb"))

# Bivariate NB (Famoye dependence)
fit <- fit_bnb(docvis ~ outwork + kids, hospvis ~ outwork + kids,
               data = d, dependence = "famoye")
summary(fit)
bnb_gof(fit)

# Random-parameter BNB (kids has a random coefficient in equation 1)
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

## Models

- `fit_bnb()` — bivariate NB with `dependence = "independence"` (two univariate
  NB2 margins) or `"famoye"` (Famoye/Sarmanov dependence).
- `rpbnb()` — random-parameter BNB through either engine (see below).
- `fit_rpbnb()` — bivariate random-parameter NB via maximum simulated likelihood
  with normal random coefficients and Halton draws.
- `simulate_rpbnb()` — simulate data from a random-parameter NB process.
- Diagnostics: `bnb_gof()`, `bnb_marginal_effects()`, `bnb_elasticities()`.
- Standard methods: `coef()`, `vcov()`, `logLik()`, `AIC()`, `BIC()`, `predict()`,
  `summary()`, `print()`.

## Two engines

`rpbnb(..., engine = )` selects between them; both underlying fitters are also
exported and can be called directly.

| | `engine = "classic"` (`fit_rpbnb()`) | `engine = "tmb"` (`fit_rpbnb_tmb()`) |
| --- | --- | --- |
| likelihood | Rcpp + OpenMP simulated likelihood | TMB automatic differentiation |
| optimizer | `maxLik::maxLik(method = "BFGS")` | `stats::nlminb` + restart polish |
| control | `rpbnb_control()` | `rpbnb_tmb_control()` |
| `offset()` in a formula | yes | no (errors) |
| `dependence = "independence"` | no (use `fit_bnb()`) | yes |
| extras | boundary tests, LR tests, residual plots | Laplace estimator, dependence profiling, memory-aware workload sizing |

```r
# Same model, either engine. The return object is the engine's native class.
a <- rpbnb(docvis ~ outwork + kids, hospvis ~ outwork + kids, data = d,
           engine = "classic", random_1 = "kids", draws = 400, seed = 1234)
b <- rpbnb(docvis ~ outwork + kids, hospvis ~ outwork + kids, data = d,
           engine = "tmb", random_1 = "kids", draws = 400, seed = 1234)
```

`rpbnb()` checks every extra argument against the selected engine's own
formals, so an argument meant for the other engine — or a typo — is an error
rather than a silently ignored `...` entry.

### TMB engine: inference and memory

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

All of the above bounds the *simulated* likelihood, whose tape scales with
`nrow(data) * draws`. `method = "laplace"` integrates the random coefficients
with TMB's Laplace approximation instead, taping one conditional evaluation per
observation and integrating the latents through a sparse Hessian. Tape size then
scales with `nrow(data)` alone and the draw budget stops binding. That is the
option to reach for when a fit exhausts memory during `MakeADFun()`, rather than
raising `max_workload` against RAM you do not have. It supports normal and
lognormal random coefficients, and it is a different approximation to the same
integral — see `?fit_rpbnb_tmb`.

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
disables the guard.

### Known issue: multithreaded Gaussian copula

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

See `vignette("rpbnb-intro")` for a worked example.

## Scope

This project ports the mature Famoye BNB MLE and random-parameter BNB SIML
implementations into an installable, tested package.
Implemented: Famoye/Sarmanov and discrete-copula (Frank / Gaussian / Clayton)
dependence for both the fixed and random-parameter models, additional
random-coefficient distributions (normal, lognormal, uniform, triangular), and a
multithreaded Rcpp/OpenMP core for the copula RP likelihood. As of 0.4.0 the
former `rpbnb.tmb` package is merged in as a second engine, adding a TMB
automatic-differentiation likelihood with a Laplace approximation. Further Monte
Carlo drivers remain planned.
