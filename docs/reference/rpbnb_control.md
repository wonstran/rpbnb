# Control parameters for every rpbnb estimator

One control object for all of the package's fitters –
[`fit_bnb()`](fit_bnb.md), [`fit_rpbnb()`](fit_rpbnb.md),
[`fit_rpbnb_tmb()`](fit_rpbnb_tmb.md), and [`rpbnb()`](rpbnb.md) with
either `engine`. It carries the union of the tuning knobs the estimators
use; each fitter reads the ones that apply to it and **ignores the
rest**, reporting the ignored names in
[`print()`](https://rdrr.io/r/base/print.html)/[`summary()`](https://rdrr.io/r/base/summary.html)
of the resulting fit rather than erroring. That is what makes a script
able to flip `engine = "classic"` to `"tmb"` without rewriting its
control call.

## Usage

``` r
rpbnb_control(
  method = c("BFGS"),
  iterlim = NULL,
  reltol = 1e-08,
  print_level = NULL,
  draws_hessian = 100L,
  halton_burn = 300L,
  n_cores = 1L,
  compute_se = TRUE,
  hessian = c("numeric", "analytic"),
  se_method = c("numeric", "opg", "analytic"),
  hess_eps = 1e-05,
  hess_r = 4L,
  gradtol = 1e-05,
  restarts = 10L,
  max_threads = NULL,
  max_workload = NULL,
  parallel_tape = FALSE
)
```

## Arguments

- method:

  Optimizer used by the `maxLik` fitters. Only "BFGS" is implemented and
  wired through; it is the sole accepted value.

- iterlim:

  Maximum optimizer iterations. `NULL` (default) uses 300 for the
  `maxLik` fitters and 500 for the TMB engine's `nlminb`.

- reltol:

  Relative convergence tolerance.

- print_level:

  Optimizer verbosity, and the switch that silences the boundary-test
  progress messages. `NULL` (default) uses 2 for the `maxLik` fitters
  and 0 (silent) for the TMB engine.

- draws_hessian:

  Retained for backward compatibility but unused by every estimator: the
  random-parameter numeric Hessian is taken with the same optimization
  draws that produced the estimate (same-draw curvature), so it no
  longer resimulates a separate Hessian draw set. Supplying it is always
  reported as an ignored setting.

- halton_burn:

  Number of leading Halton points discarded before forming the
  simulation draws.

- n_cores:

  Worker processes for [`fit_rpbnb()`](fit_rpbnb.md)'s optional cluster
  path, or OpenMP threads for [`fit_rpbnb_tmb()`](fit_rpbnb_tmb.md) (1 =
  sequential in both cases).

- compute_se:

  If FALSE, skip the Hessian and standard errors. The TMB engine has its
  own `inference` argument for this instead.

- hessian:

  How [`fit_bnb()`](fit_bnb.md) (famoye) computes the Hessian for
  standard errors: "numeric" (default,
  [`numDeriv::hessian()`](https://rdrr.io/pkg/numDeriv/man/hessian.html))
  or "analytic" (the closed-form Famoye (2010) Appendix Hessian). Both
  freeze the lambda-bounds at the optimum and yield the same
  observed-information SEs.

- se_method:

  Standard-error method for [`fit_rpbnb()`](fit_rpbnb.md) (the
  random-parameter model): "numeric" (default) uses the
  [`numDeriv::hessian()`](https://rdrr.io/pkg/numDeriv/man/hessian.html)
  observed- information Hessian; "analytic" uses the closed-form
  observed-information Hessian (Famoye (2010) per-draw second
  derivatives assembled via the Louis mixture formula) – exact and much
  faster than "numeric" for larger models; "opg" uses the BHHH /
  outer-product-of-gradients information from the per-observation scores
  – fastest, but relies on the information-matrix equality so it is
  unreliable for parameters at a boundary (e.g. a random- coefficient SD
  estimated near 0). For copula dependence
  ([`fit_rpbnb()`](fit_rpbnb.md) with `dependence = copula(...)`), only
  "opg" (recommended) and "numeric" are available; "analytic" is not
  implemented for the copula path and errors.

- hess_eps, hess_r:

  Step and Richardson order for
  [`numDeriv::hessian()`](https://rdrr.io/pkg/numDeriv/man/hessian.html)
  (used only when the numeric Hessian is selected).

- gradtol:

  Stationarity tolerance for the TMB engine's score, applied *relative*
  to the objective: the fit is treated as stationary once
  `max(abs(gradient))` falls below `gradtol * max(1, abs(nll))`.
  [`nlminb()`](https://rdrr.io/r/stats/nlminb.html) declares convergence
  from its relative function test, which can fire while the score is
  still far from zero; a Hessian taken there is not a curvature and its
  inverse can carry negative variances. The fit is restarted until the
  gradient clears this tolerance; missing it is not warned about on its
  own, because the copula families legitimately optimise against a clamp
  or a probability floor where no step improves the objective, but it is
  reported in the warning raised when the Hessian actually fails to
  factor, and is kept on the fit as `optimizer$max_abs_gradient`. The
  scaling matters because the score is a sum over observations, so an
  absolute cutoff would tighten with sample size for no statistical
  reason.

- restarts:

  Maximum number of times to restart
  [`nlminb()`](https://rdrr.io/r/stats/nlminb.html) from its own answer
  while `gradtol` is unmet. Restarting resets the trust region and step
  scaling that stalled the first solve. Set to `0L` for the single-call
  behaviour of earlier versions.

- max_threads:

  Maximum OpenMP threads permitted for one TMB fit. `NULL` (default)
  means `n_cores`, so threads are not capped unless set explicitly below
  `n_cores`.

- max_workload:

  Maximum weighted observation-draw evaluations permitted before TMB
  tape construction; `Inf` disables the guard. The default and every
  figure here are derived from `TAPE_CALIBRATION`, so this text cannot
  drift from the shipped behaviour.

  With the default `parallel_tape = FALSE` the budget is per fit; with
  `parallel_tape = TRUE` the tapes are built concurrently and the guard
  multiplies the workload by the realized thread count.

  One unit is one weighted observation-draw. All figures are measured by
  `inst/benchmark_memory.R`, whose raw results are stored in
  `inst/extdata/memory_calibration.csv`.

  Retained tape size depends on `n * draws` alone: tape (MiB) = 13.374 +
  0.001164 \* units, with R^2 = 0.9994 (1221 bytes per unit over the
  largest workloads; per-unit cost is higher at small workloads because
  of the fixed intercept).

  The budget is set on *peak* working set, not on retained tape, because
  peak is what exhausts memory: TMB records the whole likelihood before
  pruning it to each parallel region, so peak runs about 6.11 times the
  retained tape plus first-evaluation growth, or 12083 bytes per unit
  measured directly against peak. The default of 700,000 units therefore
  targets a peak of about 8 GiB for one fit.

  Families cost different amounts per unit, so each carries a weight –
  the largest *peak* ratio to Famoye observed at matched workload,
  rounded up to the next tenth: independence 0.7, famoye 1, frank 3.6,
  gaussian 0.9, clayton 1.1. Frank peaks at over three and a half times
  Famoye, so treating it as unweighted would under-budget it more than
  threefold.

  Raise `max_workload` deliberately against the memory you actually
  have, not to make one particular dataset fit.

  As of [`rpbnb_tmb_max_workload()`](rpbnb_tmb_max_workload.md), the
  value [`rpbnb_tmb_control()`](rpbnb_tmb_control.md) actually uses by
  default is no longer the fixed figure above: it will auto-detect
  available memory and budget 80% of it, falling back to the fixed 8 GiB
  figure only when detection is unavailable on the current platform.
  Call [`rpbnb_tmb_max_workload`](rpbnb_tmb_max_workload.md) directly to
  set your own budget or detection fraction.

- parallel_tape:

  Construct per-thread TMB tapes concurrently. The default `FALSE`
  constructs them sequentially to reduce peak memory; objective and
  gradient evaluation remains parallel.

## Value

An object of class `c("rpbnb_control", "rpbnb_tmb_control")` (a named
list). It carries both class names so that every historical
`inherits(control, ...)` check in the package and in user code accepts
it.

## Which parameters apply to which estimator

|  |  |  |  |
|----|----|----|----|
| Parameter | [`fit_bnb()`](fit_bnb.md) | [`fit_rpbnb()`](fit_rpbnb.md) | [`fit_rpbnb_tmb()`](fit_rpbnb_tmb.md) |
| `method`, `compute_se`, `hess_eps`, `hess_r` | yes | yes | ignored |
| `iterlim`, `reltol`, `print_level` | yes | yes | yes |
| `halton_burn`, `n_cores` | ignored | yes | yes |
| `hessian` | yes | ignored | ignored |
| `se_method` | ignored | yes | ignored |
| `gradtol`, `restarts`, `max_threads`, `max_workload`, `parallel_tape` | ignored | ignored | yes |
| `draws_hessian` | ignored | ignored | ignored |

Note that `iterlim` and `n_cores` mean different things to different
estimators – a `maxLik` BFGS iteration limit versus an `nlminb` one,
worker *processes* versus OpenMP *threads*. They are not translated;
each estimator reads the number and applies its own meaning to it.

## Defaults that depend on the estimator

`iterlim` and `print_level` default to `NULL`, which means "this
estimator's own long-standing default": `iterlim` is 300 under `maxLik`
and 500 under `nlminb`; `print_level` is 2 (progress) under `maxLik` and
0 (silent) under `nlminb`. Supplying either explicitly overrides that
for every estimator. `max_threads` defaults to `n_cores` and
`max_workload` is computed from available memory by
[`rpbnb_tmb_max_workload()`](rpbnb_tmb_max_workload.md) the first time a
TMB fit needs it (so a non-TMB fit never pays for the memory probe).

## See also

[`rpbnb_tmb_max_workload()`](rpbnb_tmb_max_workload.md),
[`rpbnb()`](rpbnb.md), [`fit_rpbnb()`](fit_rpbnb.md),
[`fit_rpbnb_tmb()`](fit_rpbnb_tmb.md), [`fit_bnb()`](fit_bnb.md)

## Examples

``` r
rpbnb_control(method = "BFGS", iterlim = 200)
#> rpbnb control settings
#>   method         BFGS
#>   iterlim        200
#>   reltol         1e-08
#>   print_level    <estimator default>
#>   draws_hessian  100
#>   halton_burn    300
#>   n_cores        1
#>   compute_se     TRUE
#>   hessian        numeric
#>   se_method      numeric
#>   hess_eps       1e-05
#>   hess_r         4
#>   gradtol        1e-05
#>   restarts       10
#>   max_threads    1
#>   max_workload   <estimator default>
#>   parallel_tape  FALSE
#>   supplied by caller: method, iterlim 
rpbnb_control(hessian = "analytic")
#> rpbnb control settings
#>   method         BFGS
#>   iterlim        <estimator default>
#>   reltol         1e-08
#>   print_level    <estimator default>
#>   draws_hessian  100
#>   halton_burn    300
#>   n_cores        1
#>   compute_se     TRUE
#>   hessian        analytic
#>   se_method      numeric
#>   hess_eps       1e-05
#>   hess_r         4
#>   gradtol        1e-05
#>   restarts       10
#>   max_threads    1
#>   max_workload   <estimator default>
#>   parallel_tape  FALSE
#>   supplied by caller: hessian 
# The same object drives either engine; the TMB-only knobs are simply
# ignored by the classic one (and reported as ignored in its summary).
rpbnb_control(n_cores = 4, gradtol = 1e-6, se_method = "opg")
#> rpbnb control settings
#>   method         BFGS
#>   iterlim        <estimator default>
#>   reltol         1e-08
#>   print_level    <estimator default>
#>   draws_hessian  100
#>   halton_burn    300
#>   n_cores        4
#>   compute_se     TRUE
#>   hessian        numeric
#>   se_method      opg
#>   hess_eps       1e-05
#>   hess_r         4
#>   gradtol        1e-06
#>   restarts       10
#>   max_threads    4
#>   max_workload   <estimator default>
#>   parallel_tape  FALSE
#>   supplied by caller: n_cores, se_method, gradtol 
```
