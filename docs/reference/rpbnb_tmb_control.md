# Control parameters for the TMB engine (alias of `rpbnb_control()`)

Retained so that code written against the pre-unification API keeps
working. The two control objects were merged in 0.4.1: this function
forwards to [`rpbnb_control()`](rpbnb_control.md) and returns exactly
the same object, which every estimator in the package accepts. New code
should call [`rpbnb_control()`](rpbnb_control.md) directly.

## Usage

``` r
rpbnb_tmb_control(
  iterlim = NULL,
  reltol = 1e-08,
  gradtol = 1e-05,
  restarts = 10L,
  print_level = NULL,
  n_cores = 1L,
  max_threads = NULL,
  max_workload = NULL,
  parallel_tape = FALSE,
  halton_burn = 300L
)
```

## Arguments

- iterlim:

  Maximum optimizer iterations. `NULL` (default) uses 300 for the
  `maxLik` fitters and 500 for the TMB engine's `nlminb`.

- reltol:

  Relative convergence tolerance.

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

- print_level:

  Optimizer verbosity, and the switch that silences the boundary-test
  progress messages. `NULL` (default) uses 2 for the `maxLik` fitters
  and 0 (silent) for the TMB engine.

- n_cores:

  Worker processes for [`fit_rpbnb()`](fit_rpbnb.md)'s optional cluster
  path, or OpenMP threads for [`fit_rpbnb_tmb()`](fit_rpbnb_tmb.md) (1 =
  sequential in both cases).

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
  value `rpbnb_tmb_control()` actually uses by default is no longer the
  fixed figure above: it will auto-detect available memory and budget
  80% of it, falling back to the fixed 8 GiB figure only when detection
  is unavailable on the current platform. Call
  [`rpbnb_tmb_max_workload`](rpbnb_tmb_max_workload.md) directly to set
  your own budget or detection fraction.

- parallel_tape:

  Construct per-thread TMB tapes concurrently. The default `FALSE`
  constructs them sequentially to reduce peak memory; objective and
  gradient evaluation remains parallel.

- halton_burn:

  Number of leading Halton points discarded before forming the
  simulation draws.

## Value

The [`rpbnb_control()`](rpbnb_control.md) object.

## Details

Only the arguments you actually supply are forwarded, so an untouched
`iterlim`/`print_level` still resolves to the TMB engine's own defaults
(500 and 0) when the object is used for a TMB fit – and to the `maxLik`
defaults if the same object is handed to [`fit_rpbnb()`](fit_rpbnb.md).

## See also

[`rpbnb_control()`](rpbnb_control.md)

## Examples

``` r
identical(unclass(rpbnb_tmb_control(n_cores = 2)),
          unclass(rpbnb_control(n_cores = 2)))
#> [1] TRUE
```
