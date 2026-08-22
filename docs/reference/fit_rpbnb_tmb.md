# Fit a bivariate random-parameter negative binomial model (TMB)

Maximum simulated likelihood via TMB (Template Model Builder) with
automatic differentiation. Supports Famoye/Sarmanov and discrete-copula
(Frank, Gaussian, Clayton) dependence.

## Usage

``` r
fit_rpbnb_tmb(
  formula_1,
  formula_2,
  data,
  random_1 = NULL,
  random_2 = NULL,
  draws = 400L,
  seed = 1234L,
  start = NULL,
  dependence = "famoye",
  control = rpbnb_tmb_control(),
  inference = c("full", "diag", "none"),
  keep = c("postfit", "compact", "full"),
  poisson_1 = FALSE,
  poisson_2 = FALSE,
  method = c("sml", "laplace"),
  force_parallel_gaussian = FALSE,
  .fixed = NULL
)
```

## Arguments

- formula_1, formula_2:

  Model formulas for the two count outcomes.

- data:

  A data frame.

- random_1, random_2:

  Random coefficient specs per equation. NULL (all fixed), character
  vector (all Normal), or named list with per-variable distribution
  specs (`"normal"`, `"lognormal"`, `"uniform"`, `"triangular"`).

- draws:

  Number of Halton simulation draws. Under `method = "sml"` this sets
  the simulation grid the likelihood is averaged over, and tape size
  scales with `nrow(data) * draws`. Under `method = "laplace"` it does
  not affect the likelihood or the tape, but still sizes the Halton grid
  used for the frozen Famoye lambda bounds and for the post-estimation
  averaging in [`predict()`](https://rdrr.io/r/stats/predict.html) and
  the marginal-effect functions.

- seed:

  Random seed for draws.

- start:

  Optional starting parameter vector (named or positional).

- dependence:

  Dependence structure: "famoye", "independence", or a
  [`copula()`](copula.md) object for copula dependence.

- control:

  An [`rpbnb_control()`](rpbnb_control.md) object
  ([`rpbnb_tmb_control()`](rpbnb_tmb_control.md) is a retained alias
  that returns the same object). One control object serves every
  estimator in the package; settings this engine does not read –
  `se_method`, `hessian`, `compute_se`, `method`, `hess_eps`, `hess_r`,
  `draws_hessian` – are ignored and listed by
  [`print()`](https://rdrr.io/r/base/print.html)/[`summary()`](https://rdrr.io/r/base/summary.html)
  of the fit rather than rejected.

- inference:

  Inference storage: `"full"` for a full covariance, `"diag"` for
  standard errors only, or `"none"` to skip Hessian calculations. In
  diagonal mode, [`vcov()`](https://rdrr.io/r/stats/vcov.html) returns
  `NA` for covariance cross-terms.

- keep:

  Retained fit state: `"postfit"` keeps data needed for marginal
  effects, `"compact"` keeps estimates only, and `"full"` also retains
  the TMB objective and responses. Low-level diagnostics that access
  `fit$obj` require `"full"`.

- poisson_1, poisson_2:

  Fit the corresponding margin at its exact Poisson limit (NB2
  dispersion m = 0, held fixed). Available for every dependence
  structure, copulas included. This is the remedy for a dispersion that
  collapses toward zero on its own: at m ~ 1e-7 the NB2 size 1/m is ~1e7
  and the marginal CDF is computed as a difference of logarithms
  agreeing to eleven digits, which leaves the value intact but the
  curvature – and so the standard errors – numerically worthless.
  Pinning the limit removes the parameter instead of estimating it into
  that regime.

- method:

  Estimator for the random-coefficient integral. `"sml"` (default) uses
  simulated maximum likelihood over Halton draws. `"laplace"` uses TMB's
  Laplace approximation with a sparse Hessian over one latent vector per
  observation, which removes `draws` from the memory cost and makes tape
  size scale with `nrow(data)` alone.

  The two are different approximations to the same integral. They agree
  asymptotically but need not agree closely on any given dataset, so a
  Laplace fit is not a drop-in reproduction of an SML fit. `"laplace"`
  supports `"normal"` and `"lognormal"` random coefficients only, and
  requires at least one random coefficient.

  [`summary()`](https://rdrr.io/r/base/summary.html) reports AIC and BIC
  for either estimator, but under `"laplace"` they are computed from an
  approximated marginal likelihood rather than the exact one, so an AIC
  from a Laplace fit is not meaningful to compare against an AIC from an
  SML fit of the same model.

- force_parallel_gaussian:

  Opt-in override of the Gaussian-copula
  (`dependence = copula("normal")`) single-thread safety cap. Default
  `FALSE`: whenever `control$n_cores > 1` or `control$parallel_tape` is
  requested together with a Gaussian copula, the request is silently
  capped to one thread with a
  [`warning()`](https://rdrr.io/r/base/warning.html) instead of being
  honored. This is not a performance knob – it exists because evaluating
  a Gaussian-copula TMB object built with more than one OpenMP thread
  has reliably crashed the R process (SIGSEGV) on the first objective
  evaluation, a defect in the registered Gaussian atomic
  (`REGISTER_ATOMIC(gauss_cell_vec)` in `src/rpbnb_tmb.cpp`) that is not
  fixed by the existing `#pragma omp critical` force-init. Frank and
  Clayton copulas are unaffected at any thread count and never see this
  cap.

  Setting `force_parallel_gaussian = TRUE` honors the requested thread
  count instead of capping it, with a
  [`warning()`](https://rdrr.io/r/base/warning.html) naming the crash
  risk explicitly. This is an escape hatch for someone who has read this
  paragraph and still wants to try it (e.g. to test whether a particular
  TMB/OpenMP build is actually affected) – it does not fix the
  underlying defect, and a crash under this override can still corrupt
  memory and lose unsaved work in the R session. Ignored for every other
  dependence structure.

- .fixed:

  Internal. A named numeric vector of parameters (in the optimization
  parameterization, e.g. `c("log_sd1:x" = -20)`) to pin at the supplied
  values and hold fixed during estimation, so they leave the
  free-parameter count and
  [`logLik()`](https://rdrr.io/r/stats/logLik.html)'s `df`. Used by
  [`rpbnb_tmb_boundary_tests()`](rpbnb_tmb_boundary_tests.md) to
  construct scale-zero restricted fits; not intended for direct use. The
  classic engine's [`fit_rpbnb()`](fit_rpbnb.md) carries an identically
  named argument.

## Value

An object of class `rpbnb_tmb_fit`. The `sdreport` field is a compact
package-owned summary and does not retain a second TMB tape.

`optimizer` is the [`nlminb()`](https://rdrr.io/r/stats/nlminb.html)
return value, with three fields added: `restarts` counts how many times
the solve was restarted from its own answer to clear `control$gradtol`,
`max_abs_gradient` is the score norm actually achieved, and
`gradient_tolerance` is the threshold it was judged against.
`convergence` and `message` come from nlminb's relative *function* test
and can report success at a point that is not stationary, so
`max_abs_gradient` is the field to check before trusting a standard
error.

`method` records which estimator produced the fit, echoing the `method`
argument (`"sml"` or `"laplace"`). Both
[`print()`](https://rdrr.io/r/base/print.html) and
[`summary()`](https://rdrr.io/r/base/summary.html) display it, so two
fits of the same model under different estimators are distinguishable in
their printed output.

`boundary_report` is a character vector naming the reported quantities
whose estimates are pinned against an implementation bound – the frozen
Famoye interval, the Frank overflow guard, or an
[`exp()`](https://rdrr.io/r/base/Log.html) clamp – or whose delta-method
derivative has collapsed numerically. Those estimates are set by the
implementation rather than identified by the data, so their standard
errors and covariances are reported as `NA` and a warning is raised. An
empty vector means no such constraint bound.

`boundary_sides` names, for each entry of `boundary_report`, which end
was reached: `"lower"`, `"upper"`, or `"degenerate"` when the derivative
collapsed rather than a bound being hit. The two sides call for opposite
remedies – a lower dispersion clamp means the margin is effectively
Poisson, an upper one means it is degenerately over-dispersed – so this
is retained on the fit rather than only mentioned in the warning.

`lambda_bounds` is a named numeric vector `c(lower =, upper =)` giving
the admissible Famoye dependence interval, and `NULL` for every other
dependence structure. The bounds *used by the likelihood* are computed
once at the starting values and held fixed for the whole fit. A `lam`
estimate at either end is therefore an artefact of the starting values
rather than a property of the data, which is why the field is exposed.
[`rpbnb_tmb_dependence_profile`](rpbnb_tmb_dependence_profile.md)
reports a likelihood-based interval where the delta-method standard
error collapses to `NA`, but for Famoye that interval is mapped through
this same frozen box and so cannot escape it: widen the box by refitting
from better starting values before treating such an interval as
informative.

`lambda_bounds_at_optimum` is the interval admissible at the *fitted*
parameters, recomputed after optimization for validation only — it never
enters the likelihood. `lambda_admissible` records whether the fitted
`lam` (mapped through the frozen `lambda_bounds`, i.e. the value the
objective actually used) lies inside it; `FALSE` means the optimizer
left the region where the joint pmf is a valid probability model, a
warning was raised, and the fit should not be interpreted — refit from
starting values closer to the optimum. When the bound is
parameter-independent (normal or lognormal coefficients loaded in both
margins, where it is the constant `c(-1, 1)`), the two intervals
coincide and the flag is trivially `TRUE`. Both fields are `NULL`/`NA`
outside Famoye dependence.

## Examples

``` r
if (FALSE) { # \dontrun{
sim <- simulate_rpbnb_tmb(n = 300,
  beta1 = c("(Intercept)" = 0.2, x1 = 0.4),
  beta2 = c("(Intercept)" = 0.1, x1 = -0.3),
  dispersion = c(m1 = 0.4, m2 = 0.5), seed = 1)
fit <- fit_rpbnb_tmb(y1 ~ x1, y2 ~ x1, data = sim$data, draws = 100)
coef(fit)
} # }
```
