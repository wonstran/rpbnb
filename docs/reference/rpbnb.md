# Fit a random-parameter bivariate NB model with either engine

A common front end over the package's two estimation engines.
`engine = "classic"` calls [`fit_rpbnb()`](fit_rpbnb.md) (Rcpp/OpenMP
simulated likelihood, `maxLik` BFGS); `engine = "tmb"` calls
[`fit_rpbnb_tmb()`](fit_rpbnb_tmb.md) (TMB automatic differentiation,
`nlminb` with restart polish). Both fitters remain exported and can be
called directly; with `standardize = FALSE` (the default) this wrapper
adds nothing to the fit itself and returns exactly what the chosen
fitter returns.

## Usage

``` r
rpbnb(
  formula_1,
  formula_2,
  data,
  engine = c("classic", "tmb"),
  random_1 = NULL,
  random_2 = NULL,
  draws = 400,
  seed = 1234,
  start = NULL,
  dependence = "famoye",
  poisson_1 = FALSE,
  poisson_2 = FALSE,
  standardize = FALSE,
  continuous_vars = NULL,
  boundary_tests = FALSE,
  boundary_draws = NULL,
  control = NULL,
  ...
)
```

## Arguments

- formula_1, formula_2:

  Model formulas for the two count responses.

- data:

  A data frame containing the model variables.

- engine:

  Estimation engine: `"classic"` (default) or `"tmb"`.

- random_1, random_2:

  Random-coefficient specifications for each equation.

- draws:

  Number of simulation draws.

- seed:

  Random seed for the draw sequence.

- start:

  Optional named or unnamed starting values.

- dependence:

  `"famoye"` (default), a [`copula()`](copula.md) object, or
  `"independence"` (TMB engine only).

- poisson_1, poisson_2:

  Restrict the corresponding margin to its Poisson limit.

- standardize:

  Centre and scale continuous predictors before fitting, and display
  fitted coefficients back-transformed to their original units (see
  "Automatic centring and scaling" above). Default `FALSE`.

- continuous_vars:

  Optional character vector overriding automatic continuous-predictor
  detection when `standardize = TRUE`. Must be columns of `data`;
  ignored when `standardize = FALSE`.

- boundary_tests:

  Which groups of parameters to LR-test after fitting, via
  [`rpbnb_boundary_tests()`](rpbnb_boundary_tests.md)
  (`engine = "classic"`) or
  [`rpbnb_tmb_boundary_tests()`](rpbnb_tmb_boundary_tests.md)
  (`engine = "tmb"`); the result is attached as `$boundary_tests` and
  [`summary()`](https://rdrr.io/r/base/summary.html)/[`print()`](https://rdrr.io/r/base/print.html)
  show the test (rather than `NA`, or rather than a Wald `z`) on the
  corresponding rows. Accepts:

  - `FALSE` (default) or `"none"` – run nothing. Each restricted refit
    costs roughly another full fit.

  - `TRUE` – `c("sd", "dispersion")`, the two boundary-null groups. This
    is what `TRUE` has always meant and it does not silently grow.

  - a character vector, any of `"sd"` (random-coefficient scales),
    `"dispersion"` (the NB2 overdispersions `m1`, `m2`), `"dependence"`
    (the association parameter), or `"all"` for all three.

  So `boundary_tests = "dispersion"` tests overdispersion only,
  `boundary_tests = c("dispersion", "dependence")` tests overdispersion
  and association without paying for one refit per random-coefficient
  scale, and `boundary_tests = "all"` tests everything. See "Boundary LR
  tests" below.

- boundary_draws:

  Number of Halton simulation draws for the boundary tests' restricted
  refits, `engine = "tmb"` only. `NULL` (default) uses the main fit's
  `draws`. Ignored when no boundary test was requested (nothing to apply
  it to); an error if supplied under `engine = "classic"` alongside a
  boundary test (see "Boundary LR tests" above – that engine has no
  `draws` to override).

- control:

  An [`rpbnb_control()`](rpbnb_control.md) object – one object for both
  engines, as of 0.4.1 ([`rpbnb_tmb_control()`](rpbnb_tmb_control.md) is
  a retained alias that returns the same thing). Settings the chosen
  engine does not read are ignored and listed by
  [`print()`](https://rdrr.io/r/base/print.html)/[`summary()`](https://rdrr.io/r/base/summary.html)
  of the fit; `iterlim` and `print_level`, whose defaults differ between
  the engines, resolve to the chosen engine's own default unless you set
  them. Fields sharing a name are still not translated: `iterlim` is a
  `maxLik` BFGS limit under `engine = "classic"` and an `nlminb` limit
  under `"tmb"`, and `n_cores` is worker processes versus OpenMP
  threads.

- ...:

  Further arguments passed to the selected fitter. Names are validated
  against that fitter's formals; an argument belonging to the other
  engine, or an unrecognised name, is an error. Exception: the TMB
  tuning knobs `method` and `force_parallel_gaussian` are dropped with a
  warning (not an error) under `engine = "classic"`, so a call can
  switch engines without stripping them.

## Value

The engine-native fit object, identical to what a direct call to the
underlying fitter would return: an object of class `rpbnb_fit` for
`engine = "classic"`, or `rpbnb_tmb_fit` for `engine = "tmb"`. The class
therefore depends on `engine`; test with
`inherits(fit, "rpbnb_tmb_fit")` if you need to branch. No wrapper class
is introduced, so every existing S3 method and post-estimation function
works unchanged. With `standardize = TRUE`, two extra fields are
attached – `$scaling` and `$continuous_vars` – and
[`print()`](https://rdrr.io/r/base/print.html)/[`summary()`](https://rdrr.io/r/base/summary.html)
use them to display original-units coefficients. With any
`boundary_tests` group requested, a third field `$boundary_tests` (the
[`rpbnb_boundary_tests()`](rpbnb_boundary_tests.md) result) is attached,
and
[`print()`](https://rdrr.io/r/base/print.html)/[`summary()`](https://rdrr.io/r/base/summary.html)
use it to show the LR test on the corresponding rows. Both fitters also
attach `$control_ignored` / `$control_engine`, the control settings the
chosen engine did not read. Nothing else on the object changes.

## Details

What it does add is argument checking across the two APIs. The engines
do not take the same arguments, and passing one engine's argument to the
other is a mistake that is easy to make and expensive to notice — a
standardized coefficient table printed under an "original units" heading
looks perfectly plausible. Every extra argument is therefore matched by
name against the selected fitter's own formals, and anything that does
not belong is an error rather than a silently ignored `...` entry.

## Automatic centring and scaling

`standardize = TRUE` automates the pattern in `inst/rpbnb_frank_open.R`
and `inst/tmb_rpbnb_frank_open.R`: continuous predictors are centred and
scaled (mean 0, SD 1) before fitting, which keeps a bounded
random-coefficient carrier from acting as a disguised random intercept
(see those scripts' headers) and fixes the design matrix's conditioning
when regressors span very different ranges. Continuous predictors are
identified automatically — numeric, non-factor columns used by either
formula with more than two distinct values, so 0/1 (or any two-level
numeric) indicators are left alone — or supplied explicitly via
`continuous_vars`. Variables that appear only inside an
[`offset()`](https://rdrr.io/r/stats/offset.html) are never
standardized.

The fitted design itself (the stored `X1`/`X2`, `mu1`/`mu2`, simulation
draws) stays on the standardized scale, exactly as in the two scripts
above, so [`predict()`](https://rdrr.io/r/stats/predict.html), marginal
effects, and boundary/LR tests keep working unchanged. Only the
coefficient table [`print()`](https://rdrr.io/r/base/print.html) and
[`summary()`](https://rdrr.io/r/base/summary.html) display is affected:
it is back-transformed to the covariates' original units by the exact
affine chain rule (no refit, no numerical differentiation) and is the
*only* coefficient table shown — there is no separate standardized-scale
table to reconcile.
[`coef()`](https://rdrr.io/r/stats/coef.html)/[`vcov()`](https://rdrr.io/r/stats/vcov.html)
still return the standardized-scale values that match the stored design;
call `.rpbnb_orig_units()` (internal, mirrors what
[`print()`](https://rdrr.io/r/base/print.html) displays) if the numeric
original-units vector is needed directly. The scaling actually used is
stored on the fit as `$scaling` (a named list of `c(center=, scale=)`)
and `$continuous_vars`.

## Boundary LR tests

`boundary_tests` runs a likelihood-ratio test on the fit as soon as it
converges (against the same data the fit itself used – the standardized
copy, when `standardize = TRUE`) and attaches the result as
`$boundary_tests`. It is a switch over three independent groups, so the
cost is paid only for the parameters actually in question:

|  |  |  |
|----|----|----|
| `boundary_tests` | tests | restricted refits |
| `FALSE` / `"none"` | nothing | 0 |
| `TRUE` | `c("sd", "dispersion")` | one per scale + one per free `m` |
| `"sd"` | random-coefficient scales | one per scale |
| `"dispersion"` | overdispersions `m1`, `m2` | one per free `m` |
| `"dependence"` | the association parameter | 1 |
| `"all"` | all three groups | all of the above |

The random-coefficient SDs and the NB2 dispersions (`m1`, `m2`) have a
null that sits on the boundary of the parameter space, so an ordinary
Wald `z`/`p` does not test it; the boundary test refits each restricted
model instead and
[`summary()`](https://rdrr.io/r/base/summary.html)/[`print()`](https://rdrr.io/r/base/print.html)
show that test's `LR`/`df`/`p` for those rows in the natural-scale
block, in place of the `NA` they would otherwise carry.

The **dependence** test is the odd one out and is therefore opt-in even
under `boundary_tests = TRUE`. Its null is "no association", i.e. the
independence model, which for Famoye (`lam`), Frank (`theta`), and the
Gaussian copula (`rho`) sits in the *interior* of the parameter space –
the Wald `z` those rows already show is valid, and the LR statistic is
an ordinary chi-square(1), not the 50:50 boundary mixture. Only Clayton
/ Kimeldorf (`theta > 0`) has a genuine boundary null there. Requesting
it replaces the dependence row's Wald `z`/`p` with the LR test on both
engines, since the two answer the same question.

Both engines test the same parameters via
[`rpbnb_boundary_tests()`](rpbnb_boundary_tests.md)
(`engine = "classic"`) or
[`rpbnb_tmb_boundary_tests()`](rpbnb_tmb_boundary_tests.md)
(`engine = "tmb"`). They differ only in how each restricted fit is
constructed while preserving common random numbers: for a scale, the
classic engine zeroes that coefficient's draw column while the TMB
engine pins its `log_sd` and maps it out of the free parameters; for the
dependence parameter, the TMB engine refits with its own
`dependence = "independence"` family while the classic engine (which has
no such fitter) pins the working-scale dependence parameter at its
family's independence value.

A [`message()`](https://rdrr.io/r/base/message.html) reports how many
restricted refits are about to run before they start, unless
`control$print_level` is `0`; suppress it with
[`suppressMessages()`](https://rdrr.io/r/base/message.html) if needed.

Under `engine = "tmb"` with `method = "laplace"`, a restricted refit
that fails to converge is automatically retried with both sides of that
one LR estimated by `method = "sml"` rather than reported `NA` – some
restrictions leave Laplace no valid optimum at all (see
[`rpbnb_tmb_boundary_tests()`](rpbnb_tmb_boundary_tests.md)'s
`sml_fallback` argument, which is where to turn this off).

`force_parallel_gaussian` (`engine = "tmb"` only, passed via `...`) is
forwarded to every restricted refit, so a Gaussian-copula fit's boundary
tests honor `control$n_cores` the same way the original fit did instead
of silently re-capping each refit to one thread – see
[`rpbnb_tmb_boundary_tests()`](rpbnb_tmb_boundary_tests.md)'s own
`force_parallel_gaussian` argument for why this needs forwarding at all
(the fit object does not record whether the override was used).

Each restricted refit costs roughly as much as the original fit (more
for a [`copula()`](copula.md) dependence than for `"famoye"`; see
[`rpbnb_boundary_tests()`](rpbnb_boundary_tests.md)'s timing note), so
this defaults to `FALSE`. `boundary_draws` (`engine = "tmb"` only) sets
a `draws` for those refits other than the main fit's – e.g. more draws
for a more precise boundary test without re-fitting the whole model at
that `draws`. It has no classic-engine counterpart:
[`rpbnb_boundary_tests()`](rpbnb_boundary_tests.md) reuses the full
fit's exact stored draw matrix (zeroing a column) rather than
regenerating draws from a count, so passing `boundary_draws` under
`engine = "classic"` is an error. For finer control still – testing only
`"sd"` or only `"dispersion"` (both engines take a `which` argument), or
reusing one boundary-test run across several summaries – call
[`rpbnb_boundary_tests()`](rpbnb_boundary_tests.md)/[`rpbnb_tmb_boundary_tests()`](rpbnb_tmb_boundary_tests.md)
directly on the fit and assign its result to `fit$boundary_tests` (with
`standardize = TRUE`, reconstruct the fitting-scale data first:
`rpbnb:::.apply_scaling(data, fit$scaling)`).

## Which arguments go with which engine

|  |  |  |
|----|----|----|
| Argument | `engine = "classic"` | `engine = "tmb"` |
| `draw_type`, `.fixed`, `.opt_draws` | yes | error |
| `inference`, `keep` | error | yes |
| `method`, `force_parallel_gaussian` | ignored with a warning | yes |
| [`offset()`](https://rdrr.io/r/stats/offset.html) in a formula | yes | error |
| `dependence = "independence"` | error | yes |
| `boundary_draws` (non-`NULL`) | error | yes |
| `control` class | `rpbnb_control` | `rpbnb_control` (same object) |
| optimizer | `maxLik::maxLik(method = "BFGS")` | [`stats::nlminb`](https://rdrr.io/r/stats/nlminb.html) + restarts |

Both engines freeze the Famoye admissible lambda interval at the
starting values (so the analytic gradient is exactly the derivative of
the optimized objective) and validate the fitted lambda against the
interval admissible at the optimum afterwards — check
`lambda_admissible` on the fit. The fixed-parameter
[`fit_bnb()`](fit_bnb.md) takes the opposite trade-off (moving bounds,
admissible by construction); see the decision note in
`R/bnb_likelihood.R`.

## See also

[`fit_rpbnb()`](fit_rpbnb.md), [`fit_rpbnb_tmb()`](fit_rpbnb_tmb.md),
[`rpbnb_control()`](rpbnb_control.md),
[`rpbnb_tmb_control()`](rpbnb_tmb_control.md), [`copula()`](copula.md),
[`fit_bnb()`](fit_bnb.md)

## Examples

``` r
# \donttest{
d <- read.csv(system.file("extdata", "rwm1984_bnb.csv", package = "rpbnb"))
fit <- rpbnb(docvis ~ outwork, hospvis ~ outwork, data = d,
             engine = "tmb", random_1 = "outwork", draws = 50)
#> Warning: The fitted Famoye lambda (1.70507) lies outside the admissible interval recomputed at the fitted parameters [-1.06667, 1.06667]. The bounds passed to the likelihood were frozen at the starting values ([-1.73201, 1.73201]), so the optimizer was free to leave the valid region: the joint pmf is negative somewhere in the count tails and this fit should not be interpreted. Refit from starting values closer to the optimum (so the frozen box is tighter), or use a different dependence structure. See fit$lambda_admissible and fit$lambda_bounds_at_optimum.
#> Warning: Dependence estimate(s) lam are pinned at a boundary of the Famoye bounds, which are frozen at the starting values. The estimates are constrained by the implementation rather than identified by the data, so their standard errors are reported as NA. Refit from different starting values, or use a dependence family whose range covers the association in these data. rpbnb_tmb_dependence_profile() reports a likelihood-based interval in place of the NA, but for Famoye that interval is mapped through this same frozen box, so widen the box first rather than reading it as a rescue.
# }
```
