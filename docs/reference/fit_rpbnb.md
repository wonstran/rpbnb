# Fit a bivariate random-parameter negative binomial model

Maximum simulated likelihood estimation with normal random coefficients
and Famoye/Sarmanov dependence. Random coefficients are selected per
equation by name via `random_1` / `random_2`.

## Usage

``` r
fit_rpbnb(
  formula_1,
  formula_2,
  data,
  random_1 = NULL,
  random_2 = NULL,
  draws = 400,
  draw_type = "halton",
  seed = 1234,
  start = NULL,
  control = rpbnb_control(),
  dependence = "famoye",
  poisson_1 = FALSE,
  poisson_2 = FALSE,
  .fixed = NULL,
  .opt_draws = NULL
)
```

## Arguments

- formula_1, formula_2:

  Model formulas for the two count outcomes. An equation-specific
  [`offset()`](https://rdrr.io/r/stats/offset.html) term (e.g.
  `y ~ x + offset(log(exposure))`) is supported: the offset enters that
  margin's linear predictor additively (integrated mean
  `E[exp(x'beta + offset)]`) during estimation and is carried through
  the stored fitted means and
  [`predict()`](https://rdrr.io/r/stats/predict.html).

- data:

  A data frame.

- random_1, random_2:

  Random coefficients per equation. Either a character vector of
  `model.matrix` column names (all Normal), or a named list whose values
  are a distribution name (`"normal"`, `"lognormal"`, `"uniform"`,
  `"triangular"`) or a list `list(dist = ..., sign = ...)` (`sign` is
  -1/1 and lognormal-only). NULL means all-fixed for that equation.

- draws:

  Number of simulation draws for the optimization.

- draw_type:

  Quasi-random draw type. Only "halton" is supported in this version.

- seed:

  Random seed for the simulation draws.

- start:

  Optional starting parameter vector.

- control:

  An [`rpbnb_control()`](rpbnb_control.md) object. Estimation uses BFGS,
  the only optimizer `control$method` accepts. One control object serves
  every estimator in the package; settings this one does not read – the
  TMB knobs (`gradtol`, `restarts`, `max_threads`, `max_workload`,
  `parallel_tape`), `hessian`, and `draws_hessian` – are ignored and
  listed by
  [`print()`](https://rdrr.io/r/base/print.html)/[`summary()`](https://rdrr.io/r/base/summary.html)
  of the fit rather than rejected.

- dependence:

  Dependence structure: "famoye" (default; Famoye/Sarmanov) or an
  [`copula()`](copula.md) object for copula dependence (Frank / Gaussian
  / Clayton). Both paths use the multithreaded (OpenMP) C++ simulated
  likelihood; the copula path is more numerically expensive per
  evaluation (discrete-copula pmf + per-draw NB CDF corners), so fits
  typically take noticeably longer than the Famoye path at comparable
  `draws`/`n`. Random coefficients on 0/1 dummy regressors are weakly
  identified under the copula path (NB dispersion trades off against the
  random-coefficient scale); prefer random coefficients on continuous
  regressors.

- poisson_1, poisson_2:

  Fit the corresponding margin at its exact Poisson limit (NB2
  dispersion `m = 0`): the margin's `log_m` is held fixed, so it is not
  a free parameter and the fit is a properly nested restriction of the
  NB model. Pair with [`lr_test()`](lr_test.md) (`boundary = TRUE`) to
  test a margin for overdispersion (`H0: m = 0`). The simulated
  likelihood uses the exact `m = 0` branch – the per-draw margin log-pmf
  is `dpois` and its Famoye dependence constant is `exp(-d*mu)` – so it
  is accurate at any fitted mean, not a fixed-dispersion approximation.
  Supported with both Famoye/Sarmanov and [`copula()`](copula.md)
  dependence (the copula path uses the same exact m = 0 branch).

- .fixed:

  Internal. A named numeric vector of parameters (in the
  optimization/log-scale parameterization) to pin at the supplied values
  and hold fixed during estimation. Used by
  [`rpbnb_boundary_tests()`](rpbnb_boundary_tests.md) to construct
  scale-zero (SD boundary) restricted fits; not intended for direct use.

- .opt_draws:

  Internal. A list `list(Z1, Z2)` of uniform Halton draw matrices to use
  verbatim instead of generating them, so a restricted refit reuses a
  full fit's draws (common random numbers). Used by
  [`rpbnb_boundary_tests()`](rpbnb_boundary_tests.md); not intended for
  direct use.

## Value

An object of class `rpbnb_fit`. Under Famoye dependence three fields
describe the admissible lambda interval: `bounds` is the interval the
optimized likelihood actually used, frozen at the starting values (its
width also feeds [`summary()`](https://rdrr.io/r/base/summary.html)'s
delta-method standard error for lambda); `bounds_at_optimum` is the
interval admissible at the fitted parameters, recomputed after
optimization for validation only; and `lambda_admissible` records
whether `lambda` — the fitted value, mapped through `bounds` — lies
inside `bounds_at_optimum`. `FALSE` means the optimizer escaped the
valid region (a warning is raised) and the fit should be re-run from
starting values closer to the optimum. For normal or lognormal random
coefficients loaded in both margins the bound is the constant
`c(-1, 1)`, the two intervals coincide, and escape cannot occur.

## Examples

``` r
sim <- simulate_rpbnb(n = 600,
  beta1 = c("(Intercept)" = 0.2, x1 = 0.4),
  beta2 = c("(Intercept)" = 0.1, x1 = -0.3),
  random_1 = list(x1 = list(sd = 0.5)),
  dispersion = c(m1 = 0.4, m2 = 0.5), seed = 1)
fit <- fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim$data, random_1 = "x1",
                 draws = 100, control = rpbnb_control(compute_se = FALSE))
#> initial  value 2004.254281 
#> iter   2 value 1893.236582
#> iter   3 value 1892.174243
#> iter   4 value 1885.984661
#> iter   5 value 1885.511406
#> iter   6 value 1884.309399
#> iter   7 value 1884.169782
#> iter   8 value 1883.428494
#> iter   9 value 1881.770693
#> iter  10 value 1880.478679
#> iter  11 value 1880.272268
#> iter  12 value 1880.222694
#> iter  13 value 1880.207021
#> iter  14 value 1880.206316
#> iter  15 value 1880.206283
#> iter  15 value 1880.206283
#> iter  15 value 1880.206283
#> final  value 1880.206283 
#> converged
coef(fit)
#> b1:(Intercept)          b1:x1 b2:(Intercept)          b2:x1     log_sd1:x1 
#>      0.2495537      0.3613666      0.1481139     -0.2144669     -0.5696078 
#>         log_m1         log_m2       z_lambda 
#>     -1.2597146     -0.9084395      0.3757306 

# \donttest{
# Copula dependence instead of Famoye/Sarmanov (slower; fewer draws here
# for a quick example -- use more in practice)
sim_cop <- simulate_rpbnb_copula(n = 600,
  beta1 = c("(Intercept)" = 0.2, x1 = 0.4),
  beta2 = c("(Intercept)" = 0.1, x1 = -0.3),
  random_1 = list(x1 = list(sd = 0.3)),
  dispersion = c(m1 = 0.4, m2 = 0.5),
  copula = copula("normal", par = 0.5), seed = 1)
fit_cop <- fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim_cop$data, random_1 = "x1",
                     dependence = copula("normal"), draws = 100,
                     control = rpbnb_control(compute_se = FALSE))
#> initial  value 1879.567596 
#> iter   2 value 1763.577380
#> iter   3 value 1760.412380
#> iter   4 value 1760.182545
#> iter   5 value 1759.459428
#> iter   6 value 1759.315678
#> iter   7 value 1758.584033
#> iter   8 value 1757.546326
#> iter   9 value 1755.044896
#> iter  10 value 1752.119663
#> iter  11 value 1752.076240
#> iter  12 value 1750.292446
#> iter  13 value 1750.238890
#> iter  14 value 1750.228544
#> iter  15 value 1750.222392
#> iter  16 value 1750.221844
#> iter  16 value 1750.221833
#> iter  16 value 1750.221833
#> final  value 1750.221833 
#> converged
tanh(coef(fit_cop)[["z_theta"]])  # estimated copula rho
#> [1] 0.5136254
# }
```
