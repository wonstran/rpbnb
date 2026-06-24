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
  control = rpbnb_control()
)
```

## Arguments

- formula_1, formula_2:

  Model formulas for the two count outcomes.

- data:

  A data frame.

- random_1, random_2:

  Character vectors of coefficient names (matching `model.matrix`
  columns) to treat as normal random coefficients in each equation. NULL
  means all-fixed for that equation.

- draws:

  Number of simulation draws for the optimization.

- draw_type:

  Quasi-random draw type. Only "halton" is supported in this version.

- seed:

  Random seed for the simulation draws.

- start:

  Optional starting parameter vector.

- control:

  An [`rpbnb_control()`](rpbnb_control.md) object. Estimation uses BFGS;
  `control$method` is not currently honored by `fit_rpbnb` (the
  simulated-likelihood objective returns an aggregate gradient).

## Value

An object of class `rpbnb_fit`.

## Examples

``` r
sim <- simulate_rpbnb(n = 600,
  beta1 = c("(Intercept)" = 0.2, x1 = 0.4),
  beta2 = c("(Intercept)" = 0.1, x1 = -0.3),
  random_1 = list(x1 = list(sd = 0.5)),
  dispersion = c(m1 = 0.4, m2 = 0.5), seed = 1)
fit <- fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim$data, random_1 = "x1",
                 draws = 100, control = rpbnb_control(compute_se = FALSE))
coef(fit)
#> b1:(Intercept)          b1:x1 b2:(Intercept)          b2:x1     log_sd1:x1 
#>      0.2496146      0.3613532      0.1481217     -0.2144693     -0.5695825 
#>         log_m1         log_m2       z_lambda 
#>     -1.2598219     -0.9085006      0.5684488 
```
