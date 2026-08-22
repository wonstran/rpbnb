# Simulate data from a random-parameter bivariate NB process

Simulate data from a random-parameter bivariate NB process

## Usage

``` r
simulate_rpbnb(
  n,
  beta1,
  beta2,
  random_1 = NULL,
  random_2 = NULL,
  dispersion = c(m1 = 0.5, m2 = 0.5),
  lambda = 0,
  covariates = NULL,
  seed = NULL
)
```

## Arguments

- n:

  Number of observations.

- beta1, beta2:

  Named numeric vectors of fixed coefficient means per equation; must
  include "(Intercept)".

- random_1, random_2:

  Named lists giving random coefficients. Each value is a list with
  `dist` (one of "normal", "lognormal", "uniform", "triangular"; default
  "normal"), `scale` (or `sd`) for the dispersion, and `sign` (-1/1,
  lognormal only). Means come from `beta1`/`beta2`; for a lognormal
  coefficient the `beta` entry is the log-location and the realized
  coefficient is `sign * exp(log_location + scale * z)`.

- dispersion:

  Named numeric `c(m1 = ..., m2 = ...)` NB2 dispersions.

- lambda:

  Famoye dependence parameter (0 = independent margins).

- covariates:

  Optional data frame of covariates; if NULL, standard-normal columns
  are generated for every non-intercept name.

- seed:

  Optional random seed. If `NULL` (default) the RNG is left untouched
  and draws continue from the caller's current stream, so repeated calls
  yield distinct datasets; supply an integer for reproducible output.

## Value

A list with `data` (y1, y2, covariates), `coef_realized` (per-obs
coefficients per equation), `mu` (conditional means), `true`
(parameters), `settings`, and `meta` (R/seed/timestamp passed by
caller).

## Examples

``` r
sim <- simulate_rpbnb(n = 500,
  beta1 = c("(Intercept)" = 0.2, x1 = 0.4),
  beta2 = c("(Intercept)" = 0.1, x1 = -0.3),
  random_1 = list(x1 = list(sd = 0.5)),
  dispersion = c(m1 = 0.4, m2 = 0.5), seed = 1)
head(sim$data)
#>   y1 y2         x1
#> 1  3  2 -0.6264538
#> 2  1  1  0.1836433
#> 3  0  2 -0.8356286
#> 4  2  1  1.5952808
#> 5  0  0  0.3295078
#> 6  0  1 -0.8204684
```
