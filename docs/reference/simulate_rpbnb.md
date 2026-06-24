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
  seed = 1234
)
```

## Arguments

- n:

  Number of observations.

- beta1, beta2:

  Named numeric vectors of fixed coefficient means per equation; must
  include "(Intercept)".

- random_1, random_2:

  Named lists giving random coefficients, e.g.
  `list(x1 = list(sd = 0.5))`. Means come from `beta1`/`beta2`.

- dispersion:

  Named numeric `c(m1 = ..., m2 = ...)` NB2 dispersions.

- lambda:

  Famoye dependence parameter (0 = independent margins).

- covariates:

  Optional data frame of covariates; if NULL, standard-normal columns
  are generated for every non-intercept name.

- seed:

  Random seed.

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
