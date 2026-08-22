# Simulate data from a copula RP-BNB process

Simulate data from a copula RP-BNB process

## Usage

``` r
simulate_rpbnb_copula(
  n,
  beta1,
  beta2,
  random_1 = NULL,
  random_2 = NULL,
  dispersion = c(m1 = 0.5, m2 = 0.5),
  copula,
  covariates = NULL,
  seed = NULL
)
```

## Arguments

- n:

  Number of observations.

- beta1, beta2:

  Named coefficient means; must include "(Intercept)".

- random_1, random_2:

  Random-coefficient specs (see
  [`simulate_rpbnb()`](simulate_rpbnb.md)).

- dispersion:

  Named `c(m1=, m2=)` NB2 dispersions.

- copula:

  An [`copula()`](copula.md) object giving the family and native
  parameter `par`.

- covariates:

  Optional covariate data frame; NULL -\> standard-normal columns.

- seed:

  Optional RNG seed.

## Value

list(data, mu, true, settings).
