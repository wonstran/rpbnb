# Simulate data from a bivariate NB process

Generates count data from a bivariate negative binomial model with
random coefficients and Famoye/Sarmanov or copula dependence.

## Usage

``` r
simulate_rpbnb_tmb(
  n,
  beta1,
  beta2,
  random_1 = NULL,
  random_2 = NULL,
  dispersion = c(m1 = 0.5, m2 = 0.5),
  dependence = "famoye",
  lambda = 0,
  covariates = NULL,
  seed = NULL
)
```

## Arguments

- n:

  Number of observations.

- beta1, beta2:

  Named coefficient vectors (must include "(Intercept)").

- random_1, random_2:

  Random coefficient specs (same format as fit_rpbnb_tmb).

- dispersion:

  Named vector c(m1 = , m2 = ) of NB2 dispersions.

- dependence:

  "famoye", "independence", or a copula() object.

- lambda:

  Famoye dependence parameter (0 = independence).

- covariates:

  Optional data frame. If NULL, standard-normal covariates generated.

- seed:

  Optional integer seed.

## Value

List with \$data (data.frame), \$truth (parameters), \$settings.

## Examples

``` r
sim <- simulate_rpbnb_tmb(n = 200,
  beta1 = c("(Intercept)" = 0.2, x1 = 0.4),
  beta2 = c("(Intercept)" = 0.1, x1 = -0.3),
  dispersion = c(m1 = 0.4, m2 = 0.5), seed = 1)
head(sim$data)
#>   y1 y2         x1
#> 1  3  1 -0.6264538
#> 2  3  2  0.1836433
#> 3  2  1 -0.8356286
#> 4  8  1  1.5952808
#> 5  2  0  0.3295078
#> 6  1  4 -0.8204684
```
