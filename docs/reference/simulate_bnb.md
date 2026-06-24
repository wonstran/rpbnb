# Simulate data from the Famoye/Sarmanov bivariate NB distribution

Generates paired count outcomes `(y1, y2)` from the fixed-parameter
Famoye/Sarmanov bivariate NB2 joint PMF:

## Usage

``` r
simulate_bnb(
  n,
  beta1,
  beta2,
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

  Named numeric vectors of coefficients for each equation; must include
  `"(Intercept)"`.

- dispersion:

  Named numeric `c(m1 = ..., m2 = ...)` NB2 dispersion parameters
  (variance = mu + m \* mu^2).

- lambda:

  Famoye/Sarmanov dependence parameter. Must lie within the valid bounds
  implied by the marginal means and dispersions.

- covariates:

  Optional data frame of covariates. If `NULL`, standard- normal columns
  are generated for every non-intercept coefficient name.

- seed:

  Random seed for reproducibility.

## Value

A list with:

- `data`:

  data frame with `y1`, `y2`, and covariate columns

- `mu`:

  data frame with `mu1`, `mu2` (per-obs conditional means)

- `true`:

  list of true parameters: `beta1`, `beta2`, `dispersion`, `lambda`

- `settings`:

  list with `n` and `seed`

- `meta`:

  list with `seed` and `r_version`

## Details

\$\$P(Y_1=y_1, Y_2=y_2) = p_1(y_1)\\p_2(y_2)\\ \[1 +
\lambda(e^{-y_1}-c_1)(e^{-y_2}-c_2)\]\$\$

where \\c_k = E\[e^{-Y_k}\]\\ under NB2(\\\mu_k, m_k\\).

## Examples

``` r
sim <- simulate_bnb(n = 500,
  beta1 = c("(Intercept)" = 0.5, x1 = 0.3),
  beta2 = c("(Intercept)" = 0.2, x1 = -0.2),
  dispersion = c(m1 = 0.4, m2 = 0.5), lambda = 0.1, seed = 1)
head(sim$data)
#>   y1 y2         x1
#> 1  1  0 -0.6264538
#> 2  1  4  0.1836433
#> 3  2  1 -0.8356286
#> 4  3  1  1.5952808
#> 5  1  0  0.3295078
#> 6  1  0 -0.8204684
```
