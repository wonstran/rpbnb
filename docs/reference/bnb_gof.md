# Goodness of fit for a bivariate NB model

Reports log-likelihood, AIC, BIC, and four pseudo-R-squared measures
(McFadden, McFadden adjusted, Cox-Snell, Nagelkerke) relative to an
intercept-only null model refit with the same dependence structure.

## Usage

``` r
bnb_gof(fit, digits = 4, print_output = TRUE)
```

## Arguments

- fit:

  A `bnb_fit` object from [`fit_bnb()`](fit_bnb.md).

- digits:

  Number of decimal places for printed output.

- print_output:

  Logical; if `FALSE`, suppress printing and return the result
  invisibly.

## Value

Invisibly, a list with `n`, `k`, `logLik_full`, `logLik_null`, `AIC`,
`BIC`, a named numeric vector `pseudoR2`, and the `null_fit`.

## Examples

``` r
d <- read.csv(system.file("extdata", "rwm1984_clean.csv", package = "rpbnb"))
fit <- fit_bnb(docvis ~ outwork + age, hospvis ~ outwork, data = d,
               dependence = "famoye")
bnb_gof(fit)
#> 
#> --- Goodness of Fit ---
#> n = 3874, k = 8
#> logLik(full) = -9620.0001
#> AIC = 19256.0001    BIC = 19306.0965
#> 
#> Pseudo R-squared:
#>        Metric  Value
#>      McFadden 0.0095
#>  McFadden_adj 0.0086
#>      CoxSnell 0.0463
#>    Nagelkerke 0.0466
```
