# Goodness of fit for a bivariate NB model

Reports log-likelihood, AIC, BIC, and four pseudo-R-squared measures
(McFadden, McFadden adjusted, Cox-Snell, Nagelkerke) relative to an
intercept-only null model refit with the same dependence structure
(copula fits rebuild the copula from `fit$cop_family`). A margin fitted
with an [`offset()`](https://rdrr.io/r/stats/offset.html) keeps that
offset in the null (an intercept-plus-offset model), so the null shares
the full model's exposure/sampling structure and the pseudo-R-squared
values remain comparable. Pseudo-R-squared values are returned raw and
are not clamped to `[0, 1]`; a negative value flags a full model that
fits worse than the null.

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
#> initial  value 16577.658491 
#> iter   2 value 13271.264393
#> iter   3 value 11329.413288
#> iter   4 value 10927.948782
#> iter   5 value 10460.070098
#> iter   6 value 10268.013084
#> iter   7 value 10239.090291
#> iter   8 value 10217.702577
#> iter   9 value 10098.173611
#> iter  10 value 10011.189900
#> iter  11 value 9742.281069
#> iter  12 value 9697.323089
#> iter  13 value 9638.136548
#> iter  14 value 9621.201100
#> iter  15 value 9620.312447
#> iter  16 value 9620.114617
#> iter  17 value 9620.104409
#> iter  18 value 9620.098183
#> iter  19 value 9620.089991
#> iter  19 value 9620.089991
#> iter  19 value 9620.089991
#> final  value 9620.089991 
#> converged
#> initial  value 9667.499495 
#> iter   2 value 9655.410932
#> iter   3 value 9649.351194
#> iter   4 value 9649.083518
#> iter   5 value 9648.593995
#> iter   6 value 9648.079321
#> iter   7 value 9643.429855
#> iter   8 value 9632.852428
#> iter   9 value 9627.281386
#> iter  10 value 9620.485655
#> iter  11 value 9620.372605
#> iter  12 value 9620.226082
#> iter  13 value 9620.184059
#> iter  14 value 9620.034085
#> iter  15 value 9619.983585
#> iter  16 value 9619.962489
#> iter  17 value 9619.956429
#> iter  18 value 9619.954798
#> iter  18 value 9619.954798
#> final  value 9619.954798 
#> converged
bnb_gof(fit)
#> initial  value 16577.658491 
#> iter   2 value 14733.497221
#> iter   3 value 14016.919440
#> iter   4 value 13702.922008
#> iter   5 value 13606.765438
#> iter   6 value 13566.187688
#> iter   7 value 13371.878283
#> iter   8 value 13168.988190
#> iter   9 value 12888.819381
#> iter  10 value 12556.341683
#> iter  11 value 11993.172079
#> iter  12 value 11944.951434
#> iter  13 value 10298.911552
#> iter  14 value 10270.789974
#> iter  15 value 9920.128844
#> iter  16 value 9893.295099
#> iter  17 value 9756.488764
#> iter  17 value 9756.488764
#> final  value 9756.488764 
#> converged
#> initial  value 9764.540322 
#> iter   2 value 9728.967786
#> iter   2 value 9728.967786
#> iter   2 value 9728.967786
#> final  value 9728.967786 
#> converged
#> Warning: Observed information for the famoye BNB is not positive definite (min eigenvalue -30900); a ridge of 30900 was added before inversion. The resulting standard errors are regularized, not observed-information SEs -- inspect fit$hessian_diag.
#> 
#> --- Goodness of Fit ---
#> n = 3874, k = 8
#> logLik(full) = -9619.9548
#> AIC = 19255.9096    BIC = 19306.0059
#> 
#> Pseudo R-squared:
#>        Metric  Value
#>      McFadden 0.0112
#>  McFadden_adj 0.0104
#>      CoxSnell 0.0547
#>    Nagelkerke 0.0551
```
