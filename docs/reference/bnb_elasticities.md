# Elasticities and semi-elasticities for a bivariate NB model

For continuous regressors, the elasticity is \\\beta_j E\[x_j\]\\; for
binary (0/1) regressors, the semi-elasticity is \\\exp(\beta_j) - 1\\
(proportional change moving from 0 to 1).

## Usage

``` r
bnb_elasticities(
  fit,
  which = c("y1", "y2", "both"),
  type = c("AME", "MEM"),
  vars = NULL,
  include_intercept = FALSE,
  digits = 4,
  print_output = TRUE
)
```

## Arguments

- fit:

  A `bnb_fit` object from [`fit_bnb()`](fit_bnb.md).

- which:

  Which margin(s): "y1", "y2", or "both".

- type:

  "AME" (uses sample mean of x) or "MEM" (uses mean design row).

- vars:

  Optional variable names to restrict output.

- include_intercept:

  Logical; include the intercept term.

- digits:

  Number of decimal places for printed output.

- print_output:

  Logical; if `FALSE`, suppress printing.

## Value

A data frame (single margin) or a named list of data frames (both), each
with columns `Name`, `Estimate`, `StdErr`, `z`, `p`, `Signif`,
`var_type`.

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
bnb_elasticities(fit, which = "both", type = "AME")
#> 
#> --- Elasticities / semi-elasticities (y1, AME) ---
#>     Name Estimate StdErr      z      p Signif                   var_type
#>  outwork   0.5236 0.0840 6.2351 0.0000    *** binary (0->1): exp(beta)-1
#>      age   0.9459 0.1037 9.1198 0.0000    ***    continuous: beta * E[x]
#> Continuous vars: elasticity = beta * E[x].
#> 0/1 vars: semi-elasticity = exp(beta) - 1 (percent change from 0->1).
#> 
#> --- Elasticities / semi-elasticities (y2, AME) ---
#>     Name Estimate StdErr      z      p Signif                   var_type
#>  outwork   0.3585 0.1878 1.9087 0.0563      . binary (0->1): exp(beta)-1
#> Continuous vars: elasticity = beta * E[x].
#> 0/1 vars: semi-elasticity = exp(beta) - 1 (percent change from 0->1).
```
