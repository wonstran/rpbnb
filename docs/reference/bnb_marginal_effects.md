# Marginal effects for a bivariate NB model

Average marginal effects (AME) or marginal effects at the mean (MEM) for
each margin. Continuous and binary (0/1) regressors are auto-detected;
continuous effects use \\\partial E\[Y\]/\partial x_j = \beta_j \mu\\
and binary effects use \\E\[Y\|x_j=1\] - E\[Y\|x_j=0\]\\.

## Usage

``` r
bnb_marginal_effects(
  fit,
  which = c("y1", "y2", "both", "all"),
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

  Which margin(s): "y1", "y2", "both", or "all".

- type:

  "AME" (average marginal effect) or "MEM" (effect at the mean).

- vars:

  Optional variable names or indices to restrict output.

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

## Details

For a model fitted with an
[`offset()`](https://rdrr.io/r/stats/offset.html), absolute marginal
effects scale with the fitted mean \\\mu = \exp(x'\beta +
\text{offset})\\: AME uses each observation's training offset, and MEM
uses the mean training offset.

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
bnb_marginal_effects(fit, which = "y1", type = "AME")
#> 
#> --- Marginal effects (auto) for y1 (AME) ---
#>     Name Estimate StdErr      z      p Signif     var_type
#>      age   0.0680 0.0080 8.4706 0.0000    ***   continuous
#>  outwork   1.3719 0.1930 7.1068 0.0000    *** binary(0->1)
#> continuous: dE[Y]/dx_j = beta_j * mu
#> binary: E[Y|x_j=1] - E[Y|x_j=0]
```
