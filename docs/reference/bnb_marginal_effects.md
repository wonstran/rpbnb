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

## Examples

``` r
d <- read.csv(system.file("extdata", "rwm1984_clean.csv", package = "rpbnb"))
fit <- fit_bnb(docvis ~ outwork + age, hospvis ~ outwork, data = d,
               dependence = "famoye")
bnb_marginal_effects(fit, which = "y1", type = "AME")
#> 
#> --- Marginal effects (auto) for y1 (AME) ---
#>     Name Estimate StdErr      z      p Signif     var_type
#>      age   0.0680 0.0379 1.7940 0.0728      .   continuous
#>  outwork   1.3719 0.2096 6.5446 0.0000    *** binary(0->1)
#> continuous: dE[Y]/dx_j = beta_j * mu
#> binary: E[Y|x_j=1] - E[Y|x_j=0]
```
