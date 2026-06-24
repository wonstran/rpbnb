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
bnb_elasticities(fit, which = "both", type = "AME")
#> 
#> --- Elasticities / semi-elasticities (y1, AME) ---
#>     Name Estimate StdErr      z      p Signif                   var_type
#>  outwork   0.5235 0.1038 5.0408 0.0000    *** binary (0->1): exp(beta)-1
#>      age   0.9453 0.3763 2.5124 0.0120      *    continuous: beta * E[x]
#> Continuous vars: elasticity = beta * E[x].
#> 0/1 vars: semi-elasticity = exp(beta) - 1 (percent change from 0->1).
#> 
#> --- Elasticities / semi-elasticities (y2, AME) ---
#>     Name Estimate StdErr      z      p Signif                   var_type
#>  outwork   0.3552 0.6159 0.5767 0.5641        binary (0->1): exp(beta)-1
#> Continuous vars: elasticity = beta * E[x].
#> 0/1 vars: semi-elasticity = exp(beta) - 1 (percent change from 0->1).
```
