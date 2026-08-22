# Elasticities and semi-elasticities for a random-parameter bivariate NB model

Continuous elasticities \\x\_{ij}\\(\partial \mu_i/\partial
x\_{ij})/\mu_i\\ and binary semi-elasticities
\\\mu_i(x_j=1)/\mu_i(x_j=0) - 1\\, built on the Monte-Carlo integrated
population mean \\\mu_i = E\_\beta\[\exp(x_i'\beta)\]\\ of an
[`fit_rpbnb()`](fit_rpbnb.md) fit. Under fixed coefficients these reduce
to \\\beta_j E\[x_j\]\\ and \\\exp(\beta_j) - 1\\. Standard errors use a
numeric delta method over the equation's mean and log-scale parameters.

## Usage

``` r
rpbnb_elasticities(
  fit,
  which = c("y1", "y2", "both"),
  type = c("AME", "MEM"),
  vars = NULL,
  include_intercept = FALSE,
  digits = 4,
  print_output = TRUE,
  n_cores = 1L,
  scaling = NULL,
  log_vars = NULL
)
```

## Arguments

- fit:

  An `rpbnb_fit` object from [`fit_rpbnb()`](fit_rpbnb.md).

- which:

  Which margin(s): "y1", "y2", or "both".

- type:

  "AME" (average over the sample) or "MEM" (evaluated at the mean row).

- vars:

  Optional variable names or indices to restrict output.

- include_intercept:

  Logical; include the intercept term.

- digits:

  Number of decimal places for printed output.

- print_output:

  Logical; if `FALSE`, suppress printing.

- n_cores:

  Number of worker processes for the delta-method standard-error
  jacobian (1 = sequential, the default). When `n_cores > 1`, the
  jacobian's independent per-parameter columns are dispatched across a
  [`parallel::makeCluster()`](https://rdrr.io/r/parallel/makeCluster.html)
  cluster (one cluster per call, shared across every equation `which`
  computes); results are numerically identical to the sequential path.
  Falls back to sequential with a warning if the `parallel` package is
  unavailable.

- scaling:

  Optional named list of `c(center =, scale =)` pairs, one per covariate
  that was centred and/or scaled before fitting (e.g. via
  `rpbnb(standardize = TRUE)`, whose `$scaling` field is exactly this
  shape), used to report the results in the covariate's original units.
  See [`rpbnb_marginal_effects()`](rpbnb_marginal_effects.md)'s
  `scaling` documentation for the full explanation (elasticities and
  marginal effects share it verbatim).

- log_vars:

  Optional character vector naming covariates that are ALREADY a log;
  see [`rpbnb_marginal_effects()`](rpbnb_marginal_effects.md)'s
  `log_vars` documentation.

## Value

A data frame (single margin, invisibly) or a named list of data frames
(`both`), each with columns `Name`, `Estimate`, `StdErr`, `z`, `p`,
`Signif`, `var_type`.

## See also

[`bnb_elasticities()`](bnb_elasticities.md) for fixed-coefficient
`bnb_fit` models.

## Examples

``` r
sim <- simulate_rpbnb(n = 400,
  beta1 = c("(Intercept)" = 0.2, x1 = 0.4),
  beta2 = c("(Intercept)" = 0.1, x1 = -0.3),
  random_1 = list(x1 = list(sd = 0.5)),
  dispersion = c(m1 = 0.4, m2 = 0.5), seed = 1)
fit <- fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim$data, random_1 = "x1",
                 draws = 100)
#> initial  value 1368.539926 
#> iter   2 value 1297.006636
#> iter   3 value 1295.477493
#> iter   4 value 1291.484139
#> iter   5 value 1289.089478
#> iter   6 value 1287.277840
#> iter   7 value 1285.643388
#> iter   8 value 1276.959179
#> iter   9 value 1276.783700
#> iter  10 value 1272.578163
#> iter  11 value 1272.010250
#> iter  12 value 1270.828218
#> iter  13 value 1270.639435
#> iter  14 value 1270.479304
#> iter  15 value 1270.469751
#> iter  16 value 1270.460411
#> iter  17 value 1270.460258
#> iter  18 value 1270.460242
#> iter  18 value 1270.460242
#> final  value 1270.460242 
#> converged
rpbnb_elasticities(fit, which = "both", type = "AME")
#> 
#> --- Elasticities (RP integrated mean) for y1 (AME) ---
#>  Name Estimate StdErr      z      p Signif            var_type
#>    x1   0.3655 0.1055 3.4647 0.0005    *** elasticity (random)
#> continuous: elasticity = x_j * (dE[Y]/dx_j) / E[Y]
#> binary: semi-elasticity = E[Y|x_j=1]/E[Y|x_j=0] - 1
#> '(random)' vars use the draw-integrated formula.
#> 
#> --- Elasticities (RP integrated mean) for y2 (AME) ---
#>  Name Estimate StdErr       z      p Signif   var_type
#>    x1  -0.0137 0.0024 -5.6599 0.0000    *** elasticity
#> continuous: elasticity = x_j * (dE[Y]/dx_j) / E[Y]
#> binary: semi-elasticity = E[Y|x_j=1]/E[Y|x_j=0] - 1
#> '(random)' vars use the draw-integrated formula.
```
