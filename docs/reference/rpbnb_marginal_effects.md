# Marginal effects for a random-parameter bivariate NB model

Average marginal effects (AME) or marginal effects at the mean (MEM) for
each margin of an [`fit_rpbnb()`](fit_rpbnb.md) fit, built on the
Monte-Carlo integrated population mean \\\mu_i =
E\_\beta\[\exp(x_i'\beta)\]\\ (the same estimand as
`predict.rpbnb_fit()`, reusing the fit's stored draws). Continuous
effects are \\\partial \mu_i/\partial x\_{ij} = \mathrm{mean}\_r\\
\mathrm{coef}\_{rj} \exp(\mathrm{lp}\_{ir})\\ (the realized coefficient
per draw, so random and fixed columns are handled uniformly); binary
(0/1) effects are the integrated discrete difference \\E\[Y\|x_j=1\] -
E\[Y\|x_j=0\]\\. Standard errors use a numeric delta method over the
equation's mean and log-scale parameters.

## Usage

``` r
rpbnb_marginal_effects(
  fit,
  which = c("y1", "y2", "both", "all"),
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

  Which margin(s): "y1", "y2", "both", or "all".

- type:

  "AME" (average over the sample) or "MEM" (effect at the mean row).

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
  Columns not named are left alone, so binary indicators and
  untransformed variables need no entry.

  Only the reported quantity is rescaled; the fitted design is not
  rebuilt. That distinction is not cosmetic when a standardized
  covariate also carries a random coefficient: the random term enters as
  `x_std * dev`, so substituting `x_raw = x_std * s + c` would add a
  `(c/s) * dev` random intercept the model never estimated – the
  disguised random intercept centring was meant to remove, back again.
  The chain rule avoids that: `d(mu)/d(x_raw) = (d(mu)/d(x_std))/s`, and
  the elasticity's leading factor becomes `x_raw/s = x_std + c/s`.
  Binary effects are untouched.

- log_vars:

  Optional character vector naming covariates that are ALREADY a log, so
  results are reported per unit of the underlying variable
  `v_j = exp(x_j)` rather than per unit of `x_j` itself. Without this
  the elasticity formula treats `log v` as an ordinary regressor and
  returns `xbar * b`, the elasticity with respect to the LOG – for a
  log-traffic column that can be an order of magnitude off from the
  elasticity with respect to traffic itself (the coefficient). Rejects
  binary columns (a 0/1 indicator is not a log-transformed variable).

## Value

A data frame (single margin, invisibly) or a named list of data frames
(`both`/`all`), each with columns `Name`, `Estimate`, `StdErr`, `z`,
`p`, `Signif`, `var_type`.

## Details

For a model fitted with an
[`offset()`](https://rdrr.io/r/stats/offset.html), absolute marginal
effects scale with the offset-inclusive mean: AME uses each
observation's training offset, and MEM uses the mean training offset.

## See also

[`bnb_marginal_effects()`](bnb_marginal_effects.md) for
fixed-coefficient `bnb_fit` models.

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
rpbnb_marginal_effects(fit, which = "y1", type = "AME")
#> 
#> --- Marginal effects (RP integrated mean) for y1 (AME) ---
#>  Name Estimate StdErr      z      p Signif            var_type
#>    x1   1.0818 0.3872 2.7941 0.0052     ** continuous (random)
#> continuous: dE[Y]/dx_j = mean_r coef_rj * exp(lp_r)
#> binary: E[Y|x_j=1] - E[Y|x_j=0] (integrated over draws)
#> '(random)' vars use the draw-integrated formula.
```
