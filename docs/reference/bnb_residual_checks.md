# Residual checks for a bivariate NB model

Formal residual diagnostics for a [`fit_bnb()`](fit_bnb.md) or
[`fit_rpbnb()`](fit_rpbnb.md) fit: a normality test on the randomized
quantile residuals (Shapiro-Wilk for `n <= 5000`, else
Kolmogorov-Smirnov vs N(0,1)); the NB2 dispersion statistic
(`sum(pearson^2) / (n - k)`) per margin; the cross-margin RQR
correlation (a check that the Famoye/copula dependence captured the
association); an outlier count/index list (`|RQR| > outlier_z`); and a
composite misspecification verdict combining these signals.

## Usage

``` r
bnb_residual_checks(
  fit,
  seed = NULL,
  outlier_z = 3,
  digits = 4,
  print_output = TRUE
)
```

## Arguments

- fit:

  A `bnb_fit` or `rpbnb_fit` object.

- seed:

  Optional integer seed for the RQR randomization.

- outlier_z:

  Absolute-RQR threshold flagging an outlier (default 3).

- digits:

  Decimal places for printed output.

- print_output:

  Logical; if `FALSE`, suppress printing.

## Value

Invisibly, an object of class `bnb_residual_checks`.
