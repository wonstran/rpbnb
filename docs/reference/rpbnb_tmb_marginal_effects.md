# Marginal effects for a rpbnb_tmb model

Computes average marginal effects (AME) or marginal effects at the mean
(MEM) for a fitted random-parameter bivariate negative binomial model.
For continuous covariates the effect is \\\partial E\[Y\]/\partial
x_j\\; for binary covariates it is the discrete difference \\E\[Y\|x_j =
1\] - E\[Y\|x_j = 0\]\\. When a coefficient is random the per-draw
realized coefficients are used to form the Monte-Carlo integrated mean.

## Usage

``` r
rpbnb_tmb_marginal_effects(
  fit,
  which = c("y1", "y2", "both"),
  type = c("AME", "MEM"),
  vars = NULL,
  include_intercept = FALSE,
  digits = 4L,
  scaling = NULL,
  log_vars = NULL,
  ...
)
```

## Arguments

- fit:

  An object of class `rpbnb_tmb_fit`.

- which:

  Which margin: `"y1"`, `"y2"`, or `"both"`.

- type:

  `"AME"` (average over the sample) or `"MEM"` (at the sample mean of
  covariates).

- vars:

  Optional character vector of variable names to restrict output.

- include_intercept:

  Logical; include the intercept term in output.

- digits:

  Number of decimal places for printed table entries.

- scaling:

  Optional named list of `c(center =, scale =)` pairs, one per covariate
  that was centred and/or scaled before fitting, used to report the
  results in the covariate's original units. Columns not named are left
  alone, so binary indicators and untransformed variables need no entry.

  Only the reported quantity is rescaled; the fitted design is not
  rebuilt. That distinction is not cosmetic when a standardized
  covariate also carries a random coefficient. The random term enters as
  `x_std * dev`, so substituting `x_raw = x_std * s + c` would add a
  `(c/s) * dev` random intercept the model never estimated – which is
  how a centred random slope turns back into the disguised random
  intercept that centring was meant to remove. The chain rule avoids
  that entirely: \\\partial\mu/\partial x\_{raw} = (\partial\mu/\partial
  x\_{std})/s\\, and the elasticity's leading factor becomes
  \\x\_{raw}/s = x\_{std} + c/s\\. Binary effects are untouched.

- log_vars:

  Optional character vector naming covariates that are ALREADY a log, so
  that results are reported per unit of the underlying variable \\v_j =
  \exp(x_j)\\ rather than per unit of \\x_j\\ itself.

  Without this the elasticity formula treats \\\log v\\ as an ordinary
  regressor and returns \\\bar{x} b\\, the elasticity with respect to
  the LOG – for the truck model's `LNAADT_3` that is 8.59, where the
  elasticity with respect to AADT is the coefficient itself, 0.871. A
  ten-fold overstatement that looks perfectly plausible in a table.

  Because \\\partial\log\mu/\partial\log v = (\partial\mu/\partial
  x)/\mu\\, the elasticity is \\E\[d\mu/\mu\]/s\\ and the marginal
  effect is \\E\[d\mu/(s v)\]\\, with \\v = \exp(x\_{std} s + c)\\
  picking up any `scaling` applied to the logged column. Named columns
  are reported with type `"log-continuous"`. Binary columns are
  rejected.

- ...:

  Not used.

## Value

A data frame (single margin) or named list of two data frames
(`"both"`).
