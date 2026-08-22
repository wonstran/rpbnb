# Residuals for a bivariate NB model

Per-margin residuals for a fixed-coefficient [`fit_bnb()`](fit_bnb.md)
model. Each margin is NB2 with fitted mean `mu` and dispersion `m`
(`size = 1/m`). `"quantile"` returns randomized quantile residuals (Dunn
& Smyth 1996), which are approximately N(0,1) under a correct model and
are the recommended residual for normality-style diagnostics on count
data.

## Usage

``` r
# S3 method for class 'bnb_fit'
residuals(
  object,
  type = c("quantile", "pearson", "deviance", "response"),
  margin = c("both", "y1", "y2"),
  seed = NULL,
  ...
)
```

## Arguments

- object:

  A `bnb_fit` object from [`fit_bnb()`](fit_bnb.md).

- type:

  Residual type: `"quantile"` (default), `"pearson"`, `"deviance"`, or
  `"response"`.

- margin:

  Which margin: `"both"` (default), `"y1"`, or `"y2"`.

- seed:

  Optional integer seed for the quantile-residual randomization (ignored
  for other types); does not disturb the caller's RNG stream.

- ...:

  Unused.

## Value

A numeric vector for a single margin, or a two-column data frame (`y1`,
`y2`) for `margin = "both"`.
