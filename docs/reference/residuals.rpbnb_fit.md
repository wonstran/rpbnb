# Residuals for a random-parameter bivariate NB model

Per-margin residuals for an [`fit_rpbnb()`](fit_rpbnb.md) model. Each
margin is a mixture of NB2 distributions over the random-coefficient
draws. `"quantile"` returns randomized quantile residuals (Dunn & Smyth
1996) from the exact mixture predictive CDF (the recommended residual);
`"pearson"` uses the exact mixture marginal variance. `"deviance"` is
not defined for the mixture and errors.

## Usage

``` r
# S3 method for class 'rpbnb_fit'
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

  An `rpbnb_fit` object from [`fit_rpbnb()`](fit_rpbnb.md).

- type:

  Residual type: `"quantile"` (default), `"pearson"`, or `"response"`.
  `"deviance"` is not supported for `rpbnb_fit`.

- margin:

  Which margin: `"both"` (default), `"y1"`, or `"y2"`.

- seed:

  Optional integer seed for the quantile-residual randomization; does
  not disturb the caller's RNG stream.

- ...:

  Unused.

## Value

A numeric vector for a single margin, or a two-column data frame (`y1`,
`y2`) for `margin = "both"`.
