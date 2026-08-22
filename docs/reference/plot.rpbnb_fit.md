# Residual diagnostic plots for a random-parameter bivariate NB model

Four base-graphics panels per margin, as for
[`plot.bnb_fit()`](plot.bnb_fit.md), built on the mixture-based
randomized quantile residuals. `resid_type = "deviance"` is not
available for `rpbnb_fit`.

## Usage

``` r
# S3 method for class 'rpbnb_fit'
plot(
  x,
  margin = c("both", "y1", "y2"),
  which = 1:4,
  resid_type = "quantile",
  seed = NULL,
  ...
)
```

## Arguments

- x:

  An `rpbnb_fit` object from [`fit_rpbnb()`](fit_rpbnb.md).

- margin:

  Which margin to plot: `"both"` (default), `"y1"`, or `"y2"`.

- which:

  Integer subset of panels `1:4`.

- resid_type:

  Residual type for panels 1 and 4: `"quantile"` (default), `"pearson"`,
  or `"response"`.

- seed:

  Optional integer seed for the RQR randomization.

- ...:

  Unused.

## Value

`NULL`, invisibly.
