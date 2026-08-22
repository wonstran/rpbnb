# Residual diagnostic plots for a bivariate NB model

Four base-graphics panels per margin: residuals-vs-fitted, a normal QQ
plot of the randomized quantile residuals, a histogram of the RQR with
an N(0,1) overlay, and a scale-location plot. The QQ and histogram
panels always use RQR (only these are approximately N(0,1) under a
correct count model).

## Usage

``` r
# S3 method for class 'bnb_fit'
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

  A `bnb_fit` object from [`fit_bnb()`](fit_bnb.md).

- margin:

  Which margin to plot: `"both"` (default), `"y1"`, or `"y2"`.

- which:

  Integer subset of panels `1:4` (1 = residuals-vs-fitted, 2 = QQ, 3 =
  histogram, 4 = scale-location).

- resid_type:

  Residual type for panels 1 and 4: `"quantile"` (default), `"pearson"`,
  `"deviance"`, or `"response"`.

- seed:

  Optional integer seed for the RQR randomization.

- ...:

  Unused.

## Value

`NULL`, invisibly (called for the side effect of drawing).
