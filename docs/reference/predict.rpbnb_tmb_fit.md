# Predict from a fitted bivariate count model

Predictions integrate over the retained simulation draws when random
coefficients are present. Link predictions are the log of the integrated
response mean, so `exp(predict(fit, type = "link"))` equals response
predictions.

## Usage

``` r
# S3 method for class 'rpbnb_tmb_fit'
predict(
  object,
  newdata = NULL,
  type = c("response", "link"),
  which = c("both", "y1", "y2"),
  ...
)
```

## Arguments

- object:

  A fitted `rpbnb_tmb_fit` object.

- newdata:

  Optional data frame. If omitted, the retained fitting design is used;
  compact fits require `newdata`.

- type:

  Either `"response"` or `"link"`.

- which:

  Return both margins or only `"y1"` or `"y2"`.

- ...:

  Reserved for future use.

## Value

A two-column numeric matrix for `which = "both"`, otherwise a numeric
vector.
