# Confidence interval for a fitted dependence parameter

Returns a profile-likelihood interval for the dependence parameter of a
fitted model, computed on the unconstrained working scale and mapped
through the family's monotone link. This is the tool to reach for when
[`summary()`](https://rdrr.io/r/base/summary.html) reports `NA` for the
dependence standard error: at a boundary the delta-method derivative
collapses, so \\SE = \|d\theta/dz\| \cdot SE(z)\\ is a \\0 \times
\infty\\ product and no symmetric standard error exists. A profile
interval needs no derivative.

## Usage

``` r
rpbnb_tmb_dependence_profile(
  fit,
  level = 0.95,
  method = c("profile", "wald"),
  ...
)
```

## Arguments

- fit:

  An object of class `rpbnb_tmb_fit`.

- level:

  Coverage level; one number strictly between 0 and 1.

- method:

  `"profile"` (default) for a profile-likelihood interval, or `"wald"`
  for a Wald interval on the working scale mapped through the same link.
  `"profile"` requires the TMB objective, retained only under
  `keep = "full"`; when it is unavailable this degrades to `"wald"` with
  a warning rather than failing.

- ...:

  Passed to [`tmbprofile`](https://rdrr.io/pkg/TMB/man/tmbprofile.html),
  e.g. `ytol` or `parm.range`. `lincomb` and `slice` are not supported
  and raise an error: the first would profile a different quantity than
  `z_dep` while still being reported as if it were, and the second
  returns a likelihood slice rather than a profile.

## Value

A data frame with one row per reported dependence quantity and columns
`parameter`, `estimate`, `lower`, `upper`, `level`, and `method` – the
method actually used, which may differ from the one requested. The raw
profile, when one was computed, is attached as `attr(, "profile")` so it
can be plotted.

## Details

For Famoye this is the usual situation, because the admissible lambda
interval is frozen at the starting values (see `lambda_bounds` in
[`fit_rpbnb_tmb`](fit_rpbnb_tmb.md)). An interval around a pinned
estimate is still an interval around an artefact – widen the box first
by refitting from better starting values.

## See also

[`fit_rpbnb_tmb`](fit_rpbnb_tmb.md)

## Examples

``` r
if (FALSE) { # \dontrun{
fit <- fit_rpbnb_tmb(y1 ~ x, y2 ~ x, data = d,
                     dependence = "famoye", keep = "full")
rpbnb_tmb_dependence_profile(fit)
plot(attr(rpbnb_tmb_dependence_profile(fit), "profile"))
} # }
```
