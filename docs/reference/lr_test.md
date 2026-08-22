# Likelihood-ratio test between two nested model fits

Compares a restricted fit against a full (nesting) fit by the
likelihood-ratio statistic. Works for any [`fit_bnb()`](fit_bnb.md) /
[`fit_rpbnb()`](fit_rpbnb.md) objects, since both carry a
[`logLik()`](https://rdrr.io/r/stats/logLik.html) with a `"df"`
attribute equal to the number of estimated parameters.

## Usage

``` r
lr_test(restricted, full, boundary = FALSE)
```

## Arguments

- restricted:

  The smaller (restricted) fit – fewer estimated parameters.

- full:

  The larger (full) fit that nests `restricted`.

- boundary:

  Logical. When the restriction pins a variance/dispersion-type
  parameter (a random-coefficient SD, or NB2 dispersion `m`) to its zero
  boundary, the null distribution is not a plain chi-square. Set
  `boundary = TRUE` to use the 50:50 mixture of `chisq(df)` and
  `chisq(df - 1)` (Self & Liang, 1987); for a single boundary parameter
  (`df = 1`) this halves the naive p-value. Default `FALSE` (ordinary
  interior restriction, e.g. dropping a fixed covariate). The mixture is
  exact only for a single parameter on the boundary; simultaneous
  boundary restrictions of several parameters need different weights and
  are not handled.

## Value

An object of class `rpbnb_lrtest` with the LR `statistic`, degrees of
freedom `df`, `p.value`, the two log-likelihoods and their df, and the
`boundary` flag. Has a `print` method.

## Details

The restricted model is one you fit yourself with a term removed – for
example dropping a name from `random_1` (testing a random-coefficient
SD), or a plain NB / independence fit (testing an NB2 dispersion). This
is the statistically appropriate replacement for the Wald z/p that the
natural-scale summary suppresses on positive scale/dispersion
parameters.

A likelihood-ratio test requires two *maximized* likelihoods. When
either argument is a package fit (`bnb_fit` / `rpbnb_fit`) that records
a failed optimization (`convergence$converged = FALSE`), `lr_test()`
errors rather than returning a p-value from an unfinished fit – the sign
of the statistic cannot establish convergence. Generic objects that only
carry a [`logLik()`](https://rdrr.io/r/stats/logLik.html) (no
convergence record) are still accepted, but their convergence cannot be
validated and is the caller's responsibility.

## Famoye caveat for bounded random-coefficient distributions

Each Famoye fit maximizes its likelihood over an admissible lambda
interval frozen at that fit's own starting values. With normal or
lognormal random coefficients loaded in both margins the interval is the
constant `c(-1, 1)` for every fit, so the comparison is unaffected. With
uniform or triangular coefficients (or a single varying margin) the
bound moves with the parameters, so two fits frozen at different
starting values maximize over slightly different lambda ranges and the
LR statistic inherits that second-order discrepancy. Check
`lambda_admissible` on both fits before relying on a comparison in that
regime.

## References

Self, S. G. and Liang, K.-Y. (1987). Asymptotic properties of maximum
likelihood estimators and likelihood ratio tests under nonstandard
conditions. *JASA* 82(398), 605–610.

## Examples

``` r
sim <- simulate_rpbnb(n = 600,
  beta1 = c("(Intercept)" = 0.2, x1 = 0.4),
  beta2 = c("(Intercept)" = 0.1, x1 = -0.3),
  random_1 = list(x1 = list(sd = 0.5)),
  dispersion = c(m1 = 0.4, m2 = 0.5), seed = 1)
ctrl <- rpbnb_control(compute_se = FALSE)
full <- fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim$data, random_1 = "x1",
                  draws = 100, control = ctrl)
#> initial  value 2004.254281 
#> iter   2 value 1893.236582
#> iter   3 value 1892.174243
#> iter   4 value 1885.984661
#> iter   5 value 1885.511406
#> iter   6 value 1884.309399
#> iter   7 value 1884.169782
#> iter   8 value 1883.428494
#> iter   9 value 1881.770693
#> iter  10 value 1880.478679
#> iter  11 value 1880.272268
#> iter  12 value 1880.222694
#> iter  13 value 1880.207021
#> iter  14 value 1880.206316
#> iter  15 value 1880.206283
#> iter  15 value 1880.206283
#> iter  15 value 1880.206283
#> final  value 1880.206283 
#> converged
rest <- fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim$data,
                  draws = 100, control = ctrl)   # no random coefficient
#> initial  value 2074.419885 
#> iter   2 value 1910.295949
#> iter   3 value 1907.320569
#> iter   4 value 1903.828518
#> iter   5 value 1903.683691
#> iter   6 value 1903.561472
#> iter   7 value 1903.501125
#> iter   8 value 1903.307292
#> iter   9 value 1901.481826
#> iter  10 value 1901.243860
#> iter  11 value 1901.241848
#> iter  12 value 1901.241820
#> iter  12 value 1901.241820
#> iter  12 value 1901.241820
#> final  value 1901.241820 
#> converged
lr_test(rest, full, boundary = TRUE)             # test sd(x1) = 0
#> Likelihood-ratio test
#>   full model:       logLik = -1880.2063  (df = 8)
#>   restricted model: logLik = -1901.2418  (df = 7)
#>   --------------------------------------------------
#>   LR statistic = 42.0711  on 1 df   p = 0.0000  ***
#>   (boundary-corrected 50:50 chi-square mixture)
```
