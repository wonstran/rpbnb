# Boundary-corrected LR tests for all boundary parameters of an rpbnb_fit

Runs a likelihood-ratio test for every random-coefficient standard
deviation (`sd1:*`, `sd2:*`) and NB2 dispersion (`m1`, `m2`) of a fitted
random-parameter bivariate NB model, and merges them into one table.
These are the parameters whose null lies on the boundary of the
parameter space (SD = 0, or dispersion `m = 0` = Poisson), for which the
natural-scale summary reports no Wald `z`/`p`; each test uses the 50:50
chi-square boundary correction of [`lr_test()`](lr_test.md)
(`boundary = TRUE`).

## Usage

``` r
rpbnb_boundary_tests(
  fit,
  data,
  control = rpbnb_control(compute_se = FALSE),
  which = c("sd", "dispersion")
)
```

## Arguments

- fit:

  A converged `rpbnb_fit` (the full model), from a Famoye or a
  [`copula()`](copula.md) dependence. Both paths use the exact `m = 0`
  branch for the dispersion tests.

- data:

  The data frame the model was fit on. Required – the fit object does
  not store it, and every restricted model is refit on the same data.

- control:

  An [`rpbnb_control()`](rpbnb_control.md) for the restricted refits.
  Defaults to `compute_se = FALSE` (the LR test needs only `logLik` and
  the degrees of freedom). `draws`/`draw_type`/`seed` are taken from
  `fit`, not `control`.

- which:

  Which parameter groups to test, any subset of `"sd"` (the
  random-coefficient scales), `"dispersion"` (the NB2 dispersions `m1`,
  `m2`), and `"dependence"` (the association parameter). The default is
  `c("sd", "dispersion")` – the two boundary-null groups. `"dependence"`
  is opt-in because it costs another full refit and, for three of the
  four families, tests an *interior* null whose ordinary Wald `z` in
  [`summary()`](https://rdrr.io/r/base/summary.html) is already valid.

  The dependence test restricts the model to independence and compares
  it to `fit`: one row labelled `lam` (Famoye), `theta` (Frank /
  Clayton), or `rho` (Gaussian). The restriction pins the working-scale
  dependence parameter at its family's independence value rather than
  refitting a different family, so both fits keep the same draws and the
  same parameter block and the comparison is a clean 1-df restriction.
  Only Clayton's null (`theta > 0`, so `theta = 0` is a boundary) takes
  the 50:50 mixture; Famoye, Frank, and Gaussian nulls are interior and
  get an ordinary chi-square(1).

## Value

An object of class `rpbnb_boundary_tests`: a data frame with columns
`Parameter`, `LR`, `df`, `p.value`, `Signif` (one row per boundary
parameter), and a `print` method.

Under Famoye dependence with uniform or triangular random coefficients
(or a single varying margin), the full and restricted fits' admissible
lambda intervals are frozen at different starting values, so each LR
statistic compares maxima over slightly different lambda ranges; see the
"Famoye caveat" section of [`lr_test()`](lr_test.md). Normal/lognormal
coefficients in both margins are unaffected (the interval is the
constant `c(-1, 1)`).

## Details

Each test refits a properly nested restricted model. With **multiple
random coefficients in an equation, each SD is tested individually** (a
1-df restriction). The restricted fit keeps the full random
specification but sets the tested coefficient's draw column to the
distribution median (`u = 0.5`), which zeroes that coefficient's
per-draw *deviation* exactly for every supported distribution (`u = 0.5`
maps to base 0 for normal/lognormal/ triangular, and to the centered
value `2 * 0.5 - 1 = 0` for uniform). The coefficient therefore
collapses to its SD-zero null – the ordinary fixed coefficient `b` for
normal/uniform/triangular, and `sign * exp(b)` for lognormal – on every
draw, independent of the covariate scale. Its (now inert) log-scale is
pinned only to drop it from the free-parameter count. Every other draw
column is the full model's exact stored draw, so the two simulated
log-likelihoods are compared on common random numbers, and each
restricted fit is **warm-started from the full fit's coefficients** so
the start-sensitive simulated objective does not settle at an inferior
optimum. A restricted fit that fails to converge yields `NA` inference
(with a warning) rather than a p-value, and a non-converged full `fit`
is rejected.

## See also

[`lr_test()`](lr_test.md), [`fit_rpbnb()`](fit_rpbnb.md)

## Examples

``` r
# \donttest{
sim <- simulate_rpbnb(n = 600,
  beta1 = c("(Intercept)" = 0.2, x1 = 0.4),
  beta2 = c("(Intercept)" = 0.1, x1 = -0.3),
  random_1 = list(x1 = list(sd = 0.5)),
  dispersion = c(m1 = 0.4, m2 = 0.5), seed = 1)
fit <- fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim$data, random_1 = "x1",
                 draws = 200, seed = 1)
#> initial  value 2009.832764 
#> iter   2 value 1893.836136
#> iter   3 value 1892.715910
#> iter   4 value 1886.982367
#> iter   5 value 1886.393516
#> iter   6 value 1885.271955
#> iter   7 value 1885.109995
#> iter   8 value 1884.705979
#> iter   9 value 1882.643001
#> iter  10 value 1880.961838
#> iter  11 value 1880.668967
#> iter  12 value 1880.580056
#> iter  13 value 1880.545950
#> iter  14 value 1880.543657
#> iter  15 value 1880.543577
#> iter  15 value 1880.543576
#> iter  15 value 1880.543576
#> final  value 1880.543576 
#> converged
rpbnb_boundary_tests(fit, sim$data)
#> initial  value 1947.816465 
#> iter   2 value 1929.535548
#> iter   3 value 1909.461118
#> iter   4 value 1906.913903
#> iter   5 value 1906.868878
#> iter   6 value 1906.814172
#> iter   7 value 1904.951886
#> iter   8 value 1904.492674
#> iter   9 value 1902.485228
#> iter  10 value 1901.685794
#> iter  11 value 1901.259202
#> iter  12 value 1901.242308
#> iter  13 value 1901.241853
#> iter  14 value 1901.241820
#> iter  14 value 1901.241820
#> iter  14 value 1901.241820
#> final  value 1901.241820 
#> converged
#> initial  value 1893.844512 
#> iter   2 value 1890.094205
#> iter   3 value 1889.911119
#> iter   4 value 1889.908537
#> iter   5 value 1889.897732
#> iter   6 value 1889.702160
#> iter   7 value 1889.594404
#> iter   8 value 1889.562967
#> iter   9 value 1889.516944
#> iter  10 value 1889.493464
#> iter  11 value 1889.483387
#> iter  12 value 1889.483362
#> iter  12 value 1889.483361
#> iter  12 value 1889.483361
#> final  value 1889.483361 
#> converged
#> initial  value 1904.066691 
#> iter   2 value 1904.058773
#> iter   3 value 1904.057644
#> iter   4 value 1904.056081
#> iter   5 value 1904.052416
#> iter   6 value 1904.050524
#> iter   7 value 1904.046822
#> iter   8 value 1904.043978
#> iter   9 value 1904.040569
#> iter  10 value 1904.038653
#> iter  10 value 1904.038653
#> iter  10 value 1904.038653
#> final  value 1904.038653 
#> converged
#> Boundary-parameter LR tests (boundary-corrected, 50:50 chi-square mixture)
#> H0: parameter = 0 (random SD absent, or margin Poisson)
#> 
#>  Parameter      LR df p.value Signif
#>     sd1:x1 41.3965  1  0.0000    ***
#>         m1 17.8796  1  0.0000    ***
#>         m2 46.9902  1  0.0000    ***
#> 
#> Signif: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
# }
```
