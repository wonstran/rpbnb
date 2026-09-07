# Boundary-corrected LR tests for the NB2 dispersions of a fit_bnb() fit

Runs a likelihood-ratio test for each free NB2 dispersion (`m1`, `m2`)
of a fitted fixed-coefficient bivariate NB model, against a restricted
refit at that margin's exact Poisson limit (`m = 0`). `m = 0` sits on
the boundary of the parameter space, so the natural-scale summary
reports no Wald `z`/`p` for `m1`/`m2` (see [`fit_bnb()`](fit_bnb.md));
the valid test is [`lr_test()`](lr_test.md)'s `boundary = TRUE` 50:50
chi-square mixture. This is the fixed-coefficient counterpart of
[`rpbnb_boundary_tests()`](rpbnb_boundary_tests.md) – the same test,
restricted to the one boundary-null group [`fit_bnb()`](fit_bnb.md) has
(no random-coefficient scales).

## Usage

``` r
bnb_boundary_tests(fit, data, control = rpbnb_control(compute_se = FALSE))
```

## Arguments

- fit:

  A converged `bnb_fit` (from [`fit_bnb()`](fit_bnb.md)) with
  `dependence = "independence"` or `"famoye"`. Not supported for a
  [`copula()`](copula.md) dependence: [`fit_bnb()`](fit_bnb.md) itself
  refuses to combine `poisson_1`/`poisson_2` with a copula dependence,
  so there is no Poisson-limit restriction to test.

- data:

  The data frame the model was fit on. Required – the fit object does
  not store it, and every restricted model is refit on it.

- control:

  An [`rpbnb_control()`](rpbnb_control.md) for the restricted refits.
  Defaults to `compute_se = FALSE` (the LR test needs only `logLik` and
  the degrees of freedom).

## Value

An object of class `bnb_boundary_tests` (a data frame with columns
`Parameter`, `LR`, `df`, `p.value`, `Signif`, one row per tested margin)
and a `print` method. A margin already fit at
`poisson_1`/`poisson_2 = TRUE` has no free dispersion left to test and
is skipped, matching
[`rpbnb_boundary_tests()`](rpbnb_boundary_tests.md)'s dispersion-test
behaviour.

## Details

Each restricted refit is warm-started from the full fit's own
coefficients (`start = fit$coef`); [`fit_bnb()`](fit_bnb.md) pins the
tested margin's `log_m` to the Poisson placeholder regardless of what
`start` supplies for it, so passing the full vector is safe and needs no
per-test trimming. Unlike
[`rpbnb_boundary_tests()`](rpbnb_boundary_tests.md), there is no
warm-start "polish" step for a negative LR statistic:
[`fit_bnb()`](fit_bnb.md)'s likelihood is exact (`glm.nb` for
independence, closed-form-gradient BFGS for Famoye), not simulated, so a
warm-started restricted refit does not out-climb the full fit's own
optimum the way a simulated objective occasionally can.

## See also

[`lr_test()`](lr_test.md), [`fit_bnb()`](fit_bnb.md),
[`rpbnb_boundary_tests()`](rpbnb_boundary_tests.md)

## Examples

``` r
d <- read.csv(system.file("extdata", "rwm1984_clean.csv", package = "rpbnb"))
fit <- fit_bnb(docvis ~ outwork, hospvis ~ outwork, data = d,
               dependence = "famoye")
#> initial  value 16577.658491 
#> iter   2 value 15142.325204
#> iter   3 value 14710.503331
#> iter   4 value 14333.907717
#> iter   5 value 14162.447652
#> iter   6 value 14076.950085
#> iter   7 value 13623.894678
#> iter   8 value 13436.688689
#> iter   9 value 13264.517777
#> iter  10 value 13214.495326
#> iter  11 value 13075.322014
#> iter  12 value 12780.993726
#> iter  13 value 12491.582897
#> iter  14 value 11921.168011
#> iter  15 value 11753.944382
#> iter  16 value 11377.499493
#> iter  17 value 10317.762362
#> iter  18 value 10251.001528
#> iter  19 value 10047.658983
#> iter  20 value 9921.406917
#> iter  21 value 9883.025635
#> iter  22 value 9841.589759
#> iter  23 value 9777.676419
#> iter  24 value 9740.193415
#> iter  25 value 9671.855990
#> iter  26 value 9661.429609
#> iter  27 value 9660.718514
#> iter  28 value 9660.625970
#> iter  29 value 9660.618441
#> iter  30 value 9660.618315
#> iter  30 value 9660.618304
#> iter  30 value 9660.618304
#> final  value 9660.618304 
#> converged
#> initial  value 9711.685648 
#> iter   2 value 9688.919104
#> iter   2 value 9688.919104
#> iter   2 value 9688.919104
#> final  value 9688.919104 
#> converged
bnb_boundary_tests(fit, d)
#> initial  value 17292.432268 
#> iter   2 value 17291.726970
#> iter   3 value 17288.703701
#> iter   4 value 17288.629354
#> iter   5 value 17286.688728
#> iter   6 value 17283.759374
#> iter   7 value 17282.069177
#> iter   8 value 17279.858703
#> iter   9 value 17278.021219
#> iter  10 value 17277.695292
#> iter  11 value 17277.402397
#> iter  12 value 17277.364008
#> iter  13 value 17277.359683
#> iter  13 value 17277.359596
#> iter  13 value 17277.359596
#> final  value 17277.359596 
#> converged
#> initial  value 10024.934408 
#> iter   2 value 10024.653786
#> iter   3 value 10024.643436
#> iter   4 value 10024.638917
#> iter   5 value 10024.630360
#> iter   6 value 10024.615995
#> iter   7 value 10024.485963
#> iter   8 value 10024.477983
#> iter   9 value 10024.463312
#> iter   9 value 10024.463307
#> iter   9 value 10024.463307
#> final  value 10024.463307 
#> converged
#> NB2 dispersion LR tests (boundary-corrected, 50:50 chi-square mixture)
#> H0: m = 0 (margin is Poisson)
#> 
#>  Parameter         LR df p.value Signif
#>         m1 15233.4826  1  0.0000    ***
#>         m2   727.6900  1  0.0000    ***
#> 
#> Signif: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```
