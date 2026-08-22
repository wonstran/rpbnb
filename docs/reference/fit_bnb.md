# Fit a bivariate negative binomial regression model

Fit a bivariate negative binomial regression model

## Usage

``` r
fit_bnb(
  formula_1,
  formula_2,
  data,
  dependence = c("independence", "famoye"),
  start = NULL,
  control = rpbnb_control(),
  poisson_1 = FALSE,
  poisson_2 = FALSE
)
```

## Arguments

- formula_1, formula_2:

  Model formulas for the two count outcomes. An equation-specific
  [`offset()`](https://rdrr.io/r/stats/offset.html) term (e.g.
  `y ~ x + offset(log(exposure))`) is supported on every dependence
  path: the offset enters that margin's linear predictor additively
  (mean `exp(x'beta + offset)`) during estimation, and is carried
  through the stored fitted means and both
  [`predict()`](https://rdrr.io/r/stats/predict.html) methods.

- data:

  A data frame.

- dependence:

  Dependence structure: "independence" (two univariate NB2 margins),
  "famoye" (Famoye/Sarmanov bivariate NB), or a [`copula()`](copula.md)
  object (Frank / Gaussian / Clayton discrete-copula bivariate NB; the
  dependence parameter is estimated).

- start:

  Optional starting parameter vector. May be positional (length equal to
  the number of parameters) or named; a named vector is reordered to the
  canonical parameter order and a named partial vector is merged into
  the defaults (unknown or duplicate names are rejected). When `start`
  is `NULL`, the famoye path uses a multi-start policy: it optimizes
  from both an all-zero mean-coefficient start and marginal `glm.nb`
  starts and keeps the better converged objective (the frozen-bounds
  gradient makes the objective start-sensitive and neither start
  dominates).

- control:

  An [`rpbnb_control()`](rpbnb_control.md) object. The famoye and copula
  estimators both use BFGS, the only optimizer `control$method` accepts.
  One control object serves every estimator in the package; settings
  this one does not read – `se_method`, `n_cores`, `halton_burn`,
  `draws_hessian`, and the TMB knobs – are ignored and listed by
  [`print()`](https://rdrr.io/r/base/print.html)/[`summary()`](https://rdrr.io/r/base/summary.html)
  of the fit.

- poisson_1, poisson_2:

  Fit the corresponding margin at its exact Poisson limit (NB2
  dispersion `m = 0`) instead of estimating the dispersion. The margin's
  `log_m` is held fixed, so it is not a free parameter and the fit is a
  properly nested restriction of the NB model – pair it with
  [`lr_test()`](lr_test.md) (`boundary = TRUE`) to test for
  overdispersion. This is the exact `m = 0` restriction at any fitted
  mean: the margin's log-pmf is `dpois` and its Famoye dependence
  constant is `exp(-d*mu)` (the `m -> 0` limit), not an NB2 at a tiny
  pinned dispersion. The famoye and independence paths are both exact
  (the independence path fits a Poisson GLM margin). Not supported with
  a [`copula()`](copula.md) dependence.

## Value

An object of class `bnb_fit`.

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
summary(fit)
#> Bivariate NB (famoye) - summary
#> 
#> --- Equation 1: docvis ---
#>    Parameter Estimate StdErr       z      p Signif
#>  (Intercept)   0.9192 0.0336 27.3602 0.0000    ***
#>      outwork   0.5412 0.0543  9.9682 0.0000    ***
#> 
#> --- Equation 2: hospvis ---
#>    Parameter Estimate StdErr        z      p Signif
#>  (Intercept)  -2.2454 0.0899 -24.9627 0.0000    ***
#>      outwork   0.3069 0.1381   2.2224 0.0263      *
#> 
#> Natural-scale dispersion / dependence (delta-method SE):
#>            Parameter Estimate StdErr    LR df       z      p Signif
#>      m1 (dispersion)   2.3751 0.0725    NA NA      NA     NA       
#>      m2 (dispersion)   9.9295 1.0636    NA NA      NA     NA       
#>  lambda (dependence)   1.6906 0.1208    NA NA 13.9949 0.0000    ***
#> Note: no Wald z/p or boundary LR test for positive scale/dispersion
#>       parameters (SDs, m) above without one; their null is a boundary.
#>       Use rpbnb_boundary_tests() (or rpbnb(boundary_tests = TRUE)) to
#>       test these.
#> 
#> n = 3874   k = 7   logLik = -9660.6183   AIC = 19335.2366   BIC = 19379.0709

# Overdispersion test for margin 1 (H0: m1 = 0, Poisson)
fit_p1 <- fit_bnb(docvis ~ outwork, hospvis ~ outwork, data = d,
                  dependence = "famoye", poisson_1 = TRUE)
#> initial  value 25581.990475 
#> iter   2 value 20616.385772
#> iter   3 value 18884.874824
#> iter   4 value 18642.717397
#> iter   5 value 18217.421957
#> iter   6 value 17933.118740
#> iter   7 value 17654.086815
#> iter   8 value 17582.731279
#> iter   9 value 17444.526478
#> iter  10 value 17344.689555
#> iter  11 value 17298.609706
#> iter  12 value 17280.642784
#> iter  13 value 17277.496870
#> iter  14 value 17277.420780
#> iter  15 value 17277.416915
#> iter  16 value 17277.416623
#> iter  17 value 17277.416339
#> iter  18 value 17277.412561
#> iter  19 value 17277.389987
#> iter  20 value 17277.364231
#> iter  21 value 17277.359697
#> iter  21 value 17277.359597
#> iter  21 value 17277.359597
#> final  value 17277.359597 
#> converged
#> initial  value 17462.029005 
#> iter   2 value 17449.876714
#> iter   3 value 17447.492778
#> iter   4 value 17394.160398
#> iter   5 value 17376.014919
#> iter   6 value 17363.733784
#> iter   7 value 17359.462196
#> iter   8 value 17295.465761
#> iter   9 value 17277.821424
#> iter  10 value 17277.547811
#> iter  11 value 17277.448854
#> iter  12 value 17277.429390
#> iter  13 value 17277.407595
#> iter  14 value 17277.381789
#> iter  15 value 17277.380761
#> iter  16 value 17277.380319
#> iter  17 value 17277.378189
#> iter  18 value 17277.376333
#> iter  19 value 17277.376061
#> iter  20 value 17277.363922
#> iter  21 value 17277.359603
#> iter  21 value 17277.359596
#> iter  21 value 17277.359596
#> final  value 17277.359596 
#> converged
lr_test(fit_p1, fit, boundary = TRUE)
#> Likelihood-ratio test
#>   full model:       logLik = -9660.6183  (df = 7)
#>   restricted model: logLik = -17277.3596  (df = 6)
#>   --------------------------------------------------
#>   LR statistic = 15233.4826  on 1 df   p = 0.0000  ***
#>   (boundary-corrected 50:50 chi-square mixture)

# Gaussian copula dependence instead of Famoye/Sarmanov
fit_cop <- fit_bnb(docvis ~ outwork, hospvis ~ outwork, data = d,
                   dependence = copula("normal"))
#> initial  value 9707.814763 
#> iter   2 value 9676.568833
#> iter   3 value 9675.718342
#> iter   4 value 9675.111657
#> iter   5 value 9674.410132
#> iter   6 value 9671.936448
#> iter   7 value 9669.352861
#> iter   8 value 9662.086943
#> iter   9 value 9638.110220
#> iter  10 value 9633.973407
#> iter  11 value 9633.861811
#> iter  12 value 9633.859385
#> iter  12 value 9633.859346
#> iter  12 value 9633.859346
#> final  value 9633.859346 
#> converged
fit_cop$cop_tau  # estimated Kendall's tau
#> [1] 0.233012
```
