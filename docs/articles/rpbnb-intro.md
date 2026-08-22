# Introduction to rpbnb

``` r

library(rpbnb)
d <- read.csv(system.file("extdata", "rwm1984_clean.csv", package = "rpbnb"))
```

## Bivariate NB

Fit a Famoye/Sarmanov bivariate negative binomial model to doctor and
hospital visits:

``` r

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
```

Goodness of fit:

``` r

g <- bnb_gof(fit, print_output = FALSE)
#> initial  value 16577.658491 
#> iter   2 value 14733.497221
#> iter   3 value 14016.919440
#> iter   4 value 13702.922008
#> iter   5 value 13606.765438
#> iter   6 value 13566.187688
#> iter   7 value 13371.878283
#> iter   8 value 13168.988190
#> iter   9 value 12888.819381
#> iter  10 value 12556.341683
#> iter  11 value 11993.172079
#> iter  12 value 11944.951434
#> iter  13 value 10298.911552
#> iter  14 value 10270.789974
#> iter  15 value 9920.128844
#> iter  16 value 9893.295099
#> iter  17 value 9756.488764
#> iter  17 value 9756.488764
#> final  value 9756.488764 
#> converged
#> initial  value 9764.540322 
#> iter   2 value 9728.967786
#> iter   2 value 9728.967786
#> iter   2 value 9728.967786
#> final  value 9728.967786 
#> converged
#> Warning: Observed information for the famoye BNB is not positive definite (min
#> eigenvalue -30900); a ridge of 30900 was added before inversion. The resulting
#> standard errors are regularized, not observed-information SEs -- inspect
#> fit$hessian_diag.
g$AIC
#> [1] 19335.24
g$pseudoR2
#>     McFadden McFadden_adj     CoxSnell   Nagelkerke 
#>  0.007025358  0.006305857  0.034670956  0.034900839
```

Compare against the independence model (two univariate NB2 margins):

``` r

fit_ind <- fit_bnb(docvis ~ outwork, hospvis ~ outwork, data = d,
                   dependence = "independence")
c(famoye = as.numeric(logLik(fit)), independence = as.numeric(logLik(fit_ind)))
#>       famoye independence 
#>    -9660.618    -9707.815
```

## Copula dependence

As an alternative to Famoye/Sarmanov dependence,
[`fit_bnb()`](../reference/fit_bnb.md) also supports a discrete-copula
bivariate NB via `dependence = copula(family)`, with `family` one of
`"frank"`, `"normal"` (Gaussian), or `"kimeldorf"` (Clayton). The
dependence parameter is estimated from the data rather than fixed.

``` r

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
fit_cop$cop_par   # estimated native parameter (rho, for the Gaussian copula)
#> [1] 0.3578966
fit_cop$cop_tau   # implied Kendall's tau
#> [1] 0.233012
```

[`fit_rpbnb()`](../reference/fit_rpbnb.md) accepts the same
`dependence = copula(...)` argument to combine copula dependence with
random coefficients (see the copula example at the end of the next
section).

## Marginal effects

``` r

bnb_marginal_effects(fit, which = "y1", type = "AME", print_output = FALSE)
```

## Residual diagnostics

Because these are count models, ordinary residuals are not normal even
under a correct fit; use **randomized quantile residuals** (Dunn–Smyth),
which are approximately N(0,1) when the model is right.
[`residuals()`](https://rdrr.io/r/stats/residuals.html),
[`plot()`](https://rdrr.io/r/graphics/plot.default.html), and
[`bnb_residual_checks()`](../reference/bnb_residual_checks.md) are
provided for both `bnb_fit` and `rpbnb_fit`:

``` r

rq <- residuals(fit, type = "quantile")      # randomized quantile residuals
plot(fit, margin = "both")                   # 4 panels per margin
bnb_residual_checks(fit)                      # normality, dispersion, outliers, ...
```

The QQ and histogram panels always use the RQR; the scale-location and
residuals-vs-fitted panels can use Pearson or (for `bnb_fit`) deviance
residuals via `resid_type=` – `rpbnb_fit` does not support
`resid_type = "deviance"`. For `rpbnb_fit` the residuals are built on
the mixture predictive distribution over the random-coefficient draws
(consistent with [`predict()`](https://rdrr.io/r/stats/predict.html)).

## Random parameters

Let the coefficient on `kids` vary across individuals in the
doctor-visits equation. For a fast vignette we use a small subsample and
a modest number of simulation draws; in practice use the full data with
more draws (e.g. `draws = 1000`) and `compute_se = TRUE`.

``` r

set.seed(1)
d_small <- d[sample(nrow(d), 600), ]
rp <- fit_rpbnb(docvis ~ outwork + kids, hospvis ~ outwork, data = d_small,
                random_1 = "kids", draws = 100, seed = 1,
                control = rpbnb_control(compute_se = FALSE))
#> initial  value 2531.234135 
#> iter   2 value 1987.035624
#> iter   3 value 1975.786641
#> iter   4 value 1946.770081
#> iter   5 value 1922.218782
#> iter   6 value 1887.318176
#> iter   7 value 1882.602996
#> iter   8 value 1846.501645
#> iter   9 value 1842.397385
#> iter  10 value 1812.496828
#> iter  11 value 1757.975445
#> iter  12 value 1665.440493
#> iter  13 value 1625.413067
#> iter  14 value 1607.359047
#> iter  15 value 1526.741489
#> iter  16 value 1491.807214
#> iter  17 value 1478.183001
#> iter  18 value 1466.001798
#> iter  19 value 1450.887136
#> iter  20 value 1449.787564
#> iter  21 value 1447.961141
#> iter  22 value 1447.259269
#> iter  23 value 1446.738334
#> iter  24 value 1444.883883
#> iter  25 value 1443.150578
#> iter  26 value 1442.603624
#> iter  27 value 1442.409881
#> iter  28 value 1442.240825
#> iter  29 value 1442.113512
#> iter  30 value 1442.072796
#> iter  31 value 1442.065296
#> iter  32 value 1442.032413
#> iter  33 value 1442.021477
#> iter  34 value 1442.014125
#> iter  35 value 1442.010310
#> iter  36 value 1442.008436
#> iter  37 value 1442.007642
#> iter  38 value 1442.007225
#> iter  39 value 1442.006947
#> iter  39 value 1442.006947
#> final  value 1442.006947 
#> converged
#> Warning: The fitted Famoye lambda (1.43392) lies outside the admissible
#> interval recomputed at the fitted parameters [-1.03742, 1.03742]. The interval
#> used by the likelihood was frozen at the starting values ([-1.73201, 1.73201]),
#> so the optimizer was free to leave the valid region: the joint pmf is negative
#> somewhere in the count tails and this fit should not be interpreted. Refit from
#> starting values closer to the optimum.
coef(rp)
#> b1:(Intercept)     b1:outwork        b1:kids b2:(Intercept)     b2:outwork 
#>      1.1983321      0.3408809     -0.4984653     -2.5936057      0.5931566 
#>   log_sd1:kids         log_m1         log_m2       z_lambda 
#>     -4.7848101      0.9696582      2.5778295      2.3628078
```

### Interpreting random-parameter fits

For [`fit_rpbnb()`](../reference/fit_rpbnb.md) models the conditional
mean is the population average `E[exp(x'beta)]` integrated over the
random-coefficient distribution, so marginal effects and elasticities
are computed on that integrated mean rather than on `exp(x'beta)`. Use
[`rpbnb_marginal_effects()`](../reference/rpbnb_marginal_effects.md) and
[`rpbnb_elasticities()`](../reference/rpbnb_elasticities.md) (the
random-parameter analogues of
[`bnb_marginal_effects()`](../reference/bnb_marginal_effects.md) /
[`bnb_elasticities()`](../reference/bnb_elasticities.md)):

``` r

rpbnb_marginal_effects(rp, which = "both", type = "AME")
rpbnb_elasticities(rp, which = "both", type = "AME")
```

For a continuous covariate the marginal effect averages the per-draw
realized coefficient times `exp(lp)`; for a 0/1 covariate it is the
integrated difference `E[Y | x = 1] - E[Y | x = 0]`. Standard errors
come from a numeric delta method over each equation’s mean and log-scale
parameters.

[`fit_rpbnb()`](../reference/fit_rpbnb.md) also accepts
`dependence = copula(...)` to pair random coefficients with copula
dependence. Random coefficients are only well identified on
**continuous** regressors under the copula path – a random coefficient
on a 0/1 dummy trades off against the NB dispersion and is weakly
identified at realistic sample sizes, so we put the random coefficient
on the continuous `age` variable instead of a dummy:

``` r

rp_cop <- fit_rpbnb(docvis ~ outwork + age, hospvis ~ outwork, data = d_small,
                    random_1 = "age", dependence = copula("normal"),
                    draws = 100, seed = 1,
                    control = rpbnb_control(compute_se = FALSE))
#> initial  value 2189.526579 
#> iter   2 value 2137.601131
#> iter   3 value 1821.190915
#> iter   4 value 1785.054267
#> iter   5 value 1566.372762
#> iter   6 value 1548.103056
#> iter   7 value 1535.358009
#> iter   8 value 1519.348827
#> iter   9 value 1494.945886
#> iter  10 value 1477.367335
#> iter  11 value 1472.280336
#> iter  12 value 1468.223569
#> iter  13 value 1459.606468
#> iter  14 value 1455.420748
#> iter  15 value 1449.228918
#> iter  16 value 1445.636157
#> iter  17 value 1442.170692
#> iter  18 value 1436.951464
#> iter  19 value 1434.359101
#> iter  20 value 1430.440516
#> iter  21 value 1430.055460
#> iter  22 value 1427.697754
#> iter  23 value 1426.942690
#> iter  24 value 1426.855621
#> iter  25 value 1426.594423
#> iter  26 value 1426.372298
#> iter  27 value 1426.191034
#> iter  28 value 1426.058654
#> iter  29 value 1426.052541
#> iter  30 value 1426.030318
#> iter  31 value 1426.027231
#> iter  32 value 1426.016001
#> iter  33 value 1426.015893
#> iter  33 value 1426.015893
#> iter  33 value 1426.015893
#> final  value 1426.015893 
#> converged
coef(rp_cop)
#> b1:(Intercept)     b1:outwork         b1:age b2:(Intercept)     b2:outwork 
#>    -0.46790857     0.25965926     0.03033793    -2.57797458     0.59294841 
#>    log_sd1:age         log_m1         log_m2        z_theta 
#>    -4.56614275     0.76720525     2.57475911     0.33641336
tanh(coef(rp_cop)[["z_theta"]])  # estimated copula rho
#> [1] 0.3242716
```

## Simulation

Generate data from a known random-parameter process and recover the
parameters:

``` r

sim <- simulate_rpbnb(n = 1000,
  beta1 = c("(Intercept)" = 0.2, x1 = 0.4),
  beta2 = c("(Intercept)" = 0.1, x1 = -0.3),
  random_1 = list(x1 = list(sd = 0.5)),
  dispersion = c(m1 = 0.4, m2 = 0.5), seed = 1)
head(sim$data)
#>   y1 y2         x1
#> 1  4  0 -0.6264538
#> 2  1  0  0.1836433
#> 3  3  1 -0.8356286
#> 4  7  0  1.5952808
#> 5  0  3  0.3295078
#> 6  2  0 -0.8204684
```

[`simulate_rpbnb_copula()`](../reference/simulate_rpbnb_copula.md) does
the same, but from a copula-dependent process (the joint pmf is built
from the two NB2 marginal CDFs and a copula CDF, rather than the
Famoye/Sarmanov tilt):

``` r

sim_cop <- simulate_rpbnb_copula(n = 1000,
  beta1 = c("(Intercept)" = 0.2, x1 = 0.4),
  beta2 = c("(Intercept)" = 0.1, x1 = -0.3),
  random_1 = list(x1 = list(sd = 0.5)),
  dispersion = c(m1 = 0.4, m2 = 0.5),
  copula = copula("normal", par = 0.5), seed = 1)
head(sim_cop$data)
#>   y1 y2         x1
#> 1  0  0 -0.6264538
#> 2  1  0  0.1836433
#> 3  0  0 -0.8356286
#> 4  2  1  1.5952808
#> 5  4  1  0.3295078
#> 6  1  2 -0.8204684
sim_cop$true$tau   # true Kendall's tau implied by rho = 0.5
#> [1] 0.3333333
```
