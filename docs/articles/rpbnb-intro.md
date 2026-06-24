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
summary(fit)
#> Bivariate NB (famoye) - summary
#> 
#> --- Equation 1: docvis ---
#>    Parameter Estimate StdErr       z      p Signif
#>  (Intercept)   0.9194 0.0336 27.3700 0.0000    ***
#>      outwork   0.5401 0.0543  9.9506 0.0000    ***
#> 
#> --- Equation 2: hospvis ---
#>    Parameter Estimate StdErr        z      p Signif
#>  (Intercept)  -2.2442 0.0900 -24.9496 0.0000    ***
#>      outwork   0.3079 0.1381   2.2292 0.0258      *
#> 
#> Natural-scale dispersion / dependence (delta-method SE):
#>            Parameter Estimate StdErr
#>      m1 (dispersion)   2.3738 0.0725
#>      m2 (dispersion)   9.9291 1.0636
#>  lambda (dependence)   1.6816 0.1208
#> 
#> n = 3874   k = 7   logLik = -9660.6217   AIC = 19335.2435   BIC = 19379.0778
```

Goodness of fit:

``` r

g <- bnb_gof(fit, print_output = FALSE)
g$AIC
#> [1] 19335.24
g$pseudoR2
#>     McFadden McFadden_adj     CoxSnell   Nagelkerke 
#>  0.005275862  0.004555093  0.026105666  0.026280303
```

Compare against the independence model (two univariate NB2 margins):

``` r

fit_ind <- fit_bnb(docvis ~ outwork, hospvis ~ outwork, data = d,
                   dependence = "independence")
c(famoye = as.numeric(logLik(fit)), independence = as.numeric(logLik(fit_ind)))
#>       famoye independence 
#>    -9660.622    -9707.815
```

## Marginal effects

``` r

bnb_marginal_effects(fit, which = "y1", type = "AME", print_output = FALSE)
```

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
coef(rp)
#> b1:(Intercept)     b1:outwork        b1:kids b2:(Intercept)     b2:outwork 
#>      1.1982893      0.3409856     -0.4985433     -2.5934669      0.5928088 
#>   log_sd1:kids         log_m1         log_m2       z_lambda 
#>     -6.4316348      0.9696919      2.5778940      4.3090962
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
