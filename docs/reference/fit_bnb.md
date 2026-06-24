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
  control = rpbnb_control()
)
```

## Arguments

- formula_1, formula_2:

  Model formulas for the two count outcomes.

- data:

  A data frame.

- dependence:

  Dependence structure: "independence" (two univariate NB2 margins) or
  "famoye" (Famoye/Sarmanov bivariate NB).

- start:

  Optional starting parameter vector.

- control:

  An [`rpbnb_control()`](rpbnb_control.md) object. The famoye estimator
  uses BFGS; `control$method` is currently honored only by the
  optimizer's internal setup.

## Value

An object of class `bnb_fit`.

## Examples

``` r
d <- read.csv(system.file("extdata", "rwm1984_clean.csv", package = "rpbnb"))
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
