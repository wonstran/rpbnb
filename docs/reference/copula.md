# Specify a copula dependence structure

Pass the result as the `dependence` argument to
[`fit_bnb()`](fit_bnb.md) or [`fit_rpbnb()`](fit_rpbnb.md) to estimate a
discrete-copula bivariate NB model instead of the default
Famoye/Sarmanov dependence, or as the `copula` argument to
[`simulate_rpbnb_copula()`](simulate_rpbnb_copula.md) to simulate from
one. The joint pmf is built from the two NB2 marginal CDFs and the
chosen copula CDF (rectangle differencing); see the package vignette for
the "Copula dependence" section.

## Usage

``` r
copula(family = c("frank", "normal", "kimeldorf"), par = NULL)
```

## Arguments

- family:

  One of `"frank"`, `"normal"` (Gaussian), or `"kimeldorf"` (Clayton).

- par:

  Optional native dependence parameter (Frank's theta, the Gaussian rho,
  or Clayton's theta) used by
  [`simulate_rpbnb_copula()`](simulate_rpbnb_copula.md) to generate
  data. Ignored by [`fit_bnb()`](fit_bnb.md) and
  [`fit_rpbnb()`](fit_rpbnb.md), which estimate the parameter from the
  data.

## Value

An object of class `rpbnb_copula`.

## Examples

``` r
copula("frank")
#> $family
#> [1] "frank"
#> 
#> $par
#> NULL
#> 
#> attr(,"class")
#> [1] "rpbnb_copula"
copula("normal", par = 0.3)
#> $family
#> [1] "normal"
#> 
#> $par
#> [1] 0.3
#> 
#> attr(,"class")
#> [1] "rpbnb_copula"
copula("kimeldorf")
#> $family
#> [1] "kimeldorf"
#> 
#> $par
#> NULL
#> 
#> attr(,"class")
#> [1] "rpbnb_copula"
```
