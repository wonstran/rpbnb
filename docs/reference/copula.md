# Specify a copula dependence structure for fit_bnb()

Specify a copula dependence structure for fit_bnb()

## Usage

``` r
copula(family = c("frank", "normal", "kimeldorf"))
```

## Arguments

- family:

  One of `"frank"`, `"normal"`, or `"kimeldorf"` (Clayton).

## Value

An object of class `rpbnb_copula`.

## Examples

``` r
copula("frank")
#> $family
#> [1] "frank"
#> 
#> attr(,"class")
#> [1] "rpbnb_copula"
copula("normal")
#> $family
#> [1] "normal"
#> 
#> attr(,"class")
#> [1] "rpbnb_copula"
copula("kimeldorf")
#> $family
#> [1] "kimeldorf"
#> 
#> attr(,"class")
#> [1] "rpbnb_copula"
```
