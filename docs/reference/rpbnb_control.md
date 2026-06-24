# Control parameters for rpbnb estimators

Control parameters for rpbnb estimators

## Usage

``` r
rpbnb_control(
  method = c("BFGS", "NR", "BHHH", "NM"),
  iterlim = 300L,
  reltol = 1e-08,
  print_level = 0L,
  draws_hessian = 100L,
  halton_burn = 300L,
  n_cores = 1L,
  compute_se = TRUE,
  hessian = c("numeric", "analytic"),
  hess_eps = 1e-05,
  hess_r = 4L
)
```

## Arguments

- method:

  Optimizer passed to
  [`maxLik::maxLik()`](https://rdrr.io/pkg/maxLik/man/maxLik.html). One
  of "BFGS", "NR", "BHHH", "NM".

- iterlim:

  Maximum optimizer iterations.

- reltol:

  Relative convergence tolerance.

- print_level:

  Verbosity passed to the optimizer (0 = silent).

- draws_hessian:

  Number of simulation draws used for the random-parameter Hessian
  (smaller than the optimization draws for speed). Ignored by
  [`fit_bnb()`](fit_bnb.md).

- halton_burn:

  Number of leading Halton points discarded before forming the
  simulation draws.

- n_cores:

  Worker processes for the optional cluster path (1 = sequential).

- compute_se:

  If FALSE, skip the Hessian and standard errors.

- hessian:

  How [`fit_bnb()`](fit_bnb.md) (famoye) computes the Hessian for
  standard errors: "numeric" (default,
  [`numDeriv::hessian()`](https://rdrr.io/pkg/numDeriv/man/hessian.html))
  or "analytic" (the closed-form Famoye (2010) Appendix Hessian). Both
  freeze the lambda-bounds at the optimum and yield the same
  observed-information SEs.

- hess_eps, hess_r:

  Step and Richardson order for
  [`numDeriv::hessian()`](https://rdrr.io/pkg/numDeriv/man/hessian.html)
  (used only when `hessian = "numeric"`).

## Value

An object of class `rpbnb_control` (a named list).

## Examples

``` r
rpbnb_control(method = "BFGS", iterlim = 200)
#> $method
#> [1] "BFGS"
#> 
#> $iterlim
#> [1] 200
#> 
#> $reltol
#> [1] 1e-08
#> 
#> $print_level
#> [1] 0
#> 
#> $draws_hessian
#> [1] 100
#> 
#> $halton_burn
#> [1] 300
#> 
#> $n_cores
#> [1] 1
#> 
#> $compute_se
#> [1] TRUE
#> 
#> $hessian
#> [1] "numeric"
#> 
#> $hess_eps
#> [1] 1e-05
#> 
#> $hess_r
#> [1] 4
#> 
#> attr(,"class")
#> [1] "rpbnb_control"
rpbnb_control(hessian = "analytic")
#> $method
#> [1] "BFGS"
#> 
#> $iterlim
#> [1] 300
#> 
#> $reltol
#> [1] 1e-08
#> 
#> $print_level
#> [1] 0
#> 
#> $draws_hessian
#> [1] 100
#> 
#> $halton_burn
#> [1] 300
#> 
#> $n_cores
#> [1] 1
#> 
#> $compute_se
#> [1] TRUE
#> 
#> $hessian
#> [1] "analytic"
#> 
#> $hess_eps
#> [1] 1e-05
#> 
#> $hess_r
#> [1] 4
#> 
#> attr(,"class")
#> [1] "rpbnb_control"
```
