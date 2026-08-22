# rpbnb: Random-Parameter Bivariate Negative Binomial Regression

Maximum-likelihood estimation of bivariate negative binomial models with
Famoye/Sarmanov or discrete-copula (Frank, Gaussian, Clayton) dependence
(see [`fit_bnb()`](fit_bnb.md), [`copula()`](copula.md)), and
maximum-simulated-likelihood estimation of a bivariate random-parameter
negative binomial model under either dependence structure.

## Details

Two estimation engines are available for the random-parameter model:
[`fit_rpbnb()`](fit_rpbnb.md) (Rcpp/OpenMP simulated likelihood,
`maxLik` BFGS, supports
[`offset()`](https://rdrr.io/r/stats/offset.html)) and
[`fit_rpbnb_tmb()`](fit_rpbnb_tmb.md) (TMB automatic differentiation,
`nlminb`, adds a Laplace approximation and dependence profiling).
[`rpbnb()`](rpbnb.md) is a common front end that dispatches to either.

## Author

**Maintainer**: Zhenyu Wang <wonstran@hotmail.com>

Authors:

- Zhenyu Wang <wonstran@hotmail.com>
