#' rpbnb: Random-Parameter Bivariate Negative Binomial Regression
#'
#' Maximum-likelihood estimation of bivariate negative binomial models with
#' Famoye/Sarmanov or discrete-copula (Frank, Gaussian, Clayton) dependence
#' (see [fit_bnb()], [copula()]), and maximum-simulated-likelihood estimation
#' of a bivariate random-parameter negative binomial model under either
#' dependence structure.
#'
#' Two estimation engines are available for the random-parameter model:
#' [fit_rpbnb()] (Rcpp/OpenMP simulated likelihood, `maxLik` BFGS, supports
#' `offset()`) and [fit_rpbnb_tmb()] (TMB automatic differentiation, `nlminb`,
#' adds a Laplace approximation and dependence profiling). [rpbnb()] is a common
#' front end that dispatches to either.
#'
#' @keywords internal
#' @aliases rpbnb-package
#' @useDynLib rpbnb, .registration = TRUE
#' @importFrom Rcpp sourceCpp
#' @importFrom TMB MakeADFun
#' @importFrom numDeriv jacobian
"_PACKAGE"

#' @importFrom stats model.frame
#' @importFrom stats model.matrix
#' @importFrom stats model.response
#' @importFrom stats pnorm
#' @importFrom stats plogis
#' @importFrom stats qnorm
#' @importFrom stats symnum
#' @importFrom stats coef
#' @importFrom stats logLik
#' @importFrom stats AIC
#' @importFrom stats BIC
#' @importFrom stats predict
#' @importFrom stats residuals
#' @importFrom stats vcov
#' @importFrom stats dnbinom
#' @importFrom stats dnorm
#' @importFrom stats pnbinom
#' @importFrom stats qnbinom
#' @importFrom stats rnbinom
#' @importFrom stats rnorm
#' @importFrom stats runif
#' @importFrom stats rbinom
#' @importFrom stats var
#' @importFrom stats sd
#' @importFrom stats setNames
#' @importFrom stats confint
#' @importFrom stats nlminb
#' @importFrom compiler cmpfun
NULL
