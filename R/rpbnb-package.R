#' rpbnb: Random-Parameter Bivariate Negative Binomial Regression
#'
#' Maximum-likelihood estimation of bivariate negative binomial models with
#' Famoye/Sarmanov or discrete-copula (Frank, Gaussian, Clayton) dependence
#' (see [fit_bnb()], [copula()]), and maximum-simulated-likelihood estimation
#' of a bivariate random-parameter negative binomial model under either
#' dependence structure (see [fit_rpbnb()]).
#'
#' @keywords internal
#' @useDynLib rpbnb, .registration = TRUE
#' @importFrom Rcpp sourceCpp
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
#' @importFrom compiler cmpfun
NULL
