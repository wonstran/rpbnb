# Public fit_bnb() plus the bnb_fit constructor and the two internal fitters
# (Famoye/Sarmanov bivariate NB and independence = two univariate NB2 margins).

#' Construct a bnb_fit object
#' @keywords internal
#' @noRd
new_bnb_fit <- function(coef, vcov, se, logLik, nobs, npar, dependence,
                        lambda, bounds, mu1, mu2, X1, X2, Y1, Y2,
                        formula_1, formula_2, ll_trace, convergence, call) {
  structure(
    list(coef = coef, vcov = vcov, se = se, logLik = logLik,
         nobs = nobs, npar = npar, dependence = dependence,
         lambda = lambda, bounds = bounds, mu1 = mu1, mu2 = mu2,
         X1 = X1, X2 = X2, Y1 = Y1, Y2 = Y2,
         formula_1 = formula_1, formula_2 = formula_2,
         ll_trace = ll_trace, convergence = convergence,
         AIC = -2 * logLik + 2 * npar, BIC = -2 * logLik + log(nobs) * npar,
         call = call),
    class = "bnb_fit"
  )
}

#' Internal estimator: Famoye/Sarmanov BNB via maxLik BFGS + analytic gradient
#'
#' Receives pre-built response vectors and design matrices. Returns a plain
#' list with fields consumed by [fit_bnb()].
#' @keywords internal
#' @noRd
fit_bnb_famoye <- function(Y1, Y2, X1, X2, cn1, cn2, start, control) {
  p1 <- NCOL(X1); p2 <- NCOL(X2)

  if (is.null(start)) start <- c(rep(0, p1 + p2), log(0.5), log(0.5), 0)
  names(start) <- c(paste0("b1:", cn1), paste0("b2:", cn2),
                    "log_m1", "log_m2", "z_lambda")

  # --- capture logLik at every evaluation ---
  .ll_eval <- numeric(0)
  ll_fun <- function(p) {
    v <- bnb_loglik_vec(p, Y1, Y2, X1, X2)   # per-obs vector
    s <- sum(v)
    .ll_eval <<- c(.ll_eval, s)              # record total logLik
    v
  }
  grad_fun <- function(p) bnb_grad_vec(p, Y1, Y2, X1, X2)

  ml_control <- list(iterlim = control$iterlim,
                     reltol = control$reltol,
                     printLevel = control$print_level)

  fit <- maxLik::maxLik(
    logLik  = ll_fun,
    grad    = grad_fun,      # analytic gradient (frozen-bounds objective)
    start   = start,
    method  = "BFGS",
    control = ml_control
  )
  par_hat <- stats::coef(fit)

  # Back-transform and lambda at fitted bounds
  beta1_hat <- par_hat[paste0("b1:", cn1)]
  beta2_hat <- par_hat[paste0("b2:", cn2)]
  m1_hat <- unname(exp(par_hat["log_m1"]))
  m2_hat <- unname(exp(par_hat["log_m2"]))
  z_hat  <- unname(par_hat["z_lambda"])

  mu1_hat <- as.vector(exp(X1 %*% beta1_hat))
  mu2_hat <- as.vector(exp(X2 %*% beta2_hat))
  c1_hat  <- c_val(mu1_hat, m1_hat)
  c2_hat  <- c_val(mu2_hat, m2_hat)
  bnds_hat <- lambda_bounds_vec(c1_hat, c2_hat)
  eps <- 1e-6
  lambda_hat <- unname(bnds_hat[1] + (bnds_hat[2] - bnds_hat[1]) *
                         (eps + (1 - 2*eps) * plogis(z_hat)))

  ll_hat <- as.numeric(stats::logLik(fit))

  # Standard errors via numeric Hessian with lambda-bounds frozen at optimum
  if (isTRUE(control$compute_se)) {
    lamLo_h <- bnds_hat[1]; lamHi_h <- bnds_hat[2]
    ll_fb <- function(p) bnbr_loglik_fixed_bounds(p, Y1, Y2, X1, X2, lamLo_h, lamHi_h)

    H <- numDeriv::hessian(ll_fb, par_hat,
                           method.args = list(r = control$hess_r, eps = control$hess_eps))
    info <- -H; info <- (info + t(info)) / 2
    if (any(!is.finite(info))) {
      warning("Non-finite information; retrying Hessian with larger step.")
      H <- numDeriv::hessian(ll_fb, par_hat,
                             method.args = list(r = max(6, control$hess_r + 2),
                                                eps = max(1e-4, 5 * control$hess_eps)))
      info <- -H; info <- (info + t(info)) / 2
    }
    ok_eig <- try(eigen(info, symmetric = TRUE, only.values = TRUE), silent = TRUE)
    if (inherits(ok_eig, "try-error") || any(!is.finite(ok_eig$values)) ||
        min(ok_eig$values) <= 0) {
      ridge <- if (inherits(ok_eig, "try-error") || any(!is.finite(ok_eig$values))) {
        1e-2
      } else {
        1e-8 - min(ok_eig$values)
      }
      info <- info + diag(ridge, nrow(info))
    }
    vc <- try(solve(info), silent = TRUE)
    if (inherits(vc, "try-error")) vc <- MASS::ginv(info)
    se <- sqrt(pmax(diag(vc), 0))
  } else {
    vc <- matrix(NA_real_, length(par_hat), length(par_hat))
    se <- rep(NA_real_, length(par_hat))
  }
  dimnames(vc) <- list(names(par_hat), names(par_hat))
  names(se) <- names(par_hat)

  convergence <- list(converged = fit$code <= 2L, code = fit$code,
                      message = fit$message, iterations = fit$iterations)

  list(coef = par_hat, vcov = vc, se = se, logLik = ll_hat,
       npar = length(par_hat),
       lambda = lambda_hat, bounds = c(bnds_hat[1], bnds_hat[2]),
       mu1 = mu1_hat, mu2 = mu2_hat,
       ll_trace = .ll_eval, convergence = convergence)
}

#' Internal estimator: independence = two univariate NB2 margins via MASS::glm.nb
#' @keywords internal
#' @noRd
fit_bnb_independence <- function(formula_1, formula_2, data, cn1, cn2,
                                 X1, X2, Y1, Y2) {
  g1 <- tryCatch(MASS::glm.nb(formula_1, data = data),
                 error = function(e) stop("Margin 1 (", deparse(formula_1),
                   ") glm.nb failed: ", conditionMessage(e), call. = FALSE))
  g2 <- tryCatch(MASS::glm.nb(formula_2, data = data),
                 error = function(e) stop("Margin 2 (", deparse(formula_2),
                   ") glm.nb failed: ", conditionMessage(e), call. = FALSE))

  p1 <- length(stats::coef(g1)); p2 <- length(stats::coef(g2))

  log_m1 <- log(1 / g1$theta)
  log_m2 <- log(1 / g2$theta)

  # Independence has no dependence parameter: the coefficient vector holds only
  # the two betas and the two dispersions (no fabricated z_lambda), so
  # length(coef) == npar. lambda = 0 is recorded separately on the fit object.
  par <- c(stats::coef(g1), stats::coef(g2), log_m1, log_m2)
  names(par) <- c(paste0("b1:", cn1), paste0("b2:", cn2),
                  "log_m1", "log_m2")

  npar_total <- length(par)            # p1 + p2 + 2, all free
  vc <- matrix(0, npar_total, npar_total,
               dimnames = list(names(par), names(par)))
  vc[1:p1, 1:p1] <- stats::vcov(g1)
  vc[p1 + (1:p2), p1 + (1:p2)] <- stats::vcov(g2)
  # log_m = log(1/theta); delta-method var(log_m) = (SE.theta / theta)^2
  vc[p1 + p2 + 1, p1 + p2 + 1] <- (g1$SE.theta / g1$theta)^2
  vc[p1 + p2 + 2, p1 + p2 + 2] <- (g2$SE.theta / g2$theta)^2

  se <- sqrt(pmax(diag(vc), 0))
  names(se) <- names(par)

  ll <- as.numeric(stats::logLik(g1)) + as.numeric(stats::logLik(g2))

  convergence <- list(converged = isTRUE(g1$converged) && isTRUE(g2$converged),
                      code = NA_integer_, message = "glm.nb",
                      iterations = NA_integer_)

  list(coef = par, vcov = vc, se = se, logLik = ll,
       npar = npar_total,               # all parameters free (no lambda)
       lambda = 0, bounds = c(NA_real_, NA_real_),
       mu1 = unname(stats::fitted(g1)), mu2 = unname(stats::fitted(g2)),
       ll_trace = numeric(0), convergence = convergence)
}

#' Fit a bivariate negative binomial regression model
#'
#' @param formula_1,formula_2 Model formulas for the two count outcomes.
#' @param data A data frame.
#' @param dependence Dependence structure: "independence" (two univariate NB2
#'   margins) or "famoye" (Famoye/Sarmanov bivariate NB).
#' @param start Optional starting parameter vector.
#' @param control An [rpbnb_control()] object. The famoye estimator uses BFGS;
#'   `control$method` is currently honored only by the optimizer's internal setup.
#' @return An object of class `bnb_fit`.
#' @export
#' @examples
#' d <- read.csv(system.file("extdata", "rwm1984_clean.csv", package = "rpbnb"))
#' fit <- fit_bnb(docvis ~ outwork, hospvis ~ outwork, data = d,
#'                dependence = "famoye")
#' summary(fit)
fit_bnb <- function(formula_1, formula_2, data,
                    dependence = c("independence", "famoye"),
                    start = NULL, control = rpbnb_control()) {
  dependence <- match.arg(dependence)

  prep <- .prepare_bnb_data(formula_1, formula_2, data)
  Y1 <- prep$Y1; Y2 <- prep$Y2
  X1 <- prep$X1; X2 <- prep$X2
  cn1 <- prep$cn1; cn2 <- prep$cn2

  res <- if (dependence == "famoye") {
    fit_bnb_famoye(Y1, Y2, X1, X2, cn1, cn2, start, control)
  } else {
    fit_bnb_independence(formula_1, formula_2, prep$data, cn1, cn2, X1, X2, Y1, Y2)
  }

  new_bnb_fit(coef = res$coef, vcov = res$vcov, se = res$se,
              logLik = res$logLik, nobs = length(Y1), npar = res$npar,
              dependence = dependence, lambda = res$lambda, bounds = res$bounds,
              mu1 = res$mu1, mu2 = res$mu2, X1 = X1, X2 = X2, Y1 = Y1, Y2 = Y2,
              formula_1 = formula_1, formula_2 = formula_2,
              ll_trace = res$ll_trace, convergence = res$convergence,
              call = match.call())
}
