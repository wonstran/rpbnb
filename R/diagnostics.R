# Diagnostics for bnb_fit objects: goodness of fit, marginal effects, and
# elasticities. Ported from the legacy bnbr_gof / bnbr_me / bnbr_elasticities,
# adapted to the new bnb_fit object structure (flat coef vector, stored vcov).

# --- internal: extract adapted pieces from a bnb_fit ------------------------
.bnb_diag_parts <- function(fit) {
  p1 <- ncol(fit$X1); p2 <- ncol(fit$X2)
  beta1 <- fit$coef[grep("^b1:", names(fit$coef))]
  names(beta1) <- sub("^b1:", "", names(beta1))
  beta2 <- fit$coef[grep("^b2:", names(fit$coef))]
  names(beta2) <- sub("^b2:", "", names(beta2))
  Vb1 <- fit$vcov[1:p1, 1:p1, drop = FALSE]
  Vb2 <- fit$vcov[(p1 + 1):(p1 + p2), (p1 + 1):(p1 + p2), drop = FALSE]
  list(p1 = p1, p2 = p2, beta1 = beta1, beta2 = beta2, Vb1 = Vb1, Vb2 = Vb2)
}

# --- internal: tidy a marginal-effect / elasticity result ------------------
.bnb_me_tidy <- function(names, est, se, digits = 4) {
  z <- est / se
  p <- 2 * stats::pnorm(-abs(z))
  data.frame(Name = names, Estimate = est, StdErr = se, z = z, p = p,
             Signif = signif_stars(p), check.names = FALSE)
}

# --- internal: print a tidy table with fixed decimals ----------------------
.bnb_me_print <- function(df, digits = 4) {
  num <- vapply(df, is.numeric, logical(1))
  df[num] <- lapply(df[num], formatC, format = "f", digits = digits)
  print(df, row.names = FALSE, right = TRUE)
}

#' Goodness of fit for a bivariate NB model
#'
#' Reports log-likelihood, AIC, BIC, and four pseudo-R-squared measures
#' (McFadden, McFadden adjusted, Cox-Snell, Nagelkerke) relative to an
#' intercept-only null model refit with the same dependence structure.
#'
#' @param fit A `bnb_fit` object from [fit_bnb()].
#' @param digits Number of decimal places for printed output.
#' @param print_output Logical; if `FALSE`, suppress printing and return the
#'   result invisibly.
#' @return Invisibly, a list with `n`, `k`, `logLik_full`, `logLik_null`,
#'   `AIC`, `BIC`, a named numeric vector `pseudoR2`, and the `null_fit`.
#' @export
#' @examples
#' d <- read.csv(system.file("extdata", "rwm1984_clean.csv", package = "rpbnb"))
#' fit <- fit_bnb(docvis ~ outwork + age, hospvis ~ outwork, data = d,
#'                dependence = "famoye")
#' bnb_gof(fit)
bnb_gof <- function(fit, digits = 4, print_output = TRUE) {
  stopifnot(inherits(fit, "bnb_fit"))

  # Full model metrics (use stored values)
  ll_full <- as.numeric(fit$logLik)
  k       <- fit$npar
  n       <- fit$nobs
  AIC_val <- fit$AIC
  BIC_val <- fit$BIC

  # ---------- Null model (intercept-only), same dependence structure --------
  df_null  <- data.frame(Y1 = fit$Y1, Y2 = fit$Y2)
  fit_null <- fit_bnb(Y1 ~ 1, Y2 ~ 1, data = df_null,
                      dependence = fit$dependence)
  ll_null  <- as.numeric(fit_null$logLik)

  # ---------- Pseudo R^2 ----------
  R2_MF  <- 1 - (ll_full / ll_null)
  R2_MFa <- 1 - ((ll_full - k) / ll_null)
  R2_CS  <- 1 - exp((2 / n) * (ll_null - ll_full))
  denom  <- 1 - exp((2 / n) * ll_null)
  R2_NK  <- if (abs(denom) < .Machine$double.eps) NA_real_ else R2_CS / denom

  clamp01 <- function(x) ifelse(is.na(x), NA_real_, pmin(pmax(x, 0), 1))
  R2_MF  <- clamp01(R2_MF)
  R2_MFa <- clamp01(R2_MFa)
  R2_CS  <- clamp01(R2_CS)
  R2_NK  <- clamp01(R2_NK)

  pseudoR2 <- c(McFadden = R2_MF, McFadden_adj = R2_MFa,
                CoxSnell = R2_CS, Nagelkerke = R2_NK)

  if (print_output) {
    cat("\n--- Goodness of Fit ---\n")
    cat(sprintf("n = %d, k = %d\n", n, k))
    cat(sprintf("logLik(full) = %s\n",
                formatC(ll_full, format = "f", digits = digits)))
    cat(sprintf("AIC = %s    BIC = %s\n",
                formatC(AIC_val, format = "f", digits = digits),
                formatC(BIC_val, format = "f", digits = digits)))
    cat("\nPseudo R-squared:\n")
    gof_tab <- data.frame(
      Metric = names(pseudoR2),
      Value  = formatC(pseudoR2, format = "f", digits = digits),
      check.names = FALSE
    )
    print(gof_tab, row.names = FALSE, right = TRUE)
  }

  invisible(list(
    n = n, k = k,
    logLik_full = ll_full,
    logLik_null = ll_null,
    AIC = AIC_val, BIC = BIC_val,
    pseudoR2 = pseudoR2,
    null_fit = fit_null
  ))
}

#' Marginal effects for a bivariate NB model
#'
#' Average marginal effects (AME) or marginal effects at the mean (MEM) for
#' each margin. Continuous and binary (0/1) regressors are auto-detected;
#' continuous effects use \eqn{\partial E[Y]/\partial x_j = \beta_j \mu} and
#' binary effects use \eqn{E[Y|x_j=1] - E[Y|x_j=0]}.
#'
#' @param fit A `bnb_fit` object from [fit_bnb()].
#' @param which Which margin(s): "y1", "y2", "both", or "all".
#' @param type "AME" (average marginal effect) or "MEM" (effect at the mean).
#' @param vars Optional variable names or indices to restrict output.
#' @param include_intercept Logical; include the intercept term.
#' @param digits Number of decimal places for printed output.
#' @param print_output Logical; if `FALSE`, suppress printing.
#' @return A data frame (single margin) or a named list of data frames (both),
#'   each with columns `Name`, `Estimate`, `StdErr`, `z`, `p`, `Signif`,
#'   `var_type`.
#' @export
#' @examples
#' d <- read.csv(system.file("extdata", "rwm1984_clean.csv", package = "rpbnb"))
#' fit <- fit_bnb(docvis ~ outwork + age, hospvis ~ outwork, data = d,
#'                dependence = "famoye")
#' bnb_marginal_effects(fit, which = "y1", type = "AME")
bnb_marginal_effects <- function(fit,
                                 which = c("y1", "y2", "both", "all"),
                                 type  = c("AME", "MEM"),
                                 vars  = NULL,
                                 include_intercept = FALSE,
                                 digits = 4,
                                 print_output = TRUE) {
  stopifnot(inherits(fit, "bnb_fit"))
  type  <- match.arg(type, c("AME", "MEM"))
  which <- match.arg(which, c("y1", "y2", "both", "all"))
  parts <- .bnb_diag_parts(fit)

  # worker for ONE response -----------------------------------------
  .one_resp <- function(X, beta, Vb, resp_name) {
    cn <- colnames(X)
    p  <- ncol(X)

    # ---- select variables first ----
    if (is.null(vars)) {
      idx <- seq_len(p)
    } else if (is.numeric(vars)) {
      if (any(vars < 1 | vars > p)) stop("Index out of range in `vars`.")
      idx <- vars
    } else {  # character
      idx <- match(vars, cn)
      if (anyNA(idx)) {
        stop("Unknown vars in ", resp_name, ": ",
             paste(vars[is.na(idx)], collapse = ", "),
             ". Available: ", paste(cn, collapse = ", "))
      }
    }

    # drop intercept if asked
    if (!include_intercept) {
      idx <- idx[cn[idx] != "(Intercept)"]
    }
    if (!length(idx)) stop("No variables selected after filtering the intercept.")

    # ---- now auto-judge *within* the selected vars ----
    is_bin <- vapply(
      idx,
      function(j) {
        x <- X[, j]; x <- x[!is.na(x)]
        length(x) > 0 && all(x %in% c(0, 1))
      },
      logical(1)
    )

    pos_bin  <- idx[is_bin]
    pos_cont <- idx[!is_bin]

    mu <- as.vector(exp(X %*% beta))
    n  <- NROW(X)
    out_list <- list()

    # ========== continuous part ==========
    if (length(pos_cont)) {
      est_c <- se_c <- numeric(length(pos_cont))
      nm_c  <- cn[pos_cont]

      if (type == "AME") {
        mu_bar  <- mean(mu)
        Xmubar  <- as.numeric(t(X) %*% (mu / n))   # E[mu * x_k]
        for (h in seq_along(pos_cont)) {
          j <- pos_cont[h]
          est_c[h] <- mean(beta[j] * mu)
          g <- beta[j] * Xmubar
          g[j] <- g[j] + mu_bar
          se_c[h] <- sqrt(as.numeric(t(g) %*% Vb %*% g))
        }
      } else {  # MEM
        Xbar   <- colMeans(X)
        mu_bar <- exp(sum(Xbar * beta))
        for (h in seq_along(pos_cont)) {
          j <- pos_cont[h]
          est_c[h] <- beta[j] * mu_bar
          g <- beta[j] * (mu_bar * Xbar)
          g[j] <- g[j] + mu_bar
          se_c[h] <- sqrt(as.numeric(t(g) %*% Vb %*% g))
        }
      }

      tab_c <- .bnb_me_tidy(nm_c, est_c, se_c, digits)
      tab_c$var_type <- "continuous"
      out_list[[length(out_list) + 1]] <- tab_c
    }

    # ========== binary 0->1 part ==========
    if (length(pos_bin)) {
      est_b <- se_b <- numeric(length(pos_bin))
      nm_b  <- cn[pos_bin]

      for (k in seq_along(pos_bin)) {
        j <- pos_bin[k]

        if (type == "AME") {
          X0 <- X; X0[, j] <- 0
          X1 <- X; X1[, j] <- 1

          fval <- function(b) mean(exp(X1 %*% b) - exp(X0 %*% b))
          est_b[k] <- fval(beta)

          g <- as.numeric(numDeriv::jacobian(fval, beta))
          se_b[k] <- sqrt(as.numeric(t(g) %*% Vb %*% g))
        } else {  # MEM
          Xbar <- matrix(colMeans(X), nrow = 1)
          X0   <- Xbar; X0[, j] <- 0
          X1   <- Xbar; X1[, j] <- 1

          mu0  <- as.numeric(exp(X0 %*% beta))
          mu1  <- as.numeric(exp(X1 %*% beta))

          est_b[k] <- mu1 - mu0

          g <- as.numeric(mu1 * X1 - mu0 * X0)
          se_b[k] <- sqrt(as.numeric(t(g) %*% Vb %*% g))
        }
      }

      tab_b <- .bnb_me_tidy(nm_b, est_b, se_b, digits)
      tab_b$var_type <- "binary(0->1)"
      out_list[[length(out_list) + 1]] <- tab_b
    }

    out <- do.call(rbind, out_list)
    rownames(out) <- NULL

    if (print_output) {
      cat(sprintf("\n--- Marginal effects (auto) for %s (%s) ---\n",
                  resp_name, type))
      .bnb_me_print(out, digits = digits)
      cat("continuous: dE[Y]/dx_j = beta_j * mu\n")
      cat("binary: E[Y|x_j=1] - E[Y|x_j=0]\n")
    }

    out
  }

  # y1 + y2
  if (which %in% c("both", "all")) {
    res <- list()
    res$y1 <- .one_resp(fit$X1, parts$beta1, parts$Vb1, "y1")
    res$y2 <- .one_resp(fit$X2, parts$beta2, parts$Vb2, "y2")
    return(invisible(res))
  }

  # single response
  if (which == "y1") {
    out <- .one_resp(fit$X1, parts$beta1, parts$Vb1, "y1")
  } else {
    out <- .one_resp(fit$X2, parts$beta2, parts$Vb2, "y2")
  }
  invisible(out)
}

#' Elasticities and semi-elasticities for a bivariate NB model
#'
#' For continuous regressors, the elasticity is \eqn{\beta_j E[x_j]}; for
#' binary (0/1) regressors, the semi-elasticity is \eqn{\exp(\beta_j) - 1}
#' (proportional change moving from 0 to 1).
#'
#' @param fit A `bnb_fit` object from [fit_bnb()].
#' @param which Which margin(s): "y1", "y2", or "both".
#' @param type "AME" (uses sample mean of x) or "MEM" (uses mean design row).
#' @param vars Optional variable names to restrict output.
#' @param include_intercept Logical; include the intercept term.
#' @param digits Number of decimal places for printed output.
#' @param print_output Logical; if `FALSE`, suppress printing.
#' @return A data frame (single margin) or a named list of data frames (both),
#'   each with columns `Name`, `Estimate`, `StdErr`, `z`, `p`, `Signif`,
#'   `var_type`.
#' @export
#' @examples
#' d <- read.csv(system.file("extdata", "rwm1984_clean.csv", package = "rpbnb"))
#' fit <- fit_bnb(docvis ~ outwork + age, hospvis ~ outwork, data = d,
#'                dependence = "famoye")
#' bnb_elasticities(fit, which = "both", type = "AME")
bnb_elasticities <- function(fit,
                             which = c("y1", "y2", "both"),
                             type  = c("AME", "MEM"),
                             vars  = NULL,
                             include_intercept = FALSE,
                             digits = 4,
                             print_output = TRUE) {
  stopifnot(inherits(fit, "bnb_fit"))
  type  <- match.arg(type)
  which <- match.arg(which)
  parts <- .bnb_diag_parts(fit)

  # worker for ONE response
  .one_resp <- function(X, beta, Vb, resp_name) {
    cn <- colnames(X)

    # restrict to user vars if given
    if (!is.null(vars)) {
      pos <- match(vars, cn)
      if (anyNA(pos)) {
        stop("These vars not found in design matrix for '", resp_name, "': ",
             paste(vars[is.na(pos)], collapse = ", "))
      }
      idxs      <- pos
      out_names <- vars
    } else {
      idxs      <- seq_along(cn)
      out_names <- cn
    }

    # drop intercept if requested
    if (!include_intercept) {
      keep <- out_names != "(Intercept)"
      idxs      <- idxs[keep]
      out_names <- out_names[keep]
    }
    if (!length(idxs)) {
      stop("No variables selected after intercept filtering for '", resp_name, "'.")
    }

    xbar <- colMeans(X)
    est  <- se <- numeric(length(idxs))
    vtype <- character(length(idxs))

    for (m in seq_along(idxs)) {
      j    <- idxs[m]
      xj   <- X[, j]
      bj   <- beta[j]
      varb <- Vb[j, j]
      is_bin <- all(xj %in% c(0, 1, NA))

      if (is_bin) {
        # 0/1 var -> semi-elasticity for dummy: exp(beta) - 1
        eff     <- exp(bj) - 1
        se_j    <- exp(bj) * sqrt(max(varb, 0))
        est[m]  <- eff
        se[m]   <- se_j
        vtype[m] <- "binary (0->1): exp(beta)-1"
      } else {
        # continuous var -> elasticity = beta * E[x_j]
        if (type == "AME") {
          m_xj    <- mean(xj, na.rm = TRUE)
          est[m]  <- bj * m_xj
          se[m]   <- sqrt(max(varb, 0)) * abs(m_xj)
        } else {
          est[m]  <- bj * xbar[j]
          se[m]   <- sqrt(max(varb, 0)) * abs(xbar[j])
        }
        vtype[m] <- "continuous: beta * E[x]"
      }
    }

    tab <- .bnb_me_tidy(out_names, est, se, digits)
    tab$var_type <- vtype

    if (print_output) {
      cat(sprintf("\n--- Elasticities / semi-elasticities (%s, %s) ---\n",
                  resp_name, type))
      .bnb_me_print(tab, digits = digits)
      cat("Continuous vars: elasticity = beta * E[x].\n")
      cat("0/1 vars: semi-elasticity = exp(beta) - 1 (percent change from 0->1).\n")
    }

    tab
  }

  # y1, y2, or both
  if (which == "both") {
    tab1 <- .one_resp(fit$X1, parts$beta1, parts$Vb1, "y1")
    tab2 <- .one_resp(fit$X2, parts$beta2, parts$Vb2, "y2")
    return(invisible(list(y1 = tab1, y2 = tab2)))
  }

  # single response
  if (which == "y1") {
    tab <- .one_resp(fit$X1, parts$beta1, parts$Vb1, "y1")
  } else {
    tab <- .one_resp(fit$X2, parts$beta2, parts$Vb2, "y2")
  }
  invisible(tab)
}
