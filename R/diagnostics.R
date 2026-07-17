# Diagnostics for bnb_fit objects: goodness of fit, marginal effects, and
# elasticities. Ported from the legacy bnbr_gof / bnbr_me / bnbr_elasticities,
# adapted to the new bnb_fit object structure (flat coef vector, stored vcov).

# --- internal: usable null-model log-likelihood for pseudo-R^2 --------------
# Returns the null log-likelihood only if the null fit exists, converged, and is
# finite; otherwise NA (with a warning for a non-converged/non-finite fit). A
# finite value from a non-converged optimizer must not be used as the baseline.
.null_model_loglik <- function(fit_null) {
  if (is.null(fit_null)) return(NA_real_)
  ll <- as.numeric(fit_null$logLik)
  converged <- isTRUE(fit_null$convergence$converged)
  if (!is.finite(ll) || !converged) {
    warning("Null model did not converge to a finite log-likelihood",
            if (!converged) " (optimizer did not converge)" else "",
            "; pseudo-R^2 set to NA.", call. = FALSE)
    return(NA_real_)
  }
  ll
}

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
#' intercept-only null model refit with the same dependence structure (copula
#' fits rebuild the copula from `fit$cop_family`). Pseudo-R-squared values are
#' returned raw and are not clamped to `[0, 1]`; a negative value flags a full
#' model that fits worse than the null.
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
  # A degenerate response (e.g. a constant outcome in a subsample) can make the
  # intercept-only glm.nb / optimizer fail. Treat that as missing pseudo-R^2
  # rather than aborting goodness-of-fit entirely.
  df_null  <- data.frame(Y1 = fit$Y1, Y2 = fit$Y2)
  # Reconstruct the dependence structure the null must share. A copula fit stores
  # dependence as a bare family string (e.g. "normal") in fit$dependence, which
  # fit_bnb() would reject; rebuild the copula() object from fit$cop_family.
  null_dep <- if (!is.null(fit$cop_family)) copula(fit$cop_family) else fit$dependence
  fit_null <- tryCatch(
    fit_bnb(Y1 ~ 1, Y2 ~ 1, data = df_null, dependence = null_dep),
    error = function(e) {
      warning("Null model could not be fitted (", conditionMessage(e),
              "); pseudo-R^2 set to NA.", call. = FALSE)
      NULL
    })
  ll_null <- .null_model_loglik(fit_null)

  # ---------- Pseudo R^2 ----------
  # Returned raw (not clamped to [0, 1]): a negative McFadden or adjusted value
  # is diagnostically meaningful -- it signals a full model that fits worse than
  # the null (e.g. under a penalty), which clamping to 0 would silently hide.
  R2_MF  <- 1 - (ll_full / ll_null)
  R2_MFa <- 1 - ((ll_full - k) / ll_null)
  R2_CS  <- 1 - exp((2 / n) * (ll_null - ll_full))
  denom  <- 1 - exp((2 / n) * ll_null)
  R2_NK  <- if (is.na(denom) || abs(denom) < .Machine$double.eps) NA_real_ else R2_CS / denom

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

# ============================================================================
# Random-parameter interpretation (rpbnb_fit): marginal effects & elasticities
# ----------------------------------------------------------------------------
# Built on the Monte-Carlo integrated population mean mu_i = E_beta[exp(x_i'b)]
# that predict.rpbnb_fit() computes over the stored draws. The mean depends only
# on an equation's mean coefficients (b{e}:col) and its log-scale parameters
# (log_sd/log_w/log_s), never on log_m or z_lambda/z_theta, so the delta-method
# parameter block is exactly (b{e}, log-scale{e}).
# ============================================================================

# Per-draw n x R matrix of pmin(exp(x'b + XR %*% dev_r), cap) for one design X.
# `dev` is the R x q deviation matrix from rand_realize(); q == 0 collapses to R
# identical columns of exp(x'b) (so a fully-fixed equation reduces cleanly).
.rp_g_matrix <- function(X, b, rand_idx, dev, cap = RP_PRED_CAP) {
  xb <- as.vector(X %*% b)
  R  <- nrow(dev)
  if (length(rand_idx) == 0) {
    return(matrix(pmin(exp(xb), cap), nrow = length(xb), ncol = R))
  }
  XR <- X[, rand_idx, drop = FALSE]
  g  <- vapply(seq_len(R),
               function(r) pmin(exp(xb + as.vector(XR %*% dev[r, ])), cap),
               numeric(length(xb)))
  matrix(g, nrow = length(xb))
}

# Rows whose integrated mean is analytically infinite: a lognormal random
# coefficient j with sign_j * X_ij > 0 (mirrors .rp_integrated_mu in methods.R).
.rp_inf_rows <- function(X, rand_idx, dist, sign) {
  inf <- logical(nrow(X))
  for (j in seq_along(rand_idx)) {
    if (identical(dist[j], "lognormal")) {
      inf <- inf | (sign[j] * X[, rand_idx[j]] > 0)
    }
  }
  inf
}

# Assemble the static metadata for one equation's interpretation computation:
# design, coefficient names, random metadata, the delta-method parameter names,
# the selected-variable indices and their binary/continuous classification.
.rp_diag_meta <- function(fit, eq, type, vars, include_intercept) {
  X  <- if (eq == 1L) fit$X1 else fit$X2
  cn <- colnames(X)
  p  <- ncol(X)
  b_names  <- paste0("b", eq, ":", cn)
  rand_idx <- if (eq == 1L) fit$rand_idx1 else fit$rand_idx2
  dist <- if (eq == 1L) fit$rp_meta$dist1 else fit$rp_meta$dist2
  sign <- if (eq == 1L) fit$rp_meta$sign1 else fit$rp_meta$sign2
  Z    <- if (eq == 1L) fit$rp_meta$Z1 else fit$rp_meta$Z2

  scale_names <- if (length(rand_idx)) {
    vapply(seq_along(rand_idx), function(j) {
      lbl <- rand_dist_registry[[dist[j]]]$scale_label
      paste0(lbl, eq, ":", cn[rand_idx[j]])
    }, character(1))
  } else character(0)

  # --- variable selection (mirrors bnb_marginal_effects) ---
  if (is.null(vars)) {
    idx <- seq_len(p)
  } else if (is.numeric(vars)) {
    if (any(vars < 1 | vars > p)) stop("Index out of range in `vars`.", call. = FALSE)
    idx <- vars
  } else {
    idx <- match(vars, cn)
    if (anyNA(idx)) {
      stop("Unknown vars in equation ", eq, ": ",
           paste(vars[is.na(idx)], collapse = ", "),
           ". Available: ", paste(cn, collapse = ", "), call. = FALSE)
    }
  }
  if (!include_intercept) idx <- idx[cn[idx] != "(Intercept)"]
  if (!length(idx)) stop("No variables selected after filtering the intercept.",
                         call. = FALSE)

  is_bin <- vapply(idx, function(j) {
    x <- X[, j]; x <- x[!is.na(x)]
    length(x) > 0 && all(x %in% c(0, 1))
  }, logical(1))
  is_rand <- idx %in% rand_idx

  Xbar <- matrix(colMeans(X), nrow = 1, dimnames = list(NULL, cn))

  list(X = X, Xbar = Xbar, cn = cn, b_names = b_names, scale_names = scale_names,
       rand_idx = rand_idx, dist = dist, sign = sign, Z = Z, type = type,
       sel = idx, is_bin = is_bin, is_rand = is_rand)
}

# Deterministic, vector-valued estimand over theta = c(b, log-scales), holding
# the draws Z fixed. `quantity` is "me" (marginal effect) or "elas" (elasticity/
# semi-elasticity). `mark_inf = TRUE` propagates analytic Inf on lognormal rows
# (for the reported point estimate); the jacobian path uses `mark_inf = FALSE`
# so the finite-difference gradient stays smooth.
.rp_estimand <- function(theta, meta, quantity, mark_inf) {
  cn <- meta$cn
  b  <- theta[meta$b_names]; names(b) <- cn
  scales <- if (length(meta$rand_idx)) exp(theta[meta$scale_names]) else numeric(0)
  Z   <- meta$Z
  rr  <- if (length(meta$rand_idx))
           rand_realize(Z, meta$dist, meta$sign, b[meta$rand_idx], scales)
         else NULL
  dev      <- if (is.null(rr)) matrix(0, nrow(Z), 0) else rr$dev
  coef_mat <- if (is.null(rr)) NULL else rr$coef

  X   <- if (meta$type == "MEM") meta$Xbar else meta$X
  inf_rows <- if (mark_inf && length(meta$rand_idx))
                .rp_inf_rows(X, meta$rand_idx, meta$dist, meta$sign)
              else logical(nrow(X))

  g  <- .rp_g_matrix(X, b, meta$rand_idx, dev)
  mu <- rowMeans(g)

  out <- numeric(length(meta$sel))
  for (m in seq_along(meta$sel)) {
    j <- meta$sel[m]
    if (meta$is_bin[m]) {
      X0 <- X; X0[, j] <- 0
      X1 <- X; X1[, j] <- 1
      mu0 <- rowMeans(.rp_g_matrix(X0, b, meta$rand_idx, dev))
      mu1 <- rowMeans(.rp_g_matrix(X1, b, meta$rand_idx, dev))
      val <- if (quantity == "me") mu1 - mu0 else mu1 / mu0 - 1
    } else {
      rk <- match(j, meta$rand_idx)
      coef_row <- if (!is.na(rk)) coef_mat[, rk] else rep(b[[j]], nrow(Z))
      dmu <- rowMeans(sweep(g, 2, coef_row, `*`))
      val <- if (quantity == "me") dmu else X[, j] * dmu / mu
    }
    if (any(inf_rows)) val[inf_rows] <- Inf
    out[m] <- mean(val)
  }
  names(out) <- cn[meta$sel]
  out
}

# One column of the delta-method jacobian: the Richardson-extrapolated
# derivative of .rp_estimand() with respect to theta_hat[[k]], holding every
# other parameter fixed. Mathematically identical to column k of
# numDeriv::jacobian(f_vector, theta_hat) -- verified bit-for-bit against it
# (max abs diff 0) -- splitting the jacobian into these independent
# per-parameter calls is what makes the standard-error step embarrassingly
# parallel across a cluster.
#
# Uses a BARE (unqualified) call to .rp_estimand, NOT rpbnb:::.rp_estimand:
# on a parallel::makeCluster() worker the rpbnb namespace is not loaded (this
# project's dev workflow runs off pkgload::load_all(), not an installed
# package), so a `:::`-qualified call fails on a worker. A bare call resolves
# via ordinary lexical/global-env lookup, which parallel::clusterExport()
# (see .rp_make_cluster()) satisfies by placing .rp_estimand and its own
# transitive callees into each worker's .GlobalEnv.
.rp_jac_col <- function(k, theta_hat, theta_names, meta, quantity) {
  fk <- function(tk) {
    t <- theta_hat; t[[k]] <- tk
    names(t) <- theta_names
    .rp_estimand(t, meta, quantity, mark_inf = FALSE)
  }
  numDeriv::jacobian(fk, theta_hat[[k]])
}

# Compute point estimates + delta-method SEs for one equation and assemble the
# tidy output frame (Name/Estimate/StdErr/z/p/Signif/var_type). `cl` is an
# optional parallel cluster (see .rp_make_cluster()); when non-NULL, the SE
# jacobian is computed as independent per-parameter columns dispatched across
# the cluster instead of one sequential numDeriv::jacobian() call -- the two
# paths are numerically identical (see .rp_jac_col()).
.rp_diag_one <- function(fit, eq, quantity, type, vars, include_intercept,
                         digits, print_output, resp_name, cl = NULL) {
  meta <- .rp_diag_meta(fit, eq, type, vars, include_intercept)

  est <- .rp_estimand(fit$coef, meta, quantity, mark_inf = TRUE)

  # Warn once if any reported estimate is a lognormal analytic infinity.
  if (any(!is.finite(est))) {
    warning(sum(!is.finite(est)), " selected variable(s) in ", resp_name,
            " have an analytically infinite estimate (a lognormal random ",
            "coefficient with sign * covariate > 0); reporting Inf.",
            call. = FALSE)
  }

  theta_names <- c(meta$b_names, meta$scale_names)
  V <- fit$vcov[theta_names, theta_names, drop = FALSE]
  if (any(!is.finite(V))) {
    warning("vcov is unavailable (fit made with compute_se = FALSE?); ",
            "standard errors set to NA for ", resp_name, ".", call. = FALSE)
    se <- rep(NA_real_, length(est))
  } else {
    theta_hat <- fit$coef[theta_names]
    G <- if (!is.null(cl)) {
      do.call(cbind, parallel::parLapply(cl, seq_along(theta_hat), .rp_jac_col,
                                         theta_hat = theta_hat, theta_names = theta_names,
                                         meta = meta, quantity = quantity))
    } else {
      numDeriv::jacobian(
        function(t) { names(t) <- theta_names
                      .rp_estimand(t, meta, quantity, mark_inf = FALSE) },
        theta_hat)
    }
    se <- vapply(seq_len(nrow(G)), function(m) {
      g <- G[m, ]
      sqrt(as.numeric(t(g) %*% V %*% g))
    }, numeric(1))
    # A non-finite point estimate (analytic Inf) has no meaningful SE.
    se[!is.finite(est)] <- NA_real_
  }

  tab <- .bnb_me_tidy(names(est), as.numeric(est), se, digits)
  base <- if (quantity == "me") {
    ifelse(meta$is_bin, "binary(0->1)", "continuous")
  } else {
    ifelse(meta$is_bin, "semi-elasticity", "elasticity")
  }
  tab$var_type <- ifelse(meta$is_rand, paste0(base, " (random)"), base)

  if (print_output) {
    label <- if (quantity == "me") "Marginal effects" else "Elasticities"
    cat(sprintf("\n--- %s (RP integrated mean) for %s (%s) ---\n",
                label, resp_name, type))
    .bnb_me_print(tab, digits = digits)
    if (quantity == "me") {
      cat("continuous: dE[Y]/dx_j = mean_r coef_rj * exp(lp_r)\n")
      cat("binary: E[Y|x_j=1] - E[Y|x_j=0] (integrated over draws)\n")
    } else {
      cat("continuous: elasticity = x_j * (dE[Y]/dx_j) / E[Y]\n")
      cat("binary: semi-elasticity = E[Y|x_j=1]/E[Y|x_j=0] - 1\n")
    }
    cat("'(random)' vars use the draw-integrated formula.\n")
  }
  tab
}

# Build (if n_cores > 1) and export-configure a PSOCK cluster for the parallel
# delta-method jacobian in .rp_diag_one() (see .rp_jac_col()). Returns NULL for
# n_cores <= 1 (the caller then uses .rp_diag_one()'s unchanged sequential
# path) or when the `parallel` package is unavailable (with a warning,
# matching the fallback message used in fit_rpbnb.R). The caller owns the
# cluster's lifetime and must parallel::stopCluster() it when done (typically
# via on.exit()), and should reuse the SAME cluster across every equation
# computed in one call (y1/y2/both/all) rather than building one per equation.
.rp_make_cluster <- function(n_cores) {
  if (n_cores <= 1) return(NULL)
  if (!requireNamespace("parallel", quietly = TRUE)) {
    warning("Package 'parallel' not available; running sequentially.", call. = FALSE)
    return(NULL)
  }
  cl <- parallel::makeCluster(as.integer(n_cores))
  ok <- FALSE
  on.exit(if (!ok) parallel::stopCluster(cl))
  parallel::clusterExport(cl,
    c(".rp_estimand", ".rp_g_matrix", ".rp_inf_rows", "rand_realize", "rand_dist_registry",
      "RP_PRED_CAP", "tri_icdf"),
    envir = environment())
  ok <- TRUE
  cl
}

#' Marginal effects for a random-parameter bivariate NB model
#'
#' Average marginal effects (AME) or marginal effects at the mean (MEM) for each
#' margin of an [fit_rpbnb()] fit, built on the Monte-Carlo integrated population
#' mean \eqn{\mu_i = E_\beta[\exp(x_i'\beta)]} (the same estimand as
#' [predict.rpbnb_fit()], reusing the fit's stored draws). Continuous effects are
#' \eqn{\partial \mu_i/\partial x_{ij} = \mathrm{mean}_r\, \mathrm{coef}_{rj}
#' \exp(\mathrm{lp}_{ir})} (the realized coefficient per draw, so random and fixed
#' columns are handled uniformly); binary (0/1) effects are the integrated
#' discrete difference \eqn{E[Y|x_j=1] - E[Y|x_j=0]}. Standard errors use a
#' numeric delta method over the equation's mean and log-scale parameters.
#'
#' @param fit An `rpbnb_fit` object from [fit_rpbnb()].
#' @param which Which margin(s): "y1", "y2", "both", or "all".
#' @param type "AME" (average over the sample) or "MEM" (effect at the mean row).
#' @param vars Optional variable names or indices to restrict output.
#' @param include_intercept Logical; include the intercept term.
#' @param digits Number of decimal places for printed output.
#' @param print_output Logical; if `FALSE`, suppress printing.
#' @param n_cores Number of worker processes for the delta-method standard-error
#'   jacobian (1 = sequential, the default). When `n_cores > 1`, the jacobian's
#'   independent per-parameter columns are dispatched across a
#'   `parallel::makeCluster()` cluster (one cluster per call, shared across
#'   every equation `which` computes); results are numerically identical to the
#'   sequential path. Falls back to sequential with a warning if the `parallel`
#'   package is unavailable.
#' @return A data frame (single margin, invisibly) or a named list of data frames
#'   (`both`/`all`), each with columns `Name`, `Estimate`, `StdErr`, `z`, `p`,
#'   `Signif`, `var_type`.
#' @seealso [bnb_marginal_effects()] for fixed-coefficient `bnb_fit` models.
#' @export
#' @examples
#' sim <- simulate_rpbnb(n = 400,
#'   beta1 = c("(Intercept)" = 0.2, x1 = 0.4),
#'   beta2 = c("(Intercept)" = 0.1, x1 = -0.3),
#'   random_1 = list(x1 = list(sd = 0.5)),
#'   dispersion = c(m1 = 0.4, m2 = 0.5), seed = 1)
#' fit <- fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim$data, random_1 = "x1",
#'                  draws = 100)
#' rpbnb_marginal_effects(fit, which = "y1", type = "AME")
rpbnb_marginal_effects <- function(fit,
                                   which = c("y1", "y2", "both", "all"),
                                   type  = c("AME", "MEM"),
                                   vars  = NULL,
                                   include_intercept = FALSE,
                                   digits = 4,
                                   print_output = TRUE,
                                   n_cores = 1L) {
  stopifnot(inherits(fit, "rpbnb_fit"))
  which <- match.arg(which)
  type  <- match.arg(type)
  cl <- .rp_make_cluster(n_cores)
  on.exit(if (!is.null(cl)) parallel::stopCluster(cl), add = TRUE)
  if (which %in% c("both", "all")) {
    return(invisible(list(
      y1 = .rp_diag_one(fit, 1L, "me", type, vars, include_intercept,
                        digits, print_output, "y1", cl = cl),
      y2 = .rp_diag_one(fit, 2L, "me", type, vars, include_intercept,
                        digits, print_output, "y2", cl = cl))))
  }
  eq <- if (which == "y1") 1L else 2L
  invisible(.rp_diag_one(fit, eq, "me", type, vars, include_intercept,
                         digits, print_output, which, cl = cl))
}

#' Elasticities and semi-elasticities for a random-parameter bivariate NB model
#'
#' Continuous elasticities \eqn{x_{ij}\,(\partial \mu_i/\partial x_{ij})/\mu_i}
#' and binary semi-elasticities \eqn{\mu_i(x_j=1)/\mu_i(x_j=0) - 1}, built on the
#' Monte-Carlo integrated population mean \eqn{\mu_i = E_\beta[\exp(x_i'\beta)]}
#' of an [fit_rpbnb()] fit. Under fixed coefficients these reduce to
#' \eqn{\beta_j E[x_j]} and \eqn{\exp(\beta_j) - 1}. Standard errors use a numeric
#' delta method over the equation's mean and log-scale parameters.
#'
#' @param fit An `rpbnb_fit` object from [fit_rpbnb()].
#' @param which Which margin(s): "y1", "y2", or "both".
#' @param type "AME" (average over the sample) or "MEM" (evaluated at the mean row).
#' @param vars Optional variable names or indices to restrict output.
#' @param include_intercept Logical; include the intercept term.
#' @param digits Number of decimal places for printed output.
#' @param print_output Logical; if `FALSE`, suppress printing.
#' @param n_cores Number of worker processes for the delta-method standard-error
#'   jacobian (1 = sequential, the default). When `n_cores > 1`, the jacobian's
#'   independent per-parameter columns are dispatched across a
#'   `parallel::makeCluster()` cluster (one cluster per call, shared across
#'   every equation `which` computes); results are numerically identical to the
#'   sequential path. Falls back to sequential with a warning if the `parallel`
#'   package is unavailable.
#' @return A data frame (single margin, invisibly) or a named list of data frames
#'   (`both`), each with columns `Name`, `Estimate`, `StdErr`, `z`, `p`,
#'   `Signif`, `var_type`.
#' @seealso [bnb_elasticities()] for fixed-coefficient `bnb_fit` models.
#' @export
#' @examples
#' sim <- simulate_rpbnb(n = 400,
#'   beta1 = c("(Intercept)" = 0.2, x1 = 0.4),
#'   beta2 = c("(Intercept)" = 0.1, x1 = -0.3),
#'   random_1 = list(x1 = list(sd = 0.5)),
#'   dispersion = c(m1 = 0.4, m2 = 0.5), seed = 1)
#' fit <- fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim$data, random_1 = "x1",
#'                  draws = 100)
#' rpbnb_elasticities(fit, which = "both", type = "AME")
rpbnb_elasticities <- function(fit,
                               which = c("y1", "y2", "both"),
                               type  = c("AME", "MEM"),
                               vars  = NULL,
                               include_intercept = FALSE,
                               digits = 4,
                               print_output = TRUE,
                               n_cores = 1L) {
  stopifnot(inherits(fit, "rpbnb_fit"))
  which <- match.arg(which)
  type  <- match.arg(type)
  cl <- .rp_make_cluster(n_cores)
  on.exit(if (!is.null(cl)) parallel::stopCluster(cl), add = TRUE)
  if (which == "both") {
    return(invisible(list(
      y1 = .rp_diag_one(fit, 1L, "elas", type, vars, include_intercept,
                        digits, print_output, "y1", cl = cl),
      y2 = .rp_diag_one(fit, 2L, "elas", type, vars, include_intercept,
                        digits, print_output, "y2", cl = cl))))
  }
  eq <- if (which == "y1") 1L else 2L
  invisible(.rp_diag_one(fit, eq, "elas", type, vars, include_intercept,
                         digits, print_output, which, cl = cl))
}
