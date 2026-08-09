# Automatic centring/scaling for rpbnb(standardize = TRUE): identify
# continuous predictors, standardize them before fitting, and map fitted
# coefficients back to the covariates' original units for display.
#
# inst/rpbnb_frank_open.R and inst/tmb_rpbnb_frank_open.R do this by hand --
# list the continuous carriers, centre/scale them before the fit, then
# rebuild an "original units" coefficient table from the affine chain rule
# afterwards. This file generalizes that pattern into rpbnb() itself: the
# fitted design (X1/X2, mu1/mu2, rp_meta, stored draws) stays on the
# standardized scale exactly as those scripts keep it -- predict(),
# marginal effects, and boundary/LR tests all still operate consistently on
# that internal representation -- but the *displayed* coefficient table
# (print()/summary()) shows original units, computed on demand from
# `$coef`/`$vcov` and never written back into them.

# Continuous predictor columns used by either equation: numeric, non-factor,
# with more than two distinct non-NA values. A two-level numeric column
# (0/1, 1/2, -1/1, ...) is treated as a fixed indicator, not a carrier to
# standardize -- centring/scaling it would only relabel the indicator, not
# change the model. Variables that appear only inside an offset() are
# excluded: an offset enters the linear predictor at face value (e.g.
# log(exposure)), and standardizing it would silently break that accounting.
.identify_continuous_vars <- function(formula_1, formula_2, data) {
  rhs_vars <- function(f) {
    tt  <- stats::terms(f, data = data)
    v   <- all.vars(stats::delete.response(tt))
    off <- attr(tt, "variables")
    oi  <- attr(tt, "offset")
    off_vars <- if (length(oi)) {
      unlist(lapply(oi, function(i) all.vars(off[[i + 1L]])))
    } else {
      character(0)
    }
    setdiff(v, off_vars)
  }
  vars <- unique(c(rhs_vars(formula_1), rhs_vars(formula_2)))
  vars <- intersect(vars, names(data))
  keep <- vapply(vars, function(v) {
    x <- data[[v]]
    is.numeric(x) && !is.factor(x) && length(unique(x[!is.na(x)])) > 2L
  }, logical(1))
  vars[keep]
}

# center = mean, scale = sd for each continuous variable. Errors on a
# zero-variance or non-finite column rather than silently dividing by zero.
.compute_scaling <- function(data, vars) {
  scaling <- lapply(vars, function(v) {
    x <- data[[v]]
    c(center = mean(x, na.rm = TRUE), scale = stats::sd(x, na.rm = TRUE))
  })
  names(scaling) <- vars
  bad <- vapply(scaling, function(s) {
    !is.finite(s[["scale"]]) || s[["scale"]] == 0
  }, logical(1))
  if (any(bad)) {
    stop("standardize = TRUE: zero-variance or non-finite column(s): ",
         paste(vars[bad], collapse = ", "), call. = FALSE)
  }
  scaling
}

.apply_scaling <- function(data, scaling) {
  for (v in names(scaling)) {
    data[[v]] <- (data[[v]] - scaling[[v]][["center"]]) / scaling[[v]][["scale"]]
  }
  data
}

# Map a fitted rpbnb_fit / rpbnb_tmb_fit's coefficient (and, where possible,
# standard-error) vector back to the covariates' original units. Both
# engines name parameters the same way (b<eq>:<var>, log_sd<eq>:<var>, etc.),
# so one implementation serves both. Centring and scaling are an affine
# column transform, so this is EXACT and needs no refit: a continuous
# slope divides by its scale, the intercept absorbs the centring shift
# -sum_j(center_j/scale_j * beta_j), a random coefficient's scale parameter
# (log_sd/log_w/log_s -- whichever distribution it names) shifts by
# -log(scale) (its SE is unaffected, since it is a constant additive shift),
# and every other parameter -- dispersion, dependence, binary/categorical
# coefficients -- is untouched.
#
# Standard errors are exact for every row except the intercept when the
# fit's full covariance matrix (`$vcov`) is unavailable: the intercept's
# variance needs its covariance with the slopes, which a diagonal-only or
# absent `$se` cannot supply. In that case the intercept row's SE is NA
# rather than silently wrong.
#
# Returns NULL when the fit carries no `$scaling` (i.e. was not fit with
# rpbnb(standardize = TRUE)).
.rpbnb_orig_units <- function(object) {
  scaling <- object$scaling
  if (is.null(scaling)) return(NULL)
  cont <- object$continuous_vars
  est  <- object$coef
  se   <- object$se
  V    <- object$vcov
  has_full_vcov <- is.matrix(V) &&
    all(names(est) %in% rownames(V)) && all(names(est) %in% colnames(V))

  est_o <- est
  se_o  <- stats::setNames(rep(NA_real_, length(est)), names(est))
  if (!is.null(se)) se_o[names(se)] <- se

  sc <- vapply(cont, function(v) scaling[[v]][["scale"]],  numeric(1))
  ce <- vapply(cont, function(v) scaling[[v]][["center"]], numeric(1))
  names(sc) <- names(ce) <- cont

  for (eq in c(1L, 2L)) {
    nm  <- names(est)
    idx <- grep(paste0("^b", eq, ":"), nm)
    if (length(idx)) {
      var <- sub(paste0("^b", eq, ":"), "", nm[idx])
      A <- diag(length(idx))
      dimnames(A) <- list(nm[idx], nm[idx])
      is_c <- which(var %in% cont)
      is_i <- which(var == "(Intercept)")
      if (length(is_c)) diag(A)[is_c] <- 1 / sc[var[is_c]]
      if (length(is_i) && length(is_c)) {
        A[is_i, is_c] <- -ce[var[is_c]] / sc[var[is_c]]
      }
      est_o[idx] <- as.vector(A %*% est[idx])
      if (has_full_vcov) {
        Vsub <- V[nm[idx], nm[idx], drop = FALSE]
        se_o[idx] <- sqrt(pmax(diag(A %*% Vsub %*% t(A)), 0))
      } else if (!is.null(se)) {
        # Exact off the intercept row (A is diagonal there); the intercept
        # needs the slopes' covariance, unavailable without a full vcov.
        se_o[idx] <- se[nm[idx]] * abs(diag(A))
        if (length(is_i)) se_o[idx[is_i]] <- NA_real_
      }
    }
    for (pfx in paste0(c("log_sd", "log_w", "log_s"), eq)) {
      sidx <- grep(paste0("^", pfx, ":"), nm)
      if (!length(sidx)) next
      svar <- sub(paste0("^", pfx, ":"), "", nm[sidx])
      in_c <- svar %in% cont
      if (any(in_c)) {
        est_o[sidx[in_c]] <- est[sidx[in_c]] - log(sc[svar[in_c]])
        # se_o unchanged: constant additive shift on the log scale.
      }
    }
  }
  list(coef = est_o, se = se_o)
}

# One-line note for print()/summary() when `$scaling` is present, naming the
# standardized variables so the reader knows the coefficient table below is
# already back-transformed rather than assuming a standardized-scale table.
.print_standardize_note <- function(object) {
  if (is.null(object$scaling)) return(invisible(NULL))
  cat("Continuous predictors standardized during fitting (",
      paste(object$continuous_vars, collapse = ", "),
      "); coefficients below are in ORIGINAL covariate units.\n", sep = "")
  invisible(NULL)
}
