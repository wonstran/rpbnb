# Shared response/design preparation for the bivariate fitters.
# Ensures the two outcomes are taken from the SAME complete-case rows and that
# responses are valid non-negative integer counts. Internal.

#' Validate and coerce a response vector to non-negative integer counts
#'
#' Rejects non-finite, negative, or non-integer-valued responses rather than
#' silently truncating them.
#'
#' @keywords internal
#' @noRd
.check_counts <- function(y, label) {
  y <- as.numeric(y)
  if (any(!is.finite(y))) {
    stop("Response ", label, " contains non-finite values; counts must be ",
         "finite non-negative integers.", call. = FALSE)
  }
  if (any(y < 0)) {
    stop("Response ", label, " contains negative values; counts must be ",
         "non-negative.", call. = FALSE)
  }
  if (any(abs(y - round(y)) > 1e-8)) {
    stop("Response ", label, " contains non-integer values; counts must be ",
         "integer-valued (use round() upstream if this is intended).",
         call. = FALSE)
  }
  as.integer(round(y))
}

#' Build aligned responses and design matrices for both equations
#'
#' Subsets `data` to the rows that are valid (complete) across BOTH formulas, so
#' the two outcomes and design matrices always share the same rows. Both model
#' frames are evaluated with `na.action = na.pass` and a single common valid-row
#' mask is derived from the *evaluated* responses, design terms, and offsets --
#' not from the raw variables. This closes a desynchronization hole: a
#' transformation (e.g. `log(x)` with `x <= 0`) can create `NA`/`NaN` *after* the
#' raw variables are complete, which would otherwise drop different rows in the
#' two equations and silently pair outcomes from mismatched rows (and, in the
#' compiled cores, index past the shorter response).
#'
#' Row selection is two-stage. Stage A applies a `complete.cases()` mask over the
#' union of the raw formula variables BEFORE any term is evaluated -- this
#' restores the historical complete-case behavior and, crucially, prevents a
#' stateful term that validates its input (e.g. `poly()` raises "missing values
#' are not allowed in 'poly'") from aborting during the first evaluation, before
#' any mask exists. Stage B then evaluates both frames with `na.action = na.pass`
#' on the raw-complete rows and keeps a row only when every evaluated design entry
#' and offset is finite -- a finiteness check (not merely missing-value-aware), so
#' transformed `NA`/`NaN` (`log(x)`, `x < 0`) AND transformed infinities
#' (`log(0) = -Inf`, which `complete.cases()` treats as complete and which would
#' otherwise seed a non-finite objective via `0 * -Inf = NaN`) are both rejected.
#' Responses drop only `NA`/`NaN`; a non-finite `Inf` response is kept so
#' `.check_counts()` raises its explicit error rather than silently dropping it.
#' A nested term whose outer function rejects an inner non-finite result
#' (e.g. `poly(log(x))` with `log(x) = NaN`) cannot be masked row-wise and is
#' reported with an actionable error.
#'
#' The final designs are REBUILT from the retained raw rows (not by subsetting
#' the full-data frames), so a stateful term (e.g. `poly(x, 2)`, `scale(x)`) is
#' evaluated over the final sample -- keeping fitting, `predict()`, and `glm.nb`
#' on one consistent basis -- and `droplevels()` removes factor levels that occur
#' only on rejected rows (no stray all-zero dummy column). Re-evaluation is
#' validated for finiteness.
#'
#' Also extracts equation-specific offsets (`model.offset`) and captures the
#' rebuilt-frame `terms` (with `predvars` fixed to the retained-sample basis),
#' factor `xlevels`, and `contrasts` so that newdata designs in `predict()` are
#' column-stable and use the same stateful-term basis the fit used.
#'
#' @return A list with `Y1`, `Y2` (integer), `X1`, `X2`, `cn1`, `cn2`, `off1`,
#'   `off2` (numeric offset vectors, all-zero when the formula has no offset),
#'   `terms1`, `terms2`, `xlevels1`, `xlevels2`, `contrasts1`, `contrasts2`, `n`,
#'   and `data` (the aligned complete-case subset, for downstream refits such as
#'   glm.nb).
#' @keywords internal
#' @noRd
.prepare_bnb_data <- function(formula_1, formula_2, data) {
  if (!is.data.frame(data)) stop("`data` must be a data frame.", call. = FALSE)
  # Expand the formulas against the data so a "." RHS resolves to the actual
  # columns before collecting variable names.
  t1 <- stats::terms(formula_1, data = data)
  t2 <- stats::terms(formula_2, data = data)
  vars <- unique(c(all.vars(t1), all.vars(t2)))
  missing_vars <- vars[!vars %in% names(data)]
  if (length(missing_vars)) {
    stop("Variable(s) not found in data: ",
         paste(missing_vars, collapse = ", "), ".", call. = FALSE)
  }

  # Stage A -- raw missing-value mask, applied BEFORE any term is evaluated.
  # A stateful term validates its input and aborts on a missing value
  # (e.g. poly(x, 2) raises "missing values are not allowed in 'poly'"), so the
  # na.pass evaluation below would fail before the evaluated finite mask could
  # run. Dropping raw-NA rows up front over the union of BOTH formulas' variables
  # restores the historical complete-case behavior and lets such terms evaluate.
  raw_ok <- stats::complete.cases(data[, vars, drop = FALSE])
  if (!any(raw_ok)) {
    stop("No complete cases across both formulas.", call. = FALSE)
  }
  data_raw <- data[raw_ok, , drop = FALSE]

  # Stage B -- evaluate BOTH frames with na.pass (on the raw-complete rows) so
  # transformed NAs/Infs are not dropped independently, then derive ONE common
  # finite-row mask over the evaluated responses, designs, and offsets of both
  # equations. A finiteness check (not complete.cases) is used so transformed
  # infinities (log(0) = -Inf) are treated as invalid, not "complete". A term
  # whose OUTER function rejects an inner non-finite value (e.g. poly(log(x)) with
  # log(x) = NaN) cannot be masked row-wise here -- diagnose it clearly.
  eval_frame <- function(tt, eqn) {
    tryCatch(stats::model.frame(tt, data = data_raw, na.action = stats::na.pass),
             error = function(e)
               stop("Could not evaluate the equation ", eqn, " model terms: ",
                    conditionMessage(e), ". A term such as poly()/bs()/scale() ",
                    "rejects a non-finite value produced by an inner transformation ",
                    "(e.g. poly(log(x)) where log(x) is NaN/Inf); precompute or ",
                    "clean that term as a column before fitting.", call. = FALSE))
  }
  mf1_full <- eval_frame(t1, 1L)
  mf2_full <- eval_frame(t2, 2L)
  y1_full  <- stats::model.response(mf1_full)
  y2_full  <- stats::model.response(mf2_full)
  X1_full  <- stats::model.matrix(t1, mf1_full)
  X2_full  <- stats::model.matrix(t2, mf2_full)

  # Design and offset entries must be FINITE (drops NA, NaN, AND Inf rows). The
  # RESPONSE mask drops only NA/NaN (matching the historical complete.cases
  # behavior); a non-finite Inf response is deliberately kept so `.check_counts()`
  # raises its explicit "non-finite response" error rather than silently dropping
  # a corrupt count.
  row_finite <- function(x) {
    if (is.null(x)) return(NULL)
    if (is.matrix(x)) apply(is.finite(x), 1L, all) else is.finite(as.numeric(x))
  }
  row_present <- function(y) if (is.null(y)) NULL else !is.na(as.numeric(y))
  masks <- Filter(Negate(is.null),
                  list(row_present(y1_full), row_present(y2_full),
                       row_finite(X1_full), row_finite(X2_full),
                       row_finite(stats::model.offset(mf1_full)),
                       row_finite(stats::model.offset(mf2_full))))
  ok <- Reduce(`&`, masks)
  if (!any(ok)) {
    stop("No rows are finite and complete across both formulas (after evaluating ",
         "any transformations and offsets).", call. = FALSE)
  }

  # REBUILD the designs from the retained RAW rows, not by subsetting the
  # already-evaluated full-data frames. This matters for stateful terms: a basis
  # such as poly(x, 2) or scale(x) is computed over the whole sample when
  # model.frame() first evaluates it, so subsetting the full-data frame would
  # keep a basis that depends on rows we are about to reject -- and, because
  # `predict()` re-evaluates the term on newdata, the stored design and an
  # explicit-newdata prediction (and glm.nb, which re-evaluates on `data`) would
  # then describe different parameterizations. Re-evaluating on the retained data
  # gives one consistent basis for fitting, prediction, and glm.nb.
  #
  # `droplevels()` removes factor levels that occur only on rejected rows, so a
  # rejected-only level does not leave a stray all-zero dummy column (a
  # rank-deficient design) or a phantom `xlevels` entry.
  # `ok` indexes the raw-complete subset (data_raw), so subset that, not `data`.
  data_cc <- droplevels(data_raw[ok, , drop = FALSE])
  mf1 <- stats::model.frame(t1, data = data_cc, na.action = stats::na.pass)
  mf2 <- stats::model.frame(t2, data = data_cc, na.action = stats::na.pass)

  build_design <- function(tt, mf, eq) {
    tryCatch(stats::model.matrix(tt, mf),
             error = function(e) {
               stop("Could not build the equation ", eq, " design after dropping ",
                    "invalid rows: ", conditionMessage(e), ". A factor may have ",
                    "fewer than two observed levels once invalid rows are removed.",
                    call. = FALSE)
             })
  }
  X1 <- build_design(t1, mf1, 1L)
  X2 <- build_design(t2, mf2, 2L)

  off1 <- .as_offset(stats::model.offset(mf1), nrow(X1))
  off2 <- .as_offset(stats::model.offset(mf2), nrow(X2))

  # Re-evaluating a stateful term on the reduced sample can itself introduce new
  # non-finite values (e.g. scale() on a now-constant column) that the first-pass
  # mask could not foresee. Validate the final designs/offsets and fail clearly
  # rather than hand a corrupt design to an estimator.
  if (!all(is.finite(X1)) || !all(is.finite(X2)) ||
      any(!is.finite(off1)) || any(!is.finite(off2))) {
    stop("A stateful formula term (e.g. poly()/scale()) produced non-finite ",
         "design or offset values after dropping invalid rows. Precompute such ",
         "terms as columns on the full sample, or remove the offending rows ",
         "before fitting.", call. = FALSE)
  }

  Y1 <- .check_counts(stats::model.response(mf1), "1")
  Y2 <- .check_counts(stats::model.response(mf2), "2")

  # Terms from the REBUILT frames carry `predvars` fixed to the retained-sample
  # basis, so predict() re-evaluates stateful terms with the SAME basis the fit
  # used. Store these (not the original t1/t2) in the prediction metadata.
  tt1 <- attr(mf1, "terms"); tt2 <- attr(mf2, "terms")

  # Hard alignment invariant: the two equations MUST enter the likelihood on the
  # same rows. A violation here would silently mispair outcomes (and, in C++,
  # index past the shorter response), so assert before returning.
  if (!(length(Y1) == length(Y2) && nrow(X1) == nrow(X2) &&
        length(Y1) == nrow(X1) && length(off1) == nrow(X1) &&
        length(off2) == nrow(X2))) {
    stop("Internal error: the two equations did not align to a common row set ",
         "(y1=", length(Y1), ", y2=", length(Y2), ", X1=", nrow(X1),
         ", X2=", nrow(X2), "). Please report this.", call. = FALSE)
  }

  list(Y1 = Y1, Y2 = Y2, X1 = X1, X2 = X2,
       cn1 = colnames(X1), cn2 = colnames(X2),
       off1 = off1, off2 = off2,
       terms1 = tt1, terms2 = tt2,
       xlevels1 = stats::.getXlevels(tt1, mf1),
       xlevels2 = stats::.getXlevels(tt2, mf2),
       contrasts1 = attr(X1, "contrasts"),
       contrasts2 = attr(X2, "contrasts"),
       n = length(Y1), data = data_cc)
}

#' Bundle the per-equation prediction metadata from a `.prepare_bnb_data()`
#' result: training `terms`, factor `xlevels`, `contrasts`, and the fitted
#' offsets. Stored on the fit object as `predict_meta` so `predict()` can rebuild
#' column-stable newdata designs (absent factor levels reproduce the training
#' columns as zeros) and re-derive newdata offsets.
#' @keywords internal
#' @noRd
.prep_predict_meta <- function(prep) {
  list(terms1 = prep$terms1, terms2 = prep$terms2,
       xlevels1 = prep$xlevels1, xlevels2 = prep$xlevels2,
       contrasts1 = prep$contrasts1, contrasts2 = prep$contrasts2,
       off1 = prep$off1, off2 = prep$off2)
}

#' Stored training offset for one equation of a fit, or NULL when the fit has no
#' `predict_meta` (objects fit before offsets were tracked). Used by residual and
#' diagnostic code so those paths see the same offset the fit and `predict()` use.
#' @keywords internal
#' @noRd
.fit_offset <- function(fit, eq) {
  pm <- fit$predict_meta
  if (is.null(pm)) return(NULL)
  if (eq == 1L) pm$off1 else pm$off2
}

#' Coerce an optional offset (possibly NULL) to a plain length-n numeric vector.
#' `model.offset()` returns NULL when a formula has no offset; downstream code
#' treats a zero vector as "no offset", so normalize here.
#' @keywords internal
#' @noRd
.as_offset <- function(off, n) {
  if (is.null(off)) return(numeric(n))
  off <- as.numeric(off)
  if (length(off) != n) {
    stop("Offset length (", length(off), ") does not match the number of ",
         "observations (", n, ").", call. = FALSE)
  }
  if (any(!is.finite(off))) {
    stop("Offset contains non-finite values.", call. = FALSE)
  }
  off
}
