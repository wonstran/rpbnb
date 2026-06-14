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
#' Subsets `data` to the rows that are complete across the union of variables in
#' BOTH formulas, so the two outcomes and design matrices share the same rows.
#' This prevents pairing outcomes from different rows when missing values fall in
#' different columns.
#'
#' @return A list with `Y1`, `Y2` (integer), `X1`, `X2`, `cn1`, `cn2`, `n`, and
#'   `data` (the complete-case subset, for downstream refits such as glm.nb).
#' @keywords internal
#' @noRd
.prepare_bnb_data <- function(formula_1, formula_2, data) {
  if (!is.data.frame(data)) stop("`data` must be a data frame.", call. = FALSE)
  # Expand the formulas against the data so a "." RHS resolves to the actual
  # columns before collecting variable names.
  vars <- unique(c(all.vars(stats::terms(formula_1, data = data)),
                   all.vars(stats::terms(formula_2, data = data))))
  missing_vars <- vars[!vars %in% names(data)]
  if (length(missing_vars)) {
    stop("Variable(s) not found in data: ",
         paste(missing_vars, collapse = ", "), ".", call. = FALSE)
  }
  ok <- stats::complete.cases(data[, vars, drop = FALSE])
  if (!any(ok)) {
    stop("No complete cases across both formulas.", call. = FALSE)
  }
  data_cc <- data[ok, , drop = FALSE]

  mf1 <- stats::model.frame(formula_1, data = data_cc)
  mf2 <- stats::model.frame(formula_2, data = data_cc)
  Y1 <- .check_counts(stats::model.response(mf1), "1")
  Y2 <- .check_counts(stats::model.response(mf2), "2")
  X1 <- stats::model.matrix(formula_1, mf1)
  X2 <- stats::model.matrix(formula_2, mf2)

  list(Y1 = Y1, Y2 = Y2, X1 = X1, X2 = X2,
       cn1 = colnames(X1), cn2 = colnames(X2),
       n = length(Y1), data = data_cc)
}
