# Design: `lr_test()` — likelihood-ratio comparison of two nested fits

Date: 2026-07-18

## Motivation

The natural-scale summary of `bnb_fit` / `rpbnb_fit` objects reports `z` and `p`
as `NA`/NaN for the positive-scale parameters — random-coefficient SDs
(`sd1:`, `sd2:`) and NB2 dispersions (`m1`, `m2`):

```
Natural-scale dispersion / dependence (delta-method SE):
           Parameter Estimate StdErr   z  p Signif
          sd1:hhninc   0.1113 0.0164  NA NA
            sd2:educ   0.0951 0.0168  NA NA
     m1 (dispersion)   2.0728 0.0787  NA NA
     m2 (dispersion)   3.2946 1.2882  NA NA
```

This is **intentional** (commit `24710fb`): each of these is a positive
parameter `a = exp(eta)` estimated on a log scale, so the Wald ratio
`z = a / SE(a)` reduces to `1/SE(eta)` and does *not* test the natural null
`a = 0` — that null is `eta = -Inf`, a boundary outside the interior
parameterization the delta method assumes. Reporting a Wald z/p there would be
statistically meaningless, so the code suppresses them.

The statistically correct test for these boundary parameters is a
**likelihood-ratio test** comparing the full fit against a restricted fit that
removes the term. This spec adds a helper for exactly that comparison.

## Public interface

```r
lr_test(restricted, full, boundary = FALSE)
```

- `restricted`, `full` — fitted model objects (`bnb_fit` or `rpbnb_fit`, any
  mix). Both expose a `logLik()` method whose result carries a `"df"` attribute
  equal to the number of estimated parameters (`object$npar`), so the helper is
  distribution/estimator agnostic.
- `boundary` — logical, default `FALSE`.
  - `FALSE`: standard interior chi-square test,
    `p = pchisq(stat, df, lower.tail = FALSE)`. Correct for ordinary
    restrictions (e.g. dropping a fixed covariate).
  - `TRUE`: 50:50 mixture of `chisq(df)` and `chisq(df - 1)` (Self & Liang
    1987), the correct null when the restriction pins a variance/dispersion-type
    parameter (a random-coefficient SD, or NB2 dispersion `m`) to its zero
    boundary. For `df = 1` this halves the naive p-value. This is the mode to
    use for the `sd*`/`m*` parameters that motivated the helper.

Returns an object of class `rpbnb_lrtest` (a list) with a `print` method.

## Behavior / algorithm

1. Extract `ll_f = as.numeric(logLik(full))`, `ll_r = as.numeric(logLik(restricted))`,
   and `df_f`, `df_r` from the `"df"` attribute of each `logLik`.
2. `df = df_f - df_r`. Error if `df <= 0` (`full` must have strictly more
   parameters than `restricted`; a non-positive df means the arguments are
   swapped or the models are not nested as expected).
3. `stat = 2 * (ll_f - ll_r)`. If `stat < 0` (restricted fit has the higher
   log-likelihood — a sign of a convergence problem or mis-specified nesting),
   warn and clamp `stat` to 0 (so the reported p-value is a conservative 1
   rather than a nonsensical value from a negative statistic).
4. p-value:
   - `boundary = FALSE`: `pchisq(stat, df, lower.tail = FALSE)`.
   - `boundary = TRUE`: `0.5 * pchisq(stat, df, lower.tail = FALSE) +
     0.5 * pchisq(stat, df - 1, lower.tail = FALSE)`, with the `df - 1 == 0`
     term contributing a point mass at 0 (`pchisq(stat, 0, ...)` is 0 for
     `stat > 0`, so the mixture correctly reduces to `0.5 * pchisq(stat, 1)`
     when `df = 1`).
5. Return `structure(list(statistic, df, p.value, logLik_full, logLik_restricted,
   df_full, df_restricted, boundary), class = "rpbnb_lrtest")`.

## Print method

`print.rpbnb_lrtest` shows a compact block:

```
Likelihood-ratio test
  full model:       logLik = -1234.56  (df = 12)
  restricted model: logLik = -1240.11  (df = 10)
  --------------------------------------------------
  LR statistic = 11.10  on 2 df   p = 0.0039  **
  (boundary-corrected 50:50 chi-square mixture)   # only when boundary = TRUE
```

Significance stars reuse the existing `signif_stars()` helper for consistency
with the coefficient tables.

## Scope boundaries (YAGNI)

- **No auto-refit.** The user constructs `restricted` themselves by calling
  `fit_rpbnb` / `fit_bnb` with the term removed (drop the name from `random_1`,
  or use a plain-NB / independence fit for the `m = 0` case). This keeps the
  helper small and avoids silently guessing which parameter to drop or rewriting
  the original call.
- **No `anova()` S3 method.** A single exported function is clearer than
  overloading `anova` for a two-model comparison; can be added later if wanted.
- **No multi-parameter boundary geometry.** The 50:50 mixture is exact only for
  a single parameter on the boundary (`df` interior + up to 1 boundary). For
  restrictions that place several parameters on the boundary simultaneously the
  correct weights are more complex; that is out of scope. The `df = 1` boundary
  case (one SD or one dispersion) is the motivating and correct use.

## Files

- `R/lr_test.R` — new: `lr_test()` + `print.rpbnb_lrtest()`.
- `NAMESPACE` — export `lr_test`, register `S3method(print, rpbnb_lrtest)`
  (via roxygen `@export` / `#' @export`).
- `tests/testthat/test-lr-test.R` — new:
  - unit tests on the chi-square math (interior vs. boundary p-values, the
    `df = 1` halving, the negative-statistic clamp+warning, the `df <= 0`
    error);
  - an integration test fitting a small `rpbnb_fit` with vs. without a random
    coefficient and checking a sane LR result;
  - a `print` snapshot.
- `man/lr_test.Rd` — generated by roxygen.

## Testing strategy

TDD: write the failing unit tests for the pure math first (they need only a
tiny stub object carrying `logLik`/`df`, or a hand-built `logLik` structure),
implement `lr_test()`, then add the slower integration test behind the existing
slow-test helper if it exceeds the fast-suite budget.
