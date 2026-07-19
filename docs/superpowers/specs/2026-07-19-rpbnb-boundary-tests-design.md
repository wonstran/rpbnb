# Design: `rpbnb_boundary_tests()` — merged boundary LR tests for rpbnb_fit

Date: 2026-07-19

## Motivation

The natural-scale summary of an `rpbnb_fit` leaves `z`/`p` blank for every
positive-scale/boundary parameter: the random-coefficient SDs (`sd1:*`, `sd2:*`)
and the NB2 dispersions (`m1`, `m2`). The valid test for each is a
boundary-corrected likelihood-ratio test against a nested restricted fit
([lr_test()] with `boundary = TRUE`). Running these by hand means constructing
one restricted `fit_rpbnb` call per parameter — tedious and, with **multiple
random coefficients per equation**, error-prone: to test `sd1:hhninc` when
`random_1 = c("hhninc", "age")` you must drop *only* `hhninc` and keep `age`
(a df-1 restriction), not drop the whole equation's random spec.

This adds one helper that builds and runs all those nested fits and merges the
results into a single table.

## Public interface

```r
rpbnb_boundary_tests(fit, data,
                     control = rpbnb_control(compute_se = FALSE),
                     which   = c("sd", "dispersion"))
```

- `fit` — a famoye `rpbnb_fit` (the full model; its `logLik`/`npar` are the
  full side of every LR test).
- `data` — the original data frame the model was fit on. Required: the fit
  object does not store the data, and every restricted model must be refit on
  the same data. A column/level mismatch surfaces as the usual `fit_rpbnb`
  preparation error.
- `control` — control for the restricted refits; default skips SEs
  (`compute_se = FALSE`) since `lr_test` needs only `logLik` + df. `draws`,
  `draw_type`, and `seed` are **not** taken from here — they are reused from
  `fit` so each comparison shares common random numbers.
- `which` — subset of `c("sd", "dispersion")` (both by default; `several.ok`).

Returns an object of class `rpbnb_boundary_tests`: a data frame with columns
`Parameter`, `LR`, `df`, `p.value`, `Signif`, and a `print` method. All rows use
the boundary correction (`df = 1` each, so the 50:50 mixture halves the naive
chi-square p-value).

## Behavior / algorithm

1. **Validate.** `fit` inherits `rpbnb_fit`; `data` is a data frame. Error if
   the fit is a copula fit (`!is.null(fit$cop_family)`): Poisson-limit margins
   are not supported there, so the dispersion tests can't be built. `which` via
   `match.arg(..., several.ok = TRUE)`.

2. **Reconstruct the full random specs** from the fit (no stored spec object):
   - `names_k = colnames(fit$X_k)[fit$rand_idx_k]`,
     `dist_k = fit$rp_meta$dist_k`, `sign_k = fit$rp_meta$sign_k`.
   - `build_spec(names, dist, sign)` → a named list
     `setNames(Map(\(d, s) list(dist = d, sign = s), dist, sign), names)`, or
     `NULL` when there are no random coefficients. This is the exact form
     `parse_rand_spec()` accepts, so a round-trip reproduces the full fit.

3. **Refit closure** reused for every restricted model:
   ```r
   refit <- function(spec1, spec2, poisson_1 = FALSE, poisson_2 = FALSE)
     fit_rpbnb(fit$formula_1, fit$formula_2, data = data,
               random_1 = spec1, random_2 = spec2,
               draws = fit$draws, draw_type = fit$draw_type, seed = fit$seed,
               control = control, dependence = "famoye",
               poisson_1 = poisson_1, poisson_2 = poisson_2)
   ```

4. **SD rows** (if `"sd"` in `which`), per equation, one coefficient at a time:
   for each `k`, refit with that coefficient removed from its equation's spec
   (the other equation keeps its full spec), `lr_test(rest, fit, boundary =
   TRUE)`. Label = `paste0(scale_label(dist_k[k]), eq, ":", names_k[k])` where
   `scale_label` is `rand_dist_registry[[dist]]$scale_label` (`sd`/`w`/`s`), so
   labels match the natural-scale table. Dropping the last remaining coefficient
   yields `spec = NULL` for that equation.

5. **Dispersion rows** (if `"dispersion"` in `which`): refit with `poisson_1 =
   TRUE` (full random specs retained) → `m1`; then `poisson_2 = TRUE` → `m2`.

6. **Assemble** the rows (SD eq1, SD eq2, then m1, m2) into the data frame,
   class it `rpbnb_boundary_tests`, return.

`print.rpbnb_boundary_tests` renders the table (4-digit `LR`/`p.value`, stars via
`signif_stars`) under a header, with the boundary-correction note.

## Cost

The helper performs `q1 + q2 + 2` full simulated-likelihood refits (one per
boundary parameter). This is inherent to nested LR testing; `compute_se = FALSE`
keeps each refit as cheap as possible.

## Scope (YAGNI)

- **rpbnb_fit, famoye only.** Copula fits error. `bnb_fit` (no random
  coefficients) is out of scope — its only boundary params are `m1`/`m2`, which
  a user can test directly with two `fit_bnb(..., poisson_k = TRUE)` calls.
- No parallelization across refits beyond whatever `control$n_cores` gives each
  fit; the refits run sequentially.

## Files

- `R/boundary_tests.R` — new: `rpbnb_boundary_tests()`, the `build_spec`
  reconstruction helper (internal), and `print.rpbnb_boundary_tests()`.
- `NAMESPACE` — export `rpbnb_boundary_tests`; `S3method(print,
  rpbnb_boundary_tests)` (roxygen).
- `man/*` — generated.
- `tests/testthat/test-boundary-tests.R` — new (below).
- `inst/fit_rpbnb_diff_famoye.R` — replace the four hand-written restricted
  fits with a single `rpbnb_boundary_tests(fit, data)` call (dogfood + shorter).

## Testing strategy (TDD)

Pure/fast unit tests:
1. `build_spec` round-trips: names/dist/sign in → `parse_rand_spec()` recovers
   the same names/dist/sign. Single and multiple coefficients; `NULL` on empty.
2. Input validation: non-`rpbnb_fit` errors; copula fit errors ("copula");
   invalid `which` errors.
3. Label construction matches the natural-scale table for normal/uniform/
   lognormal distributions.

Slow tier (`skip_slow`), the key multiple-random-coefficient correctness check:
4. Fit with `random_1 = c("x1", "x2")`, `random_2 = "x1"`; run
   `rpbnb_boundary_tests(fit, data)`. Assert rows are exactly
   `sd1:x1, sd1:x2, sd2:x1, m1, m2`; every `df == 1`; the `sd1:x1` restricted
   fit dropped only `x1` (verified by re-deriving: a fit with `random_1 = "x2"`
   has `npar == fit$npar - 1`, and its logLik matches the table's implied
   restricted logLik). All `p.value` in `[0, 1]`.
5. `which = "dispersion"` returns only `m1`, `m2`.
