# Design: Poisson-limit fits for overdispersion LR tests

Date: 2026-07-19

## Motivation

The natural-scale summary leaves `z`/`p` blank for the NB2 dispersions `m1`,
`m2` (as it does for the random-coefficient SDs). Their null `m = 0` is the
**Poisson** limit and sits on the boundary of the parameter space, so the Wald
ratio does not test it. `lr_test()` already handles the boundary correction, but
it needs a *nested restricted fit*. For the SDs that restricted fit is trivial
(drop the random slope). For a dispersion the restricted model has a **Poisson
margin** (`m = 0`) instead of NB — and `fit_bnb`/`fit_rpbnb` currently have no
way to fit that. This spec adds it, so users can run an overdispersion LR test
for each margin.

## Key facts established during exploration

- **The NB2 likelihood converges smoothly and numerically stably to Poisson as
  `m -> 0`.** Measured max per-observation `|nb_logpmf - dpois|`: 4.0e-5 at
  `m = 1e-6`, 4.0e-6 at `1e-7`, 2.8e-7 at `1e-8`, with no overflow/cancellation
  (`R/famoye_core.R:nb_logpmf_y_mu_r` uses a cancellation-free form). Likewise
  `c_val(mu, m) = (1 + d*m*mu)^(-1/m) -> exp(-d*mu)` (error 2.7e-6 at `m = 1e-5`).
- **`maxLik` supports `fixed=`**: holding a named parameter at its start value,
  zeroing its vcov row/col, and setting the `logLik` `"df"` to the number of
  *free* parameters. It composes with this package's gradient-carrying `logLik`.

Together these mean the Poisson-limit fit needs **no new Poisson pmf and no C++
change**: pin `log_m_k` at `log(POISSON_M)` for a tiny `POISSON_M` and hold it
fixed, and the existing NB machinery evaluates the (numerically) Poisson model.

## Public interface

Add to both fitters:

```r
fit_bnb(...,  poisson_1 = FALSE, poisson_2 = FALSE)
fit_rpbnb(..., poisson_1 = FALSE, poisson_2 = FALSE)
```

- `poisson_k = TRUE` fits margin `k` at its Poisson limit: `log_m_k` is pinned at
  `log(POISSON_M)` and held fixed, so it is **not a free parameter**. The fit is
  a properly nested restriction of the corresponding NB fit (one fewer free
  parameter per pinned margin).
- Default `FALSE` reproduces today's behavior exactly (no pinning, no code-path
  change when both are `FALSE`).
- `POISSON_M` is an internal constant (`1e-6`; see "Numerical constant" below).

Intended use — overdispersion test for margin 1:

```r
full <- fit_rpbnb(f1, f2, data, random_1 = "hhninc", random_2 = "educ",
                  dependence = "famoye", draws = 500, seed = 20240712)
rest <- fit_rpbnb(f1, f2, data, random_1 = "hhninc", random_2 = "educ",
                  dependence = "famoye", draws = 500, seed = 20240712,
                  poisson_1 = TRUE)                 # margin 1 Poisson (m1 = 0)
lr_test(rest, full, boundary = TRUE)                # H0: m1 = 0 (no overdispersion)
```

## Scope

- **In scope:** the Famoye path of `fit_bnb` and `fit_rpbnb`, and the
  `fit_bnb` independence path (two NB2 margins). These are the paths that carry
  `log_m1`/`log_m2` and drive the demo.
- **Out of scope (error clearly):** the copula paths (`fit_bnb` /
  `fit_rpbnb` with a `copula()` dependence). `poisson_k = TRUE` combined with a
  copula raises an informative "not yet supported" error. Can be added later.

## Behavior / algorithm

### `fit_bnb` (famoye) and `fit_rpbnb` (famoye)

1. Validate: `poisson_k` logical scalars; error on `copula` dependence.
2. Build `par_names` / `start` as today. For each pinned margin set
   `start["log_m_k"] <- log(POISSON_M)`, collect the pinned names into
   `fixed_names`.
3. Pass `fixed = fixed_names` to the `maxLik::maxLik` call (both fitters already
   call `maxLik`). maxLik holds those coordinates constant during BFGS.
4. `npar <- <total> - length(fixed_names)` so `logLik` df, `AIC`, `BIC` count
   only free parameters. (`new_*_fit` computes AIC/BIC from `npar`.)
5. **Standard errors:** the C++/numeric score/Hessian are full-dimensional.
   Before inverting, drop the pinned rows/columns (subset to free indices), pass
   the free-only matrix + free names to `opg_vcov` / `.observed_info_vcov`, then
   scatter the resulting `se`/`vcov` back into full-size objects with `NA`
   (vcov: `NA`, or 0 row/col) for the pinned parameter. A pinned `m_k` reports
   **`NA` SE** in the natural-scale table (it is fixed, not estimated).
   - Applies to all three `fit_rpbnb` SE methods (opg, analytic, numeric) and
     the `fit_bnb` famoye Hessian path.
6. Everything downstream (`mu_hat`, frozen bounds, natural-scale `m_k =
   exp(log_m_k) ≈ POISSON_M`) works unchanged from the full-length `coef`.

### `fit_bnb` independence

The independence path fits each margin with `MASS::glm.nb`. For a Poisson
margin, fit that margin with `stats::glm(family = poisson)` instead: drop its
`log_m` from `par`/`npar`, take `logLik`/`vcov` from the Poisson GLM. (Exact
Poisson here — no limit needed — since the margins are independent.)

## Numerical constant

`POISSON_M = 1e-6`. Rationale:
- Approximation error to true Poisson: `<= 4e-5` per obs, so `<= ~0.16` total
  logLik at `n ≈ 3874`; the LR statistic is thus accurate to `< ~0.3`,
  negligible against overdispersion statistics (hundreds+) for these data.
- Well-conditioned: `r = 1/m = 1e6`, so `lgamma(y + r) - lgamma(r)` avoids the
  catastrophic cancellation that appears at `r = 1e8` for large `y`.
- Implementation will **empirically verify** on the real data that the restricted
  logLik is stable (agrees to `< 1` unit) across `POISSON_M in {1e-6, 1e-7}`;
  if not, revisit.

Documented as an m→0 numerical limit, not an exact Poisson reparameterization.

## Files

- `R/fit_bnb.R` — `poisson_1`/`poisson_2` args; pinning + fixed= in
  `fit_bnb_famoye`; Poisson-GLM branch in `fit_bnb_independence`; copula error;
  internal `POISSON_M` constant (or in `R/utilities.R`).
- `R/fit_rpbnb.R` — `poisson_1`/`poisson_2` args; pinning + `fixed=`; free-index
  SE subsetting across the opg/analytic/numeric branches; copula error.
- `R/utilities.R` — small helper `.free_index_vcov()` that subsets an
  info/score matrix to free indices, inverts via the existing helper, and
  scatters `se`/`vcov` back to full size with `NA` for pinned params (shared by
  both fitters to avoid duplicating the scatter logic).
- `man/*` — regenerated roxygen for the new args.
- `tests/testthat/test-poisson-overdispersion.R` — new (see below).
- `inst/fit_rpbnb_diff_famoye.R` — add m1/m2 overdispersion LR tests alongside
  the existing SD tests.

## Testing strategy (TDD)

Fast unit tests via `fit_bnb` (no MSL):
1. `poisson_1 = TRUE` drops one free parameter: `npar` and `attr(logLik, "df")`
   decrease by exactly 1 vs the full NB fit; `poisson_1 = poisson_2 = TRUE`
   drops 2.
2. The pinned `m_k` is `≈ POISSON_M` (`< 1e-5`) and its natural-scale SE is `NA`.
3. Default `FALSE` path is byte-for-byte unchanged (same coef/logLik as a fit
   without the new args) — guards against accidental behavior change.
4. `lr_test(poisson_fit, full_fit, boundary = TRUE)` on overdispersed simulated
   data yields `df = 1` and a small p-value; on Poisson-generated data the test
   does not reject at conventional levels (sanity of direction).
5. `poisson_k = TRUE` with a `copula()` dependence errors informatively.
6. Independence path: `fit_bnb(..., dependence = "independence", poisson_1 =
   TRUE)` matches a direct `glm(poisson)` margin logLik.

Slow tier (`skip_slow`): one `fit_rpbnb` full-vs-Poisson overdispersion LR test
end-to-end, plus the `POISSON_M` stability check across `{1e-6, 1e-7}`.
