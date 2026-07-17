# Multithreading for RP marginal effects & elasticities — design

**Date:** 2026-07-17
**Status:** Approved (brainstorming)
**Scope:** Parallelize the standard-error computation in `rpbnb_marginal_effects()`
and `rpbnb_elasticities()` (both in `R/diagnostics.R`) across an R-level cluster.

## Problem

`.rp_diag_one()` computes standard errors via `numDeriv::jacobian()` on the
vector-valued estimand `.rp_estimand()`. numDeriv's default (Richardson) method
re-evaluates the estimand roughly 8 times per parameter for extrapolated
accuracy, so for an equation with `npar` delta-method parameters (mean
coefficients + log-scale parameters), the SE step costs on the order of
`8 * npar` evaluations of `.rp_estimand`, each doing `O(n_obs * R_draws *
n_selected_vars)` matrix work. This dominates the wall-clock cost of the
interpretation functions for models with many covariates and/or many
simulation draws; the point-estimate step is a single evaluation and is not
the bottleneck.

## Decision: R-level parallel jacobian (not a C++ kernel)

Two mechanisms were considered:
- **R-level `parallel` cluster** (chosen): distribute the jacobian's
  per-parameter finite-difference evaluations across worker processes. No new
  compiled code; directly targets the dominant cost (confirmed by profiling
  the evaluation count above).
- **C++/OpenMP kernel**: port `.rp_g_matrix`/`.rp_estimand`'s per-draw loop
  into a compiled, OpenMP-parallel function (mirroring `src/halton_parallel.cpp`).
  Bigger engineering lift, and by itself does not parallelize the jacobian's
  repeated evaluations (would need to be combined with R-level jacobian
  parallelism anyway for the SE cost) — deferred.

## Mathematical decomposition (verified exact)

`numDeriv::jacobian(f, theta_hat)` for a vector-valued `f: R^npar -> R^nvars`
returns an `nvars x npar` matrix, computed column-by-column: column `k` is the
Richardson-extrapolated finite-difference derivative of `f` with respect to
`theta[k]`, with all other parameters held at `theta_hat`. This decomposes
exactly into `npar` independent 1-D jacobian calls:

```r
jac_col <- function(k, theta_hat, theta_names, meta, quantity) {
  fk <- function(tk) {
    t <- theta_hat; t[[k]] <- tk
    names(t) <- theta_names
    .rp_estimand(t, meta, quantity, mark_inf = FALSE)   # bare call -- see below
  }
  numDeriv::jacobian(fk, theta_hat[[k]])   # nvars x 1
}
G <- do.call(cbind, lapply(seq_along(theta_hat), jac_col, ...))
```

This was verified empirically (scratch probe against `make_rp_fixture()`) to
be **bit-identical** (max abs diff = 0) to the current single-call
`numDeriv::jacobian(f_vector, theta_hat)`. It is a pure reshuffling of the
same computation, not a numerical approximation change. Each `jac_col(k, ...)`
call is independent of every other `k` and is the unit of parallelism.

## Worker resolution: bare calls + `clusterExport`, verified empirically

This project's dev workflow loads the package with `pkgload::load_all()`
(see `tools/test-tiers.R`, the `inst/*.R` demo scripts), not a formal
`R CMD INSTALL`. A `parallel::makeCluster()` PSOCK worker starts as a bare R
process with no access to that in-memory dev-loaded namespace. Two consequences,
both confirmed by direct experiment against this codebase before writing this
spec:

1. A **namespace-qualified** call inside a worker-dispatched closure
   (`rpbnb:::.rp_estimand(...)`) fails on the worker (`object '.rp_estimand'
   not found` initially, or worse if it silently tried to load a
   nonexistent-on-disk namespace) — `:::` requires the `rpbnb` namespace to be
   loadable on that process, and it is not.
2. A **bare** call (`.rp_estimand(...)`) combined with
   `parallel::clusterExport(cl, c(".rp_estimand", ...), envir = environment())`
   called from inside the package **does** resolve correctly on the worker
   and reproduces the sequential result exactly. This is exactly the pattern
   already used in `R/fit_rpbnb.R`'s R-fallback parallel path (see its
   comment: "workers run in a fresh globalenv without the rpbnb namespace, so
   all helpers (and their callees) must be exported by name").

`numDeriv::jacobian` is called via `numDeriv::` (not bare) inside `jac_col`
and needs no export: `numDeriv` is a properly installed package, resolvable
via `::` on demand on any worker regardless of `rpbnb`'s load state — this
was also confirmed in the same experiment.

**Required export list** (transitively reachable from `.rp_estimand`):
`.rp_estimand`, `.rp_g_matrix`, `.rp_inf_rows`, `rand_realize`,
`rand_dist_registry`.

## API

New trailing parameter on both exported functions:

```r
rpbnb_marginal_effects(fit, which = c("y1","y2","both","all"),
                       type = c("AME","MEM"), vars = NULL,
                       include_intercept = FALSE, digits = 4,
                       print_output = TRUE, n_cores = 1L)

rpbnb_elasticities(fit, which = c("y1","y2","both"),
                   type = c("AME","MEM"), vars = NULL,
                   include_intercept = FALSE, digits = 4,
                   print_output = TRUE, n_cores = 1L)
```

- `n_cores = 1L` (default): **unchanged** existing sequential code path — the
  current single-call `numDeriv::jacobian(f_vector, theta_hat)` inside
  `.rp_diag_one()` is left exactly as-is. Zero behavior change, zero risk, for
  every existing caller (including the two demo scripts and the full test
  suite as they stand today).
- `n_cores > 1`: the exported function builds ONE `parallel::makeCluster(n_cores)`
  (PSOCK), `clusterExport`s the fixed helper list above, and passes the cluster
  handle `cl` down through `.rp_diag_one(..., cl = cl)` for every equation it
  computes (`which = "both"/"all"` reuses the same cluster across y1 and y2,
  rather than paying cluster-startup cost twice). The cluster is torn down via
  `on.exit(parallel::stopCluster(cl))` at the top-level exported function.
- `.rp_diag_one()` gains a `cl = NULL` parameter. When `cl` is non-NULL, the SE
  step builds `G` via `do.call(cbind, parallel::parLapply(cl, seq_along(theta_hat),
  .rp_jac_col, theta_hat = theta_hat, theta_names = theta_names, meta = meta,
  quantity = quantity))` instead of the single `numDeriv::jacobian(...)` call.
  `theta_hat`/`meta`/`quantity` are passed as explicit `parLapply(...)` extra
  args (not captured via closure), avoiding any dependence on environment
  serialization.
- New internal helper `.rp_jac_col(k, theta_hat, theta_names, meta, quantity)`
  in `R/diagnostics.R`, implementing the decomposition above.
- If `n_cores > 1` and the `parallel` package is not installed: warn (matching
  the existing message style in `R/fit_rpbnb.R`: `"Package 'parallel' not
  available; running sequentially."`) and fall back to the `n_cores = 1` path.

## Testing

- **Fast tier, fixture-based** (`make_rp_fixture()`): an exact-equivalence test
  — `n_cores = 1` vs `n_cores = 3` (or similar) on both `rpbnb_marginal_effects`
  and `rpbnb_elasticities` must produce **identical** `Estimate` and `StdErr`
  columns (`tolerance = 0`), proving the parallel path is a pure reshuffling.
  This spins up a real cluster (~1-2s overhead observed in the verification
  probe) but is fixture-based (no model fitting), so it belongs in the fast
  tier alongside the rest of `test-rpbnb-interpretation.R`.
- A `parallel`-unavailable fallback path is not independently testable without
  mocking `requireNamespace`; rely on code inspection matching the established
  `fit_rpbnb.R` pattern (no new test needed beyond what covers the
  `n_cores = 1` default already).
- No change needed to the `NA`-vcov or lognormal-`Inf` tests: those paths are
  independent of `n_cores` (the `NA`-vcov path returns before building `G` at
  all; the `Inf`-marking step operates on `mark_inf = TRUE`, outside the
  jacobian).

## Demo scripts

Update the Step 7 calls added earlier in `inst/fit_rpbnb_complex.R` and
`inst/fit_rpbnb_copula_complex.R` to pass `n_cores = rpbnb_threads()`,
demonstrating the new parameter on the same real fits already used to
validate the interpretation functions.

## Out of scope

- A C++/OpenMP kernel for `.rp_g_matrix`/`.rp_estimand` itself (deferred; see
  Decision section).
- Parallelizing the point-estimate step (`mark_inf = TRUE` single evaluation)
  — not the bottleneck.
- Persisting/reusing a cluster across multiple `rpbnb_marginal_effects()` /
  `rpbnb_elasticities()` calls (each call builds and tears down its own
  cluster; out of scope for this pass, consistent with `fit_rpbnb.R`'s own
  per-call cluster lifecycle).
