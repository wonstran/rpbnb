# rpbnb — Phase 1 Design (Port + Packaging)

**Date:** 2026-06-13
**Status:** Approved (brainstorming) — pending spec review
**Parent scope:** `docs/scope_rpnbn.md`

## 1. Purpose and Phasing

The full scope (`docs/scope_rpnbn.md`) is too large for one design/implementation
pass. It is decomposed into phases, each with its own spec → plan → implementation
cycle. **This document covers Phase 1 only.**

**Phase 1 goal:** turn the two mature existing scripts into a clean, installable,
tested R package named **`rpbnb`**, *without changing the underlying mathematics*.

The two mature sources being ported:

- `Rcodes/bnbr_v2-4.R` — Famoye/Sarmanov bivariate NB (NB2) MLE with analytic
  gradient, numeric Hessian (frozen λ-bounds), GOF, marginal effects, elasticities.
- `Rcodes/rpbnbr_faymore.R` — bivariate random-parameter BNB via maximum simulated
  likelihood (Halton draws, log-sum-exp, optional cluster parallelism).

### Decisions locked during brainstorming

| Decision | Choice |
|---|---|
| Sequencing | Port mature code first (this phase) |
| Package name | `rpbnb` |
| Bayesian (Stan/JAGS) | Excluded from Phase 1; deferred to a later phase |
| Simulator in Phase 1 | Yes — minimal normal-random-coefficient `simulate_rpbnb()` + one recovery test |
| `dependence = "independence"` | Yes — included in Phase 1 `fit_bnb()` |
| Parallelism | Sequential default; preserve existing optional cluster path as-is; `future` backend deferred |
| `fit_rpbnb()` interface | **Bivariate** (match existing code), reconcile with scope's univariate signature in a later phase |

## 2. Out of Scope for Phase 1

Each item below becomes a later phase with its own spec:

- Copula dependence (Gaussian / Clayton / Frank / Gumbel) and the copula rectangle
  probability.
- Formal Famoye/Sarmanov mathematical documentation distinguishing variants.
- Documentation deliverables: `literature_review.md`, `code_review.md`,
  `model_formulation.md`, `user_guide.md`.
- Additional random-coefficient distributions (lognormal, triangular, uniform,
  truncated normal) and correlated random coefficients.
- Full Monte Carlo driver and `results/bnb/`, `results/rpbnb/` outputs.
- `future`/`future.apply` cross-platform backend and `set_rpbnb_threads()`.
- Rcpp / RcppArmadillo / OpenMP compiled likelihood.
- Bayesian backends (Stan / JAGS).
- YAML configuration system and config validation.
- Continuous-integration matrix (Windows + Ubuntu + multiple R versions).
- Univariate single-formula `fit_rpbnb()` signature from the scope text.

Phase 1 must not introduce Windows-only code, so the deferred Linux/CI phase stays
cheap. Cross-platform path handling (`file.path`, `system.file`) is used from the start.

## 3. Statistical Specification (unchanged from existing code)

**Parameterization:** NB2. For a margin with mean `μ` and dispersion `m`, the size is
`r = 1/m`, and

```
P(Y = y) = Γ(y + r) / (Γ(r) y!) · p^r · (1 − p)^y,   p = r / (r + μ)
```

implemented on the log scale via `lgamma` and `log1p`.

**Famoye/Sarmanov dependence.** With `d = 1 − e⁻¹` and
`c(μ, m) = (1 + d·m·μ)^(−1/m) = E[e^{−Y}]`, the joint pmf is

```
P(Y1=y1, Y2=y2) = f1(y1) · f2(y2) · [1 + λ (e^{−y1} − c1)(e^{−y2} − c2)]
```

`λ` is constrained to keep the bracket positive. Data-adaptive global bounds:

```
λ_min = max_i [ −1 / ((1 − c1_i)(1 − c2_i)) ]
λ_max = min_i [  1 / max(c1_i(1 − c2_i), c2_i(1 − c1_i)) ]
```

estimated through an interior logistic map of an unconstrained `z_lambda`:

```
λ = λ_lo + (λ_hi − λ_lo) · (ε + (1 − 2ε)·plogis(z_lambda)),   ε = 1e-6
```

**BNB parameter vector:** `[β1 (p1), β2 (p2), log_m1, log_m2, z_lambda]`.

**RP-BNB parameter vector:**
`[β1 (k1), β2 (k2), log_sd1 (q1), log_sd2 (q2), log_m1, log_m2, z_lambda]`,
where random coefficients are `β_random,i = β_mean + sd · z_draw`, `z_draw` standard
normal (Halton). Conditional mean per draw `μ_ir = exp(Xβ + XR·(sd⊙z_r))`. The
simulated log-likelihood averages over draws with `rowLogSumExp`.

**Independence dependence** = two univariate NB2 MLEs (equivalently `λ = 0`).

**Estimation:** `maxLik` + BFGS with analytic gradient; numeric Hessian
(`numDeriv::hessian`) computed with λ-bounds **frozen at the optimum** for stable SEs;
information matrix symmetrized; ridge or `MASS::ginv` fallback when non-positive-definite.
Positive parameters (`m`, `sd`) estimated on the log scale; SEs back-transformed by the
delta method for display. The same Halton draws are retained across the optimization for
a smooth, reproducible simulated likelihood; a smaller independent draw set is used for
the Hessian.

## 4. Package Architecture

```
rpbnb/
├── DESCRIPTION
├── NAMESPACE                # roxygen2-generated
├── LICENSE
├── README.md
├── NEWS.md
├── R/
│   ├── famoye_core.R        # INTERNAL shared math (single source of truth)
│   ├── control.R            # rpbnb_control()
│   ├── simulation_draws.R   # halton_normal() draws
│   ├── fit_bnb.R            # fit_bnb(): independence | famoye
│   ├── fit_rpbnb.R          # fit_rpbnb(): bivariate RP-BNB (SIML)
│   ├── simulate_rpbnb.R     # simulate_rpbnb(): normal RC simulator
│   ├── diagnostics.R        # bnb_gof(), bnb_marginal_effects(), bnb_elasticities()
│   ├── methods.R            # S3 methods for bnb_fit / rpbnb_fit
│   └── utilities.R          # rowLogSumExp, pretty-print, LL-trace helpers
├── man/                     # generated
├── tests/testthat/
├── vignettes/rpbnb-intro.Rmd
├── inst/
│   ├── extdata/rwm1984_clean.csv
│   └── legacy/              # untouched copies of original Rcodes scripts
├── data-raw/
└── docs/
```

**De-duplication is the key internal change.** Both scripts currently define their own
copies of `d_const`, `c_val`, `nb_logpmf_y_mu_r`, `lambda_bounds_vec`, `dct_dm`,
`dc_dbeta_mat`. Phase 1 hoists these into `R/famoye_core.R` as internal functions that
both estimators and the test suite call. No mathematical change — byte-for-byte
equivalent formulas, validated by the reproduction tests.

**Isolation / boundaries.** Each unit has one purpose and a documented interface:
`famoye_core.R` knows only math (no I/O, no optimizer); `fit_*` build design matrices,
drive `maxLik`, and assemble result objects; `methods.R` only reads result objects;
`diagnostics.R` consume a fitted object. A consumer can use `fit_bnb()` without reading
the core internals; core internals can change without breaking consumers as long as the
documented signatures hold.

## 5. User-Facing API (Phase 1 subset)

```r
fit_bnb(
  formula_1, formula_2, data,
  dependence = c("independence", "famoye"),
  start = NULL,
  control = rpbnb_control()
)                                   # -> class "bnb_fit"

fit_rpbnb(
  formula_1, formula_2, data,
  random_1 = NULL, random_2 = NULL, # coefficient names to make random, per equation
  draws = 400,
  draw_type = "halton",
  seed = 1234,
  start = NULL,
  control = rpbnb_control()
)                                   # -> class "rpbnb_fit"

simulate_rpbnb(
  n,
  beta1, beta2,                     # fixed-coefficient means per equation (named)
  random_1 = NULL, random_2 = NULL, # list(name = list(sd = ...)) normal RCs
  dispersion = c(m1 = 0.5, m2 = 0.5),
  lambda = 0,
  covariates = NULL,                # optional supplied X; else generated
  seed = 1234
)                                   # -> list(data, coef_realized, mu, true, settings, meta)

rpbnb_control(
  method = "BFGS",
  iterlim = 300, reltol = 1e-8, print_level = 0,
  draws_hessian = 100,
  halton_burn = 300, halton_skip = 100,
  n_cores = 1L,
  compute_se = TRUE,
  hess_eps = 1e-5, hess_r = 4
)

# S3 methods on bnb_fit and rpbnb_fit:
print(), summary(), coef(), vcov(), logLik(), AIC(), BIC(), predict()

# Diagnostics (exported functions):
bnb_gof(fit, ...)
bnb_marginal_effects(fit, which, type = c("AME","MEM"), vars = NULL, ...)
bnb_elasticities(fit, which, type = c("AME","MEM"), vars = NULL, ...)
```

- `draw_type` accepts `"halton"` in Phase 1; the argument exists so later phases can add
  `"sobol"` / `"random"` / antithetic without an API break.
- `predict()` returns predicted conditional means (`μ1`, `μ2`); for `rpbnb_fit` it
  returns the unconditional (draw-averaged) mean by default with an option for
  conditional means. `type = "response"` default.
- `n_cores > 1` preserves the existing `parallel::makeCluster` path unchanged.
- Diagnostics (`bnb_gof`, `bnb_marginal_effects`, `bnb_elasticities`) target `bnb_fit`
  in Phase 1, ported directly from `bnbr_v2-4.R`. Marginal effects/elasticities for
  `rpbnb_fit` (which require integrating over the random-coefficient distribution) are
  deferred to a later phase; `bnb_gof` (LL/AIC/BIC) works for both.

## 6. Result Object Contract

Both fit classes expose, at minimum (satisfying scope §6F.14):

- `coef` (named estimates; dispersion and λ on natural scale for display, plus the raw
  unconstrained vector for `vcov` alignment),
- `vcov` (covariance matrix on the estimation scale),
- `logLik`, `AIC`, `BIC`, `npar`, `nobs`,
- `convergence` diagnostics (optimizer code/message, gradient norm, iterations),
- design matrices and responses (`X1`, `X2`, `Y1`, `Y2`) for diagnostics/prediction,
- fitted means and the LL trace,
- metadata: formulas, dependence type, draws/draw_type/seed (RP), R/package versions,
  timestamp passed in, optimizer settings (scope §7.3).

## 7. Numerical Stability Requirements (scope §7.2)

Carried over from the existing code and asserted in tests where practical: log-scale
pmf with `lgamma`/`log1p`; `pmax(dep, 1e-300)` floor on the dependence factor;
log-sum-exp for the simulated likelihood; symmetrized information matrix; ridge/`ginv`
fallback for singular Hessians; finite-value guard returning a large negative objective
with zero gradient when λ-bounds collapse.

## 8. Testing Plan (scope §8)

`testthat`:

1. **Core math** — `nb_logpmf_y_mu_r` vs `dnbinom(..., size = r, mu, log = TRUE)`;
   `c_val` vs a direct expectation on a truncated grid; `lambda_bounds_vec` validity.
2. **Independence limit** — Famoye with `λ → 0` (via `z_lambda → −∞` mapping check and a
   fitted comparison) matches two univariate NB2 fits within tolerance.
3. **BNB reproduction** — `fit_bnb(dependence="famoye")` on rwm1984 (`docvis`, `hospvis`)
   reproduces the legacy `bnbr_v2-4.R` estimates within tolerance.
4. **Simulator** — `simulate_rpbnb` reproducibility (same seed ⇒ identical data); realized
   random coefficients match requested mean/sd; counts show overdispersion when `m > 0`.
5. **RP-BNB recovery** — `fit_rpbnb` on one small simulated dataset recovers true β, sd, m
   within Monte Carlo tolerance.
6. **Degenerate RP** — `sd → 0` reduces RP-BNB toward the fixed-parameter BNB.
7. **Input validation** — bad formulas, negative counts, unknown random names, NA handling,
   offset handling produce informative errors.
8. **S3 methods** — `coef/vcov/logLik/predict` dimensions and names are consistent.

Critical math functions are checked against an independent base-R implementation
(`dnbinom`) per scope §8.

## 9. Acceptance Criteria (Phase 1)

1. `R CMD check` (via `devtools::check()`) passes with no errors and no avoidable
   warnings on Windows.
2. Package installs from source.
3. All exported functions have roxygen2 docs with runnable `@examples`.
4. The intro vignette builds and runs end-to-end on rwm1984.
5. All `testthat` tests pass.
6. `fit_bnb` reproduces legacy BNB estimates; `fit_rpbnb` recovers simulated parameters.
7. Fixed vs random coefficients are selectable through function arguments without editing
   core source (scope §6F.13).
8. Fit objects provide coefficients, covariance, likelihood, convergence info, predictions,
   and metadata (scope §6F.14).
9. Legacy scripts preserved untouched under `inst/legacy/`.

## 10. Dependencies (Phase 1)

- **Imports:** `stats`, `maxLik`, `numDeriv`, `randtoolbox`, `MASS`.
- **Suggests:** `testthat`, `knitr`, `rmarkdown`, `parallel` (base; for the optional
  cluster path), `roxygen2`, `devtools`.

Minimal and justified per scope §6E; copula/future/Rcpp/yaml dependencies are added only
in the phases that need them.
