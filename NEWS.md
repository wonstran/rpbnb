# rpbnb 0.2.0 (development)

* `fit_rpbnb()` and `simulate_rpbnb()`: per-coefficient random distributions.
  `random_1`/`random_2` now accept a named list whose values specify a
  distribution (`"normal"`, `"lognormal"`, `"uniform"`, `"triangular"`) or a
  list `list(dist = ..., sign = ...)` for sign-constrained lognormal. The
  previous character-vector interface (all-Normal) is fully preserved.
* `rpbnb_marginal_effects()` and `rpbnb_elasticities()`: interpretation for
  random-parameter (`rpbnb_fit`) models, built on the Monte-Carlo integrated
  population mean `E[exp(x'beta)]` (consistent with `predict.rpbnb_fit()`).
  Continuous marginal effects use the per-draw realized coefficient
  (`mean_r coef_rj * exp(lp_r)`); binary effects use the integrated discrete
  difference. Standard errors use a numeric delta method over each equation's
  mean and log-scale parameters. Mirrors the existing `bnb_marginal_effects()` /
  `bnb_elasticities()` for fixed-coefficient models.
* `rpbnb_marginal_effects()` and `rpbnb_elasticities()` gain an `n_cores`
  parameter (default `1`, sequential) that parallelizes the delta-method
  standard-error computation across a `parallel` cluster for `n_cores > 1`.
  The jacobian is decomposed into independent per-parameter columns dispatched
  via `parallel::parLapply()`; results are numerically identical to the
  sequential path (verified exactly, `tolerance = 0`, in
  `tests/testthat/test-rpbnb-interpretation.R`).
* Residual diagnostics: `residuals()` methods for `bnb_fit` and `rpbnb_fit`
  (randomized quantile residuals as the primary count-model residual, plus
  Pearson/deviance/response; `rpbnb_fit` uses the exact mixture predictive
  CDF/variance over the stored draws, and does not support deviance residuals);
  `plot()` methods drawing four base-graphics panels per margin
  (residuals-vs-fitted, normal QQ of the RQR, RQR histogram, scale-location);
  and `bnb_residual_checks()` reporting normality (Shapiro-Wilk / KS on the RQR),
  the NB2 dispersion statistic, the cross-margin residual correlation, an outlier
  list, and a composite misspecification verdict.

## Review fixes (2026-07-15 model review)

* **Famoye/Sarmanov lower `lambda` bound corrected** (model-validity fix): the
  admissible lower bound now uses `max((1-c1)(1-c2), c1*c2)`, including the
  `c1*c2` corner that dominates at low means. The previous bound admitted
  `lambda` values that made the joint pmf negative in the count tails. Fixed in
  both the R core and the C++ core. Existing negative-`lambda` estimates should
  be re-validated.
* `predict.rpbnb_fit()` now returns the integrated (population) mean
  `E[exp(x'beta)]`, distribution-aware for normal, uniform, triangular, and
  lognormal random coefficients, replacing a normal-only correction that ignored
  non-normal scales. `summary()` reports uniform/triangular and lognormal scale
  rows.
* Natural-scale `summary()` no longer attaches Wald p-values / significance
  stars to positive scale and dispersion parameters (the ratio did not test the
  boundary null); dependence parameters keep their tests.
* `copula()` validates native parameters (`|rho| < 1`, Clayton `theta > 0`,
  finite Frank `theta`); `fit_rpbnb()` rejects a `dependence` that is not
  `"famoye"` or a `copula()` object; the copula simulator validates dispersions,
  random names, and scales.
* `bnb_gof()` refits the correct copula null model and returns pseudo-R-squared
  values raw (no longer clamped to `[0, 1]`).
* Hessian repair is no longer silent: SE paths record curvature diagnostics on
  the fit as `$hessian_diag` and warn when the observed information is not
  positive definite.
* `fit_rpbnb()` numeric standard errors use the optimization draws (same-draw
  curvature); `draws_hessian` is retained but unused.
* `rpbnb_control(method=)` accepts only the implemented `"BFGS"`.
* Starting values: `start` may be positional or **named** (reordered to the
  canonical order; named partial starts merge into the defaults; unknown or
  duplicate names are rejected). With no user start, `fit_bnb(dependence =
  "famoye")` uses a **multi-start** policy — it optimizes from both a zero start
  and marginal `glm.nb` starts and keeps the best converged objective.

## Follow-up review fixes (2026-07-15 23:07 review)

* `predict.rpbnb_fit()` no longer errors on a one-row `newdata`.
* `predict.rpbnb_fit()` returns `Inf` (with a warning) where the population mean
  is analytically infinite (a lognormal random coefficient with sign × covariate
  > 0), and tags the output with `estimand`, `n_draws`, and `per_draw_cap`
  attributes.
* Random-coefficient scales must be finite and strictly positive (previously
  `Inf`, zero, and negative scales were accepted).
* `summary()$coefficients` (the raw table) also suppresses Wald p-values for the
  `log_sd`/`log_w`/`log_s`/`log_m` nuisance parameters.
* `bnb_gof()` requires the null model to converge, not merely to return a finite
  log-likelihood.
* `.superpowers/` is excluded from the source tarball.
* Validation studies archived under `inst/validation/`.

# rpbnb 0.1.0

* Initial release. Phase 1 deliverables:
  * `fit_bnb()` — bivariate NB with `independence` and `famoye` dependence.
  * `fit_rpbnb()` — bivariate random-parameter NB via maximum simulated likelihood
    (normal random coefficients, Halton draws, optional cluster parallelism).
  * `simulate_rpbnb()` — normal random-coefficient data simulator.
  * `rpbnb_control()` — estimation control object.
  * Diagnostics: `bnb_gof()`, `bnb_marginal_effects()`, `bnb_elasticities()`.
  * S3 methods: `print`, `summary`, `coef`, `vcov`, `logLik`, `AIC`, `BIC`, `predict`.
  * Ported and validated against the original `Rcodes` scripts (preserved in
    `inst/legacy/`).
