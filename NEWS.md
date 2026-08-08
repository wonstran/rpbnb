# rpbnb 0.4.0

The `rpbnb.tmb` package (v0.3.5, git `c64a2ec`) has been merged into `rpbnb`.
That source tree is now superseded; everything it provided is available here.

## Review fixes (2026-08-08 07:55 response review)

See `comments/response_2026-08-08-08-07-17.md`.

* **The Famoye admissible interval is now the random coefficients' support bound
  for BOTH estimators** (model-validity fix), superseding the estimator-specific
  split introduced earlier the same day. Deriving it from the finite Halton grid
  for `method = "sml"` confused a quadrature rule with the model's support: the
  mixing distributions stay continuous under both estimators, `?fit_rpbnb_tmb`
  documents them as different approximations to the *same* integral, and the
  grid bound is monotone in the draw count — `[-2.550, 3.131]` at 5 draws,
  `[-2.142, 2.753]` at 400, `[-1.914, 2.261]` at 50,000, converging on the
  support bound `[-1, 1]`. A constraint set that depends on the number of draws
  is not a parameter space, and the latent neighbourhoods a finite grid misses
  carry positive probability.
* **Bounded random-coefficient supports are propagated rather than approximated
  by `(0, 1)`.** Uniform and triangular deviations live in `(-s, s)`, so forcing
  `(0, 1)` on them would be *over-strict* — rejecting admissible fits — not
  merely conservative. Per-distribution deviation supports are now used:
  unbounded for normal, one-sided for lognormal, `(-s, s)` for uniform and
  triangular.
* **The R-side constraint and the TMB objective now share one parameterization.**
  Variation is determined from the declared random coefficient and its design
  loading rather than from `exp(log_scale) > 0`: `exp(-1000)` underflows to `0`
  in R while the template clamps `log_sd` to `-20` and keeps
  `exp(-20) = 2.06e-9`, so a legal start dropped a random effect the objective
  retained — and a dropped effect *widens* the interval, admitting
  `lambda = 2`. Scales, dispersions and the `eta` clamp now mirror
  `src/rpbnb_tmb.cpp` exactly.
* **`BASELINE_SML_LOGLIK` re-captured**, from `-949.6478422037` to
  `-949.6478374514` (4.75e-06 nats). `lamLo`/`lamHi` are inputs to the tape, so
  changing the bound notion moves the objective; the tape itself is unchanged.
  `lambda_bounds` is now pinned separately in the same test, so a tape
  regression (log-likelihood moves, bounds do not) stays distinguishable from a
  deliberate bound change (both move).

## Review fixes (2026-08-08 07:36 response review)

See `comments/response_2026-08-08-07-36-52.md`.

* **The Famoye admissible interval is now estimator-specific** (model-validity
  fix). The interval was always derived from the finite Halton grid, which is
  the right notion for `method = "sml"` — whose likelihood really is an average
  over those draws — and the wrong one for `method = "laplace"`, which never
  evaluates that grid and instead integrates per-observation latent `u1`/`u2`.
  The grid bound is far too wide there: with one normal random coefficient per
  margin, `sd = 0.2`, `m = 0.5`, 400 draws give `[-2.142, 2.753]` while the
  latent support admits only `[-1, 1]`.

  A new latent-support bound is computed exactly rather than sampled: both
  quantities bounded by `lambda_bounds_vec()` are pointwise maxima of bilinear
  functions of `(c1, c2)`, so their suprema over the attainable rectangle sit at
  its corners. It is used for **both** the frozen bounds handed to the template
  and the post-fit re-check, so a Laplace Famoye fit is judged against the
  constraint it was optimized under — previously the objective itself was
  constrained by the too-wide grid bound.

  Consequence: when both margins carry a varying random coefficient the Laplace
  bound is `[-1, 1]` and parameter-independent, so the frozen-versus-current gap
  does not arise on that path at all. It remains for a single varying margin,
  where the other margin's `c` depends on the parameters, and for all SML fits.
  **Laplace Famoye estimates will differ from 0.4.0** on models where the
  previous bound was wider: `z_dep` maps through `[lamLo, lamHi]`.
* **Random-coefficient scales are no longer all labelled as SDs**: only `sd` is
  a standard deviation — `w` is a uniform/triangular half-width and `s` is a
  lognormal log-scale — so `summary.rpbnb_tmb_fit()` now heads the block
  "Random-coefficient scales" and keeps each row's own label (derived from the
  parameter name, itself built from `rand_dist_registry`'s `scale_label`), with
  a note stating what each means.
* `.write_truck_results_markdown()` **errors instead of overwriting** when the
  base timestamp and every suffix through `-1000` are taken. The bounded search
  previously fell through with the path still at the unsuffixed base name, so
  the write replaced the very file the guard exists to protect.

## Review fixes (2026-08-07 project review)

See `comments/response_2026-08-07-23-25-15.md` for the full response.

* **Gaussian copula fits are capped at one thread** (safety fix): evaluating a
  Gaussian-copula TMB object built with more than one thread terminates the R
  process. `fit_rpbnb_tmb()` now forces `n_cores = 1` and `parallel_tape =
  FALSE` for that family with a warning, before any TMB object is built, so the
  crash is not reachable from a public call. `fit$parallel` records both the
  requested and realized thread counts. The underlying defect in the registered
  Gaussian atomic is **not** fixed.
* **Famoye fits are checked for admissibility at the optimum** (model-validity
  fix): `lamLo`/`lamHi` are computed from the starting values and passed to the
  template as data, so the optimizer can leave the region actually admissible
  at the fitted parameters — making the joint pmf negative in the count tails
  while the objective stays finite at the observed cells. The bounds are now
  recomputed at the fitted parameters and a warning is raised when the fitted
  `lam` falls outside them; `fit$lambda_admissible` and
  `fit$lambda_bounds_at_optimum` carry the result. This detects invalid fits;
  it does not repair the optimized objective. A parameter-dependent constraint
  inside the template remains outstanding.
* **No Wald p-values or significance stars on random-coefficient scales**:
  `summary.rpbnb_tmb_fit()` was testing `log_scale = 0` (natural scale = 1) and
  labelling it under "Random-coefficient SDs". The interesting null, scale = 0,
  is a boundary at `log_scale = -Inf` where two-sided Wald inference is invalid.
  Both scales are now labelled explicitly, the natural-scale standard error is
  the delta-method transform `scale * SE(log_scale)`, and the same boundary
  footnote the Rcpp engine prints (see "Review fixes (2026-07-15 model
  review)") is shown. Use `lr_test()` for these.
* **`tools/test-tiers.R` no longer exits 0 when tests error**: it summed only
  expectation failures (`df$failed`) and ignored test-level errors
  (`df$error`), so its CI-friendly contract was false. Both are now counted,
  reported, and drive exit status.
* **Tests depending on `inst/extdata/export_dense_all.csv` now skip instead of
  erroring**: that file is local research data, gitignored and build-ignored, so
  three `test-fit-copula.R` tests errored on a clean checkout via
  `mustWork = TRUE` (their `skip_on_cran()` guards do not fire under the tier
  runner's `NOT_CRAN=true`). They now use a `dense_truck_fixture()` helper that
  skips with a reason.
* **`rpbnb_tmb_control()` validates every field**: `iterlim`, `reltol`,
  `print_level` and `halton_burn` were coerced without validation, so
  `reltol = -1`, `print_level = NA`, `iterlim = "many"` and `halton_burn = -1`
  were accepted. The last silently broke the draw contract —
  `.tmb_halton_uniform()` returned 9 rows for 10 requested draws.
  `fit_rpbnb_tmb()` also now checks that `control` inherits from
  `rpbnb_tmb_control`, since a direct call bypasses `rpbnb()`'s type check.
* `.write_truck_results_markdown()` no longer overwrites a same-second report;
  it suffixes and warns.
* Removed a dead `README.md` link to `docs/scope_rpnbn.md`; corrected the
  `AGENTS.md` specs pointer to `dev-docs/superpowers/specs/`.

Still open after this pass: the Gaussian atomic is not thread-safe (only made
unreachable); the Famoye constraint is still frozen at the starting values
(invalid optima are detected, not prevented); and Gaussian tail cells whose
NB2 CDF endpoints both reach the `safe_qnorm()` clamp still collapse to the
`1e-300` probability floor (245 of 3,487 cells on the truck fixture at its
documented starting point).

## Two engines, one package

* New `rpbnb()` front end dispatching to either engine via `engine = "cpp"`
  (the existing `fit_rpbnb()`, Rcpp/OpenMP simulated likelihood, `maxLik` BFGS)
  or `engine = "tmb"` (the new `fit_rpbnb_tmb()`, TMB automatic differentiation,
  `nlminb` with restart polish). Both fitters remain exported with unchanged
  signatures, and `rpbnb()` returns the engine-native object unaltered — no
  wrapper class, so every S3 method and post-estimation function works as
  before.
* `rpbnb()` validates every extra argument by name against the selected
  fitter's own formals. Passing a tmb-only argument (`inference`, `keep`,
  `method`) under `engine = "cpp"`, a cpp-only one (`draw_type`, `.fixed`,
  `.opt_draws`) under `engine = "tmb"`, or a misspelled name is an error rather
  than a silently ignored `...` entry. The two control objects are engine-typed
  and are never translated into one another: fields sharing a name mean
  different things.
* New from the TMB engine: a Laplace approximation (`method = "laplace"`),
  `rpbnb_tmb_dependence_profile()`, `rpbnb_tmb_max_workload()`, memory-aware
  workload sizing, and `scaling`/`log_vars` support in
  `rpbnb_tmb_marginal_effects()`/`rpbnb_tmb_elasticities()`.

## Native code

* One shared library now hosts both engines. `R_init_rpbnb` is hand-written at
  the bottom of `src/rpbnb_tmb.cpp` and registers TMB's `TMB_CALLDEFS`
  alongside the six Rcpp entry points; `Rcpp::compileAttributes()` detects it
  and no longer emits its own. `tests/testthat/test-native-registration.R`
  fails loudly if that table drifts out of sync with the `[[Rcpp::export]]`
  set.
* `src/Makevars*` now set `CXX_STD = CXX17` (required by TMB) and, on Windows,
  `-Wa,-mbig-obj`. `$(TMB_CXXFLAGS)`/`$(TMB_LIBS)` are deliberately not used —
  they are undefined make variables for a `LinkingTo: TMB` package, and a
  stray `-DTMB_LIB_INIT` would resurrect a second init and break the link.

## Shared code and behaviour changes

* `copula()` is now a single definition (the former `rpbnb` version, which
  validates `par` more strictly). `.prepare_bnb_data()`, `parse_rand_spec()`,
  `chk_rand_spec()`, `chk_dispersion()`, `.resolve_start()`, `.check_counts()`,
  `.chk_poisson_flag()`, `rand_dist_registry` and the Famoye math helpers are
  likewise shared rather than duplicated. Some error messages seen from the TMB
  engine are now the (more verbose) `rpbnb` wording.
* **The TMB engine now rejects `offset()` terms with an error.** The shared data
  prep understands offsets but the TMB template has no offset in its linear
  predictor, so accepting one would silently drop it. Use `fit_rpbnb()`
  (`engine = "cpp"`) for offset models.
* The two Halton generators are deliberately **not** unified:
  `halton_uniform()` (Rcpp engine, via `randtoolbox`) and
  `.tmb_halton_uniform()` (TMB engine, radical inverse). They are believed
  equivalent for a common `burn`, but `test-laplace.R` pins the TMB engine's
  SML log-likelihood to `1e-10` and that guard was not worth moving on a
  belief. `test-halton-equivalence.R` asserts the agreement; unify once it has
  held across platforms.
* Known duplication left in place: `signif_stars()` (returns a character
  vector) and `.signif_stars()` (returns the `symnum` object). The names differ
  and so do the return types, so collapsing them would change table formatting.

## Tests, data, docs

* New `slow-tmb` tier in `tools/test-tiers.R`. `test-dependence-profile.R` and
  `test-inference-memory.R` are gated with `skip_slow()` and excluded from the
  fast tier — the former alone runs ~6 minutes.
* `test-against-rpbnb.R` became `test-engine-agreement.R`: it cross-checked
  against `rpbnb` as a Suggests dependency, which is now an intra-package
  comparison, plus new cpp-vs-tmb agreement tests.
* All 23 scripts carried over from `rpbnb.tmb/inst/` are prefixed `tmb_`.
* Hand-written design docs live in `dev-docs/`, not `docs/` — the latter is
  pkgdown output that `build_site()` cleans.
* `inst/extdata/export_dense_all.csv` is gitignored and build-ignored, matching
  the existing treatment of `export_open_all.csv` (local research data).

# rpbnb 0.2.3

* Equation-specific `offset()` support on every model path (`fit_bnb()` under
  independence/Famoye/copula dependence, and `fit_rpbnb()` under Famoye/copula):
  the offset enters that margin's linear predictor additively
  (`mu = exp(x'beta + offset)`) during estimation and is carried through the
  stored fitted means, both `predict()` methods, `residuals()`,
  `bnb_marginal_effects()`/`rpbnb_marginal_effects()`,
  `bnb_elasticities()`/`rpbnb_elasticities()`, and the `bnb_gof()` null model
  (which now keeps the training offset so its log-likelihood, and every
  pseudo-R^2, stays comparable to the offset-aware full model).
* `.prepare_bnb_data()`: row selection now derives one common valid-row mask
  from the *evaluated* responses, design matrices, and offsets of both
  formulas, not from the raw variables. This closes a desynchronization hole
  where a transformation (e.g. `log(x)` with `x <= 0`) could produce
  `NA`/`NaN`/`Inf` after the raw variables were already complete, silently
  dropping different rows per equation. `predict()` now uses stored
  terms/factor-levels/contrasts (`predict_meta`) so newdata designs stay
  column-stable and reuse a stateful term's (`poly()`, `scale()`) training
  basis.
* `lr_test()` now errors if either fit is a `bnb_fit`/`rpbnb_fit` that recorded
  a failed optimization, instead of returning a p-value from an unfinished fit.
* Natural-scale `summary()` reports a Poisson-restricted margin's dispersion
  (`poisson_1`/`poisson_2`) as exactly `0` (SE `NA`), rather than leaking the
  internal `log(1e-6)` display placeholder onto the natural scale.
* Packaging: `.Rbuildignore` now excludes `packages/`, `data.zip`,
  `vignettes/*.pdf`, and `tests/results/`, which had been inflating the source
  tarball to ~8.3 MB (mostly copies of previously built package archives);
  it is now ~0.25 MB. Fixed an `R CMD check` WARNING (broken `predict.rpbnb_fit`
  Rd cross-reference) and NOTE (undefined global `x` in a residual histogram
  plot); `R CMD check` is now clean (0 WARNINGs, 0 NOTEs).

# rpbnb 0.2.2

* `fit_rpbnb()`: fixed a crash (`attempt to set an attribute on NULL`) when a
  random coefficient sits on a large-scale covariate. Such a covariate can drive
  some simulation draws to a near-zero fitted mean, where the negative-binomial
  log-pmf term `y * log1p(-p)` collapsed to `-Inf` (for `y > 0`) or `0 * -Inf =
  NaN` (for `y == 0`), poisoning the simulated likelihood so the optimizer never
  started. The term is now computed cancellation-free in both the R and C++
  likelihood paths. `fit_rpbnb()` also now fails with an actionable message
  instead of the opaque attribute error if the optimizer returns no estimate.

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
