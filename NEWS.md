# rpbnb 0.4.3

* **TMB engine: the multithreaded Gaussian-copula SIGSEGV is fixed.** The
  crash was never a data race in the kernel: `REGISTER_ATOMIC`'s cache
  (`atomic::forrev_derivatives`, TMB's checkpoint_macro.hpp under the CppAD
  framework) sizes its per-thread inner-tape array to `config.nthreads` ONCE,
  at first initialization, and never resizes, while evaluation indexes it by
  `omp_get_thread_num()` unchecked. Initialized at one fit's thread count,
  any later evaluation with more threads read past the end of that array and
  took the R process down — which is why the crash reproduced only on
  thread-count *escalation* within a session (a serial fit followed by a
  parallel one, or a 2-core fit followed by a 4-core fit) and never in a
  fresh process at a fixed count. `src/rpbnb_tmb.cpp`'s `FAM_GAUSSIAN` init
  block now raises `config.nthreads` to the machine's processor count for the
  one call that initializes the atomic (inside the existing
  `omp critical`, first-init only), so the array is sized once for every
  thread count a fit can realize. Verified at 2/4/8/16 threads on a
  2,321-observation, 1,000-draw fit: identical objective and gradient at
  every count, ~5x gradient speedup at 16 threads, where the same sequence
  previously segfaulted. The `test-parallel.R` Gaussian serial-vs-parallel
  comparison, skipped since the crash was first documented, now runs as the
  regression guard. The single-thread default cap and
  `force_parallel_gaussian` opt-in are unchanged for now; a follow-up may
  relax them.
* **TMB engine: exact draw chunking fixes out-of-memory failures at large
  `draws`.** SML tape size used to scale as `nrow(data) * draws` with no
  mitigation beyond a pre-flight refusal
  (`Weighted TMB workload is ... above max_workload`). `fit_rpbnb_tmb()` now
  auto-splits large-workload fits into several draw chunks replayed over one
  smaller TMB tape via `DATA_UPDATE()`, cutting peak memory to
  `nrow(data) * ceiling(draws / chunks)` — exact for the requested `draws`,
  not an approximation (see `docs/TMB_SML_large_draws_OOM_guide.md`).
  `control$tape_chunks` (new; see `?rpbnb_control`) pins a layout explicitly
  instead of relying on the auto-threshold, which — pending a follow-up
  calibration pass measuring the chunked tape's own memory profile — is
  currently derived from the pre-chunking calibration and should be treated
  as provisional. `rpbnb_tmb_boundary_tests()` propagates a chunked fit's
  memory policy to its restricted refits, so LR tests on a large fit stay
  chunked instead of rebuilding one full tape. A chunked fit has no taped
  Hessian: `confint(method = "profile")`/`rpbnb_tmb_dependence_profile()`
  fall back to a Wald interval with a warning; Wald/`optimHess` inference
  (the default) is unaffected. Also hoists several per-draw redundant
  computations in the independence/Famoye NB2/Poisson log-likelihood,
  shrinking the tape further for fits that do not chunk.
* `rpbnb_boundary_tests()` (the classic/simulated-ML engine) no longer reports
  spuriously negative LR statistics — the
  `Restricted model has the higher log-likelihood ... Clamping the statistic to 0`
  warning — for random-coefficient scale and dependence rows. The classic
  engine maximizes a simulated likelihood with a single BFGS run, so near a
  boundary parameter the full fit could stop just below a restricted refit that
  was warm-started from it; the TMB engine's exact-gradient Laplace fit with
  `restarts` did not show this. When a restricted fit now comes out ahead, the
  full model is re-optimized from the restricted optimum re-expressed in the
  full parameterization (the point where it provably attains the restricted
  likelihood) and the better of the two full-model optima defines the
  statistic, so `LR >= 0` holds by construction rather than by clamping.
* Regenerated `ref/rpbnb_0.4.2.pdf` (the CRAN-style PDF reference manual)
  from current Rd files, and clarified in README that `fit_rpbnb()`'s Halton
  draws are Cranley-Patterson-shifted (randomized quasi-Monte Carlo), not
  digit-scrambled.

# rpbnb 0.4.1

## Breaking-ish: one control object for every estimator

* `rpbnb_control()` and `rpbnb_tmb_control()` **are now the same object**.
  `rpbnb_control()` carries the union of both parameter sets and is accepted
  by `fit_bnb()`, `fit_rpbnb()`, `fit_rpbnb_tmb()`, and `rpbnb()` with either
  `engine`. `rpbnb_tmb_control()` is retained as a thin alias that forwards to
  it, so existing scripts keep working unchanged.
* `rpbnb(engine = ..., control = ...)` no longer errors on a "wrong-engine"
  control object; a script can flip `engine` without rewriting its control
  call. The returned object's class is
  `c("rpbnb_control", "rpbnb_tmb_control")`, so every historical
  `inherits(control, ...)` check still passes.
* **Settings an estimator does not read are ignored, not rejected** — and are
  reported, not silenced. Each fit records the supplied-but-unused names on
  `$control_ignored` (with `$control_engine`), and `print()`/`summary()` print
  a line naming them, e.g.
  `Control settings ignored (not used by the TMB engine): se_method, draws_hessian`.
  Only settings the caller actually wrote are reported. `draws_hessian` has
  been a documented no-op since same-draw curvature landed, so it is now
  always reported as ignored.
* `iterlim` and `print_level` — the two fields whose two constructors
  disagreed (300/500 and 2/0) — now default to `NULL`, meaning "this
  estimator's own default", resolved once the estimator is known. Each engine
  therefore behaves exactly as before when they are left alone, and an
  explicit value is honored by every engine. `rpbnb_control()$iterlim` is
  consequently `NULL` rather than `300L`; read it back off a fit's control, or
  set it explicitly, if a number is needed.
* `max_workload` also defaults to `NULL` and is computed by
  `rpbnb_tmb_max_workload()` only when a TMB fit needs it, so a `maxLik` fit
  no longer pays for the memory probe (and cannot emit its detection warning).
* Fields sharing a name are still **not translated**: `iterlim` means a
  `maxLik` BFGS limit to one estimator and an `nlminb` limit to another, and
  `n_cores` means worker processes versus OpenMP threads.

## New feature: per-group switches for the boundary LR tests, incl. the dependence parameter

* `rpbnb(boundary_tests = )` now takes a **switch over three groups** instead
  of a bare logical: `"sd"` (random-coefficient scales), `"dispersion"` (the
  NB2 overdispersions `m1`, `m2`), `"dependence"` (the association
  parameter), plus `"all"` and `"none"`. `TRUE` still means exactly what it
  always did, `c("sd", "dispersion")`, so it does not silently grow a third
  refit; `FALSE` still runs nothing. So `boundary_tests = "dispersion"` tests
  overdispersion alone, and `c("dispersion", "dependence")` tests
  overdispersion and association without paying one refit per scale.
* Both `rpbnb_boundary_tests()` and `rpbnb_tmb_boundary_tests()` accept
  `which = "dependence"` for the new test: H0 is "no association", i.e. the
  independence model, giving one row labelled `lam` (Famoye), `theta` (Frank /
  Clayton-Kimeldorf), or `rho` (Gaussian).
  * The TMB engine refits with its own `dependence = "independence"` family,
    which maps `z_dep` out of the free parameters — an exact 1-df restriction
    on the same draws.
  * The classic engine has no independence fitter, so it pins the
    working-scale dependence parameter at its family's independence value:
    `z_theta = 0` for Frank (`theta = 0`) and the Gaussian copula
    (`rho = tanh(0) = 0`), `z_theta = -30` for Clayton (below the `theta`
    threshold at which both the R and C++ kernels take their exact
    product-copula branch), and for Famoye the `z` that maps to `lambda = 0`
    under the interval the restricted refit freezes.
  * **The boundary correction is applied per family, not blanket.** Famoye's
    `lambda`, Frank's `theta`, and the Gaussian `rho` all have an *interior*
    null at 0, so their LR statistic is an ordinary chi-square(1); only
    Clayton/Kimeldorf (`theta > 0`) takes the 50:50 mixture. Using the mixture
    everywhere would have halved three of the four families' p-values.
* When a dependence LR test is present, `summary()` shows it in place of that
  row's Wald `z`/`p` on both engines — the two answer the same question, and
  printing both in one row invites reading the wrong column. Without the test
  (the default) the dependence row is the Wald table it has always been.
* Fix: the classic `rpbnb_boundary_tests()` no longer tests `m1`/`m2` on a
  margin the fit already pinned to its Poisson limit (a 0-df comparison
  `lr_test()` correctly refuses), and its restricted refits now inherit the
  full fit's `poisson_1`/`poisson_2` rather than defaulting to `FALSE` — with
  a Poisson-pinned margin the old default made every scale test compare
  non-nested fits. `rpbnb_tmb_boundary_tests()` already had both guards.

# rpbnb 0.4.0

The `rpbnb.tmb` package (v0.3.5, git `c64a2ec`) has been merged into `rpbnb`.
That source tree is now superseded; everything it provided is available here.

## Behaviour change: `rpbnb(engine = "classic")` ignores `method`/`force_parallel_gaussian`

* The TMB tuning knobs `method` and `force_parallel_gaussian` passed via
  `...` are now **dropped with a warning** under `engine = "classic"`,
  instead of the hard "only accepted by engine = \"tmb\"" error. Neither has
  any classic-engine meaning (they only tune *how* the TMB fit runs, not
  *what* is fitted), so a call can switch `engine = "tmb"` to `"classic"`
  without stripping them. All other cross-engine argument names — `inference`
  and `keep` under classic; `draw_type`, `.fixed`, `.opt_draws` under tmb —
  keep the hard error, as do unknown names.

## New feature: `rpbnb_tmb_boundary_tests()` / `rpbnb(engine = "tmb", boundary_tests = TRUE)`

* **Boundary-corrected LR tests for the TMB engine's boundary parameters** —
  every random-coefficient scale (`sd1:*`, `sd2:*`) and every unrestricted
  NB2 dispersion (`m1`, `m2`). `summary.rpbnb_tmb_fit()` has always reported
  no significance test for these (their nulls — scale = 0, or `m = 0`, the
  Poisson limit — sit on the boundary of the parameter space, where an
  ordinary Wald ratio doesn't apply). `rpbnb_tmb_boundary_tests(fit, data)`
  is the TMB-engine counterpart of the classic engine's
  [rpbnb_boundary_tests()], with the same `which = c("sd", "dispersion")`
  argument, and returns the same `rpbnb_boundary_tests` class so both share
  one `print()` method.
* Each restricted refit is otherwise identical to the full fit (same
  formulas, specification, dependence, draws, seed, estimator), warm-started
  from `fit$coef`. **Dispersions** are restricted via `poisson_1`/
  `poisson_2 = TRUE`. **Scales** are restricted by pinning that
  coefficient's `log_sd` at the parameterization's zero (`-20`; the template
  clamps `log_sd` to `[-20, 20]` and computes `sd = exp(log_sd)`, so this is
  `sd = 2.1e-9`) and mapping it out of the free parameters, giving a 1-df
  restriction.
* Pinning the scale rather than dropping the coefficient from
  `random_1`/`random_2` is what preserves **common random numbers**: the
  Halton draw matrix keeps the same width, so every *other* random
  coefficient draws from exactly the dimensions it did in the full fit. (An
  earlier version of this feature shipped dispersion-only on the assumption
  that TMB had no CRN-preserving way to restrict a scale; pinning via TMB's
  own `map` is that way.)
* `rpbnb(engine = "tmb", boundary_tests = TRUE)` now works (previously a
  documented error) and attaches the result as `$boundary_tests`, same as
  `engine = "classic"`. `summary()` merges it into **both** the
  random-coefficient scale blocks and the dispersion block automatically:
  `LR`/`df`/`Pr(>chisq)` columns replace the `NA` those rows previously
  carried.
* `fit_rpbnb_tmb()` gains an internal `.fixed` argument (a named numeric
  vector of parameters to pin, in the optimization parameterization),
  mirroring the classic engine's identically named argument. Not intended
  for direct use; it is how the scale-zero refits above are built.
* Also fixed in the same area: `summary.rpbnb_tmb_fit()`'s dispersion block
  was missing `Std. Error` entirely (a real gap, not intentional design —
  `m1`/`m2` are `ADREPORT`ed in the template, so the delta-method SE was
  already available and simply wasn't being read).
* `fit_rpbnb_tmb()` now stores `formula_1`/`formula_2`/`draws`/`seed`/
  `poisson_1`/`poisson_2` on the returned `rpbnb_tmb_fit` (mirroring what
  `fit_rpbnb()`'s `rpbnb_fit` already stores) — needed to reconstruct a
  restricted refit without the caller re-supplying them by hand, and useful
  for any other refit-based tooling built on a TMB fit going forward.
* `rpbnb_tmb_boundary_tests()`'s default `control` (when the caller passes
  none) now reuses the original fit's `n_cores` (`fit$parallel$requested`)
  instead of hardcoding `n_cores = 1`, so the restricted refits get the same
  thread budget the full fit was given — subject to the same Gaussian-copula
  single-thread safety cap, re-applied per refit exactly as it was on the
  original fit. Fits from before this field was stored fall back to `1`.
  The default `print_level` is now `1` (was `0`), so each restricted refit
  is no longer silent.
* `rpbnb_tmb_boundary_tests()` now reports a `message()` ("Boundary LR
  test: `<parameter>`...") before each restricted refit starts (suppressed
  when `control$print_level` is `0`), matching the summary message
  `rpbnb(boundary_tests = TRUE)` already prints before the whole batch.
* `rpbnb_tmb_boundary_tests()` gains a `draws` argument for the restricted
  refits, defaulting to `fit$draws` (unchanged behavior). Lets a caller
  raise (or lower) simulation draws for the boundary tests alone, without
  refitting `fit` itself at a different `draws`. A non-default value still
  shares `fit$seed` but no longer reproduces `fit`'s exact simulated
  log-likelihood surface, so the LR statistic then also reflects simulation
  noise beyond the restriction under test — the default remains the
  statistically clean choice.
* `rpbnb()` gains a `boundary_draws` argument (`engine = "tmb"` only) that
  forwards to `rpbnb_tmb_boundary_tests()`'s new `draws` argument above, so
  the boundary tests folded in via `boundary_tests = TRUE` can use a
  different `draws` than the main fit without a separate manual
  `rpbnb_tmb_boundary_tests()` call. `NULL` (default) keeps prior behavior
  (the main fit's `draws`). An error under `engine = "classic"`: that
  engine's `rpbnb_boundary_tests()` reuses the full fit's exact stored draw
  matrix rather than regenerating draws from a count, so there is nothing
  for `boundary_draws` to override there.
* `summary.rpbnb_tmb_fit()`'s "--- Dependence ---" block now reports an
  ordinary Wald `z value`/`Pr(>|z|)`/significance stars for the dependence
  parameter (`theta` for Frank/Kimeldorf, `rho` for the Gaussian copula,
  `lam` for Famoye), not just `Estimate`/`Std. Error` as before — mirroring
  the classic engine's `summary.rpbnb_fit()`, which has always computed this
  (`R/methods.R`, `add_dispersion()`'s copula/lambda rows). By design (same
  as the classic engine) this is always an ordinary Wald test, never a
  boundary-corrected LR test: the dependence null is interior for Frank
  (`theta` unrestricted) and the Gaussian copula (`rho` in `(-1, 1)`); for
  Kimeldorf/Clayton (`theta > 0`) the null is technically a boundary case,
  but both engines deliberately test it the same (ordinary Wald) way so they
  do not disagree about what they report.
* Fixed: `summary.rpbnb_tmb_fit()` printed its "Random-coefficient scales"
  explanatory note (the `sd`/`w`/`s` legend, plus the LR/df/`Pr(>chisq)`
  footnote) once per equation when both equations had random coefficients —
  identical text, back to back. It now prints once, after both equations'
  tables.
* `summary.rpbnb_tmb_fit()`'s "--- Dependence ---" block now explains an `NA`
  standard error instead of printing a bare `Estimate / NA / NA / NA` row.
  The "Dispersion" block above it has always explained its own `NA`s, so the
  silent one read as a malfunction rather than as the deliberate refusal it
  is: the boundary scan in `R/tmb_inference.R` nulls the standard error when
  the estimate is set by an implementation bound rather than identified by
  the data. Reported from a Famoye fit whose `z_dep` reached 24.9, saturating
  the logistic map onto the frozen lambda interval — `lam` came out
  `1 - 2.0e-06` (printing as a tidy `1.0000`, which is what made it look like
  a placeholder) with a delta-method derivative of `3.1e-11`, so
  `near_smooth_cap()`'s 2% margin flagged it `"upper"` and blanked the SE.
  The note names the cause, points at `fit$boundary_report` /
  `fit$boundary_sides`, and refers `rpbnb_tmb_dependence_profile()` for a
  likelihood-based interval.

  Two cases that produce identical-looking `NA`s are deliberately kept apart,
  because they call for opposite remedies. A *pinned* dependence parameter is
  specific to that parameter and means the family cannot represent the
  association in the data. A *non-positive-definite Hessian* says nothing
  about the dependence parameter at all — it blanks every standard error in
  the summary — so the note reports that case as a property of the fit as a
  whole and points at rank deficiency instead. Attributing the second to an
  implementation bound would be a confidently wrong diagnosis aimed at the
  wrong fix.

  For Famoye the note adds that the profile interval is mapped through the
  same frozen box, and that when the box is parameter-independent
  (`lambda_bounds` identical to `lambda_bounds_at_optimum`) refitting from
  better starting values *cannot* widen it — the documented "widen the box"
  remedy does not apply there and the cap is structural, so a copula
  dependence is the actual answer.
* Fixed: `fit_rpbnb_tmb()` (and therefore `rpbnb(engine = "tmb")`) could
  abort outright with `Error in stats::nlminb(...) : NA/NaN gradient
  evaluation` (or `NA/NaN function evaluation`), discarding an entire
  in-progress fit, if the optimizer's very first `nlminb()` call stepped into
  a non-finite region deep in its search — `nlminb()` raises this as a hard R
  error rather than returning a sentinel, and only the *restart* loop's
  `nlminb()` calls were already guarded against it. Observed on the truck
  data under a Kimeldorf (Clayton) copula after nearly an hour of otherwise
  productive optimization. The first call is now wrapped the same way: on
  that specific error, it recovers at `obj$env$last.par.best` (TMB's own
  running best-finite-objective parameter vector) and hands control to the
  existing restart loop from there, marked not-converged, instead of losing
  the run. When even that recovery point is non-finite (never observed a
  finite objective at all -- the failure mode below), `fit_rpbnb_tmb()`
  still re-raises rather than fabricate a result.
* Fixed: `rpbnb_tmb_boundary_tests()` (and therefore
  `rpbnb(boundary_tests = TRUE)` under `engine = "tmb"`) could lose an
  entire batch of restricted refits to the same `nlminb()` abort, even after
  the `fit_rpbnb_tmb()` fix above, when a restricted refit's warm start
  itself (the full fit's coefficients with one boundary parameter pinned) is
  non-finite at every point `nlminb()` tries -- there is then no finite
  `obj$env$last.par.best` to recover to. Observed on the truck data's `m1`
  dispersion test under a Kimeldorf copula: the very first outer gradient
  evaluation came back `NaN`. `test_row()` already had a well-defined
  "restricted fit did not converge" path (`NA` for that row, with a
  `warning()`) for an ordinary non-convergent refit; each restricted refit
  is now wrapped in `tryCatch()` so specifically that `"NA/NaN ..."` abort
  is funneled into that same path (`convergence = NA`) instead of
  propagating and losing every other refit in the batch. Any other error
  (a real bug, a bad argument) still propagates as an ordinary error rather
  than being relabelled "did not converge".
* Fixed the root cause behind the `m1`/`m2` NaN above rather than only
  degrading gracefully around it: the template's exact `m = 0` (Poisson)
  branch (`poisson_1`/`poisson_2 = TRUE`, `src/rpbnb_tmb.cpp`) computed
  `ppois()`/`dpois()` in LINEAR space and logged the result afterward. For a
  random-coefficient draw that pushes `mu` far enough from a small observed
  count -- ordinary in count data, where most observations are small -- both
  `ppois()` and `dpois()` underflow to an exact linear-space `0.0` before
  `log()` ever runs, so a Kimeldorf/Clayton cell probability's internal ratio
  (`log_pmf - log_cdf`, in `clayton_cell_prob()`'s `log_ratio()`) can end up
  subtracting two `-Inf`s -- an `Inf - Inf` indeterminate form, `NaN` by
  construction, which then poisons the whole taped objective (every free
  parameter's gradient goes `NaN` at once, not just ones near that
  observation, since `NaN` propagates through the sum-of-observations
  reduction wherever it appears). This is exactly the failure mode the NB2
  branch's `nb2_cdf_pair()` was already hardened against; the exact-Poisson
  branch just never got the same treatment when it was added. New
  `pois_cdf_pair()` mirrors `nb2_cdf_pair()`'s log-space `log_add_exp`
  accumulation (simpler, since Poisson has no dispersion parameter to carry)
  and replaces the linear-then-log `ppois()`/`dpois()` calls in both margins'
  exact-Poisson branch. Confirmed on the truck data: the `m1` boundary
  refit's objective, `NaN` at the very first evaluation before this fix, is
  now finite (`10709.73`) with a fully finite gradient, and the boundary LR
  test reports a real `LR`/`df`/`Pr(>chisq)` instead of `NA`.

* Fixed: **a Poisson margin's Gaussian-copula cell probability collapsed in
  the tails.** Gaussian is the only family that must pass its margins through
  `qnorm()` (singular at 0 and 1), and it clamped the marginal CDF to
  `[1e-15, 1-1e-15]` before doing so. When both corners of a cell landed past
  the same clamp the integration strip collapsed to *zero width*, returning
  probability 0 — floored to `1e-300` (690.8 nats) — and, because a clamp is
  a `CondExp` step, that cell then contributed **exactly zero gradient**.
  `gaussian_cell_prob()`'s own comment had recorded this as a known
  limitation needing "a separately accumulated survival function"; this is
  that function, for the case that most needs it.
  - A Poisson margin's corners are now taken from whichever tail is still
    representable: `q = +qnorm(F)` normally, and `q = -qnorm(S)` once `F`
    passes `1 - 1e-10`, with `S = P(Y > y) = pgamma(mu, y+1)` computed
    **directly** (an already-differentiable TMB atomic) rather than as
    `1 - F`. The second corner follows from `S(y-1) = S(y) + P(Y = y)`, an
    addition of positive quantities, so it cannot cancel. The key observation
    is that `1e-15` was never the real limit: a probability near *zero* is
    representable to ~1e-308, and only a probability near *one* loses its
    information (a double's spacing at 1 is 1.1e-16).
  - **NB2 margins deliberately keep the old clamp**, and with it the old
    limitation. The NB2 survival is `pbeta(mu/(mu+r), y+1, r)`, whose shape
    `r = 1/m` runs away exactly when the data wants a Poisson margin (the
    dispersion collapses toward 0, and `log_m`'s clamp allows `r` up to
    `exp(20) = 4.85e8`). `pbeta`'s *value* is exact even at `r = 1e9`
    (checked against `pnbinom`: relative error 0), but its derivative with
    respect to `r` is ~1e-13 there — and since `CondExp` evaluates *both* of
    its branches, that call sat on the tape for every cell whether or not the
    survival was selected. With it present, a polished fit came back with a
    **non-finite** `max|gradient|` and the free dispersion stopped collapsing
    on Poisson-generated data (`m1` 0.018, `m2` 0.034, against `< 1e-3`
    before and a true value of 0). Narrowing the switch so that essentially
    no cell selected it barely moved those numbers, which is what identified
    the call's mere *presence* rather than its selection as the cause. The
    residual NB2 exposure is small: 0.05% of observation-draw cells and 0.4%
    of observations on the truck data, against the 2.69% and 16.4% a
    Poisson-pinned margin reached.
  - Validated against the invariant that pins it down without an external
    reference — the strip must carry exactly the marginal mass,
    `Phi(q(y)) - Phi(q(y-1)) == P(Y = y)` — to a worst relative error of
    **4.2e-13** over cells whose pmf spans 1.4e-296 to 0.37, where 35 of
    those same 55 cells collapsed outright before. Where the old and new
    forms disagree (`F > 1 - 1e-11`) that round trip also says which is
    right: **1.2e-14** relative error against **9.3e-4**. Where the old form
    was sound the two agree to 4e-10 over 5,000 random cells.
  - What it was costing, measured on the truck data's `m1` boundary LR test
    (margin 1 pinned Poisson against counts to 242) under a Gaussian copula:
    **2.69%** of observation-draw cells and **16.4%** of observations
    collapsed, against 0.05%/0.4% with that margin left NB2 — and 0.000% for
    the `m2` test, whose margin (`C_HV`, 78% zeros) never saturates, which is
    why `m2` succeeded while `m1` did not. The objective was a step function:
    1,285 nats of swing over parameter steps of `2e-4`, with the AD gradient
    disagreeing with a central finite difference by ~100% (51 vs 5,094; -479
    vs -1.6e6) on the coordinates that move `mu1`, against exact agreement on
    those that do not. `nlminb` stopped at `false convergence (8)` — PORT's
    code for "converging to a noncritical point; gradients possibly wrong, or
    the function discontinuous", which is literally what it was handed. After
    the fix, at the same parameter point: objective 18671.47 → **11143.02**
    (the floored mass restored), **0 of 29** coordinates disagreeing with
    finite differences, and the scan's slope jumps falling from 1.28e7 to
    **3.02**.
  - Residual limitation for the Poisson margin, 293 decades further out than
    the one it replaces: if the *small* tail itself underflows (`S < 1e-300`
    upper, `F < 1e-300` lower) both corners still floor and the cell
    collapses. The lower-tail half is in principle recoverable —
    `nb2_cdf_pair()`/`pois_cdf_pair()` already return an accurate `log F` —
    but needs a log-scale `qnorm`, which TMB does not provide (its `qnorm`
    atomic has no `log_p` argument). The truck data's largest count is 242,
    where the survival is 4e-228 and nowhere near this floor.
  - `y == 0` keeps the historical `qnorm(1e-15) = -7.94` sentinel rather than
    the more nearly-correct `qnorm(1e-300) = -37.05`. Its lower corner is a
    true `-infinity` and the mass between the two is immaterial (1e-300
    against 1e-15), but `gaussian_cell_prob()` spends a *fixed* quadrature
    node budget on `[q(a'), q(a)]`, so the deeper sentinel made that interval
    about five times wider for every zero count — and zeros are most of a
    count sample — for no gain in accuracy.
* Fixed: **a Frank copula cell probability the data genuinely produces was
  being clipped at `1e-300`.** Frank's `frank_cell_prob()` returned a *linear*
  probability, and the caller floored it at `1e-300` before taking its log —
  the same floor Clayton and Gaussian still use. That floor was written as a
  guard against underflow noise, but on this workload it clips ordinary
  observations. It is now `frank_log_cell_prob()`: it takes the margins' *log*
  masses (which `nb2_cdf_pair()`/`pois_cdf_pair()` already return) and returns
  the log cell probability, so nothing on the Frank path passes through the
  linear representation of the cell and the floor never binds.
  - Found on the truck data's `m1` boundary LR test — margin 1 forced Poisson,
    a Frank copula, `method = "laplace"`. Observation 2230 (`y = 125` against
    `mu = 0.193`, so `log P(Y1 = 125) = -687.8`) has a cell probability of
    **1.03e-300**: three ulp above the floor, not an underflow artefact. One
    inner-Newton step crosses it, and on the far side the objective is the
    constant `-log(1e-300) = 690.776` rather than the likelihood.
  - A clip is a kink, and Laplace differentiates the joint *twice*. The
    centred second difference of `-log p` in `log(mu1)` across it came back
    **-12,181** at `h = 1e-2` and **-94,836** at `h = 1e-3` where the true
    curvature is `mu1 = 0.193` — which is what the log-space form now returns,
    at every step size. That much negative curvature on one latent row costs
    the inner Hessian its positive definiteness, and TMB reports
    `PD hess?: FALSE`, then `Newton drop out: Too many failed attempts`, then
    `inner newton optimization failed during gradient calculation` and a NaN
    outer gradient. At the parameter vector where the refit first went
    non-finite, the inner Hessian's worst per-observation eigenvalue was NaN
    (and `-106.6` on the blocks that still had one); it is `-0.109` after the
    fix, and the matrix is finite throughout.
  - `M = (1 + A(a')B(b)/D)(1 + A(a)B(b')/D)` is now computed from an identity
    rather than as written. Each factor is `exp(-th * C(.,.))`, which for
    saturated corners is `exp(-th)`: 2.5e-9 at the `theta = 19.8` this test
    reaches and 6.3e-16 at the `FRANK_THETA_MAX` cap of 35, where it is under
    three ulp of 1. Recovering it by adding 1 to a quantity near -1 left
    **5.6% relative error** at the cap. Writing `p = exp(-th u)`,
    `q = exp(-th v)` and expanding `D + A(u)B(v)` gives a numerator of two
    terms that both carry the denominator's sign for either sign of `theta`,
    so the factor is positive by construction and accurate to full relative
    precision.
  - **This changes the Frank likelihood wherever the old floor bound**, under
    both estimators — those cells now contribute their true value instead of
    690.776 nats, so the objective is larger and the fit is not the same
    number. On a one-observation check (`y = (300, 0)`, Poisson margin 1,
    `mu = 1`) the objective is **1432.93** against the **706.49** the floor
    allowed. Ordinary counts are untouched: against the old linear form, over
    counts where that form is sound, the two agree to 1e-8.
  - **Not fixed, and not a numerical defect:** the truck `m1` refit still ends
    at `nlminb` `false convergence (8)`, so that row stays `NA` under
    `method = "laplace"`. With margin 1 forced Poisson the optimizer drives
    Frank's `theta` to about 20 (Kendall's tau ≈ 0.8) to recover the lost
    dispersion, and at that strength the cell probability is a smooth but very
    sharp ridge — convex only near its peak. A fine scan at the worst
    observation (`y = (38, 1)`, `mu = (26.9, 0.57)`) shows `-log p` falling
    from 4.21 to 3.09 and back to 10.03 over `eta1` steps of 0.01, with
    curvature running from +584 on the peak to negative on the shoulders. That
    is genuine non-log-concavity of a strong discrete copula in the random
    effects, which is exactly what a Laplace inner Newton cannot have.
    `method = "sml"` has no inner Newton and is not exposed to it.
  - Clayton and Gaussian still return a linear cell probability and keep the
    `1e-300` floor, so a Laplace fit whose true cell probabilities reach that
    depth remains exposed there. Clayton's internals are already log-space
    (`log_ratio()`), so it is the cheaper of the two to convert if the need
    comes up; Gaussian's cell is a quadrature and has no log form.
* `rpbnb_tmb_boundary_tests()` gains **`sml_fallback`** (default `TRUE`):
  when the fit under test was estimated by `method = "laplace"` and a
  restricted refit fails to converge, that one test's LR is re-run with
  **both sides estimated by `method = "sml"`** instead of reported `NA`.
  `rpbnb(engine = "tmb", boundary_tests = TRUE)` inherits it automatically.
  - This is the follow-up to the Frank clip fix above, which repaired the
    *numerics* of the truck data's `m1` test but left its row `NA` anyway,
    for a reason no numerical fix can reach: pinning margin 1 to Poisson
    drives Frank's `theta` to about 20 (Kendall's tau 0.8) to absorb the
    lost dispersion, and at that strength the cell probability is
    non-log-concave in the random effects — at `y = (38, 1)`,
    `mu = (26.9, 0.57)`, `-log p` runs 4.21 → 3.09 → 10.03 over `eta1`
    steps of 0.01, curvature +584 at the peak and negative on the
    shoulders. The Laplace approximation differentiates through an inner
    Newton that requires exactly the log-concavity the restricted model no
    longer has. TMB's inner-solver knobs were screened at the failing
    point: `tol10 = 0` does clear the `"Newton drop out"` (the early-exit
    probe's PD check was one failure source), but the outer fit still ends
    at nlminb `false convergence (8)` with `max|grad|` of 1e8–1e10.
    SML has no inner Newton and no third derivatives, and is not exposed.
  - The fallback refits the **full model too** under SML (once, cached
    across all of one call's tests, warm-started from the Laplace optimum,
    at the same `draws`/`seed`), and takes the LR between the two SML
    fits. A Laplace `logLik` is never paired with an SML `logLik`: they
    are different approximations of the likelihood, and their difference
    is not an LR statistic. If the SML restricted fit fails too, the row
    is `NA` with a warning and the full-model refit is skipped (its only
    purpose is to pair with the restricted one).
  - Fallback rows are announced by a `message()` (silenced by
    `control$print_level = 0`), listed in the result's `sml_fallback`
    attribute, and footnoted by `print()` — the estimator switch is never
    silent. Validated on the truck data (`m1`, Frank copula, 400 draws,
    16 threads): full SML `logLik = -7674.99` (30 min), restricted
    `-8188.79` (37 min), **both `nlminb` code 0** with `max|grad|` under
    0.05, giving LR = **1027.60** on 1 df, `p ≈ 9e-226` — where every
    Laplace attempt at the same restriction ended at
    `false convergence (8)` with `max|grad|` of 1e8–1e10. (The stalled
    Laplace pairs bounded their LR near 948.6; the SML statistic differs
    because the two estimators maximize different approximations of the
    likelihood — which is exactly why the fallback never mixes them.)
  - The interpretation caveat is the flip side of the rescue: an SML-pair
    LR is a statement about the SML likelihood, tested at these `draws`.
    It answers "is this dispersion zero" under a consistent estimator; it
    is not the Laplace fit's own LR, which is exactly why the rows are
    flagged everywhere they appear.
  - The fallback also triggers on a **converged Laplace pair with a
    negative raw LR** -- a restricted logLik above the full fit's, which a
    nested pair at true optima cannot produce. nlminb code 0 is not
    sufficient to trust a Laplace value: a full truck-data boundary run
    surfaced restricted refits beating the full fit by 0.002-1.2 nats
    (the warm-started, restart-polished refits out-polished the full
    fit's stopping point) and one by 1,919 nats (LR = -3838: near a
    singular inner Hessian the Laplace log-likelihood rises without
    bound, and the restricted refit "converged" on that spurious ridge).
    Previously these fell through to `lr_test()`, which clamped the
    statistic to 0 with a warning -- for the scale rows that silently
    replaced real LRs of 7-30 with "no evidence". Now any negative raw
    LR under a Laplace fit is treated exactly like a failed refit: the
    test re-runs as an SML pair, with a message stating the raw LR that
    disqualified the Laplace one.

## New feature: `fit_rpbnb_tmb(force_parallel_gaussian = )`

* **Opt-in override for the Gaussian-copula single-thread safety cap.**
  `dependence = copula("normal")` fits have always been silently capped to
  one TMB thread (`n_cores`/`max_threads` forced to 1, `parallel_tape`
  forced off) because multithreaded evaluation of this family has reliably
  crashed the R process (SIGSEGV) on the first objective evaluation — a
  defect in the registered Gaussian atomic
  (`REGISTER_ATOMIC(gauss_cell_vec)`, not re-entrant under OpenMP). That cap
  is unchanged by default. `force_parallel_gaussian = TRUE` (also usable via
  `rpbnb(engine = "tmb", force_parallel_gaussian = TRUE, ...)`) instead
  honors the requested thread count, with a `warning()` naming the crash
  risk explicitly rather than the earlier cap-notice warning.
* This does **not** fix the underlying defect — it is an escape hatch for
  someone who has read the documentation and still wants to try running
  multithreaded (e.g. to test whether a particular TMB/OpenMP build is
  actually affected). A crash under this override can still corrupt memory
  and lose unsaved work. Frank and Clayton copulas were never capped and are
  unaffected by this argument.
* Internally, the capping/override logic is now a pure function
  (`.resolve_gaussian_threads()`, `R/tmb_helpers.R`) with no TMB/DLL calls,
  so its warning/capping behavior is unit-tested directly
  (`tests/testthat/test-parallel.R`) without ever triggering a real
  multithreaded Gaussian-copula evaluation — no test in this package
  exercises that path, by design.

## Breaking change

* **`rpbnb(engine = )` renamed `"cpp"` to `"classic"`**. `engine = "classic"`
  now selects the Rcpp/OpenMP `fit_rpbnb()` path (unchanged behaviour); the
  old `engine = "cpp"` value is no longer accepted. `fit_rpbnb()` and
  `fit_rpbnb_tmb()` themselves are unaffected — only the `rpbnb()` dispatcher's
  `engine` argument changed name.

## New feature: `rpbnb(standardize = TRUE)`

* **Automatic centring/scaling with original-units display**, generalizing
  the by-hand pattern in `inst/rpbnb_frank_open.R` and
  `inst/tmb_rpbnb_frank_open.R`. `standardize = TRUE` auto-detects continuous
  predictors (numeric, non-factor columns with more than two distinct values
  — so 0/1 and other two-level numeric indicators are left alone — excluding
  anything used only inside an `offset()`), centres and scales them (mean 0,
  SD 1) before fitting, and attaches the scaling as `$scaling` /
  `$continuous_vars` on the fit. `continuous_vars` overrides auto-detection
  when supplied.
* The fitted design itself (`X1`/`X2`, `mu1`/`mu2`, stored draws) stays on
  the standardized scale, so `predict()`, marginal effects, and boundary/LR
  tests are unaffected. Only `print()`/`summary()` change: the coefficient
  table is back-transformed to the covariates' original units by the exact
  affine chain rule (a continuous slope divides by its scale, the intercept
  absorbs the centring shift, a random coefficient's log-scale parameter
  shifts by `-log(scale)`) — no refit, no numerical differentiation — and is
  the *only* coefficient table shown. Works identically for both engines
  (`engine = "classic"` and `engine = "tmb"`).
* `coef()`/`vcov()` continue to return the standardized-scale values that
  match the stored design (needed by `predict()` etc.); the original-units
  values shown by `print()`/`summary()` are computed on demand, not written
  back into the fit.

## New feature: `rpbnb(boundary_tests = TRUE)`

* **Boundary LR tests integrated into `rpbnb()`**, `engine = "classic"` only.
  The random-coefficient SDs and NB2 dispersions (`m1`, `m2`) have a null that
  sits on the boundary of the parameter space, so an ordinary Wald `z`/`p`
  does not test it (`summary()` has always reported `NA` for those rows).
  `boundary_tests = TRUE` runs [rpbnb_boundary_tests()] on the fit as soon as
  it converges — against the same (possibly standardized) data the fit itself
  used — and attaches the result as `$boundary_tests`.
* `print()`/`summary()` now show that boundary-corrected LR test (`LR`, `df`,
  `p`) for the SD and dispersion rows in the natural-scale block, in place of
  the `NA` they previously carried; rows without a matching boundary test
  (dependence parameters, a Poisson-pinned margin) are unaffected. Composes
  with `standardize = TRUE`.
* Default `FALSE` — each restricted refit costs roughly another full fit, more
  under a `copula()` dependence than `"famoye"` (see
  `rpbnb_boundary_tests()`'s timing note) — so existing `rpbnb()` calls are
  unaffected. `boundary_tests = TRUE` under `engine = "tmb"` is an error
  (`rpbnb_boundary_tests()` only accepts an `rpbnb_fit`).

## New feature: original-units marginal effects/elasticities for `engine = "classic"`

* **`rpbnb_marginal_effects()`/`rpbnb_elasticities()` gain `scaling =`/
  `log_vars =`**, matching the TMB engine's `rpbnb_tmb_marginal_effects()`/
  `rpbnb_tmb_elasticities()` (which have had them since the `rpbnb.tmb`
  merge). `scaling` is the same named list of `c(center =, scale =)` pairs
  `rpbnb(standardize = TRUE)` attaches as `$scaling`; `log_vars` names
  covariates that are already a log (e.g. `log(AADT)`), so results are
  reported per unit of the underlying variable rather than per unit of its
  log. Both engines now share one implementation of the restatement
  (`.scaling_vec()`/`.log_vars_flag()`, `R/tmb_marginal_effects.R`).
* Without `scaling =`, a centred continuous predictor's elasticity prints as
  ~0 (its standardized-scale sample mean is 0, so the elasticity's leading
  x-bar factor vanishes) — that reads as "no effect" but means "these units
  are arbitrary". `rpbnb_marginal_effects(fit, scaling = fit$scaling)` (and
  the same for elasticities) restates both by the exact affine/chain-rule
  transform the delta-method jacobian differentiates directly, so standard
  errors are exact too — not the ~100-line hand-rolled `raw_diag()`
  `inst/rpbnb_frank_open.R` previously needed for this (now replaced there,
  and in `inst/rpbnb_truck.R`, with `scaling = fit$scaling`).
* `var_type` gains `"log-continuous"`/`"log-elasticity"` for `log_vars`
  columns, alongside the existing `"continuous"`/`"elasticity"`,
  `"binary(0->1)"`/`"semi-elasticity"`, and `"(random)"` suffix.

## Review fixes (2026-08-09 project review)

See `comments/review_2026-08-09-08-10-18.md`. Documentation and test hygiene
only; no estimation behaviour change.

* **The moving-vs-frozen bounds trade-off is now a recorded decision, not an
  accident.** The two Famoye fitters resolve it in opposite directions:
  `fit_bnb()` recomputes the interval per evaluation (fitted lambda admissible
  *by construction*, gradient inconsistent en route, mitigated by multi-start),
  while the RP engines freeze it at the starting values (gradient exactly the
  derivative of the optimized objective, escape detected post fit). A DECISION
  note in `R/bnb_likelihood.R` states both horns and cross-references; the
  `fit_bnb` SE site documents why freezing *at the optimum* is self-consistent
  there (the converged point is a fixed point of the frozen-gradient system);
  `?rpbnb` states the RP engines' semantics.
* **`?fit_rpbnb_tmb` no longer denies fields its own warning points to**:
  `lambda_bounds_at_optimum` and `lambda_admissible` are documented, and the
  "not recomputed at the optimum" sentence now says the recomputation is for
  validation only. `?fit_rpbnb` documents `bounds`, `bounds_at_optimum` and
  `lambda_admissible` likewise.
* **`?lr_test` and `?rpbnb_boundary_tests` state the frozen-interval caveat**,
  confined to where it exists: uniform/triangular coefficients or a single
  varying margin. Normal/lognormal in both margins use the constant `[-1, 1]`
  and are unaffected.
* **`AGENTS.md` describes the two-engine reality**: single hand-written
  `R_init_rpbnb`, the TMB source layout, the engine table including `rpbnb()`,
  non-interchangeable control objects, and a corrected lambda-bounds
  convention note (the old one described pre-0.4.0 semantics).
* **`R CMD check` breakers inherited from pre-merge `master` fixed**:
  `tests/bnb_rwm1984.R` now resolves its results directory robustly instead of
  writing to a cwd-relative path that does not exist under check, and
  `test-rpbnb-interpretation.R` uses `n_cores = 2L` (the `--as-cran` core cap)
  — the parallel-equals-serial property it tests is preserved.

## Review fixes (2026-08-08 22:12 response review)

See `comments/response_2026-08-08-22-29-34.md`. Tests and one shared helper; no
estimation behaviour change.

* **The covariance regression now pins which interval each path used.** The
  previous version asserted only that the two intervals differ, that non-NA SEs
  are finite, and that `summary()` returns a table — none of which says which
  interval produced the covariance. Reverting a single covariance call site
  while leaving `fit$bounds` correct left it entirely green. (The finiteness
  assertion was itself vacuous: written over `fit$se[!is.na(fit$se)]`, and
  `all(logical(0))` is `TRUE`, it would have passed with every SE `NA`.)
  A new oracle rebuilds the covariance at `coef(fit)` on the stored draws under
  a supplied interval — `crossprod()` of scores for OPG, `bnbr_rp_hessian()` for
  analytic, the fixed-bound numeric objective for numeric — and the test asserts
  it matches under the frozen interval and differs under the optimum interval.
  Verified by mutating each covariance site independently, `fit$bounds`
  untouched: 2 failures each.
* **`famoye_lam_from_z()` is now used by every R-side implementation of the
  map** — `bnbr_rp_ll_and_grad()`, `bnbr_rp_ll_fixed_bounds()` and
  `bnbr_rp_hessian()`, alongside the post-fit guard. It previously prevented
  only guard/test drift while its documentation claimed it prevented drift from
  the objective's parameterization; the map was still written out separately at
  three R sites. The C++ core keeps its own copy, now documented as such and
  tied to the R side by the math-identity assertions in
  `test-cpp-likelihood.R`.

## Review fixes (2026-08-08 21:49 response review)

See `comments/response_2026-08-08-22-04-13.md`. Tests only; no behaviour change.

* **The Famoye guard regressions now discriminate.** As written they would have
  passed against the broken wiring they exist to catch: the arithmetic test
  reimplemented the logistic map instead of calling it, the end-to-end test
  commented that the frozen and optimum intervals differ without asserting it,
  produced no inadmissible fit, and used `compute_se = FALSE` so it entered none
  of the OPG, analytic-Hessian or numeric-Hessian branches; and the score
  identity test called both wrappers with `lam_bounds = NULL`, exercising only
  the recomputed-bound default.
* The interior map is now `famoye_lam_from_z()`, shared by the guard and its
  test. Every fit-level test asserts that the two intervals actually differ
  before testing anything that depends on it. A constructed escaped fit
  (`z_lambda` pinned; frozen `[-2.3186, 2.7647]`, optimum `[-1.7987, 1.4202]`,
  `lambda = 2.73066`) asserts the production warning fires and
  `lambda_admissible` is `FALSE`. A new test covers all three `se_method`
  branches. The score identity is checked with the frozen interval supplied to
  both wrappers, plus a positive control asserting that omitting `lam_bounds`
  breaks it — the silent-default path that caused the original defect.
* Verified by reinstating the old wiring: 8 assertions across the three new
  fit-level tests fail, and pass again once the correct source is restored.

## Review fixes (2026-08-08 13:15 response review)

See `comments/response_2026-08-08-15-26-11.md`.

* **The Rcpp post-fit admissibility guard actually works now.** It mapped
  `z_lambda` through the interval recomputed at the optimum and then asked
  whether the result lay inside that same interval — true by construction, since
  `eps + (1-2eps)*plogis(z)` is in `(0,1)`, so the warning could never fire. It
  now maps `z_lambda` through the *frozen* interval the objective actually used
  and tests that value against the interval admissible at the optimum. With
  frozen `[-2, 2]`, optimum `[-0.5, 0.5]` and `z = 2`, the objective used
  `lambda = 1.523185266` (inadmissible) while the old code reported
  `0.3807963164` as admissible.
* **`fit$lambda` is the lambda that produced `fit$logLik`.** It was previously
  `z_lambda` remapped through the recomputed interval, so it need not have been
  the value the likelihood used.
* **New fields on `rpbnb_fit`**: `bounds` is the frozen interval the objective
  used (and the width `summary()`'s delta method needs), `bounds_at_optimum` is
  the interval admissible at the fitted parameters, and `lambda_admissible`
  records the comparison. `lambda_admissible` was previously computed but never
  reached the constructor.
* **Every covariance path now uses the objective's interval.**
  `bnbr_rp_scores_cpp()` gained a `lam_bounds` argument and the analytic and
  numeric Hessian paths receive the frozen interval, instead of the one
  recomputed at the optimum. Standard errors previously described a
  reparameterized likelihood that had not been optimized whenever the support
  bound moved during the fit.

## Review fixes (2026-08-08 12:12 response review)

See `comments/response_2026-08-08-13-01-57.md`.

* **The support bound uses the true scale for bounded distributions.** Scales
  and dispersions on the bound side were `exp(clamp(x, -20, 20))`. The floor is
  conservative, but the *upper cap* is not: uniform and triangular deviations
  are supported on `(-s, s)`, so capping `s` shrinks the attainable mean range
  and *widens* the admissible interval. With one uniform coefficient per margin,
  `log_w = 30`, loading `1e-9` and `m = 0.5`, the cap gave
  `[-2.0362897, 2.5327946]` — admitting `lambda = 2` — where the true support
  gives `[-1, 1]`. Now `exp(pmax(x, -20))`: floor only, never a cap.
* **The Rcpp optimizer's objective is now genuinely fixed-bound**, so its
  analytic gradient is its actual derivative. The interval was recomputed from
  the current parameters on every objective call while the kernels differentiate
  `lam` treating `lamLo`/`lamHi` as constants, so the optimized function carried
  `d(bound)/d(par)` terms the gradient omitted. Measured on a one-margin uniform
  fixture the gap was `2.048`, on every coordinate except `z_lambda` (the only
  one the bounds do not involve); it is now `5.7e-09` in R and `2.6e-08` in C++.
  `fit_rpbnb()` freezes the bound at the starting values and passes it via a new
  `lam_bounds` argument, matching what the TMB engine does.
* **`fit_rpbnb()` gained the TMB engine's post-fit admissibility check**: the
  interval is recomputed at the fitted parameters and a warning is raised when
  the fitted `lambda` has left it. The two engines now agree on semantics as
  well as on the interval — both freeze, both check afterwards. Where the bound
  is parameter-independent (normal or lognormal loaded in both margins) freezing
  is exact and the check is trivially satisfied.

## Review fixes (2026-08-08 08:38 response review)

See `comments/response_2026-08-08-11-31-05.md`.

* **The Rcpp engine now uses the same support bound as the TMB engine**
  (model-validity fix). The previous round replaced the finite-draw bound only
  inside `fit_rpbnb_tmb()`, leaving `rpbnb()`'s default `engine = "cpp"` with a
  parameter space that moved with `draws` — measured, `[-1.4242, 1.0995]` at 50
  draws against `[-1.0796, 1.0668]` at 800. `famoye_support_bounds()` now lives
  in `R/famoye_core.R` and is used by the Rcpp objective (R and C++ kernels),
  the post-fit bound reconstruction, the analytic Hessian, and the TMB fitter.
  Fitted log-likelihoods are unchanged; `lambda` moves by about `1e-3` from the
  reparameterization of `z_dep` through a narrower interval.
  `bnbr_rp_scores_cpp()` uses the same bound, without which `colSums(scores)`
  would no longer equal the analytic gradient and every OPG standard error would
  be quietly wrong.
* **Admissibility is derived from the unclipped mean support.** Applying the
  template's `eta` clamp before mapping to `c` capped the attainable mean, which
  stopped `c` reaching 0 at large dispersion and *widened* the interval: at
  `log_m = 20` the bound was `[-1, 8.97e6]` instead of `[-1, 1]`, admitting
  `lambda = 2`. The scale clamp is retained — it cannot widen the admissible
  set, only decide whether a coefficient varies — but the mean clamp is not part
  of the model and no longer enters the constraint.
* The per-draw `lambda_bounds_vec()` reduction is removed from the Rcpp
  objective's pass-1 loop, where it had become dead work.

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
