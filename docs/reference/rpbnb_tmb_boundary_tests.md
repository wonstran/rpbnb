# Boundary-corrected LR tests for an rpbnb_tmb_fit's boundary parameters

Runs a likelihood-ratio test for every random-coefficient scale
(`sd1:*`, `sd2:*`) and NB2 dispersion (`m1`, `m2`) of a fitted
TMB-engine random-parameter bivariate NB model, and merges them into one
table. These are the parameters whose null lies on the boundary of the
parameter space (scale = 0, or dispersion `m = 0` = Poisson), for which
`summary(fit)` reports no Wald `z`/`p`; each test uses the 50:50
chi-square boundary correction of [`lr_test()`](lr_test.md)
(`boundary = TRUE`).

## Usage

``` r
rpbnb_tmb_boundary_tests(
  fit,
  data,
  control = NULL,
  which = c("sd", "dispersion"),
  draws = fit$draws,
  force_parallel_gaussian = FALSE,
  sml_fallback = TRUE
)
```

## Arguments

- fit:

  A converged `rpbnb_tmb_fit` (from
  [`fit_rpbnb_tmb()`](fit_rpbnb_tmb.md) or `rpbnb(engine = "tmb")`).

- data:

  The data frame the model was fit on. Required – the fit object does
  not store it, and every restricted model is refit on the same data.
  With `standardize = TRUE`, pass the standardized data
  (`rpbnb(engine = "tmb", boundary_tests = TRUE)` does this
  automatically; a manual call needs
  `rpbnb:::.apply_scaling(data, fit$scaling)`).

- control:

  An [`rpbnb_tmb_control()`](rpbnb_tmb_control.md) for the restricted
  refits. Defaults to
  `rpbnb_tmb_control(print_level = 1, n_cores = fit$parallel$requested, max_workload = Inf)`
  – the same `n_cores` the original fit was called with (1 if `fit`
  predates the stored `$parallel` field). `seed`/`method` are taken from
  `fit`, not `control`.

- which:

  Which parameter groups to test, any subset of `"sd"` (the
  random-coefficient scales), `"dispersion"` (the NB2 dispersions `m1`,
  `m2`), and `"dependence"` (the association parameter). The default is
  `c("sd", "dispersion")` – the two boundary-null groups. `"dependence"`
  is opt-in because it costs another full refit and, for three of the
  four families, tests an *interior* null whose ordinary Wald `z` in
  [`summary()`](https://rdrr.io/r/base/summary.html) is already valid.

  The dependence test refits the model with
  `dependence = "independence"` – the template's own family, which maps
  `z_dep` out of the free parameters, giving an exact 1-df restriction
  on the same draws – and reports one row labelled `lam` (Famoye),
  `theta` (Frank / Kimeldorf), or `rho` (Gaussian). Only Kimeldorf's
  null (`theta > 0`, so `theta = 0` is a boundary) takes the 50:50
  mixture; Famoye, Frank, and Gaussian nulls are interior and get an
  ordinary chi-square(1).

- draws:

  Number of Halton simulation draws for the restricted refits. Defaults
  to `fit$draws` – the same number `fit` itself used. Raising this
  trades refit cost for precision without needing to refit `fit` at a
  higher `draws`; lowering it is cheaper but noisier. A value other than
  `fit$draws` still shares `fit$seed` (so the Halton sequence's *prefix*
  is identical) but no longer draws the exact same simulated
  log-likelihood surface as `fit`, so the LR statistic picks up
  simulation noise beyond the restriction under test – prefer the
  default unless you have a specific reason to diverge.

- force_parallel_gaussian:

  Opt-in override of the Gaussian-copula single-thread safety cap (see
  [`?fit_rpbnb_tmb`](fit_rpbnb_tmb.md)), forwarded to every restricted
  refit. Default `FALSE`. This is intentionally a separate argument
  rather than something read off `fit`: `fit` does not record whether
  the original fit used the override, so passing
  `force_parallel_gaussian = TRUE` here is required even when the
  original [`fit_rpbnb_tmb()`](fit_rpbnb_tmb.md)/[`rpbnb()`](rpbnb.md)
  call also passed it – otherwise every refit silently falls back to one
  thread regardless of `control$n_cores`.

- sml_fallback:

  When `fit` was estimated by `method = "laplace"` and a restricted
  refit's Laplace pair cannot be trusted, re-run **that one test's** LR
  with both sides estimated by `method = "sml"` instead of reporting
  `NA` (or a clamped 0). Default `TRUE`. Two triggers: the restricted
  refit **fails to converge**, or it reports convergence with a
  restricted logLik *above* the full fit's (a negative raw LR). The
  second is not a rounding curiosity: the models are nested, so at true
  optima the restricted logLik can never exceed the full fit's – a
  negative LR proves at least one Laplace value is wrong, either a full
  fit that stopped short of its optimum (observed at -0.002 to -1.2
  nats: the warm-started restricted refit out-polished it) or a
  restricted fit that climbed a spurious ridge of the approximation
  itself (observed at -3838 nats: near a singular inner Hessian the
  Laplace log-likelihood rises without bound, and nlminb reports code 0
  there). Clamping such a statistic to 0 would silently turn a real
  effect into "no evidence".

  This exists because some restrictions leave Laplace no valid optimum
  to converge to. Pinning a margin to Poisson can push the dependence
  strong enough (observed on the truck data's `m1` test under a Frank
  copula: theta driven to about 20, Kendall's tau 0.8) that the
  per-observation cell probability is non-log-concave in the random
  effects – and the Laplace approximation differentiates through an
  inner Newton that requires exactly the log-concavity the restricted
  model no longer has. No inner-solver setting fixes that (`tol10 = 0`
  clears TMB's `"Newton drop out"` but the outer fit still ends at
  nlminb `false convergence (8)` with a gradient of 1e10); SML has no
  inner Newton and is not exposed to it.

  The fallback refits the **full model too** under SML (once, cached
  across tests) at these same `draws` and `fit$seed`, and takes the LR
  between the two SML fits – never between a Laplace logLik and an SML
  logLik, which are different approximations of the likelihood and whose
  difference is not an LR statistic. Rows that used the fallback are
  listed in the result's `sml_fallback` attribute and announced by a
  [`message()`](https://rdrr.io/r/base/message.html) (silenced by
  `control$print_level = 0` like the other announcements). If the SML
  pair fails to converge as well, the row is `NA` with a warning,
  exactly as before. Ignored when `fit` itself was estimated by SML
  (there is nothing different to fall back to).

## Value

An object of class `rpbnb_boundary_tests` – the same class
[`rpbnb_boundary_tests()`](rpbnb_boundary_tests.md) returns, with
columns `Parameter`, `LR`, `df`, `p.value`, `Signif` (one row per
boundary parameter) and the same
[`print()`](https://rdrr.io/r/base/print.html) method – so the two
engines' results are interchangeable wherever that class is consumed
(e.g. [`summary.rpbnb_tmb_fit()`](summary.rpbnb_tmb_fit.md)'s scale and
dispersion blocks, once `$boundary_tests` is attached to the fit).

Scale rows are labelled by the distribution's own scale parameter (`sd`
for normal/lognormal, `w` for uniform/triangular half-width, `s` for a
lognormal log-scale), matching
[`summary()`](https://rdrr.io/r/base/summary.html)'s row names.

The `sml_fallback` attribute is a character vector of the `Parameter`
rows whose LR came from the SML fallback pair (see the `sml_fallback`
argument); `character(0)` when every test ran under `fit`'s own
estimator.

## Details

Each test refits a properly nested restricted model, otherwise identical
to `fit` (same formulas, random-coefficient specification, dependence,
seed, and estimator, and by default the same `draws`), warm-started from
`fit$coef` and run with `inference = "none"` since the LR test needs
only `logLik` and the parameter count.

**Dispersions** are restricted with `poisson_1`/`poisson_2 = TRUE`, the
template's exact `m = 0` branch.

**Scales** are restricted by pinning that coefficient's `log_sd` at the
parameterization's zero (`-20`; the template clamps `log_sd` to
`[-20, 20]` and computes `sd = exp(log_sd)`, so this is `sd = 2.1e-9` –
numerically zero on every draw) and holding it out of the free-parameter
count, giving a 1-df restriction. With **multiple random coefficients in
an equation, each scale is tested individually.**

Pinning the scale rather than dropping the coefficient from
`random_1`/`random_2` is what preserves **common random numbers**: the
Halton draw matrix keeps the same width, so every *other* random
coefficient draws from exactly the dimensions it did in the full fit,
and the two simulated log-likelihoods differ only by the restriction
under test. Dropping the name instead would renumber the remaining
coefficients' Halton dimensions, and the LR statistic would absorb that
reshuffling as extra simulation noise.

A [`message()`](https://rdrr.io/r/base/message.html)
(`"Boundary LR test: <parameter>..."`) reports each restricted refit
right before it starts, unless `control$print_level` is `0`; suppress it
with [`suppressMessages()`](https://rdrr.io/r/base/message.html) if
needed.

## See also

[`lr_test()`](lr_test.md), [`fit_rpbnb_tmb()`](fit_rpbnb_tmb.md),
[`rpbnb_boundary_tests()`](rpbnb_boundary_tests.md) (the classic-engine
counterpart)

## Examples

``` r
# \donttest{
sim <- simulate_rpbnb(n = 600,
  beta1 = c("(Intercept)" = 0.2, x1 = 0.4),
  beta2 = c("(Intercept)" = 0.1, x1 = -0.3),
  random_1 = list(x1 = list(sd = 0.5)),
  dispersion = c(m1 = 0.4, m2 = 0.5), seed = 1)
fit <- fit_rpbnb_tmb(y1 ~ x1, y2 ~ x1, data = sim$data, random_1 = "x1",
                     draws = 100)
rpbnb_tmb_boundary_tests(fit, sim$data)
#> Boundary LR test: sd1:x1...
#> outer mgc:  139.2252 
#> outer mgc:  51.00002 
#> outer mgc:  49.88855 
#> outer mgc:  55.14521 
#> outer mgc:  16.25192 
#> outer mgc:  8.173531 
#> outer mgc:  10.5244 
#> outer mgc:  4.567736 
#> outer mgc:  2.597854 
#> outer mgc:  4.536156 
#> outer mgc:  1.124805 
#> outer mgc:  2.39283 
#> outer mgc:  0.4442394 
#> outer mgc:  0.3235357 
#> outer mgc:  0.3192301 
#> outer mgc:  0.3054224 
#> outer mgc:  0.2758239 
#> outer mgc:  0.8867828 
#> outer mgc:  0.5359577 
#> outer mgc:  0.8050756 
#> outer mgc:  0.5829464 
#> outer mgc:  0.2183853 
#> outer mgc:  0.2644181 
#> outer mgc:  0.1881673 
#> outer mgc:  0.2574333 
#> outer mgc:  0.07384581 
#> outer mgc:  0.1808604 
#> outer mgc:  0.05734191 
#> outer mgc:  0.1274127 
#> outer mgc:  0.06357337 
#> outer mgc:  0.09432142 
#> outer mgc:  0.0507132 
#> outer mgc:  0.01744722 
#> outer mgc:  0.01744722 
#> outer mgc:  0.01744722 
#> Boundary LR test: m1...
#> outer mgc:  37.26544 
#> outer mgc:  8.650418 
#> outer mgc:  1.362515 
#> outer mgc:  0.6129178 
#> outer mgc:  0.3586363 
#> outer mgc:  0.2264818 
#> outer mgc:  0.3171213 
#> outer mgc:  0.2176361 
#> outer mgc:  0.2521139 
#> outer mgc:  0.5846424 
#> outer mgc:  0.3707246 
#> outer mgc:  0.2547623 
#> outer mgc:  0.2485095 
#> outer mgc:  0.1943857 
#> outer mgc:  0.246703 
#> outer mgc:  0.2962943 
#> outer mgc:  0.3118346 
#> outer mgc:  0.2524341 
#> outer mgc:  0.3016883 
#> outer mgc:  0.1706645 
#> outer mgc:  0.1666839 
#> outer mgc:  0.1906157 
#> outer mgc:  0.2058872 
#> outer mgc:  0.1823848 
#> outer mgc:  0.09934848 
#> outer mgc:  0.02268604 
#> outer mgc:  0.0240201 
#> outer mgc:  0.01740295 
#> outer mgc:  0.06784896 
#> outer mgc:  0.04714727 
#> outer mgc:  0.04714727 
#> outer mgc:  0.04714727 
#> outer mgc:  0.006849954 
#> outer mgc:  0.002137974 
#> outer mgc:  0.002137974 
#> outer mgc:  0.002137974 
#> Boundary LR test: m2...
#> outer mgc:  2.777529 
#> outer mgc:  0.585213 
#> outer mgc:  0.4106103 
#> outer mgc:  0.5379071 
#> outer mgc:  0.4008264 
#> outer mgc:  0.448099 
#> outer mgc:  0.9764165 
#> outer mgc:  1.181307 
#> outer mgc:  1.075618 
#> outer mgc:  0.966169 
#> outer mgc:  1.500775 
#> outer mgc:  0.8690968 
#> outer mgc:  0.713576 
#> outer mgc:  0.5986374 
#> outer mgc:  0.7075973 
#> outer mgc:  0.4255183 
#> outer mgc:  0.3547595 
#> outer mgc:  0.1147076 
#> outer mgc:  0.2731667 
#> outer mgc:  0.06477574 
#> outer mgc:  0.1139779 
#> outer mgc:  0.1554963 
#> outer mgc:  0.1851498 
#> outer mgc:  0.1367059 
#> outer mgc:  0.01306486 
#> outer mgc:  0.01796246 
#> outer mgc:  0.01796246 
#> outer mgc:  0.01796246 
#> Boundary-parameter LR tests (boundary-corrected, 50:50 chi-square mixture)
#> H0: parameter = 0 (random SD absent, or margin Poisson)
#> 
#>  Parameter      LR df p.value Signif
#>     sd1:x1 42.0711  1  0.0000    ***
#>         m1 16.9739  1  0.0000    ***
#>         m2 46.9738  1  0.0000    ***
#> 
#> Signif: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
# }
```
