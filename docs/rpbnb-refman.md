|  |  |
|----|----|
| Type: | Package |
| Title: | Random-Parameter Bivariate Negative Binomial Regression |
| Version: | 0.4.1 |
| Author: | Zhenyu Wang \[aut, cre\] |
| Maintainer: | Zhenyu Wang \<wonstran@hotmail.com\> |
| Description: | Maximum-likelihood estimation of bivariate negative binomial (NB2) regression models with Famoye/Sarmanov or discrete-copula (Frank, Gaussian, or Clayton) dependence, and maximum-simulated-likelihood estimation of a bivariate random-parameter negative binomial model under either dependence structure, with a random-coefficient data simulator, diagnostics, and standard model methods. Two estimation engines are provided: an OpenMP/Rcpp maximum-simulated-likelihood engine and a TMB automatic-differentiation engine that additionally offers a Laplace approximation, with a common front end. |
| License: | [MIT](https://opensource.org/licenses/mit-license.php) + file LICENSE |
| Encoding: | UTF-8 |
| Depends: | R (≥ 4.1) |
| Imports: | stats, compiler, maxLik, numDeriv, randtoolbox, MASS, pbivnorm, Rcpp, parallel, graphics, TMB |
| LinkingTo: | Rcpp, TMB, RcppEigen |
| Suggests: | testthat (≥ 3.1.5), knitr, rmarkdown, pkgload |
| SystemRequirements: | GNU make |
| Config/testthat/edition: | 3 |
| VignetteBuilder: | knitr |
| Roxygen: | list(markdown = TRUE) |
| LazyData: | false |
| Config/roxygen2/version: | 8.0.0 |

------------------------------------------------------------------------

## rpbnb: Random-Parameter Bivariate Negative Binomial Regression

### Description

Maximum-likelihood estimation of bivariate negative binomial models with
Famoye/Sarmanov or discrete-copula (Frank, Gaussian, Clayton) dependence
(see [`fit_bnb()`](#topic+fit_bnb), [`copula()`](#topic+copula)), and
maximum-simulated-likelihood estimation of a bivariate random-parameter
negative binomial model under either dependence structure.

### Details

Two estimation engines are available for the random-parameter model:
[`fit_rpbnb()`](#topic+fit_rpbnb) (Rcpp/OpenMP simulated likelihood,
`maxLik` BFGS, supports `offset()`) and
[`fit_rpbnb_tmb()`](#topic+fit_rpbnb_tmb) (TMB automatic
differentiation, `nlminb`, adds a Laplace approximation and dependence
profiling). [`rpbnb()`](#topic+rpbnb) is a common front end that
dispatches to either.

### Author(s)

**Maintainer**: Zhenyu Wang <wonstran@hotmail.com>

Authors:

- Zhenyu Wang <wonstran@hotmail.com>

------------------------------------------------------------------------

## Elasticities and semi-elasticities for a bivariate NB model

### Description

For continuous regressors, the elasticity is `\beta_j E[x_j]`; for
binary (0/1) regressors, the semi-elasticity is `\exp(\beta_j) - 1`
(proportional change moving from 0 to 1).

### Usage

``` R
bnb_elasticities(
  fit,
  which = c("y1", "y2", "both"),
  type = c("AME", "MEM"),
  vars = NULL,
  include_intercept = FALSE,
  digits = 4,
  print_output = TRUE
)
```

### Arguments

|  |  |
|----|----|
| `fit` | A `bnb_fit` object from [`fit_bnb()`](#topic+fit_bnb). |
| `which` | Which margin(s): "y1", "y2", or "both". |
| `type` | "AME" (uses sample mean of x) or "MEM" (uses mean design row). |
| `vars` | Optional variable names to restrict output. |
| `include_intercept` | Logical; include the intercept term. |
| `digits` | Number of decimal places for printed output. |
| `print_output` | Logical; if `FALSE`, suppress printing. |

### Value

A data frame (single margin) or a named list of data frames (both), each
with columns `Name`, `Estimate`, `StdErr`, `z`, `p`, `Signif`,
`var_type`.

### Examples

``` R
d <- read.csv(system.file("extdata", "rwm1984_clean.csv", package = "rpbnb"))
fit <- fit_bnb(docvis ~ outwork + age, hospvis ~ outwork, data = d,
               dependence = "famoye")
bnb_elasticities(fit, which = "both", type = "AME")
```

------------------------------------------------------------------------

## Goodness of fit for a bivariate NB model

### Description

Reports log-likelihood, AIC, BIC, and four pseudo-R-squared measures
(McFadden, McFadden adjusted, Cox-Snell, Nagelkerke) relative to an
intercept-only null model refit with the same dependence structure
(copula fits rebuild the copula from `fit$cop_family`). A margin fitted
with an `offset()` keeps that offset in the null (an
intercept-plus-offset model), so the null shares the full model's
exposure/sampling structure and the pseudo-R-squared values remain
comparable. Pseudo-R-squared values are returned raw and are not clamped
to `⁠[0, 1]⁠`; a negative value flags a full model that fits worse than
the null.

### Usage

``` R
bnb_gof(fit, digits = 4, print_output = TRUE)
```

### Arguments

|  |  |
|----|----|
| `fit` | A `bnb_fit` object from [`fit_bnb()`](#topic+fit_bnb). |
| `digits` | Number of decimal places for printed output. |
| `print_output` | Logical; if `FALSE`, suppress printing and return the result invisibly. |

### Value

Invisibly, a list with `n`, `k`, `logLik_full`, `logLik_null`, `AIC`,
`BIC`, a named numeric vector `pseudoR2`, and the `null_fit`.

### Examples

``` R
d <- read.csv(system.file("extdata", "rwm1984_clean.csv", package = "rpbnb"))
fit <- fit_bnb(docvis ~ outwork + age, hospvis ~ outwork, data = d,
               dependence = "famoye")
bnb_gof(fit)
```

------------------------------------------------------------------------

## Marginal effects for a bivariate NB model

### Description

Average marginal effects (AME) or marginal effects at the mean (MEM) for
each margin. Continuous and binary (0/1) regressors are auto-detected;
continuous effects use `\partial E[Y]/\partial x_j = \beta_j \mu` and
binary effects use `E[Y|x_j=1] - E[Y|x_j=0]`.

### Usage

``` R
bnb_marginal_effects(
  fit,
  which = c("y1", "y2", "both", "all"),
  type = c("AME", "MEM"),
  vars = NULL,
  include_intercept = FALSE,
  digits = 4,
  print_output = TRUE
)
```

### Arguments

|  |  |
|----|----|
| `fit` | A `bnb_fit` object from [`fit_bnb()`](#topic+fit_bnb). |
| `which` | Which margin(s): "y1", "y2", "both", or "all". |
| `type` | "AME" (average marginal effect) or "MEM" (effect at the mean). |
| `vars` | Optional variable names or indices to restrict output. |
| `include_intercept` | Logical; include the intercept term. |
| `digits` | Number of decimal places for printed output. |
| `print_output` | Logical; if `FALSE`, suppress printing. |

### Details

For a model fitted with an `offset()`, absolute marginal effects scale
with the fitted mean `\mu = \exp(x'\beta + \text{offset})`: AME uses
each observation's training offset, and MEM uses the mean training
offset.

### Value

A data frame (single margin) or a named list of data frames (both), each
with columns `Name`, `Estimate`, `StdErr`, `z`, `p`, `Signif`,
`var_type`.

### Examples

``` R
d <- read.csv(system.file("extdata", "rwm1984_clean.csv", package = "rpbnb"))
fit <- fit_bnb(docvis ~ outwork + age, hospvis ~ outwork, data = d,
               dependence = "famoye")
bnb_marginal_effects(fit, which = "y1", type = "AME")
```

------------------------------------------------------------------------

## Residual checks for a bivariate NB model

### Description

Formal residual diagnostics for a [`fit_bnb()`](#topic+fit_bnb) or
[`fit_rpbnb()`](#topic+fit_rpbnb) fit: a normality test on the
randomized quantile residuals (Shapiro-Wilk for `n <= 5000`, else
Kolmogorov-Smirnov vs N(0,1)); the NB2 dispersion statistic
(`sum(pearson^2) / (n - k)`) per margin; the cross-margin RQR
correlation (a check that the Famoye/copula dependence captured the
association); an outlier count/index list (`⁠|RQR| > outlier_z⁠`); and a
composite misspecification verdict combining these signals.

### Usage

``` R
bnb_residual_checks(
  fit,
  seed = NULL,
  outlier_z = 3,
  digits = 4,
  print_output = TRUE
)
```

### Arguments

|                |                                                         |
|----------------|---------------------------------------------------------|
| `fit`          | A `bnb_fit` or `rpbnb_fit` object.                      |
| `seed`         | Optional integer seed for the RQR randomization.        |
| `outlier_z`    | Absolute-RQR threshold flagging an outlier (default 3). |
| `digits`       | Decimal places for printed output.                      |
| `print_output` | Logical; if `FALSE`, suppress printing.                 |

### Value

Invisibly, an object of class `bnb_residual_checks`.

------------------------------------------------------------------------

## Specify a copula dependence structure

### Description

Pass the result as the `dependence` argument to
[`fit_bnb()`](#topic+fit_bnb) or [`fit_rpbnb()`](#topic+fit_rpbnb) to
estimate a discrete-copula bivariate NB model instead of the default
Famoye/Sarmanov dependence, or as the `copula` argument to
[`simulate_rpbnb_copula()`](#topic+simulate_rpbnb_copula) to simulate
from one. The joint pmf is built from the two NB2 marginal CDFs and the
chosen copula CDF (rectangle differencing); see the package vignette for
the "Copula dependence" section.

### Usage

``` R
copula(family = c("frank", "normal", "kimeldorf"), par = NULL)
```

### Arguments

|  |  |
|----|----|
| `family` | One of `"frank"`, `"normal"` (Gaussian), or `"kimeldorf"` (Clayton). |
| `par` | Optional native dependence parameter (Frank's theta, the Gaussian rho, or Clayton's theta) used by [`simulate_rpbnb_copula()`](#topic+simulate_rpbnb_copula) to generate data. Ignored by [`fit_bnb()`](#topic+fit_bnb) and [`fit_rpbnb()`](#topic+fit_rpbnb), which estimate the parameter from the data. |

### Value

An object of class `rpbnb_copula`.

### Examples

``` R
copula("frank")
copula("normal", par = 0.3)
copula("kimeldorf")
```

------------------------------------------------------------------------

## Fit a bivariate negative binomial regression model

### Description

Fit a bivariate negative binomial regression model

### Usage

``` R
fit_bnb(
  formula_1,
  formula_2,
  data,
  dependence = c("independence", "famoye"),
  start = NULL,
  control = rpbnb_control(),
  poisson_1 = FALSE,
  poisson_2 = FALSE
)
```

### Arguments

|  |  |
|----|----|
| `formula_1`, `formula_2` | Model formulas for the two count outcomes. An equation-specific `offset()` term (e.g. `y ~ x + offset(log(exposure))`) is supported on every dependence path: the offset enters that margin's linear predictor additively (mean `⁠exp(x'beta + offset)⁠`) during estimation, and is carried through the stored fitted means and both `predict()` methods. |
| `data` | A data frame. |
| `dependence` | Dependence structure: "independence" (two univariate NB2 margins), "famoye" (Famoye/Sarmanov bivariate NB), or a [`copula()`](#topic+copula) object (Frank / Gaussian / Clayton discrete-copula bivariate NB; the dependence parameter is estimated). |
| `start` | Optional starting parameter vector. May be positional (length equal to the number of parameters) or named; a named vector is reordered to the canonical parameter order and a named partial vector is merged into the defaults (unknown or duplicate names are rejected). When `start` is `NULL`, the famoye path uses a multi-start policy: it optimizes from both an all-zero mean-coefficient start and marginal `glm.nb` starts and keeps the better converged objective (the frozen-bounds gradient makes the objective start-sensitive and neither start dominates). |
| `control` | An [`rpbnb_control()`](#topic+rpbnb_control) object. The famoye and copula estimators both use BFGS, the only optimizer `control$method` accepts. One control object serves every estimator in the package; settings this one does not read – `se_method`, `n_cores`, `halton_burn`, `draws_hessian`, and the TMB knobs – are ignored and listed by `print()`/`summary()` of the fit. |
| `poisson_1`, `poisson_2` | Fit the corresponding margin at its exact Poisson limit (NB2 dispersion `m = 0`) instead of estimating the dispersion. The margin's `log_m` is held fixed, so it is not a free parameter and the fit is a properly nested restriction of the NB model – pair it with [`lr_test()`](#topic+lr_test) (`boundary = TRUE`) to test for overdispersion. This is the exact `m = 0` restriction at any fitted mean: the margin's log-pmf is `dpois` and its Famoye dependence constant is `exp(-d*mu)` (the `m -> 0` limit), not an NB2 at a tiny pinned dispersion. The famoye and independence paths are both exact (the independence path fits a Poisson GLM margin). Not supported with a [`copula()`](#topic+copula) dependence. |

### Value

An object of class `bnb_fit`.

### Examples

``` R
d <- read.csv(system.file("extdata", "rwm1984_clean.csv", package = "rpbnb"))
fit <- fit_bnb(docvis ~ outwork, hospvis ~ outwork, data = d,
               dependence = "famoye")
summary(fit)

# Overdispersion test for margin 1 (H0: m1 = 0, Poisson)
fit_p1 <- fit_bnb(docvis ~ outwork, hospvis ~ outwork, data = d,
                  dependence = "famoye", poisson_1 = TRUE)
lr_test(fit_p1, fit, boundary = TRUE)

# Gaussian copula dependence instead of Famoye/Sarmanov
fit_cop <- fit_bnb(docvis ~ outwork, hospvis ~ outwork, data = d,
                   dependence = copula("normal"))
fit_cop$cop_tau  # estimated Kendall's tau
```

------------------------------------------------------------------------

## Fit a bivariate random-parameter negative binomial model

### Description

Maximum simulated likelihood estimation with normal random coefficients
and Famoye/Sarmanov dependence. Random coefficients are selected per
equation by name via `random_1` / `random_2`.

### Usage

``` R
fit_rpbnb(
  formula_1,
  formula_2,
  data,
  random_1 = NULL,
  random_2 = NULL,
  draws = 400,
  draw_type = "halton",
  seed = 1234,
  start = NULL,
  control = rpbnb_control(),
  dependence = "famoye",
  poisson_1 = FALSE,
  poisson_2 = FALSE,
  .fixed = NULL,
  .opt_draws = NULL
)
```

### Arguments

|  |  |
|----|----|
| `formula_1`, `formula_2` | Model formulas for the two count outcomes. An equation-specific `offset()` term (e.g. `y ~ x + offset(log(exposure))`) is supported: the offset enters that margin's linear predictor additively (integrated mean `⁠E[exp(x'beta + offset)]⁠`) during estimation and is carried through the stored fitted means and `predict()`. |
| `data` | A data frame. |
| `random_1`, `random_2` | Random coefficients per equation. Either a character vector of `model.matrix` column names (all Normal), or a named list whose values are a distribution name (`"normal"`, `"lognormal"`, `"uniform"`, `"triangular"`) or a list `list(dist = ..., sign = ...)` (`sign` is -1/1 and lognormal-only). NULL means all-fixed for that equation. |
| `draws` | Number of simulation draws for the optimization. |
| `draw_type` | Quasi-random draw type. Only "halton" is supported in this version. |
| `seed` | Random seed for the simulation draws. |
| `start` | Optional starting parameter vector. |
| `control` | An [`rpbnb_control()`](#topic+rpbnb_control) object. Estimation uses BFGS, the only optimizer `control$method` accepts. One control object serves every estimator in the package; settings this one does not read – the TMB knobs (`gradtol`, `restarts`, `max_threads`, `max_workload`, `parallel_tape`), `hessian`, and `draws_hessian` – are ignored and listed by `print()`/`summary()` of the fit rather than rejected. |
| `dependence` | Dependence structure: "famoye" (default; Famoye/Sarmanov) or an [`copula()`](#topic+copula) object for copula dependence (Frank / Gaussian / Clayton). Both paths use the multithreaded (OpenMP) C++ simulated likelihood; the copula path is more numerically expensive per evaluation (discrete-copula pmf + per-draw NB CDF corners), so fits typically take noticeably longer than the Famoye path at comparable `draws`/`n`. Random coefficients on 0/1 dummy regressors are weakly identified under the copula path (NB dispersion trades off against the random-coefficient scale); prefer random coefficients on continuous regressors. |
| `poisson_1`, `poisson_2` | Fit the corresponding margin at its exact Poisson limit (NB2 dispersion `m = 0`): the margin's `log_m` is held fixed, so it is not a free parameter and the fit is a properly nested restriction of the NB model. Pair with [`lr_test()`](#topic+lr_test) (`boundary = TRUE`) to test a margin for overdispersion (`H0: m = 0`). The simulated likelihood uses the exact `m = 0` branch – the per-draw margin log-pmf is `dpois` and its Famoye dependence constant is `exp(-d*mu)` – so it is accurate at any fitted mean, not a fixed-dispersion approximation. Supported with both Famoye/Sarmanov and [`copula()`](#topic+copula) dependence (the copula path uses the same exact m = 0 branch). |
| `.fixed` | Internal. A named numeric vector of parameters (in the optimization/log-scale parameterization) to pin at the supplied values and hold fixed during estimation. Used by [`rpbnb_boundary_tests()`](#topic+rpbnb_boundary_tests) to construct scale-zero (SD boundary) restricted fits; not intended for direct use. |
| `.opt_draws` | Internal. A list `list(Z1, Z2)` of uniform Halton draw matrices to use verbatim instead of generating them, so a restricted refit reuses a full fit's draws (common random numbers). Used by [`rpbnb_boundary_tests()`](#topic+rpbnb_boundary_tests); not intended for direct use. |

### Value

An object of class `rpbnb_fit`. Under Famoye dependence three fields
describe the admissible lambda interval: `bounds` is the interval the
optimized likelihood actually used, frozen at the starting values (its
width also feeds `summary()`'s delta-method standard error for lambda);
`bounds_at_optimum` is the interval admissible at the fitted parameters,
recomputed after optimization for validation only; and
`lambda_admissible` records whether `lambda` — the fitted value, mapped
through `bounds` — lies inside `bounds_at_optimum`. `FALSE` means the
optimizer escaped the valid region (a warning is raised) and the fit
should be re-run from starting values closer to the optimum. For normal
or lognormal random coefficients loaded in both margins the bound is the
constant `c(-1, 1)`, the two intervals coincide, and escape cannot
occur.

### Examples

``` R
sim <- simulate_rpbnb(n = 600,
  beta1 = c("(Intercept)" = 0.2, x1 = 0.4),
  beta2 = c("(Intercept)" = 0.1, x1 = -0.3),
  random_1 = list(x1 = list(sd = 0.5)),
  dispersion = c(m1 = 0.4, m2 = 0.5), seed = 1)
fit <- fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim$data, random_1 = "x1",
                 draws = 100, control = rpbnb_control(compute_se = FALSE))
coef(fit)


# Copula dependence instead of Famoye/Sarmanov (slower; fewer draws here
# for a quick example -- use more in practice)
sim_cop <- simulate_rpbnb_copula(n = 600,
  beta1 = c("(Intercept)" = 0.2, x1 = 0.4),
  beta2 = c("(Intercept)" = 0.1, x1 = -0.3),
  random_1 = list(x1 = list(sd = 0.3)),
  dispersion = c(m1 = 0.4, m2 = 0.5),
  copula = copula("normal", par = 0.5), seed = 1)
fit_cop <- fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim_cop$data, random_1 = "x1",
                     dependence = copula("normal"), draws = 100,
                     control = rpbnb_control(compute_se = FALSE))
tanh(coef(fit_cop)[["z_theta"]])  # estimated copula rho
```

------------------------------------------------------------------------

## Fit a bivariate random-parameter negative binomial model (TMB)

### Description

Maximum simulated likelihood via TMB (Template Model Builder) with
automatic differentiation. Supports Famoye/Sarmanov and discrete-copula
(Frank, Gaussian, Clayton) dependence.

### Usage

``` R
fit_rpbnb_tmb(
  formula_1,
  formula_2,
  data,
  random_1 = NULL,
  random_2 = NULL,
  draws = 400L,
  seed = 1234L,
  start = NULL,
  dependence = "famoye",
  control = rpbnb_tmb_control(),
  inference = c("full", "diag", "none"),
  keep = c("postfit", "compact", "full"),
  poisson_1 = FALSE,
  poisson_2 = FALSE,
  method = c("sml", "laplace"),
  force_parallel_gaussian = FALSE,
  .fixed = NULL
)
```

### Arguments

[TABLE]

### Value

An object of class `rpbnb_tmb_fit`. The `sdreport` field is a compact
package-owned summary and does not retain a second TMB tape.

`optimizer` is the `nlminb()` return value, with three fields added:
`restarts` counts how many times the solve was restarted from its own
answer to clear `control$gradtol`, `max_abs_gradient` is the score norm
actually achieved, and `gradient_tolerance` is the threshold it was
judged against. `convergence` and `message` come from nlminb's relative
*function* test and can report success at a point that is not
stationary, so `max_abs_gradient` is the field to check before trusting
a standard error.

`method` records which estimator produced the fit, echoing the `method`
argument (`"sml"` or `"laplace"`). Both `print()` and `summary()`
display it, so two fits of the same model under different estimators are
distinguishable in their printed output.

`boundary_report` is a character vector naming the reported quantities
whose estimates are pinned against an implementation bound – the frozen
Famoye interval, the Frank overflow guard, or an `exp()` clamp – or
whose delta-method derivative has collapsed numerically. Those estimates
are set by the implementation rather than identified by the data, so
their standard errors and covariances are reported as `NA` and a warning
is raised. An empty vector means no such constraint bound.

`boundary_sides` names, for each entry of `boundary_report`, which end
was reached: `"lower"`, `"upper"`, or `"degenerate"` when the derivative
collapsed rather than a bound being hit. The two sides call for opposite
remedies – a lower dispersion clamp means the margin is effectively
Poisson, an upper one means it is degenerately over-dispersed – so this
is retained on the fit rather than only mentioned in the warning.

`lambda_bounds` is a named numeric vector `c(lower =, upper =)` giving
the admissible Famoye dependence interval, and `NULL` for every other
dependence structure. The bounds *used by the likelihood* are computed
once at the starting values and held fixed for the whole fit. A `lam`
estimate at either end is therefore an artefact of the starting values
rather than a property of the data, which is why the field is exposed.
[`rpbnb_tmb_dependence_profile`](#topic+rpbnb_tmb_dependence_profile)
reports a likelihood-based interval where the delta-method standard
error collapses to `NA`, but for Famoye that interval is mapped through
this same frozen box and so cannot escape it: widen the box by refitting
from better starting values before treating such an interval as
informative.

`lambda_bounds_at_optimum` is the interval admissible at the *fitted*
parameters, recomputed after optimization for validation only — it never
enters the likelihood. `lambda_admissible` records whether the fitted
`lam` (mapped through the frozen `lambda_bounds`, i.e. the value the
objective actually used) lies inside it; `FALSE` means the optimizer
left the region where the joint pmf is a valid probability model, a
warning was raised, and the fit should not be interpreted — refit from
starting values closer to the optimum. When the bound is
parameter-independent (normal or lognormal coefficients loaded in both
margins, where it is the constant `c(-1, 1)`), the two intervals
coincide and the flag is trivially `TRUE`. Both fields are `NULL`/`NA`
outside Famoye dependence.

### Examples

``` R
## Not run: 
sim <- simulate_rpbnb_tmb(n = 300,
  beta1 = c("(Intercept)" = 0.2, x1 = 0.4),
  beta2 = c("(Intercept)" = 0.1, x1 = -0.3),
  dispersion = c(m1 = 0.4, m2 = 0.5), seed = 1)
fit <- fit_rpbnb_tmb(y1 ~ x1, y2 ~ x1, data = sim$data, draws = 100)
coef(fit)

## End(Not run)
```

------------------------------------------------------------------------

## Likelihood-ratio test between two nested model fits

### Description

Compares a restricted fit against a full (nesting) fit by the
likelihood-ratio statistic. Works for any [`fit_bnb()`](#topic+fit_bnb)
/ [`fit_rpbnb()`](#topic+fit_rpbnb) objects, since both carry a
`logLik()` with a `"df"` attribute equal to the number of estimated
parameters.

### Usage

``` R
lr_test(restricted, full, boundary = FALSE)
```

### Arguments

|  |  |
|----|----|
| `restricted` | The smaller (restricted) fit – fewer estimated parameters. |
| `full` | The larger (full) fit that nests `restricted`. |
| `boundary` | Logical. When the restriction pins a variance/dispersion-type parameter (a random-coefficient SD, or NB2 dispersion `m`) to its zero boundary, the null distribution is not a plain chi-square. Set `boundary = TRUE` to use the 50:50 mixture of `chisq(df)` and `chisq(df - 1)` (Self & Liang, 1987); for a single boundary parameter (`df = 1`) this halves the naive p-value. Default `FALSE` (ordinary interior restriction, e.g. dropping a fixed covariate). The mixture is exact only for a single parameter on the boundary; simultaneous boundary restrictions of several parameters need different weights and are not handled. |

### Details

The restricted model is one you fit yourself with a term removed – for
example dropping a name from `random_1` (testing a random-coefficient
SD), or a plain NB / independence fit (testing an NB2 dispersion). This
is the statistically appropriate replacement for the Wald z/p that the
natural-scale summary suppresses on positive scale/dispersion
parameters.

A likelihood-ratio test requires two *maximized* likelihoods. When
either argument is a package fit (`bnb_fit` / `rpbnb_fit`) that records
a failed optimization (`convergence$converged = FALSE`), `lr_test()`
errors rather than returning a p-value from an unfinished fit – the sign
of the statistic cannot establish convergence. Generic objects that only
carry a `logLik()` (no convergence record) are still accepted, but their
convergence cannot be validated and is the caller's responsibility.

### Value

An object of class `rpbnb_lrtest` with the LR `statistic`, degrees of
freedom `df`, `p.value`, the two log-likelihoods and their df, and the
`boundary` flag. Has a `print` method.

### Famoye caveat for bounded random-coefficient distributions

Each Famoye fit maximizes its likelihood over an admissible lambda
interval frozen at that fit's own starting values. With normal or
lognormal random coefficients loaded in both margins the interval is the
constant `c(-1, 1)` for every fit, so the comparison is unaffected. With
uniform or triangular coefficients (or a single varying margin) the
bound moves with the parameters, so two fits frozen at different
starting values maximize over slightly different lambda ranges and the
LR statistic inherits that second-order discrepancy. Check
`lambda_admissible` on both fits before relying on a comparison in that
regime.

### References

Self, S. G. and Liang, K.-Y. (1987). Asymptotic properties of maximum
likelihood estimators and likelihood ratio tests under nonstandard
conditions. *JASA* 82(398), 605–610.

### Examples

``` R
sim <- simulate_rpbnb(n = 600,
  beta1 = c("(Intercept)" = 0.2, x1 = 0.4),
  beta2 = c("(Intercept)" = 0.1, x1 = -0.3),
  random_1 = list(x1 = list(sd = 0.5)),
  dispersion = c(m1 = 0.4, m2 = 0.5), seed = 1)
ctrl <- rpbnb_control(compute_se = FALSE)
full <- fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim$data, random_1 = "x1",
                  draws = 100, control = ctrl)
rest <- fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim$data,
                  draws = 100, control = ctrl)   # no random coefficient
lr_test(rest, full, boundary = TRUE)             # test sd(x1) = 0
```

------------------------------------------------------------------------

## Residual diagnostic plots for a bivariate NB model

### Description

Four base-graphics panels per margin: residuals-vs-fitted, a normal QQ
plot of the randomized quantile residuals, a histogram of the RQR with
an N(0,1) overlay, and a scale-location plot. The QQ and histogram
panels always use RQR (only these are approximately N(0,1) under a
correct count model).

### Usage

``` R
## S3 method for class 'bnb_fit'
plot(
  x,
  margin = c("both", "y1", "y2"),
  which = 1:4,
  resid_type = "quantile",
  seed = NULL,
  ...
)
```

### Arguments

|  |  |
|----|----|
| `x` | A `bnb_fit` object from [`fit_bnb()`](#topic+fit_bnb). |
| `margin` | Which margin to plot: `"both"` (default), `"y1"`, or `"y2"`. |
| `which` | Integer subset of panels `1:4` (1 = residuals-vs-fitted, 2 = QQ, 3 = histogram, 4 = scale-location). |
| `resid_type` | Residual type for panels 1 and 4: `"quantile"` (default), `"pearson"`, `"deviance"`, or `"response"`. |
| `seed` | Optional integer seed for the RQR randomization. |
| `...` | Unused. |

### Value

`NULL`, invisibly (called for the side effect of drawing).

------------------------------------------------------------------------

## Residual diagnostic plots for a random-parameter bivariate NB model

### Description

Four base-graphics panels per margin, as for
[`plot.bnb_fit()`](#topic+plot.bnb_fit), built on the mixture-based
randomized quantile residuals. `resid_type = "deviance"` is not
available for `rpbnb_fit`.

### Usage

``` R
## S3 method for class 'rpbnb_fit'
plot(
  x,
  margin = c("both", "y1", "y2"),
  which = 1:4,
  resid_type = "quantile",
  seed = NULL,
  ...
)
```

### Arguments

|  |  |
|----|----|
| `x` | An `rpbnb_fit` object from [`fit_rpbnb()`](#topic+fit_rpbnb). |
| `margin` | Which margin to plot: `"both"` (default), `"y1"`, or `"y2"`. |
| `which` | Integer subset of panels `1:4`. |
| `resid_type` | Residual type for panels 1 and 4: `"quantile"` (default), `"pearson"`, or `"response"`. |
| `seed` | Optional integer seed for the RQR randomization. |
| `...` | Unused. |

### Value

`NULL`, invisibly.

------------------------------------------------------------------------

## Predict from a fitted bivariate count model

### Description

Predictions integrate over the retained simulation draws when random
coefficients are present. Link predictions are the log of the integrated
response mean, so `exp(predict(fit, type = "link"))` equals response
predictions.

### Usage

``` R
## S3 method for class 'rpbnb_tmb_fit'
predict(
  object,
  newdata = NULL,
  type = c("response", "link"),
  which = c("both", "y1", "y2"),
  ...
)
```

### Arguments

|  |  |
|----|----|
| `object` | A fitted `rpbnb_tmb_fit` object. |
| `newdata` | Optional data frame. If omitted, the retained fitting design is used; compact fits require `newdata`. |
| `type` | Either `"response"` or `"link"`. |
| `which` | Return both margins or only `"y1"` or `"y2"`. |
| `...` | Reserved for future use. |

### Value

A two-column numeric matrix for `which = "both"`, otherwise a numeric
vector.

------------------------------------------------------------------------

## Residuals for a bivariate NB model

### Description

Per-margin residuals for a fixed-coefficient
[`fit_bnb()`](#topic+fit_bnb) model. Each margin is NB2 with fitted mean
`mu` and dispersion `m` (`size = 1/m`). `"quantile"` returns randomized
quantile residuals (Dunn & Smyth 1996), which are approximately N(0,1)
under a correct model and are the recommended residual for
normality-style diagnostics on count data.

### Usage

``` R
## S3 method for class 'bnb_fit'
residuals(
  object,
  type = c("quantile", "pearson", "deviance", "response"),
  margin = c("both", "y1", "y2"),
  seed = NULL,
  ...
)
```

### Arguments

|  |  |
|----|----|
| `object` | A `bnb_fit` object from [`fit_bnb()`](#topic+fit_bnb). |
| `type` | Residual type: `"quantile"` (default), `"pearson"`, `"deviance"`, or `"response"`. |
| `margin` | Which margin: `"both"` (default), `"y1"`, or `"y2"`. |
| `seed` | Optional integer seed for the quantile-residual randomization (ignored for other types); does not disturb the caller's RNG stream. |
| `...` | Unused. |

### Value

A numeric vector for a single margin, or a two-column data frame (`y1`,
`y2`) for `margin = "both"`.

------------------------------------------------------------------------

## Residuals for a random-parameter bivariate NB model

### Description

Per-margin residuals for an [`fit_rpbnb()`](#topic+fit_rpbnb) model.
Each margin is a mixture of NB2 distributions over the
random-coefficient draws. `"quantile"` returns randomized quantile
residuals (Dunn & Smyth 1996) from the exact mixture predictive CDF (the
recommended residual); `"pearson"` uses the exact mixture marginal
variance. `"deviance"` is not defined for the mixture and errors.

### Usage

``` R
## S3 method for class 'rpbnb_fit'
residuals(
  object,
  type = c("quantile", "pearson", "deviance", "response"),
  margin = c("both", "y1", "y2"),
  seed = NULL,
  ...
)
```

### Arguments

|  |  |
|----|----|
| `object` | An `rpbnb_fit` object from [`fit_rpbnb()`](#topic+fit_rpbnb). |
| `type` | Residual type: `"quantile"` (default), `"pearson"`, or `"response"`. `"deviance"` is not supported for `rpbnb_fit`. |
| `margin` | Which margin: `"both"` (default), `"y1"`, or `"y2"`. |
| `seed` | Optional integer seed for the quantile-residual randomization; does not disturb the caller's RNG stream. |
| `...` | Unused. |

### Value

A numeric vector for a single margin, or a two-column data frame (`y1`,
`y2`) for `margin = "both"`.

------------------------------------------------------------------------

## Fit a random-parameter bivariate NB model with either engine

### Description

A common front end over the package's two estimation engines.
`engine = "classic"` calls [`fit_rpbnb()`](#topic+fit_rpbnb)
(Rcpp/OpenMP simulated likelihood, `maxLik` BFGS); `engine = "tmb"`
calls [`fit_rpbnb_tmb()`](#topic+fit_rpbnb_tmb) (TMB automatic
differentiation, `nlminb` with restart polish). Both fitters remain
exported and can be called directly; with `standardize = FALSE` (the
default) this wrapper adds nothing to the fit itself and returns exactly
what the chosen fitter returns.

### Usage

``` R
rpbnb(
  formula_1,
  formula_2,
  data,
  engine = c("classic", "tmb"),
  random_1 = NULL,
  random_2 = NULL,
  draws = 400,
  seed = 1234,
  start = NULL,
  dependence = "famoye",
  poisson_1 = FALSE,
  poisson_2 = FALSE,
  standardize = FALSE,
  continuous_vars = NULL,
  boundary_tests = FALSE,
  boundary_draws = NULL,
  control = NULL,
  ...
)
```

### Arguments

[TABLE]

### Details

What it does add is argument checking across the two APIs. The engines
do not take the same arguments, and passing one engine's argument to the
other is a mistake that is easy to make and expensive to notice — a
standardized coefficient table printed under an "original units" heading
looks perfectly plausible. Every extra argument is therefore matched by
name against the selected fitter's own formals, and anything that does
not belong is an error rather than a silently ignored `...` entry.

### Value

The engine-native fit object, identical to what a direct call to the
underlying fitter would return: an object of class `rpbnb_fit` for
`engine = "classic"`, or `rpbnb_tmb_fit` for `engine = "tmb"`. The class
therefore depends on `engine`; test with
`inherits(fit, "rpbnb_tmb_fit")` if you need to branch. No wrapper class
is introduced, so every existing S3 method and post-estimation function
works unchanged. With `standardize = TRUE`, two extra fields are
attached – `⁠$scaling⁠` and `⁠$continuous_vars⁠` – and `print()`/`summary()`
use them to display original-units coefficients. With any
`boundary_tests` group requested, a third field `⁠$boundary_tests⁠` (the
[`rpbnb_boundary_tests()`](#topic+rpbnb_boundary_tests) result) is
attached, and `print()`/`summary()` use it to show the LR test on the
corresponding rows. Both fitters also attach `⁠$control_ignored⁠` /
`⁠$control_engine⁠`, the control settings the chosen engine did not read.
Nothing else on the object changes.

### Automatic centring and scaling

`standardize = TRUE` automates the pattern in `inst/rpbnb_frank_open.R`
and `inst/tmb_rpbnb_frank_open.R`: continuous predictors are centred and
scaled (mean 0, SD 1) before fitting, which keeps a bounded
random-coefficient carrier from acting as a disguised random intercept
(see those scripts' headers) and fixes the design matrix's conditioning
when regressors span very different ranges. Continuous predictors are
identified automatically — numeric, non-factor columns used by either
formula with more than two distinct values, so 0/1 (or any two-level
numeric) indicators are left alone — or supplied explicitly via
`continuous_vars`. Variables that appear only inside an `offset()` are
never standardized.

The fitted design itself (the stored `X1`/`X2`, `mu1`/`mu2`, simulation
draws) stays on the standardized scale, exactly as in the two scripts
above, so `predict()`, marginal effects, and boundary/LR tests keep
working unchanged. Only the coefficient table `print()` and `summary()`
display is affected: it is back-transformed to the covariates' original
units by the exact affine chain rule (no refit, no numerical
differentiation) and is the *only* coefficient table shown — there is no
separate standardized-scale table to reconcile. `coef()`/`vcov()` still
return the standardized-scale values that match the stored design; call
`.rpbnb_orig_units()` (internal, mirrors what `print()` displays) if the
numeric original-units vector is needed directly. The scaling actually
used is stored on the fit as `⁠$scaling⁠` (a named list of
`c(center=, scale=)`) and `⁠$continuous_vars⁠`.

### Boundary LR tests

`boundary_tests` runs a likelihood-ratio test on the fit as soon as it
converges (against the same data the fit itself used – the standardized
copy, when `standardize = TRUE`) and attaches the result as
`⁠$boundary_tests⁠`. It is a switch over three independent groups, so the
cost is paid only for the parameters actually in question:

|  |  |  |
|:---|:---|:---|
| `boundary_tests` | tests | restricted refits |
| `FALSE` / `"none"` | nothing | 0 |
| `TRUE` | `c("sd", "dispersion")` | one per scale + one per free `m` |
| `"sd"` | random-coefficient scales | one per scale |
| `"dispersion"` | overdispersions `m1`, `m2` | one per free `m` |
| `"dependence"` | the association parameter | 1 |
| `"all"` | all three groups | all of the above |

The random-coefficient SDs and the NB2 dispersions (`m1`, `m2`) have a
null that sits on the boundary of the parameter space, so an ordinary
Wald `z`/`p` does not test it; the boundary test refits each restricted
model instead and `summary()`/`print()` show that test's `LR`/`df`/`p`
for those rows in the natural-scale block, in place of the `NA` they
would otherwise carry.

The **dependence** test is the odd one out and is therefore opt-in even
under `boundary_tests = TRUE`. Its null is "no association", i.e. the
independence model, which for Famoye (`lam`), Frank (`theta`), and the
Gaussian copula (`rho`) sits in the *interior* of the parameter space –
the Wald `z` those rows already show is valid, and the LR statistic is
an ordinary chi-square(1), not the 50:50 boundary mixture. Only Clayton
/ Kimeldorf (`theta > 0`) has a genuine boundary null there. Requesting
it replaces the dependence row's Wald `z`/`p` with the LR test on both
engines, since the two answer the same question.

Both engines test the same parameters via
[`rpbnb_boundary_tests()`](#topic+rpbnb_boundary_tests)
(`engine = "classic"`) or
[`rpbnb_tmb_boundary_tests()`](#topic+rpbnb_tmb_boundary_tests)
(`engine = "tmb"`). They differ only in how each restricted fit is
constructed while preserving common random numbers: for a scale, the
classic engine zeroes that coefficient's draw column while the TMB
engine pins its `log_sd` and maps it out of the free parameters; for the
dependence parameter, the TMB engine refits with its own
`dependence = "independence"` family while the classic engine (which has
no such fitter) pins the working-scale dependence parameter at its
family's independence value.

A [`message()`](base.html#topic+message) reports how many restricted
refits are about to run before they start, unless `control$print_level`
is `0`; suppress it with
[`suppressMessages()`](base.html#topic+suppressMessages) if needed.

Under `engine = "tmb"` with `method = "laplace"`, a restricted refit
that fails to converge is automatically retried with both sides of that
one LR estimated by `method = "sml"` rather than reported `NA` – some
restrictions leave Laplace no valid optimum at all (see
[`rpbnb_tmb_boundary_tests()`](#topic+rpbnb_tmb_boundary_tests)'s
`sml_fallback` argument, which is where to turn this off).

`force_parallel_gaussian` (`engine = "tmb"` only, passed via `...`) is
forwarded to every restricted refit, so a Gaussian-copula fit's boundary
tests honor `control$n_cores` the same way the original fit did instead
of silently re-capping each refit to one thread – see
[`rpbnb_tmb_boundary_tests()`](#topic+rpbnb_tmb_boundary_tests)'s own
`force_parallel_gaussian` argument for why this needs forwarding at all
(the fit object does not record whether the override was used).

Each restricted refit costs roughly as much as the original fit (more
for a [`copula()`](#topic+copula) dependence than for `"famoye"`; see
[`rpbnb_boundary_tests()`](#topic+rpbnb_boundary_tests)'s timing note),
so this defaults to `FALSE`. `boundary_draws` (`engine = "tmb"` only)
sets a `draws` for those refits other than the main fit's – e.g. more
draws for a more precise boundary test without re-fitting the whole
model at that `draws`. It has no classic-engine counterpart:
[`rpbnb_boundary_tests()`](#topic+rpbnb_boundary_tests) reuses the full
fit's exact stored draw matrix (zeroing a column) rather than
regenerating draws from a count, so passing `boundary_draws` under
`engine = "classic"` is an error. For finer control still – testing only
`"sd"` or only `"dispersion"` (both engines take a `which` argument), or
reusing one boundary-test run across several summaries – call
[`rpbnb_boundary_tests()`](#topic+rpbnb_boundary_tests)/[`rpbnb_tmb_boundary_tests()`](#topic+rpbnb_tmb_boundary_tests)
directly on the fit and assign its result to `fit$boundary_tests` (with
`standardize = TRUE`, reconstruct the fitting-scale data first:
`rpbnb:::.apply_scaling(data, fit$scaling)`).

### Which arguments go with which engine

|  |  |  |
|:---|:---|:---|
| Argument | `engine = "classic"` | `engine = "tmb"` |
| `draw_type`, `.fixed`, `.opt_draws` | yes | error |
| `inference`, `keep` | error | yes |
| `method`, `force_parallel_gaussian` | ignored with a warning | yes |
| `offset()` in a formula | yes | error |
| `dependence = "independence"` | error | yes |
| `boundary_draws` (non-`NULL`) | error | yes |
| `control` class | `rpbnb_control` | `rpbnb_control` (same object) |
| optimizer | `maxLik::maxLik(method = "BFGS")` | `stats::nlminb` + restarts |

Both engines freeze the Famoye admissible lambda interval at the
starting values (so the analytic gradient is exactly the derivative of
the optimized objective) and validate the fitted lambda against the
interval admissible at the optimum afterwards — check
`lambda_admissible` on the fit. The fixed-parameter
[`fit_bnb()`](#topic+fit_bnb) takes the opposite trade-off (moving
bounds, admissible by construction); see the decision note in
`R/bnb_likelihood.R`.

### See Also

[`fit_rpbnb()`](#topic+fit_rpbnb),
[`fit_rpbnb_tmb()`](#topic+fit_rpbnb_tmb),
[`rpbnb_control()`](#topic+rpbnb_control),
[`rpbnb_tmb_control()`](#topic+rpbnb_tmb_control),
[`copula()`](#topic+copula), [`fit_bnb()`](#topic+fit_bnb)

### Examples

``` R

d <- read.csv(system.file("extdata", "rwm1984_bnb.csv", package = "rpbnb"))
fit <- rpbnb(docvis ~ outwork, hospvis ~ outwork, data = d,
             engine = "tmb", random_1 = "outwork", draws = 50)
```

------------------------------------------------------------------------

## Boundary-corrected LR tests for all boundary parameters of an rpbnb_fit

### Description

Runs a likelihood-ratio test for every random-coefficient standard
deviation (`⁠sd1:*⁠`, `⁠sd2:*⁠`) and NB2 dispersion (`m1`, `m2`) of a fitted
random-parameter bivariate NB model, and merges them into one table.
These are the parameters whose null lies on the boundary of the
parameter space (SD = 0, or dispersion `m = 0` = Poisson), for which the
natural-scale summary reports no Wald `z`/`p`; each test uses the 50:50
chi-square boundary correction of [`lr_test()`](#topic+lr_test)
(`boundary = TRUE`).

### Usage

``` R
rpbnb_boundary_tests(
  fit,
  data,
  control = rpbnb_control(compute_se = FALSE),
  which = c("sd", "dispersion")
)
```

### Arguments

[TABLE]

### Details

Each test refits a properly nested restricted model. With **multiple
random coefficients in an equation, each SD is tested individually** (a
1-df restriction). The restricted fit keeps the full random
specification but sets the tested coefficient's draw column to the
distribution median (`u = 0.5`), which zeroes that coefficient's
per-draw *deviation* exactly for every supported distribution (`u = 0.5`
maps to base 0 for normal/lognormal/ triangular, and to the centered
value `2 * 0.5 - 1 = 0` for uniform). The coefficient therefore
collapses to its SD-zero null – the ordinary fixed coefficient `b` for
normal/uniform/triangular, and `sign * exp(b)` for lognormal – on every
draw, independent of the covariate scale. Its (now inert) log-scale is
pinned only to drop it from the free-parameter count. Every other draw
column is the full model's exact stored draw, so the two simulated
log-likelihoods are compared on common random numbers, and each
restricted fit is **warm-started from the full fit's coefficients** so
the start-sensitive simulated objective does not settle at an inferior
optimum. A restricted fit that fails to converge yields `NA` inference
(with a warning) rather than a p-value, and a non-converged full `fit`
is rejected.

### Value

An object of class `rpbnb_boundary_tests`: a data frame with columns
`Parameter`, `LR`, `df`, `p.value`, `Signif` (one row per boundary
parameter), and a `print` method.

Under Famoye dependence with uniform or triangular random coefficients
(or a single varying margin), the full and restricted fits' admissible
lambda intervals are frozen at different starting values, so each LR
statistic compares maxima over slightly different lambda ranges; see the
"Famoye caveat" section of [`lr_test()`](#topic+lr_test).
Normal/lognormal coefficients in both margins are unaffected (the
interval is the constant `c(-1, 1)`).

### See Also

[`lr_test()`](#topic+lr_test), [`fit_rpbnb()`](#topic+fit_rpbnb)

### Examples

``` R

sim <- simulate_rpbnb(n = 600,
  beta1 = c("(Intercept)" = 0.2, x1 = 0.4),
  beta2 = c("(Intercept)" = 0.1, x1 = -0.3),
  random_1 = list(x1 = list(sd = 0.5)),
  dispersion = c(m1 = 0.4, m2 = 0.5), seed = 1)
fit <- fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim$data, random_1 = "x1",
                 draws = 200, seed = 1)
rpbnb_boundary_tests(fit, sim$data)
```

------------------------------------------------------------------------

## Control parameters for every rpbnb estimator

### Description

One control object for all of the package's fitters –
[`fit_bnb()`](#topic+fit_bnb), [`fit_rpbnb()`](#topic+fit_rpbnb),
[`fit_rpbnb_tmb()`](#topic+fit_rpbnb_tmb), and [`rpbnb()`](#topic+rpbnb)
with either `engine`. It carries the union of the tuning knobs the
estimators use; each fitter reads the ones that apply to it and
**ignores the rest**, reporting the ignored names in
`print()`/`summary()` of the resulting fit rather than erroring. That is
what makes a script able to flip `engine = "classic"` to `"tmb"` without
rewriting its control call.

### Usage

``` R
rpbnb_control(
  method = c("BFGS"),
  iterlim = NULL,
  reltol = 1e-08,
  print_level = NULL,
  draws_hessian = 100L,
  halton_burn = 300L,
  n_cores = 1L,
  compute_se = TRUE,
  hessian = c("numeric", "analytic"),
  se_method = c("numeric", "opg", "analytic"),
  hess_eps = 1e-05,
  hess_r = 4L,
  gradtol = 1e-05,
  restarts = 10L,
  max_threads = NULL,
  max_workload = NULL,
  parallel_tape = FALSE
)
```

### Arguments

[TABLE]

### Value

An object of class `c("rpbnb_control", "rpbnb_tmb_control")` (a named
list). It carries both class names so that every historical
`inherits(control, ...)` check in the package and in user code accepts
it.

### Which parameters apply to which estimator

|  |  |  |  |
|:---|:---|:---|:---|
| Parameter | [`fit_bnb()`](#topic+fit_bnb) | [`fit_rpbnb()`](#topic+fit_rpbnb) | [`fit_rpbnb_tmb()`](#topic+fit_rpbnb_tmb) |
| `method`, `compute_se`, `hess_eps`, `hess_r` | yes | yes | ignored |
| `iterlim`, `reltol`, `print_level` | yes | yes | yes |
| `halton_burn`, `n_cores` | ignored | yes | yes |
| `hessian` | yes | ignored | ignored |
| `se_method` | ignored | yes | ignored |
| `gradtol`, `restarts`, `max_threads`, `max_workload`, `parallel_tape` | ignored | ignored | yes |
| `draws_hessian` | ignored | ignored | ignored |

Note that `iterlim` and `n_cores` mean different things to different
estimators – a `maxLik` BFGS iteration limit versus an `nlminb` one,
worker *processes* versus OpenMP *threads*. They are not translated;
each estimator reads the number and applies its own meaning to it.

### Defaults that depend on the estimator

`iterlim` and `print_level` default to `NULL`, which means "this
estimator's own long-standing default": `iterlim` is 300 under `maxLik`
and 500 under `nlminb`; `print_level` is 2 (progress) under `maxLik` and
0 (silent) under `nlminb`. Supplying either explicitly overrides that
for every estimator. `max_threads` defaults to `n_cores` and
`max_workload` is computed from available memory by
[`rpbnb_tmb_max_workload()`](#topic+rpbnb_tmb_max_workload) the first
time a TMB fit needs it (so a non-TMB fit never pays for the memory
probe).

### See Also

[`rpbnb_tmb_max_workload()`](#topic+rpbnb_tmb_max_workload),
[`rpbnb()`](#topic+rpbnb), [`fit_rpbnb()`](#topic+fit_rpbnb),
[`fit_rpbnb_tmb()`](#topic+fit_rpbnb_tmb), [`fit_bnb()`](#topic+fit_bnb)

### Examples

``` R
rpbnb_control(method = "BFGS", iterlim = 200)
rpbnb_control(hessian = "analytic")
# The same object drives either engine; the TMB-only knobs are simply
# ignored by the classic one (and reported as ignored in its summary).
rpbnb_control(n_cores = 4, gradtol = 1e-6, se_method = "opg")
```

------------------------------------------------------------------------

## Elasticities and semi-elasticities for a random-parameter bivariate NB model

### Description

Continuous elasticities `x_{ij}\,(\partial \mu_i/\partial x_{ij})/\mu_i`
and binary semi-elasticities `\mu_i(x_j=1)/\mu_i(x_j=0) - 1`, built on
the Monte-Carlo integrated population mean
`\mu_i = E_\beta[\exp(x_i'\beta)]` of an
[`fit_rpbnb()`](#topic+fit_rpbnb) fit. Under fixed coefficients these
reduce to `\beta_j E[x_j]` and `\exp(\beta_j) - 1`. Standard errors use
a numeric delta method over the equation's mean and log-scale
parameters.

### Usage

``` R
rpbnb_elasticities(
  fit,
  which = c("y1", "y2", "both"),
  type = c("AME", "MEM"),
  vars = NULL,
  include_intercept = FALSE,
  digits = 4,
  print_output = TRUE,
  n_cores = 1L,
  scaling = NULL,
  log_vars = NULL
)
```

### Arguments

|  |  |
|----|----|
| `fit` | An `rpbnb_fit` object from [`fit_rpbnb()`](#topic+fit_rpbnb). |
| `which` | Which margin(s): "y1", "y2", or "both". |
| `type` | "AME" (average over the sample) or "MEM" (evaluated at the mean row). |
| `vars` | Optional variable names or indices to restrict output. |
| `include_intercept` | Logical; include the intercept term. |
| `digits` | Number of decimal places for printed output. |
| `print_output` | Logical; if `FALSE`, suppress printing. |
| `n_cores` | Number of worker processes for the delta-method standard-error jacobian (1 = sequential, the default). When `n_cores > 1`, the jacobian's independent per-parameter columns are dispatched across a `parallel::makeCluster()` cluster (one cluster per call, shared across every equation `which` computes); results are numerically identical to the sequential path. Falls back to sequential with a warning if the `parallel` package is unavailable. |
| `scaling` | Optional named list of `c(center =, scale =)` pairs, one per covariate that was centred and/or scaled before fitting (e.g. via `rpbnb(standardize = TRUE)`, whose `⁠$scaling⁠` field is exactly this shape), used to report the results in the covariate's original units. See [`rpbnb_marginal_effects()`](#topic+rpbnb_marginal_effects)'s `scaling` documentation for the full explanation (elasticities and marginal effects share it verbatim). |
| `log_vars` | Optional character vector naming covariates that are ALREADY a log; see [`rpbnb_marginal_effects()`](#topic+rpbnb_marginal_effects)'s `log_vars` documentation. |

### Value

A data frame (single margin, invisibly) or a named list of data frames
(`both`), each with columns `Name`, `Estimate`, `StdErr`, `z`, `p`,
`Signif`, `var_type`.

### See Also

[`bnb_elasticities()`](#topic+bnb_elasticities) for fixed-coefficient
`bnb_fit` models.

### Examples

``` R
sim <- simulate_rpbnb(n = 400,
  beta1 = c("(Intercept)" = 0.2, x1 = 0.4),
  beta2 = c("(Intercept)" = 0.1, x1 = -0.3),
  random_1 = list(x1 = list(sd = 0.5)),
  dispersion = c(m1 = 0.4, m2 = 0.5), seed = 1)
fit <- fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim$data, random_1 = "x1",
                 draws = 100)
rpbnb_elasticities(fit, which = "both", type = "AME")
```

------------------------------------------------------------------------

## Marginal effects for a random-parameter bivariate NB model

### Description

Average marginal effects (AME) or marginal effects at the mean (MEM) for
each margin of an [`fit_rpbnb()`](#topic+fit_rpbnb) fit, built on the
Monte-Carlo integrated population mean
`\mu_i = E_\beta[\exp(x_i'\beta)]` (the same estimand as
`predict.rpbnb_fit()`, reusing the fit's stored draws). Continuous
effects are
`\partial \mu_i/\partial x_{ij} = \mathrm{mean}_r\, \mathrm{coef}_{rj} \exp(\mathrm{lp}_{ir})`
(the realized coefficient per draw, so random and fixed columns are
handled uniformly); binary (0/1) effects are the integrated discrete
difference `E[Y|x_j=1] - E[Y|x_j=0]`. Standard errors use a numeric
delta method over the equation's mean and log-scale parameters.

### Usage

``` R
rpbnb_marginal_effects(
  fit,
  which = c("y1", "y2", "both", "all"),
  type = c("AME", "MEM"),
  vars = NULL,
  include_intercept = FALSE,
  digits = 4,
  print_output = TRUE,
  n_cores = 1L,
  scaling = NULL,
  log_vars = NULL
)
```

### Arguments

[TABLE]

### Details

For a model fitted with an `offset()`, absolute marginal effects scale
with the offset-inclusive mean: AME uses each observation's training
offset, and MEM uses the mean training offset.

### Value

A data frame (single margin, invisibly) or a named list of data frames
(`both`/`all`), each with columns `Name`, `Estimate`, `StdErr`, `z`,
`p`, `Signif`, `var_type`.

### See Also

[`bnb_marginal_effects()`](#topic+bnb_marginal_effects) for
fixed-coefficient `bnb_fit` models.

### Examples

``` R
sim <- simulate_rpbnb(n = 400,
  beta1 = c("(Intercept)" = 0.2, x1 = 0.4),
  beta2 = c("(Intercept)" = 0.1, x1 = -0.3),
  random_1 = list(x1 = list(sd = 0.5)),
  dispersion = c(m1 = 0.4, m2 = 0.5), seed = 1)
fit <- fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim$data, random_1 = "x1",
                 draws = 100)
rpbnb_marginal_effects(fit, which = "y1", type = "AME")
```

------------------------------------------------------------------------

## Number of CPU threads available for the multithreaded likelihood

### Description

Number of CPU threads available for the multithreaded likelihood

### Usage

``` R
rpbnb_threads()
```

### Value

Integer thread count reported by OpenMP (1 if OpenMP is unavailable).

------------------------------------------------------------------------

## Boundary-corrected LR tests for an rpbnb_tmb_fit's boundary parameters

### Description

Runs a likelihood-ratio test for every random-coefficient scale
(`⁠sd1:*⁠`, `⁠sd2:*⁠`) and NB2 dispersion (`m1`, `m2`) of a fitted
TMB-engine random-parameter bivariate NB model, and merges them into one
table. These are the parameters whose null lies on the boundary of the
parameter space (scale = 0, or dispersion `m = 0` = Poisson), for which
`summary(fit)` reports no Wald `z`/`p`; each test uses the 50:50
chi-square boundary correction of [`lr_test()`](#topic+lr_test)
(`boundary = TRUE`).

### Usage

``` R
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

### Arguments

[TABLE]

### Details

Each test refits a properly nested restricted model, otherwise identical
to `fit` (same formulas, random-coefficient specification, dependence,
seed, and estimator, and by default the same `draws`), warm-started from
`fit$coef` and run with `inference = "none"` since the LR test needs
only `logLik` and the parameter count.

**Dispersions** are restricted with `poisson_1`/`poisson_2 = TRUE`, the
template's exact `m = 0` branch.

**Scales** are restricted by pinning that coefficient's `log_sd` at the
parameterization's zero (`-20`; the template clamps `log_sd` to
`⁠[-20, 20]⁠` and computes `sd = exp(log_sd)`, so this is `sd = 2.1e-9` –
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

A [`message()`](base.html#topic+message)
(`"Boundary LR test: <parameter>..."`) reports each restricted refit
right before it starts, unless `control$print_level` is `0`; suppress it
with [`suppressMessages()`](base.html#topic+suppressMessages) if needed.

### Value

An object of class `rpbnb_boundary_tests` – the same class
[`rpbnb_boundary_tests()`](#topic+rpbnb_boundary_tests) returns, with
columns `Parameter`, `LR`, `df`, `p.value`, `Signif` (one row per
boundary parameter) and the same `print()` method – so the two engines'
results are interchangeable wherever that class is consumed (e.g.
`summary.rpbnb_tmb_fit()`'s scale and dispersion blocks, once
`⁠$boundary_tests⁠` is attached to the fit).

Scale rows are labelled by the distribution's own scale parameter (`sd`
for normal/lognormal, `w` for uniform/triangular half-width, `s` for a
lognormal log-scale), matching `summary()`'s row names.

The `sml_fallback` attribute is a character vector of the `Parameter`
rows whose LR came from the SML fallback pair (see the `sml_fallback`
argument); `character(0)` when every test ran under `fit`'s own
estimator.

### See Also

[`lr_test()`](#topic+lr_test),
[`fit_rpbnb_tmb()`](#topic+fit_rpbnb_tmb),
[`rpbnb_boundary_tests()`](#topic+rpbnb_boundary_tests) (the
classic-engine counterpart)

### Examples

``` R

sim <- simulate_rpbnb(n = 600,
  beta1 = c("(Intercept)" = 0.2, x1 = 0.4),
  beta2 = c("(Intercept)" = 0.1, x1 = -0.3),
  random_1 = list(x1 = list(sd = 0.5)),
  dispersion = c(m1 = 0.4, m2 = 0.5), seed = 1)
fit <- fit_rpbnb_tmb(y1 ~ x1, y2 ~ x1, data = sim$data, random_1 = "x1",
                     draws = 100)
rpbnb_tmb_boundary_tests(fit, sim$data)
```

------------------------------------------------------------------------

## Control parameters for the TMB engine (alias of `rpbnb_control()`)

### Description

Retained so that code written against the pre-unification API keeps
working. The two control objects were merged in 0.4.1: this function
forwards to [`rpbnb_control()`](#topic+rpbnb_control) and returns
exactly the same object, which every estimator in the package accepts.
New code should call [`rpbnb_control()`](#topic+rpbnb_control) directly.

### Usage

``` R
rpbnb_tmb_control(
  iterlim = NULL,
  reltol = 1e-08,
  gradtol = 1e-05,
  restarts = 10L,
  print_level = NULL,
  n_cores = 1L,
  max_threads = NULL,
  max_workload = NULL,
  parallel_tape = FALSE,
  halton_burn = 300L
)
```

### Arguments

[TABLE]

### Details

Only the arguments you actually supply are forwarded, so an untouched
`iterlim`/`print_level` still resolves to the TMB engine's own defaults
(500 and 0) when the object is used for a TMB fit – and to the `maxLik`
defaults if the same object is handed to
[`fit_rpbnb()`](#topic+fit_rpbnb).

### Value

The [`rpbnb_control()`](#topic+rpbnb_control) object.

### See Also

[`rpbnb_control()`](#topic+rpbnb_control)

### Examples

``` R
identical(unclass(rpbnb_tmb_control(n_cores = 2)),
          unclass(rpbnb_control(n_cores = 2)))
```

------------------------------------------------------------------------

## Confidence interval for a fitted dependence parameter

### Description

Returns a profile-likelihood interval for the dependence parameter of a
fitted model, computed on the unconstrained working scale and mapped
through the family's monotone link. This is the tool to reach for when
`summary()` reports `NA` for the dependence standard error: at a
boundary the delta-method derivative collapses, so
`SE = |d\theta/dz| \cdot SE(z)` is a `0 \times \infty` product and no
symmetric standard error exists. A profile interval needs no derivative.

### Usage

``` R
rpbnb_tmb_dependence_profile(
  fit,
  level = 0.95,
  method = c("profile", "wald"),
  ...
)
```

### Arguments

|  |  |
|----|----|
| `fit` | An object of class `rpbnb_tmb_fit`. |
| `level` | Coverage level; one number strictly between 0 and 1. |
| `method` | `"profile"` (default) for a profile-likelihood interval, or `"wald"` for a Wald interval on the working scale mapped through the same link. `"profile"` requires the TMB objective, retained only under `keep = "full"`; when it is unavailable this degrades to `"wald"` with a warning rather than failing. |
| `...` | Passed to [`tmbprofile`](TMB.html#topic+tmbprofile), e.g. `ytol` or `parm.range`. `lincomb` and `slice` are not supported and raise an error: the first would profile a different quantity than `z_dep` while still being reported as if it were, and the second returns a likelihood slice rather than a profile. |

### Details

For Famoye this is the usual situation, because the admissible lambda
interval is frozen at the starting values (see `lambda_bounds` in
[`fit_rpbnb_tmb`](#topic+fit_rpbnb_tmb)). An interval around a pinned
estimate is still an interval around an artefact – widen the box first
by refitting from better starting values.

### Value

A data frame with one row per reported dependence quantity and columns
`parameter`, `estimate`, `lower`, `upper`, `level`, and `method` – the
method actually used, which may differ from the one requested. The raw
profile, when one was computed, is attached as `attr(, "profile")` so it
can be plotted.

### See Also

[`fit_rpbnb_tmb`](#topic+fit_rpbnb_tmb)

### Examples

``` R
## Not run: 
fit <- fit_rpbnb_tmb(y1 ~ x, y2 ~ x, data = d,
                     dependence = "famoye", keep = "full")
rpbnb_tmb_dependence_profile(fit)
plot(attr(rpbnb_tmb_dependence_profile(fit), "profile"))

## End(Not run)
```

------------------------------------------------------------------------

## Elasticities for a rpbnb_tmb model

### Description

Continuous elasticities `x_j / E[Y] \cdot \partial E[Y]/\partial x_j`
and binary semi-elasticities `E[Y|x_j = 1] / E[Y|x_j = 0] - 1`. Built on
the same Monte-Carlo integrated mean used by
[`rpbnb_tmb_marginal_effects`](#topic+rpbnb_tmb_marginal_effects).

### Usage

``` R
rpbnb_tmb_elasticities(
  fit,
  which = c("y1", "y2", "both"),
  type = c("AME", "MEM"),
  vars = NULL,
  include_intercept = FALSE,
  digits = 4L,
  scaling = NULL,
  log_vars = NULL,
  ...
)
```

### Arguments

[TABLE]

### Value

A data frame or named list of data frames.

------------------------------------------------------------------------

## Marginal effects for a rpbnb_tmb model

### Description

Computes average marginal effects (AME) or marginal effects at the mean
(MEM) for a fitted random-parameter bivariate negative binomial model.
For continuous covariates the effect is `\partial E[Y]/\partial x_j`;
for binary covariates it is the discrete difference
`E[Y|x_j = 1] - E[Y|x_j = 0]`. When a coefficient is random the per-draw
realized coefficients are used to form the Monte-Carlo integrated mean.

### Usage

``` R
rpbnb_tmb_marginal_effects(
  fit,
  which = c("y1", "y2", "both"),
  type = c("AME", "MEM"),
  vars = NULL,
  include_intercept = FALSE,
  digits = 4L,
  scaling = NULL,
  log_vars = NULL,
  ...
)
```

### Arguments

[TABLE]

### Value

A data frame (single margin) or named list of two data frames
(`"both"`).

------------------------------------------------------------------------

## Compute a TMB workload budget from a memory figure

### Description

Companion to `rpbnb_tmb_control()`'s `max_workload`: rather than picking
a weighted-observation-draw count directly, state a memory budget and
let this function do the arithmetic `TAPE_CALIBRATION` implies.

### Usage

``` R
rpbnb_tmb_max_workload(budget_gib = NULL, fraction = 0.8)
```

### Arguments

|  |  |
|----|----|
| `budget_gib` | Optional memory budget in GiB, stated explicitly. When supplied, used as-is – `fraction` does not apply, because a number you state yourself is not second-guessed with a discount. When omitted (the default), available memory is auto-detected and `fraction` of it is budgeted instead. |
| `fraction` | Of auto-detected available memory, the fraction to actually budget; one number in `(0, 1]`. Available memory fluctuates and competes with other processes, so budgeting all of it risks the guard passing a fit that then exhausts memory anyway. Ignored when `budget_gib` is supplied. |

### Details

With `budget_gib` omitted, this is also `rpbnb_tmb_control()`'s own
default: every fit that doesn't set `max_workload` explicitly already
goes through this function.

### Value

One positive numeric workload value, on the same scale as
`rpbnb_tmb_control()`'s `max_workload`.

### Examples

``` R
rpbnb_tmb_max_workload(budget_gib = 16)
## Not run: 
ctrl <- rpbnb_tmb_control(max_workload = rpbnb_tmb_max_workload())

## End(Not run)
```

------------------------------------------------------------------------

## Simulate data from the Famoye/Sarmanov bivariate NB distribution

### Description

Generates paired count outcomes `⁠(y1, y2)⁠` from the fixed-parameter
Famoye/Sarmanov bivariate NB2 joint PMF:

### Usage

``` R
simulate_bnb(
  n,
  beta1,
  beta2,
  dispersion = c(m1 = 0.5, m2 = 0.5),
  lambda = 0,
  covariates = NULL,
  seed = NULL
)
```

### Arguments

|  |  |
|----|----|
| `n` | Number of observations. |
| `beta1`, `beta2` | Named numeric vectors of coefficients for each equation; must include `"(Intercept)"`. |
| `dispersion` | Named numeric `c(m1 = ..., m2 = ...)` NB2 dispersion parameters (variance = mu + m \* mu^2). |
| `lambda` | Famoye/Sarmanov dependence parameter. Must lie within the valid bounds implied by the marginal means and dispersions. |
| `covariates` | Optional data frame of covariates. If `NULL`, standard- normal columns are generated for every non-intercept coefficient name. |
| `seed` | Optional random seed. If `NULL` (default) the RNG is left untouched and draws continue from the caller's current stream, so repeated calls yield distinct datasets; supply an integer for reproducible output. |

### Details

`P(Y_1=y_1, Y_2=y_2) = p_1(y_1)\,p_2(y_2)\, [1 + \lambda(e^{-y_1}-c_1)(e^{-y_2}-c_2)]`

where `c_k = E[e^{-Y_k}]` under NB2(`\mu_k, m_k`).

### Value

A list with:

- `data`:

  data frame with `y1`, `y2`, and covariate columns

- `mu`:

  data frame with `mu1`, `mu2` (per-obs conditional means)

- `true`:

  list of true parameters: `beta1`, `beta2`, `dispersion`, `lambda`

- `settings`:

  list with `n` and `seed`

- `meta`:

  list with `seed` and `r_version`

### Examples

``` R
sim <- simulate_bnb(n = 500,
  beta1 = c("(Intercept)" = 0.5, x1 = 0.3),
  beta2 = c("(Intercept)" = 0.2, x1 = -0.2),
  dispersion = c(m1 = 0.4, m2 = 0.5), lambda = 0.1, seed = 1)
head(sim$data)
```

------------------------------------------------------------------------

## Simulate data from a random-parameter bivariate NB process

### Description

Simulate data from a random-parameter bivariate NB process

### Usage

``` R
simulate_rpbnb(
  n,
  beta1,
  beta2,
  random_1 = NULL,
  random_2 = NULL,
  dispersion = c(m1 = 0.5, m2 = 0.5),
  lambda = 0,
  covariates = NULL,
  seed = NULL
)
```

### Arguments

|  |  |
|----|----|
| `n` | Number of observations. |
| `beta1`, `beta2` | Named numeric vectors of fixed coefficient means per equation; must include "(Intercept)". |
| `random_1`, `random_2` | Named lists giving random coefficients. Each value is a list with `dist` (one of "normal", "lognormal", "uniform", "triangular"; default "normal"), `scale` (or `sd`) for the dispersion, and `sign` (-1/1, lognormal only). Means come from `beta1`/`beta2`; for a lognormal coefficient the `beta` entry is the log-location and the realized coefficient is `sign * exp(log_location + scale * z)`. |
| `dispersion` | Named numeric `c(m1 = ..., m2 = ...)` NB2 dispersions. |
| `lambda` | Famoye dependence parameter (0 = independent margins). |
| `covariates` | Optional data frame of covariates; if NULL, standard-normal columns are generated for every non-intercept name. |
| `seed` | Optional random seed. If `NULL` (default) the RNG is left untouched and draws continue from the caller's current stream, so repeated calls yield distinct datasets; supply an integer for reproducible output. |

### Value

A list with `data` (y1, y2, covariates), `coef_realized` (per-obs
coefficients per equation), `mu` (conditional means), `true`
(parameters), `settings`, and `meta` (R/seed/timestamp passed by
caller).

### Examples

``` R
sim <- simulate_rpbnb(n = 500,
  beta1 = c("(Intercept)" = 0.2, x1 = 0.4),
  beta2 = c("(Intercept)" = 0.1, x1 = -0.3),
  random_1 = list(x1 = list(sd = 0.5)),
  dispersion = c(m1 = 0.4, m2 = 0.5), seed = 1)
head(sim$data)
```

------------------------------------------------------------------------

## Simulate data from a copula RP-BNB process

### Description

Simulate data from a copula RP-BNB process

### Usage

``` R
simulate_rpbnb_copula(
  n,
  beta1,
  beta2,
  random_1 = NULL,
  random_2 = NULL,
  dispersion = c(m1 = 0.5, m2 = 0.5),
  copula,
  covariates = NULL,
  seed = NULL
)
```

### Arguments

|  |  |
|----|----|
| `n` | Number of observations. |
| `beta1`, `beta2` | Named coefficient means; must include "(Intercept)". |
| `random_1`, `random_2` | Random-coefficient specs (see [`simulate_rpbnb()`](#topic+simulate_rpbnb)). |
| `dispersion` | Named `c(m1=, m2=)` NB2 dispersions. |
| `copula` | An [`copula()`](#topic+copula) object giving the family and native parameter `par`. |
| `covariates` | Optional covariate data frame; NULL -\> standard-normal columns. |
| `seed` | Optional RNG seed. |

### Value

list(data, mu, true, settings).

------------------------------------------------------------------------

## Simulate data from a bivariate NB process

### Description

Generates count data from a bivariate negative binomial model with
random coefficients and Famoye/Sarmanov or copula dependence.

### Usage

``` R
simulate_rpbnb_tmb(
  n,
  beta1,
  beta2,
  random_1 = NULL,
  random_2 = NULL,
  dispersion = c(m1 = 0.5, m2 = 0.5),
  dependence = "famoye",
  lambda = 0,
  covariates = NULL,
  seed = NULL
)
```

### Arguments

|  |  |
|----|----|
| `n` | Number of observations. |
| `beta1`, `beta2` | Named coefficient vectors (must include "(Intercept)"). |
| `random_1`, `random_2` | Random coefficient specs (same format as fit_rpbnb_tmb). |
| `dispersion` | Named vector c(m1 = , m2 = ) of NB2 dispersions. |
| `dependence` | "famoye", "independence", or a copula() object. |
| `lambda` | Famoye dependence parameter (0 = independence). |
| `covariates` | Optional data frame. If NULL, standard-normal covariates generated. |
| `seed` | Optional integer seed. |

### Value

List with \$data (data.frame), \$truth (parameters), \$settings.

### Examples

``` R
sim <- simulate_rpbnb_tmb(n = 200,
  beta1 = c("(Intercept)" = 0.2, x1 = 0.4),
  beta2 = c("(Intercept)" = 0.1, x1 = -0.3),
  dispersion = c(m1 = 0.4, m2 = 0.5), seed = 1)
head(sim$data)
```

------------------------------------------------------------------------

## Summarize a fitted TMB-engine rpbnb model

### Description

Summarize a fitted TMB-engine rpbnb model

### Usage

``` R
## S3 method for class 'rpbnb_tmb_fit'
summary(object, digits = 4L, ...)
```

### Arguments

|  |  |
|----|----|
| `object` | A fitted model object of class rpbnb_tmb_fit. |
| `digits` | Number of decimal places for numeric columns in coefficient tables. Default 4. Use a negative value (e.g. -1) for full precision. |
| `...` | Not used. |

### Value

The fitted object, invisibly.
