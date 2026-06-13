# rpbnb

Bivariate and random-parameter negative binomial regression (NB2), via maximum
likelihood (Famoye/Sarmanov dependence) and maximum simulated likelihood.

## Install

```r
# install.packages("devtools")
devtools::install("path/to/rpbnb")
```

## Quick start

```r
library(rpbnb)
d <- read.csv(system.file("extdata", "rwm1984_clean.csv", package = "rpbnb"))

# Bivariate NB (Famoye dependence)
fit <- fit_bnb(docvis ~ outwork + kids, hospvis ~ outwork + kids,
               data = d, dependence = "famoye")
summary(fit)
bnb_gof(fit)

# Random-parameter BNB (kids has a random coefficient in equation 1)
rp <- fit_rpbnb(docvis ~ outwork + kids, hospvis ~ outwork + kids,
                data = d, random_1 = "kids", draws = 400, seed = 1234)
summary(rp)

# Simulate from a known RP-BNB process
sim <- simulate_rpbnb(n = 2000,
  beta1 = c("(Intercept)" = -1.0, x1 = 0.4),
  beta2 = c("(Intercept)" = -0.5, x1 = 0.2),
  random_1 = list(x1 = list(sd = 0.5)),
  dispersion = c(m1 = 0.8, m2 = 0.8), seed = 1234)
```

## Models

- `fit_bnb()` — bivariate NB with `dependence = "independence"` (two univariate
  NB2 margins) or `"famoye"` (Famoye/Sarmanov dependence).
- `fit_rpbnb()` — bivariate random-parameter NB via maximum simulated likelihood
  with normal random coefficients and Halton draws.
- `simulate_rpbnb()` — simulate data from a random-parameter NB process.
- Diagnostics: `bnb_gof()`, `bnb_marginal_effects()`, `bnb_elasticities()`.
- Standard methods: `coef()`, `vcov()`, `logLik()`, `AIC()`, `BIC()`, `predict()`,
  `summary()`, `print()`.

See `vignette("rpbnb-intro")` for a worked example.

## Scope

This is Phase 1 of a larger project (see `docs/scope_rpnbn.md`): porting the mature
Famoye BNB MLE and random-parameter BNB SIML implementations into an installable,
tested package. Copula dependence, additional random-coefficient distributions,
Monte Carlo drivers, a `future`-based parallel backend, and Rcpp acceleration are
planned for later phases.
