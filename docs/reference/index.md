# Package index

## Model fitting

Functions for fitting bivariate NB regression models

- [`rpbnb()`](rpbnb.md) : Fit a random-parameter bivariate NB model with
  either engine
- [`fit_bnb()`](fit_bnb.md) : Fit a bivariate negative binomial
  regression model
- [`fit_rpbnb()`](fit_rpbnb.md) : Fit a bivariate random-parameter
  negative binomial model
- [`copula()`](copula.md) : Specify a copula dependence structure
- [`rpbnb_control()`](rpbnb_control.md) : Control parameters for every
  rpbnb estimator
- [`rpbnb_threads()`](rpbnb_threads.md) : Number of CPU threads
  available for the multithreaded likelihood

## TMB engine

The automatic-differentiation engine, reachable either directly or via
rpbnb(engine = “tmb”). Adds a Laplace approximation and dependence
profiling. Shares rpbnb_control() with the classic engine above
(rpbnb_tmb_control() is a retained alias for the same object).

- [`fit_rpbnb_tmb()`](fit_rpbnb_tmb.md) : Fit a bivariate
  random-parameter negative binomial model (TMB)

- [`rpbnb_tmb_control()`](rpbnb_tmb_control.md) :

  Control parameters for the TMB engine (alias of
  [`rpbnb_control()`](../reference/rpbnb_control.md))

- [`rpbnb_tmb_dependence_profile()`](rpbnb_tmb_dependence_profile.md) :
  Confidence interval for a fitted dependence parameter

- [`rpbnb_tmb_marginal_effects()`](rpbnb_tmb_marginal_effects.md) :
  Marginal effects for a rpbnb_tmb model

- [`rpbnb_tmb_elasticities()`](rpbnb_tmb_elasticities.md) : Elasticities
  for a rpbnb_tmb model

- [`rpbnb_tmb_max_workload()`](rpbnb_tmb_max_workload.md) : Compute a
  TMB workload budget from a memory figure

## Boundary and likelihood-ratio tests

Tests for the parameters (random-coefficient scales, NB2 dispersions,
the dependence parameter) whose null hides on or near a boundary of the
parameter space, where an ordinary Wald z/p does not apply.

- [`lr_test()`](lr_test.md) : Likelihood-ratio test between two nested
  model fits
- [`rpbnb_boundary_tests()`](rpbnb_boundary_tests.md) :
  Boundary-corrected LR tests for all boundary parameters of an
  rpbnb_fit
- [`rpbnb_tmb_boundary_tests()`](rpbnb_tmb_boundary_tests.md) :
  Boundary-corrected LR tests for an rpbnb_tmb_fit's boundary parameters

## Post-estimation

Marginal effects, elasticities, goodness-of-fit, and residual
diagnostics

- [`bnb_marginal_effects()`](bnb_marginal_effects.md) : Marginal effects
  for a bivariate NB model
- [`bnb_elasticities()`](bnb_elasticities.md) : Elasticities and
  semi-elasticities for a bivariate NB model
- [`bnb_gof()`](bnb_gof.md) : Goodness of fit for a bivariate NB model
- [`bnb_residual_checks()`](bnb_residual_checks.md) : Residual checks
  for a bivariate NB model
- [`rpbnb_marginal_effects()`](rpbnb_marginal_effects.md) : Marginal
  effects for a random-parameter bivariate NB model
- [`rpbnb_elasticities()`](rpbnb_elasticities.md) : Elasticities and
  semi-elasticities for a random-parameter bivariate NB model
- [`predict(`*`<rpbnb_tmb_fit>`*`)`](predict.rpbnb_tmb_fit.md) : Predict
  from a fitted bivariate count model
- [`summary(`*`<rpbnb_tmb_fit>`*`)`](summary.rpbnb_tmb_fit.md) :
  Summarize a fitted TMB-engine rpbnb model
- [`residuals(`*`<bnb_fit>`*`)`](residuals.bnb_fit.md) : Residuals for a
  bivariate NB model
- [`residuals(`*`<rpbnb_fit>`*`)`](residuals.rpbnb_fit.md) : Residuals
  for a random-parameter bivariate NB model
- [`plot(`*`<bnb_fit>`*`)`](plot.bnb_fit.md) : Residual diagnostic plots
  for a bivariate NB model
- [`plot(`*`<rpbnb_fit>`*`)`](plot.rpbnb_fit.md) : Residual diagnostic
  plots for a random-parameter bivariate NB model

## Simulation

Data simulators

- [`simulate_bnb()`](simulate_bnb.md) : Simulate data from the
  Famoye/Sarmanov bivariate NB distribution
- [`simulate_rpbnb()`](simulate_rpbnb.md) : Simulate data from a
  random-parameter bivariate NB process
- [`simulate_rpbnb_copula()`](simulate_rpbnb_copula.md) : Simulate data
  from a copula RP-BNB process
- [`simulate_rpbnb_tmb()`](simulate_rpbnb_tmb.md) : Simulate data from a
  bivariate NB process

## Package

- [`rpbnb-package`](rpbnb-package.md) : rpbnb: Random-Parameter
  Bivariate Negative Binomial Regression
