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
