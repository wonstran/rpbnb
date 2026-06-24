# Changelog

## rpbnb 0.1.0

- Initial release. Phase 1 deliverables:
  - [`fit_bnb()`](../reference/fit_bnb.md) — bivariate NB with
    `independence` and `famoye` dependence.
  - [`fit_rpbnb()`](../reference/fit_rpbnb.md) — bivariate
    random-parameter NB via maximum simulated likelihood (normal random
    coefficients, Halton draws, optional cluster parallelism).
  - [`simulate_rpbnb()`](../reference/simulate_rpbnb.md) — normal
    random-coefficient data simulator.
  - [`rpbnb_control()`](../reference/rpbnb_control.md) — estimation
    control object.
  - Diagnostics: [`bnb_gof()`](../reference/bnb_gof.md),
    [`bnb_marginal_effects()`](../reference/bnb_marginal_effects.md),
    [`bnb_elasticities()`](../reference/bnb_elasticities.md).
  - S3 methods: `print`, `summary`, `coef`, `vcov`, `logLik`, `AIC`,
    `BIC`, `predict`.
  - Ported and validated against the original `Rcodes` scripts
    (preserved in `inst/legacy/`).
