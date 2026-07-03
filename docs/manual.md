# rpbnb Package Reference Manual

**Random-Parameter Bivariate Negative Binomial Regression**

**Version:** Development  
**Last Updated:** July 3, 2026

---

## Table of Contents

1. [Package Overview](#package-overview)
2. [Core Functions](#core-functions)
3. [Methods and Utilities](#methods-and-utilities)
4. [Quick Start Examples](#quick-start-examples)
5. [Technical Details](#technical-details)
6. [Parameter Reference](#parameter-reference)

---

## Package Overview

**rpbnb** provides comprehensive tools for estimation and simulation of random-parameter bivariate negative binomial (RP-BNB) regression models. The package supports:

- **Fixed-parameter bivariate NB models** using the Famoye/Sarmanov copula with explicit dependence structure
- **Random-parameter extensions** where coefficients vary across observations
- **Per-coefficient distribution choice**: normal, lognormal, uniform, and triangular
- **Efficient computation**: analytic gradient evaluation and quasi-Monte Carlo integration
- **Simulation tools** for model development and Monte Carlo experiments

The package is designed for analyzing count data where:
- Both dependent variables are non-negative integer counts
- A dependence structure between the two outcomes is theoretically or empirically motivated
- Heterogeneity in regression coefficients across observations is plausible

---

## Core Functions

### Simulation Functions

#### `simulate_bnb()`

Generate data from a fixed-parameter bivariate negative binomial distribution with Sarmanov dependence.

```r
simulate_bnb(n, beta1, beta2, dispersion = c(m1 = 0.5, m2 = 0.5),
             lambda = 0, covariates = NULL, seed = NULL)
```

**Arguments:**

- `n`: Number of observations
- `beta1`, `beta2`: Named numeric vectors of coefficients for equations 1 and 2; must include `"(Intercept)"`
- `dispersion`: Named vector `c(m1 = ..., m2 = ...)` of NB2 dispersion parameters (variance = μ + m·μ²)
- `lambda`: Famoye/Sarmanov dependence parameter (0 = independent margins)
- `covariates`: Optional data frame of covariates; if NULL, standard-normal columns generated
- `seed`: Optional random seed for reproducibility; if NULL, RNG left untouched

**Value:** A list containing:
- `data`: data frame with y1, y2, and covariate columns
- `mu`: data frame with conditional means (mu1, mu2)
- `true`: true parameters used in simulation
- `settings`: simulation settings (n, seed)
- `meta`: metadata (R version, seed)

**Example:**

```r
sim <- simulate_bnb(
  n = 500,
  beta1 = c("(Intercept)" = 0.5, x1 = 0.3),
  beta2 = c("(Intercept)" = 0.2, x1 = -0.2),
  dispersion = c(m1 = 0.4, m2 = 0.5),
  lambda = 0.1,
  seed = 42
)
head(sim$data)
```

---

#### `simulate_rpbnb()`

Generate data from a random-parameter bivariate negative binomial process with per-coefficient distribution choice.

```r
simulate_rpbnb(n, beta1, beta2, random_1 = NULL, random_2 = NULL,
               dispersion = c(m1 = 0.5, m2 = 0.5),
               lambda = 0, covariates = NULL, seed = NULL)
```

**Arguments:**

- `n`: Number of observations
- `beta1`, `beta2`: Named numeric vectors of fixed-coefficient means per equation
- `random_1`, `random_2`: Named lists specifying random coefficients:
  - Each element: `list(dist = "normal"|"lognormal"|"uniform"|"triangular", scale = ..., sign = ±1)`
  - `dist`: default "normal"
  - `sign`: -1 or 1 (lognormal only)
  - `scale` (or `sd`): dispersion parameter
- `dispersion`: Named vector `c(m1 = ..., m2 = ...)` of NB2 dispersions
- `lambda`: Dependence parameter (currently must be 0 for Phase 1)
- `covariates`: Optional covariate data frame
- `seed`: Optional random seed

**Value:** A list containing:
- `data`: data frame with y1, y2, and covariates
- `coef_realized`: n × p matrices of realized random coefficients per equation
- `mu`: data frame with conditional means
- `true`: true parameters including random specifications
- `settings`: simulation settings
- `meta`: metadata

**Example:**

```r
sim <- simulate_rpbnb(
  n = 500,
  beta1 = c("(Intercept)" = 0.2, x1 = 0.4),
  beta2 = c("(Intercept)" = 0.1, x1 = -0.3),
  random_1 = list(x1 = list(dist = "normal", sd = 0.5)),
  random_2 = list(x1 = list(dist = "lognormal", scale = 0.3, sign = 1)),
  dispersion = c(m1 = 0.4, m2 = 0.5),
  seed = 123
)
```

---

### Model Fitting Functions

#### `fit_bnb()`

Maximum likelihood estimation of fixed-parameter bivariate NB regression.

```r
fit_bnb(formula1, formula2, data, dispersion = NULL,
        lambda = NULL, control = NULL, ...)
```

**Arguments:**

- `formula1`, `formula2`: Model formulae for equations 1 and 2
- `data`: Data frame containing the variables
- `dispersion`: Optional initial values for dispersion; estimated if NULL
- `lambda`: Optional dependence parameter; estimated if NULL
- `control`: List of control parameters (see `bnb_control()`)

**Value:** An object of class `bnb_fit` containing:
- `coefficients`: Estimated coefficients
- `vcov`: Variance-covariance matrix
- `loglik`: Maximized log-likelihood
- `data`: Model data used in estimation
- `fitted.values`: Predicted values
- And other model diagnostics

**Example:**

```r
fit <- fit_bnb(y1 ~ x1 + x2, y2 ~ x1 + x2, data = sim$data)
summary(fit)
```

---

#### `fit_rpbnb()`

Quasi-Monte Carlo maximum likelihood estimation of random-parameter BNB regression.

```r
fit_rpbnb(formula1, formula2, data,
          random_1 = NULL, random_2 = NULL,
          dispersion = NULL, lambda = NULL,
          control = rpbnb_control(), ...)
```

**Arguments:**

- `formula1`, `formula2`: Model formulae
- `data`: Data frame
- `random_1`, `random_2`: Random coefficient specifications (character vector of names, or named list with distribution configs)
  - `"x"` → normal distribution (sd to be estimated)
  - `list(x = list(dist = "lognormal", scale = 0.2))` → fixed distribution with specified scale
- `dispersion`, `lambda`: Optional initial values
- `control`: Control parameters (see `rpbnb_control()`)

**Value:** An object of class `rpbnb_fit` with:
- `coefficients`: Fixed-coefficient estimates
- `random`: Random coefficient specifications and realized draws
- `vcov`: Variance-covariance matrix
- `loglik`: Quasi-MC log-likelihood approximation
- Model diagnostics and fit statistics

**Example:**

```r
fit <- fit_rpbnb(
  y1 ~ x1 + x2,
  y2 ~ x1 + x2,
  data = sim$data,
  random_1 = list(x1 = list(dist = "normal", scale = 0.3)),
  random_2 = list(x1 = list(dist = "lognormal", scale = 0.2))
)
summary(fit)
```

---

## Methods and Utilities

### S3 Methods

| Method | Class | Description |
|--------|-------|-------------|
| `print()` | bnb_fit, rpbnb_fit | Print model object |
| `summary()` | bnb_fit, rpbnb_fit | Display model summary with coefficients and tests |
| `predict()` | bnb_fit | Predict conditional means on new data |
| `predict()` | rpbnb_fit | Predict draw-averaged conditional means |
| `coef()` | bnb_fit, rpbnb_fit | Extract coefficient estimates |
| `vcov()` | bnb_fit, rpbnb_fit | Extract variance-covariance matrix |
| `logLik()` | bnb_fit, rpbnb_fit | Extract log-likelihood |

### Utility Functions

#### `bnb_gof()`

Compute goodness-of-fit measures.

```r
bnb_gof(object)
```

**Returns:** A data frame with pseudo-R² (McFadden), BIC, and AIC.

#### `bnb_marginal_effects()`

Compute marginal effects on conditional means at the sample mean.

```r
bnb_marginal_effects(object, equation = 1)
```

**Returns:** Named numeric vector of marginal effects.

#### `bnb_elasticities()`

Compute elasticities of expected outcomes with respect to covariates.

```r
bnb_elasticities(object, equation = 1)
```

**Returns:** Named numeric vector of elasticities at sample mean.

#### `rpbnb_control()`

Control parameters for random-parameter estimation.

```r
rpbnb_control(R = 200, seed = NULL, parallel = FALSE, workers = NULL)
```

**Arguments:**

- `R`: Number of Halton quasi-Monte Carlo draws (default 200)
- `seed`: Optional seed for reproducibility
- `parallel`: Logical; use parallel computation?
- `workers`: Number of workers for parallel computation

---

## Quick Start Examples

### Example 1: Simulate and Fit a Fixed-Parameter Model

```r
library(rpbnb)

# Simulate data
sim <- simulate_bnb(
  n = 1000,
  beta1 = c("(Intercept)" = 0.5, x = 0.3),
  beta2 = c("(Intercept)" = 0.2, x = -0.1),
  dispersion = c(m1 = 0.4, m2 = 0.5),
  lambda = 0.15,
  seed = 42
)

# Fit fixed-parameter model
fit <- fit_bnb(y1 ~ x, y2 ~ x, data = sim$data)

# Summary and diagnostics
summary(fit)
bnb_gof(fit)
bnb_marginal_effects(fit, equation = 1)

# Predictions
newdata <- data.frame(x = seq(0, 1, by = 0.25))
predict(fit, newdata)
```

### Example 2: Simulate and Fit a Random-Parameter Model

```r
# Simulate with random slope on x
sim <- simulate_rpbnb(
  n = 1000,
  beta1 = c("(Intercept)" = 0.2, x1 = 0.4, x2 = -0.2),
  beta2 = c("(Intercept)" = 0.1, x1 = -0.3, x2 = 0.2),
  random_1 = list(x1 = list(dist = "normal", sd = 0.5)),
  random_2 = list(x1 = list(dist = "lognormal", scale = 0.3, sign = 1)),
  dispersion = c(m1 = 0.4, m2 = 0.5),
  seed = 99
)

# Fit random-parameter model
fit <- fit_rpbnb(
  y1 ~ x1 + x2,
  y2 ~ x1 + x2,
  data = sim$data,
  random_1 = list(x1 = list(dist = "normal")),
  random_2 = list(x1 = list(dist = "lognormal")),
  control = rpbnb_control(R = 300, seed = 123)
)

summary(fit)
bnb_gof(fit)
```

### Example 3: Distribution-Specific Random Coefficients

```r
# Mix distributions across coefficients
sim <- simulate_rpbnb(
  n = 800,
  beta1 = c("(Intercept)" = 0.3, x1 = 0.5, x2 = -0.2),
  beta2 = c("(Intercept)" = 0.2, x1 = 0.2, x2 = 0.3),
  random_1 = list(
    x1 = list(dist = "lognormal", scale = 0.3, sign = 1),
    x2 = list(dist = "triangular", scale = 0.2)
  ),
  random_2 = list(
    x1 = list(dist = "uniform", scale = 0.4)
  ),
  dispersion = c(m1 = 0.5, m2 = 0.6),
  seed = 77
)

# Fit with the same distribution specifications
fit <- fit_rpbnb(
  y1 ~ x1 + x2,
  y2 ~ x1 + x2,
  data = sim$data,
  random_1 = list(
    x1 = list(dist = "lognormal", scale = 0.3),
    x2 = list(dist = "triangular")
  ),
  random_2 = list(
    x1 = list(dist = "uniform")
  ),
  control = rpbnb_control(R = 250, seed = 42)
)

summary(fit)
```

---

## Technical Details

### Bivariate Negative Binomial (Sarmanov) Distribution

The Famoye/Sarmanov bivariate NB2 joint PMF is:

$$P(Y_1 = y_1, Y_2 = y_2) = p_1(y_1) \, p_2(y_2) \, \left[ 1 + \lambda \left( e^{-y_1} - c_1 \right) \left( e^{-y_2} - c_2 \right) \right]$$

where:
- $p_k(y_k)$ is the marginal NB2 PMF with mean $\mu_k$ and dispersion $m_k$
- $c_k = (1 + d \cdot m_k \cdot \mu_k)^{-1/m_k}$ is the moment constant ($d = \exp(-\gamma_E) \approx 0.561$)
- $\lambda$ is the dependence parameter

The valid range of $\lambda$ is determined by the marginal means and dispersions to ensure $P \geq 0$ everywhere.

### Random-Coefficient Distributions

Each random coefficient is specified independently. For a given coefficient $j$, the realized value is:

**Normal:** $\beta_j = \mu_j + \sigma_j \cdot z$, where $z \sim N(0, 1)$

**Lognormal:** $\beta_j = \text{sign} \cdot \exp(\mu_j + \sigma_j \cdot z)$, where $z \sim N(0, 1)$  
Note: $E[\beta_j] = \text{sign} \cdot \exp(\mu_j + \sigma_j^2/2)$

**Uniform:** $\beta_j = \mu_j + \sigma_j \cdot (2u - 1)$, where $u \sim U(0, 1)$

**Triangular:** $\beta_j = \mu_j + \sigma_j \cdot T$, where $T$ is symmetric triangular on $[-1, 1]$

### Quasi-Monte Carlo Integration

Random-parameter likelihoods are estimated using randomized Halton sequences for quasi-Monte Carlo integration. This provides:
- Faster convergence than standard Monte Carlo (lower variance for the same number of draws)
- Reproducibility when a seed is fixed
- Improved accuracy for the same computational budget

---

## Parameter Reference

### Distribution Scale Parameters

| Distribution | `scale` Parameter | Default Estimation | Notes |
|:------------|:-----------------|:------------------|:------|
| Normal | Standard deviation (σ) | Estimated | SD of realized coefficients |
| Lognormal | Log-scale (σ) | Estimated | Related to coefficient CV |
| Uniform | Half-width (σ) | Estimated | Coefficient support: [μ - σ, μ + σ] |
| Triangular | Half-width (σ) | Estimated | Coefficient support: [μ - σ, μ + σ] |

### Dispersion Interpretation

The NB2 dispersion parameter $m$ (also called "alpha" in some packages):
- **Variance formula:** $\text{Var}(Y) = \mu + m \cdot \mu^2$
- **Poisson reference:** $m = 0$ gives Poisson (no overdispersion)
- **Larger $m$:** More overdispersion

### Lambda (Dependence) Interpretation

- **λ = 0:** Margins independent (Poisson product in limit)
- **λ > 0:** Positive dependence (outcomes tend to co-vary positively)
- **λ < 0:** Negative dependence (outcomes tend to co-vary negatively)
- **Valid range:** Constrained by margins and dispersions

---

## References

The rpbnb package implements methodology from:

1. Famoye, F., Wulu Jr., T. (2003). "Modeling household fertility decisions with generalized Poisson regression." *Journal of Population Research*.

2. Sarmanov, O. V. (1966). "Generalized normal correlation and two-dimensional Fréchet classes." *Doklady Akademii Nauk SSSR*.

3. For quasi-Monte Carlo methods, see documentation on Halton sequences in simulation.

---

*Reference manual for rpbnb package (Development version, July 2026)*
