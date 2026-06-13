# Scope of Work: R Software for Bivariate and Random-Parameter Negative Binomial Models

## 1. Project Title

**Development and Validation of R Codes for Bivariate Negative Binomial and Random-Parameter Negative Binomial Regression Models**

## 2. Background

Negative binomial regression is commonly used to analyze overdispersed count data. In applications involving correlated outcomes or unobserved heterogeneity, standard negative binomial models may be insufficient. This project will develop a reproducible R-based modeling framework for:

1. Bivariate negative binomial (BNB) regression;
2. Random-parameter negative binomial (RPBNB) regression;
3. Simulation of count data following an RPBNB data-generating process; and
4. Flexible specification of fixed and random parameters through a reusable R project.

The work will build upon relevant methodological papers stored in `docs/literature`, existing R scripts stored in `Rcodes`, and sample datasets stored in `data`.

## 3. Objectives

The project objectives are to:

1. Review the methodological literature and document the statistical formulations required for BNB and RPBNB models.
2. Review, execute, validate, and improve the existing R codes.
3. Develop and validate an R implementation of a BNB regression model.
4. Develop an R-based simulator for generating samples from an RPBNB data-generating process.
5. Develop and validate an R implementation of an RPBNB regression model.
6. Produce a reusable R project that allows users to specify fixed and random parameters without modifying the model-estimation source code.
7. Provide documentation, examples, tests, and reproducible validation results.

## 4. Project Inputs

The project will use the following existing directories:

```text
docs/literature/   Methodological papers and technical references
Rcodes/            Existing R scripts and model implementations
data/              Sample and validation datasets
```

The development process shall preserve the original input files. Revised or replacement scripts shall be stored separately or managed through version control.

## 5. Technical Scope

### Task 1. Review Literature in `docs/literature`

Review the papers and technical references contained in `docs/literature`.

The review shall identify and document:

- Probability mass functions and likelihood formulations;
- Parameterizations of the negative binomial distribution;
- Bivariate dependence structures;
- Overdispersion parameters;
- Random-parameter specifications;
- Assumed distributions for random coefficients;
- Simulation or numerical integration methods;
- Correlation among random parameters, when applicable;
- Estimation algorithms;
- Starting-value procedures;
- Model-identification requirements;
- Convergence criteria;
- Standard-error estimation;
- Goodness-of-fit measures; and
- Validation examples reported in the literature.

A literature review summary shall be created at:

```text
docs/literature_review.md
```

The summary shall clearly distinguish alternative negative binomial parameterizations, including differences in the definitions of the mean, variance, dispersion parameter, and probability mass function.

### Task 2. Review and Validate Existing Codes in `Rcodes`

Inspect all existing scripts in `Rcodes` and determine:

- The purpose of each script;
- Required R packages;
- Input and output files;
- Model formulation;
- Parameterization;
- Optimization method;
- Starting values;
- Parameter constraints;
- Numerical integration or simulation method;
- Convergence checks;
- Standard-error calculations; and
- Known errors, incomplete components, or undocumented assumptions.

Each existing script shall be executed using the available sample data when possible.

Validation shall include:

- Syntax and dependency checks;
- Reproducibility checks;
- Verification of dimensions and variable types;
- Likelihood-function checks;
- Comparison of analytical and numerical calculations;
- Examination of optimizer convergence;
- Sensitivity to starting values;
- Detection of invalid parameter regions;
- Review of coefficient signs and magnitudes;
- Comparison with published or previously generated results; and
- Identification of numerical instability.

The findings shall be documented at:

```text
docs/code_review.md
```

Existing scripts shall not be overwritten unless a backup or version-control history is available.

### Task 3. Develop and Validate a Bivariate Negative Binomial Model

Develop R code for estimating a bivariate negative binomial regression model using paired count outcomes.

The implementation shall support:

- Two dependent count variables;
- Separate design matrices for the two outcomes;
- Outcome-specific regression coefficients;
- Outcome-specific or shared dispersion parameters, according to the selected formulation;
- A dependence or covariance parameter;
- User-specified exposure or offset terms;
- Observation weights, where appropriate;
- Missing-data checks;
- Parameter constraints;
- Maximum-likelihood estimation;
- Multiple optimization methods or fallback optimizers;
- User-specified starting values;
- Hessian-based covariance estimation;
- Robust handling of singular or non-positive-definite Hessians;
- Log-likelihood and information criteria;
- Predicted means;
- Residual or diagnostic outputs; and
- A standard model summary.

The primary fitting interface should follow an R-style formula structure, such as:

```r
fit_bnb(
  formula_1 = y1 ~ x1 + x2,
  formula_2 = y2 ~ x1 + x3,
  data = sample_data,
  offset_1 = NULL,
  offset_2 = NULL,
  control = list()
)
```

#### BNB Validation

The BNB model shall be validated using a sample dataset in `data`.

Validation shall include:

1. Data-quality and descriptive-statistics checks;
2. Comparison with two independently fitted univariate negative binomial models;
3. Verification that the bivariate model reduces to an appropriate limiting or independent case when the dependence parameter approaches its independence value;
4. Comparison of estimated parameters with known, published, or benchmark values when available;
5. Evaluation of log-likelihood, AIC, and BIC;
6. Gradient checks using finite differences;
7. Hessian and standard-error checks;
8. Sensitivity analysis using multiple starting values;
9. Prediction checks;
10. Convergence and boundary-condition checks; and
11. Reproducible generation of all validation results.

Validation outputs shall be stored under:

```text
results/bnb/
```

### Task 4. Generate Samples from an RPBNB Data-Generating Process

Develop R functions to simulate count data following a random-parameter negative binomial process.

The simulator shall allow users to specify:

- Number of observations;
- Number and names of explanatory variables;
- Fixed coefficient values;
- Random coefficient means;
- Random coefficient standard deviations;
- Random coefficient distributions;
- Correlation among random coefficients;
- Negative binomial dispersion;
- Exposure or offset variables;
- Covariate distributions;
- Covariate correlation;
- Random seed; and
- Number of simulated datasets or Monte Carlo replications.

At minimum, the simulator should support normal random coefficients. The design should allow extension to other distributions, such as lognormal, triangular, uniform, or truncated normal distributions.

A proposed interface is:

```r
simulate_rpbnb(
  n = 5000,
  beta_fixed = c(intercept = -1.0, x1 = 0.4),
  random_parameters = list(
    x2 = list(mean = -0.2, sd = 0.5, distribution = "normal")
  ),
  dispersion = 0.8,
  offset = NULL,
  seed = 1234
)
```

The simulator shall return:

- Simulated covariates;
- Observation-level coefficient realizations;
- Conditional means;
- Simulated counts;
- True parameter values;
- Simulation settings; and
- Reproducibility metadata.

Simulation validation shall verify that:

- Generated random coefficients match their specified means, standard deviations, and correlations;
- Count means and variances are consistent with the selected parameterization;
- Overdispersion is present when specified;
- Results are reproducible with the same seed; and
- Large-sample simulations recover expected theoretical properties.

### Task 5. Develop and Validate an RPBNB Regression Model

Develop R code for maximum simulated likelihood estimation of a random-parameter negative binomial regression model.

The model shall allow each coefficient to be declared as either:

- Fixed; or
- Random.

The implementation shall support:

- Formula-based model specification;
- User selection of random coefficients;
- Random-coefficient means;
- Random-coefficient standard deviations;
- At least normally distributed random coefficients;
- Optional extension to correlated random coefficients;
- Negative binomial dispersion;
- Exposure or offset terms;
- User-specified starting values;
- Parameter transformations that enforce valid standard deviations and dispersion values;
- Pseudo-random and reproducible simulation draws;
- Quasi-random draws, where feasible;
- Antithetic draws, where feasible;
- Configurable number of simulation draws;
- Maximum simulated likelihood estimation;
- Numerical stabilization using log-sum-exp or equivalent methods;
- Gradient and Hessian calculations;
- Standard and robust covariance estimates;
- Convergence diagnostics;
- Predicted conditional and unconditional means;
- Estimated observation-level parameter distributions; and
- Model-fit statistics.

A proposed fitting interface is:

```r
fit_rpbnb(
  formula = y ~ x1 + x2 + x3,
  data = sample_data,
  random = c(x2 = "normal", x3 = "normal"),
  correlated = FALSE,
  draws = 1000,
  draw_type = "halton",
  seed = 1234,
  control = list()
)
```

The software shall use stable internal parameterizations. Parameters constrained to be positive, including standard deviations and dispersion terms, should be estimated on a transformed scale.

#### RPBNB Validation

The RPBNB implementation shall be validated using the simulated samples developed in Task 4.

Validation shall include:

1. Recovery of known fixed coefficient values;
2. Recovery of random-parameter means;
3. Recovery of random-parameter standard deviations;
4. Recovery of dispersion parameters;
5. Bias, empirical standard deviation, root mean squared error, and confidence-interval coverage across Monte Carlo replications;
6. Sensitivity to the number and type of simulation draws;
7. Sensitivity to starting values;
8. Comparison with a fixed-parameter negative binomial model;
9. Likelihood-ratio or equivalent model-comparison tests, where statistically valid;
10. Gradient and Hessian checks;
11. Convergence-frequency reporting;
12. Replication under multiple random seeds; and
13. Computational-time reporting.

Monte Carlo validation results shall be stored under:

```text
results/rpbnb/
```

### Task 6. Produce a Reusable R Project

Create a self-contained R project that separates:

- User configuration;
- Data preparation;
- Probability and likelihood functions;
- Simulation functions;
- Estimation functions;
- Diagnostics;
- Reporting;
- Tests; and
- Examples.

The user shall be able to change fixed and random parameter specifications through a configuration file or function arguments without editing the core estimation code.

A recommended project structure is:

```text
project-root/
├── scope.md
├── README.md
├── DESCRIPTION
├── renv.lock
├── .Rprofile
├── config/
│   ├── bnb_example.yml
│   └── rpbnb_example.yml
├── data/
│   ├── raw/
│   ├── processed/
│   └── simulated/
├── docs/
│   ├── literature/
│   ├── literature_review.md
│   ├── code_review.md
│   ├── model_formulation.md
│   └── user_guide.md
├── R/
│   ├── data_validation.R
│   ├── distributions.R
│   ├── bnb_likelihood.R
│   ├── fit_bnb.R
│   ├── simulate_rpbnb.R
│   ├── rpbnb_likelihood.R
│   ├── simulation_draws.R
│   ├── fit_rpbnb.R
│   ├── diagnostics.R
│   ├── predict.R
│   ├── summary.R
│   └── utilities.R
├── Rcodes/
│   └── legacy/
├── scripts/
│   ├── 01_review_existing_codes.R
│   ├── 02_validate_bnb.R
│   ├── 03_generate_rpbnb_sample.R
│   ├── 04_validate_rpbnb.R
│   └── 05_run_monte_carlo.R
├── tests/
│   └── testthat/
├── results/
│   ├── bnb/
│   └── rpbnb/
└── examples/
    ├── bnb_example.R
    └── rpbnb_example.R
```

The project shall use relative paths and shall not depend on machine-specific absolute paths.

Dependency management should be implemented using `renv` or an equivalent reproducible R environment.

## 6. Configuration Requirements

The project shall provide a configuration mechanism for defining fixed and random parameters.

An illustrative YAML configuration is:

```yaml
model:
  family: "rpbnb"
  formula: "y ~ x1 + x2 + x3"
  dispersion_parameterization: "NB2"

parameters:
  fixed:
    - "(Intercept)"
    - "x1"

  random:
    x2:
      distribution: "normal"
      correlated: false
    x3:
      distribution: "normal"
      correlated: false

simulation:
  draws: 1000
  draw_type: "halton"
  antithetic: true
  seed: 1234

optimization:
  method: "BFGS"
  fallback_methods:
    - "nlminb"
    - "Nelder-Mead"
  max_iterations: 5000
  gradient_tolerance: 1.0e-6
```

Configuration files shall be validated before estimation. Invalid parameter names, unsupported distributions, duplicated parameter assignments, and incompatible options shall produce informative errors.


## 6A. R Package Project Requirements

The final implementation shall be delivered as a formal, installable R package rather than as a collection of standalone scripts only.

The package shall support:

- Installation from source on Windows and Linux;
- Standard R package structure;
- Exported user-facing functions;
- Internal helper functions;
- Package documentation generated with `roxygen2`;
- Automated testing with `testthat`;
- Dependency management;
- Reproducible examples;
- Vignettes or extended user guides;
- Cross-platform parallel processing; and
- R package checking with no errors and no avoidable warnings.

A recommended package structure is:

```text
rpbnb/
├── DESCRIPTION
├── NAMESPACE
├── LICENSE
├── README.md
├── NEWS.md
├── R/
│   ├── fit_bnb.R
│   ├── fit_rpbnb.R
│   ├── simulate_rpbnb.R
│   ├── copula_likelihood.R
│   ├── famoye_likelihood.R
│   ├── simulation_draws.R
│   ├── parallel.R
│   ├── diagnostics.R
│   ├── predict.R
│   ├── summary.R
│   └── utilities.R
├── man/
├── tests/
│   └── testthat/
├── vignettes/
├── inst/
│   ├── extdata/
│   └── config/
├── data-raw/
├── docs/
└── src/
```

Compiled code in `src/` may be added using Rcpp or another suitable interface after the pure-R implementation has been validated.

### 6A.1 Required User-Facing Functions

At minimum, the package shall provide:

```r
fit_bnb()
fit_rpbnb()
simulate_rpbnb()
rpbnb_control()
set_rpbnb_threads()
predict()
summary()
coef()
vcov()
logLik()
```

The primary fitting interfaces should support:

```r
fit_bnb(
  formula_1,
  formula_2,
  data,
  dependence = c("independence", "copula", "famoye"),
  copula_family = c("gaussian", "clayton", "frank", "gumbel"),
  control = rpbnb_control()
)
```

and:

```r
fit_rpbnb(
  formula,
  data,
  random,
  correlated = FALSE,
  draws = 1000,
  draw_type = "halton",
  workers = NULL,
  seed = 1234,
  control = rpbnb_control()
)
```

## 6B. Bivariate Dependence Model Options

The bivariate negative binomial implementation shall provide at least three dependence specifications:

1. Independent negative binomial margins;
2. Copula-based bivariate negative binomial model; and
3. Famoye-style bivariate negative binomial model.

The user shall select the dependence formulation through the `dependence` argument without changing source code.

### 6B.1 Copula-Based Model

The copula implementation shall combine two discrete negative binomial marginal distributions through a user-selected copula.

At minimum, the following copula families shall be supported:

- Gaussian;
- Clayton;
- Frank; and
- Gumbel.

Additional copula families may be added if justified by the reviewed literature.

For discrete outcomes, the joint probability shall be calculated using the copula rectangle probability:

```text
P(Y1 = y1, Y2 = y2)
  = C(F1(y1), F2(y2))
  - C(F1(y1 - 1), F2(y2))
  - C(F1(y1), F2(y2 - 1))
  + C(F1(y1 - 1), F2(y2 - 1))
```

The copula implementation shall include:

- Valid parameter transformations;
- Family-specific parameter bounds;
- Independence-limit checks;
- Stable calculation of small rectangle probabilities;
- Protection against negative probabilities caused by numerical precision;
- Dependence measures, including Kendall's tau where available;
- Model comparison across copula families; and
- Documentation of tail-dependence properties.

An illustrative interface is:

```r
fit <- fit_bnb(
  formula_1 = y1 ~ x1 + x2,
  formula_2 = y2 ~ x1 + x3,
  data = sample_data,
  dependence = "copula",
  copula_family = "gaussian"
)
```

Copula models shall be validated by:

- Simulating data from known copula parameters;
- Recovering marginal regression coefficients;
- Recovering dependence parameters;
- Verifying the independence limit;
- Comparing log-likelihood values across candidate families; and
- Checking joint probability masses over selected count grids.

### 6B.2 Famoye-Style Model

The package shall include a Famoye-style bivariate negative binomial option based on the specific mathematical formulation identified in the literature review.

Because multiple bivariate negative binomial formulations are associated with Famoye and related Sarmanov-type constructions, the exact probability mass function, dependence parameter, admissible parameter range, mean, variance, covariance, and correlation shall be documented before implementation.

A possible Sarmanov/Famoye representation is:

```text
p(y1, y2)
  = p1(y1) p2(y2)
    [1 + omega h1(y1) h2(y2)]
```

where:

- `p1()` and `p2()` are negative binomial marginal probability functions;
- `h1()` and `h2()` are bounded, centered kernel functions; and
- `omega` is a dependence parameter constrained to preserve a valid joint probability mass.

The implementation shall include:

- The exact literature-supported Famoye parameterization;
- Validity constraints for the dependence parameter;
- Parameter transformations that enforce the admissible range;
- Marginal-distribution checks;
- Covariance and correlation calculations;
- Independence-limit checks;
- Stable likelihood evaluation; and
- Comparison with copula and independent formulations.

An illustrative interface is:

```r
fit <- fit_bnb(
  formula_1 = y1 ~ x1 + x2,
  formula_2 = y2 ~ x1 + x3,
  data = sample_data,
  dependence = "famoye"
)
```

The Famoye model shall not be treated as interchangeable with a generic copula model. Its mathematical formulation and assumptions shall be documented separately.

## 6C. Multithreading and Parallel Processing

The R package shall support multithreaded or multi-process computation for computationally intensive tasks, including:

- Simulated likelihood evaluation;
- Generation of simulation draws;
- Monte Carlo replications;
- Bootstrap estimation;
- Sensitivity analyses;
- Multiple starting-value runs;
- Copula-family comparison; and
- Validation experiments.

### 6C.1 Cross-Platform Parallel Backend

The default parallel implementation shall work on both Windows and Linux.

The recommended backend is the `future` ecosystem using:

```r
future::plan(future::multisession, workers = n_workers)
```

The `multisession` backend shall be the portable default because it launches independent R processes and is supported on Windows and Linux.

Linux users may optionally use:

```r
future::plan(future::multicore, workers = n_workers)
```

The package shall prevent or clearly warn against use of `multicore` on Windows.

A user-facing configuration function shall be provided:

```r
set_rpbnb_threads(
  workers = 8,
  strategy = "multisession"
)
```

Supported strategies should include:

- `sequential`;
- `multisession`; and
- `multicore` on supported operating systems.

### 6C.2 Worker and Thread Controls

The package shall allow users to specify:

- Number of workers;
- Parallel strategy;
- Reproducible random-number generation;
- Chunk size;
- Maximum memory usage where feasible;
- Whether optimization starts are evaluated in parallel; and
- Whether Monte Carlo replications are parallelized.

The default worker count should be conservative, such as:

```r
max(1L, future::availableCores() - 1L)
```

The package shall not automatically consume all available CPU cores without an explicit user setting.

### 6C.3 Reproducible Parallel Random Numbers

Parallel simulations shall use reproducible random-number streams.

The implementation shall support:

- User-specified seeds;
- Reproducible results across repeated runs;
- Reproducible Monte Carlo replications;
- Reproducible simulated-likelihood draws; and
- Platform-aware documentation of any unavoidable numerical differences.

Appropriate methods may include:

- `future.seed = TRUE`;
- L'Ecuyer-CMRG random-number streams; or
- Pre-generated simulation draws shared across workers.

The same simulation draws should be retained during optimization to ensure a smooth and reproducible simulated likelihood.

### 6C.4 Optional Compiled Multithreading

After validation of the R implementation, computationally intensive likelihood components may be implemented in compiled code.

Possible approaches include:

- Rcpp;
- RcppArmadillo;
- OpenMP; and
- Thread-safe C++ numerical routines.

Any OpenMP implementation shall:

- Compile conditionally;
- Fall back to a single-threaded implementation when OpenMP is unavailable;
- Avoid nested oversubscription with process-level parallelism;
- Allow users to control the thread count; and
- Be tested on both Windows and Linux.

The package shall not run both a large number of parallel R processes and a large number of OpenMP threads per process by default.

## 6D. Windows and Linux Compatibility

The package shall be developed and tested for:

- Windows 10 or later;
- Windows 11;
- Current supported Ubuntu Linux releases; and
- Other common Linux distributions where feasible.

The implementation shall:

- Use platform-independent path handling;
- Avoid shell commands that exist only on Linux;
- Avoid fork-only parallel code as the default;
- Avoid hard-coded path separators;
- Avoid case-sensitive filename assumptions;
- Use UTF-8 text encoding;
- Support installation from an R source package;
- Avoid unnecessary external system dependencies; and
- Produce informative messages when optional system libraries are unavailable.

Cross-platform paths shall be constructed with functions such as:

```r
file.path()
system.file()
normalizePath()
```

The package shall not rely on absolute paths such as:

```text
C:\...
/home/...
```

### 6D.1 Cross-Platform Testing

Package validation shall include:

```r
R CMD build .
R CMD check rpbnb_*.tar.gz
```

or equivalent `devtools` commands:

```r
devtools::document()
devtools::test()
devtools::check()
```

Testing shall be performed in clean R sessions on both Windows and Linux.

Where continuous integration is available, the project should include a matrix covering:

- Windows latest;
- Ubuntu latest;
- Multiple supported R versions; and
- Release and development R versions where feasible.

## 6E. Package Dependencies

Recommended dependencies include:

```text
stats
methods
MASS
copula
future
future.apply
parallelly
numDeriv
testthat
roxygen2
```

Optional dependencies may include:

```text
Rcpp
RcppArmadillo
randtoolbox
qrng
yaml
knitr
rmarkdown
withr
```

Dependencies shall be minimized and justified. Core estimation functionality should not require packages used only for vignettes, plotting, or optional performance enhancements.

## 6F. Additional Package Validation Criteria

The R package deliverable will be accepted when:

1. The package installs successfully on Windows and Linux.
2. `R CMD check` completes without errors.
3. Exported functions are documented.
4. Examples execute successfully.
5. Unit and integration tests pass.
6. Copula models pass marginal and independence checks.
7. The Famoye model passes probability-validity and marginal checks.
8. RPBNB simulation and estimation are reproducible.
9. Parallel and sequential runs produce statistically equivalent results.
10. User-defined worker counts are respected.
11. Parallel random-number generation is reproducible.
12. The package falls back to sequential execution when parallel processing is unavailable.
13. Fixed and random parameters can be specified through function arguments or configuration files.
14. Model objects provide coefficients, covariance matrices, likelihood values, convergence diagnostics, predictions, and model metadata.


## 7. Statistical and Numerical Requirements

### 7.1 Parameterization

The selected BNB and RPBNB formulations shall be fully documented in:

```text
docs/model_formulation.md
```

The documentation shall include:

- Probability mass function;
- Conditional mean;
- Conditional variance;
- Joint distribution for the BNB model;
- Random-parameter mixing distribution;
- Log-likelihood or simulated log-likelihood;
- Parameter transformations;
- Independence assumptions;
- Correlation structures;
- Prediction equations; and
- Definitions of all reported statistics.

### 7.2 Numerical Stability

The implementation shall address:

- Overflow and underflow;
- Logarithms of zero;
- Invalid gamma-function evaluations;
- Very small or very large dispersion values;
- Invalid covariance matrices;
- Near-zero random-parameter standard deviations;
- Singular Hessians;
- Non-finite objective-function values; and
- Optimizer failure.

Where applicable, calculations shall use log-scale probability functions, `lgamma`, `log1p`, `expm1`, Cholesky decompositions, and log-sum-exp techniques.

### 7.3 Reproducibility

All simulations and simulated-likelihood estimations shall support a user-defined seed.

Validation scripts shall record:

- R version;
- Package versions;
- Operating system;
- Date and time;
- Git commit identifier, when available;
- Model configuration;
- Random seed;
- Number and type of draws; and
- Optimizer settings.

## 8. Testing Requirements

Automated tests shall be implemented using `testthat` or an equivalent framework.

Tests shall cover:

- Input validation;
- Formula parsing;
- Fixed/random parameter assignment;
- Distribution functions;
- Likelihood calculations;
- Simulation reproducibility;
- Parameter transformations;
- Gradient calculations;
- Prediction dimensions;
- Handling of offsets;
- Handling of missing values;
- Failure messages;
- BNB independence behavior;
- RPBNB behavior when random-parameter standard deviations approach zero; and
- Recovery of known parameters from simulated data.

Critical mathematical functions shall be tested against manually calculated values or an independent implementation.

## 9. Validation Acceptance Criteria

The implementation will be considered successfully validated when:

1. All core scripts run from a clean R session.
2. Required dependencies can be restored from the project environment.
3. Automated tests pass.
4. BNB estimates are reproducible and consistent with the selected mathematical formulation.
5. The BNB dependence structure is demonstrated using sample data.
6. RPBNB simulated data reproduce the specified distributional properties.
7. The RPBNB estimator recovers true parameters with acceptable Monte Carlo bias under sufficiently large samples and simulation draws.
8. Increasing the number of simulation draws produces stable estimates.
9. Alternative starting values lead to the same optimum or documented local optima.
10. The final model objects include coefficients, standard errors, likelihood values, convergence information, predictions, and configuration metadata.
11. A user can change fixed and random parameter assignments without modifying core source files.
12. All examples and validation analyses are reproducible using documented commands.

Exact numerical tolerances shall be established after reviewing the literature, sample sizes, and benchmark datasets.

## 10. Deliverables

The project shall produce:

1. `scope.md` — project scope and implementation requirements;
2. `README.md` — installation and quick-start instructions;
3. `docs/literature_review.md` — literature and methodology review;
4. `docs/code_review.md` — review of existing R codes;
5. `docs/model_formulation.md` — complete mathematical specification;
6. `docs/user_guide.md` — user instructions and model examples;
7. BNB estimation source code;
8. RPBNB simulation source code;
9. RPBNB estimation source code;
10. Configuration templates for fixed and random parameters;
11. Automated unit and integration tests;
12. Example scripts;
13. BNB validation results;
14. RPBNB Monte Carlo validation results;
15. Reproducible R environment files; and
16. A final technical summary describing methods, validation findings, limitations, and recommended future enhancements;
17. An installable R package source directory;
18. A built R source package archive;
19. Cross-platform Windows and Linux test results;
20. Copula model examples and validation results;
21. Famoye model examples and validation results; and
22. Parallel-processing examples and benchmarks.

## 11. Coding Standards

All R code shall:

- Follow a consistent style, preferably the tidyverse style guide where it does not conflict with numerical-performance requirements;
- Use descriptive function and variable names;
- Include roxygen2-style function documentation;
- Avoid undocumented global variables;
- Avoid hard-coded paths;
- Validate user inputs;
- Return informative errors and warnings;
- Separate public functions from internal helper functions;
- Include comments for non-obvious mathematical operations;
- Preserve reproducibility; and
- Be suitable for version control.

## 12. Performance Considerations

The implementation should minimize unnecessary loops and repeated matrix construction. Computationally intensive components may use vectorized R, compiled package functions, or optional Rcpp implementations after the pure-R formulation has been validated.

Performance benchmarks shall report:

- Sample size;
- Number of explanatory variables;
- Number of random parameters;
- Number and type of simulation draws;
- Number of optimizer iterations;
- Elapsed time; and
- Peak memory usage when feasible.

Parallel processing may be added for Monte Carlo replications, but model results shall remain reproducible.

## 13. Limitations and Extension Points

The initial implementation will focus on the BNB and RPBNB formulations supported by the reviewed literature and available validation materials. The project architecture should allow future extension to:

- Correlated random parameters;
- Alternative mixing distributions;
- Panel or repeated-observation models;
- Zero-inflated or hurdle formulations;
- Multivariate count models with more than two outcomes;
- Bayesian estimation;
- Cluster-robust covariance estimators;
- Marginal effects;
- Elasticities;
- Posterior parameter estimates; and
- Compiled likelihood functions.

## 14. Completion Definition

The project is complete when the literature and legacy-code reviews are documented, the BNB and RPBNB functions are implemented, simulated and real/sample-data validations are reproducible, automated tests pass, and the R project allows users to specify fixed and random coefficients through documented function arguments or configuration files.
