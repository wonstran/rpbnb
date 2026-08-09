# RP-BNB Test Suite: export_dense_all.csv and export_open_all.csv

This directory contains a comprehensive test suite for fitting random-parameter bivariate negative binomial (RP-BNB) models to two large real-world datasets using the TMB engine.

## Data Files

- **export_dense_all.csv**: Dense dataset with ~10K observations
- **export_open_all.csv**: Open dataset with ~6.5K observations

Both files are located in `inst/extdata/` and contain count outcomes (y1, y2) and multiple numeric predictors.

## Test Scripts

### TMB Engine Tests

#### Dense Dataset (TMB)

1. **tmb_export_dense_all_famoye.R**
   - Model: Famoye/Sarmanov dependence
   - Method: Simulated Maximum Likelihood (SML)
   - Purpose: Test basic SML estimation with Famoye dependence

2. **tmb_export_dense_all_frank.R**
   - Model: Frank copula dependence
   - Method: Simulated Maximum Likelihood (SML)
   - Purpose: Test discrete copula dependence structure
   - Notes: Fixed-effects model (no random coefficients)

3. **tmb_export_dense_all_gaussian.R**
   - Model: Gaussian (Normal) copula dependence
   - Method: Simulated Maximum Likelihood (SML)
   - Purpose: Test Gaussian copula parametrization

4. **tmb_export_dense_random_coef.R**
   - Models: Multiple configurations with random coefficients
   - Includes:
     - Single random coefficient per equation (SML)
     - Multiple random coefficients (SML)
     - Multiple random coefficients (Laplace approximation)
   - Purpose: Compare random specification options and estimation methods

#### Open Dataset (TMB)

1. **tmb_export_open_all_famoye_laplace.R**
   - Model: Famoye dependence
   - Method: Laplace approximation
   - Random: Yes (first 2 numeric predictors)
   - Purpose: Test Laplace method for memory-efficient estimation

2. **tmb_export_open_all_independence.R**
   - Model: Independence (separate marginals)
   - Method: Simulated Maximum Likelihood (SML)
   - Purpose: Baseline model for likelihood ratio tests
   - Notes: No dependence parameter estimated

3. **tmb_export_open_all_clayton.R**
   - Model: Clayton copula dependence
   - Method: Simulated Maximum Likelihood (SML)
   - Purpose: Test lower-tail dependence structure

### Non-TMB Engine Tests

#### Dense Dataset (Non-TMB)

1. **rpbnb_export_dense_all_famoye.R**
   - Model: Famoye/Sarmanov dependence with RP-BNB
   - Optimizer: BFGS
   - Random: Yes (2 numeric predictors)
   - Purpose: Test non-TMB (classic) engine for random-parameter model

2. **rpbnb_export_dense_all_frank.R**
   - Model: Frank copula with RP-BNB
   - Optimizer: BFGS
   - Random: None (fixed-effects)
   - Purpose: Test copula implementation in non-TMB engine

3. **bnb_export_dense_all_famoye.R**
   - Model: Bivariate NB (fixed-effects) with Famoye dependence
   - Optimizer: BFGS
   - Random: None
   - Purpose: Test simpler BNB model (no random coefficients)

#### Open Dataset (Non-TMB)

1. **rpbnb_export_open_all_famoye.R**
   - Model: Famoye dependence with RP-BNB
   - Optimizer: BFGS
   - Random: Yes (2 numeric predictors)
   - Purpose: Test non-TMB engine on open dataset

2. **rpbnb_export_open_all_clayton.R**
   - Model: Clayton copula with RP-BNB
   - Optimizer: BFGS
   - Random: Yes (1 numeric predictor)
   - Purpose: Test Clayton copula in non-TMB engine

3. **bnb_export_open_all_copulas.R**
   - Models: Multiple BNB models (Famoye, Frank, Gaussian, Clayton)
   - Optimizer: BFGS
   - Random: None (fixed-effects)
   - Purpose: Compare copula structures in simple BNB model

### Comprehensive Comparison Tests

#### TMB Comparisons

4. **tmb_export_comparison_suite.R**
   - **Scope**: Comprehensive multi-model TMB comparison
   - **Models tested**:
     - Dense dataset: Famoye (SML), Frank (SML), Gaussian (SML), Independence (SML)
     - Open dataset: Famoye (SML), Frank (SML), Clayton (SML), Independence (SML), Famoye (Laplace)
   - **Output**: Comparison table with logLik, AIC, BIC, convergence diagnostics
   - **Use case**: Model selection, checking robustness across dependence structures

5. **tmb_export_inference_comparison.R**
   - **Scope**: Impact of inference method on computation (TMB)
   - **Inference methods tested**:
     - Full: Complete covariance matrix
     - Diagonal: Standard errors only
     - None: No Hessian computation
   - **Model**: Famoye dependence, SML, open dataset
   - **Output**: Timing comparisons and uncertainty quantification options
   - **Use case**: Optimizing compute time vs. uncertainty reporting

#### Engine Comparison

6. **rpbnb_tmb_vs_nontmb_comparison.R**
   - **Scope**: Direct comparison of TMB vs non-TMB engines
   - **Models tested**:
     - Famoye (RP-BNB): Both engines with same specification
     - Frank copula (RP-BNB): Both engines
   - **Datasets**: Both dense and open
   - **Output**: Timing speedup analysis, logLik agreement verification
   - **Use case**: Engine performance benchmarking, validating TMB improvements

## Model Specifications

### Available Dependence Structures

| Structure | Type | Comment | RP-BNB (TMB) | RP-BNB (Non-TMB) | BNB (Non-TMB) |
|-----------|------|---------|--------------|------------------|---------------|
| `"famoye"` | Explicit | Famoye/Sarmanov parametrization | ✓ | ✓ | ✓ |
| `"independence"` | Explicit | Two separate negative binomials | ✓ | ✗ | ✓ |
| `copula("frank")` | Discrete | Frank copula, symmetric | ✓ | ✓ | ✓ |
| `copula("normal")` | Discrete | Gaussian copula | ✓ | ✓ | ✓ |
| `copula("clayton")` | Discrete | Clayton copula, lower-tail focus | ✓ | ✓ | ✓ |

### Available Estimation Methods

| Method | Engine | Memory | Speed | Random Coef Types | Notes |
|--------|--------|--------|-------|-------------------|-------|
| SML (fit_rpbnb) | Non-TMB | O(n·draws) | Moderate | normal, lognormal, uniform, triangular | Classic BFGS optimizer |
| `"sml"` | TMB | O(n·draws) | Fast | normal, lognormal, uniform, triangular | AD-based gradient, parallel |
| `"laplace"` | TMB | O(n) | Very Fast | normal, lognormal only | Sparse Hessian, memory-efficient |
| BNB (fit_bnb) | Non-TMB | O(n) | Very Fast | None (fixed-effects) | Simpler, no random coefficients |

### Inference Options

| Option | Hessian | Output | Use Case |
|--------|---------|--------|----------|
| `"full"` | Yes (full) | SE + covariance matrix | Uncertainty quantification |
| `"diag"` | Yes (diagonal) | SE only | Quick SE reporting |
| `"none"` | No | None | Model comparison |

## Running the Tests

### Run a single test script:
```R
source("inst/tmb_export_dense_all_famoye.R")
```

### Run the comprehensive comparison:
```R
source("inst/tmb_export_comparison_suite.R")
```

### Run inference method comparison:
```R
source("inst/tmb_export_inference_comparison.R")
```

## Example Output

Each script produces:
- Model fit summary (coefficients, SEs, dispersion parameters)
- Optimization diagnostics (convergence status, max gradient)
- Information criteria (logLik, AIC, BIC)
- Timing information
- Model comparison tables (in multi-model scripts)

## Model Classes

### RP-BNB (Random Parameter Bivariate NB)
- **TMB engine**: `fit_rpbnb_tmb()` - Fast AD-based optimization
- **Non-TMB engine**: `fit_rpbnb()` - Classic BFGS optimizer
- Both support random coefficients and multiple dependence structures
- TMB typically 2-10x faster depending on model and data

### BNB (Bivariate NB, fixed-effects)
- **Non-TMB engine only**: `fit_bnb()` 
- No random coefficients (simpler model)
- Faster and more memory-efficient than RP-BNB
- Useful for baseline comparisons and when random effects not needed

## Notes

1. **Computation Time**: Scripts automatically detect available cores and use `n_cores - 2`. Adjust `n_cores` parameter manually if needed.

2. **Random Seed**: All scripts use `seed = 20260809` for reproducibility. Change the seed to test variability.

3. **Draw Count**: 
   - SML draws default to 200-300 (TMB) or 300 (non-TMB)
   - Laplace draws default to 100-200
   - Increase draws for more stable estimates (trades off computation time)
   - Non-TMB BNB models don't use draws (fixed-effects)

4. **Data Handling**: Scripts extract first N numeric columns as predictors. If your data has categorical variables or requires transformations, modify the formula construction in the scripts.

5. **Memory Considerations**:
   - SML/RP-BNB: O(n·draws) memory
   - Laplace/TMB: O(n) memory
   - BNB: O(n) memory (most efficient)
   - Use `inference = "none"` (TMB) if you only need model comparison
   - For very large datasets, use Laplace approximation or BNB

6. **Engine Comparison**:
   - TMB is faster but requires random coefficients (for Laplace method)
   - Non-TMB is more flexible (supports more random distributions)
   - Both should produce nearly identical estimates for same specification

## Integration with devtools

TMB tests use `devtools::load_all()` to test the development version of rpbnb. Non-TMB tests use `library(rpbnb)`. For production use, change all to:
```R
library(rpbnb)
```

## Troubleshooting

- **"no random coefficients" error under Laplace**: Laplace method requires `random_1` and `random_2` to be specified
- **Convergence warnings**: Try adjusting `draws`, `seed`, or starting values
- **Out of memory**: Reduce `draws` or use `method = "laplace"`
- **Data not found**: Ensure `system.file()` paths are correct or use absolute file paths

## Contact

For questions or issues with these tests, refer to the main rpbnb documentation and vignettes.
