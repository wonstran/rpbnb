# RP-BNB Test Suite: export_dense_all.csv and export_open_all.csv

This directory contains a comprehensive test suite for fitting random-parameter bivariate negative binomial (RP-BNB) models to two large real-world datasets using the TMB engine.

## Data Files

- **export_dense_all.csv**: Dense dataset with ~10K observations
- **export_open_all.csv**: Open dataset with ~6.5K observations

Both files are located in `inst/extdata/` and contain count outcomes (y1, y2) and multiple numeric predictors.

## Test Scripts

### Individual Model Tests (Single Dataset, Single Model)

#### Dense Dataset

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

#### Open Dataset

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

### Comprehensive Comparison Tests

4. **tmb_export_comparison_suite.R**
   - **Scope**: Comprehensive multi-model comparison
   - **Models tested**:
     - Dense dataset: Famoye (SML), Frank (SML), Gaussian (SML), Independence (SML)
     - Open dataset: Famoye (SML), Frank (SML), Clayton (SML), Independence (SML), Famoye (Laplace)
   - **Output**: Comparison table with logLik, AIC, BIC, convergence diagnostics
   - **Use case**: Model selection, checking robustness across dependence structures

5. **tmb_export_inference_comparison.R**
   - **Scope**: Impact of inference method on computation
   - **Inference methods tested**:
     - Full: Complete covariance matrix
     - Diagonal: Standard errors only
     - None: No Hessian computation
   - **Model**: Famoye dependence, SML, open dataset
   - **Output**: Timing comparisons and uncertainty quantification options
   - **Use case**: Optimizing compute time vs. uncertainty reporting

## Model Specifications

### Available Dependence Structures

| Structure | Type | Comment |
|-----------|------|---------|
| `"famoye"` | Explicit | Famoye/Sarmanov parametrization |
| `"independence"` | Explicit | Two separate negative binomials |
| `copula("frank")` | Discrete | Frank copula, symmetric |
| `copula("normal")` | Discrete | Gaussian copula |
| `copula("clayton")` | Discrete | Clayton copula, lower-tail focus |

### Available Estimation Methods

| Method | Memory | Speed | Random Coef Types |
|--------|--------|-------|-------------------|
| `"sml"` | O(n·draws) | Moderate | normal, lognormal, uniform, triangular |
| `"laplace"` | O(n) | Fast | normal, lognormal only |

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

## Notes

1. **Computation Time**: Scripts automatically detect available cores and use `n_cores - 2`. Adjust `n_cores` parameter manually if needed.

2. **Random Seed**: All scripts use `seed = 20260809` for reproducibility. Change the seed to test variability.

3. **Draw Count**: SML draws default to 200-300; Laplace draws default to 100-200. Increase draws for more stable estimates (trades off computation time).

4. **Data Handling**: Scripts extract first N numeric columns as predictors. If your data has categorical variables or requires transformations, modify the formula construction in the scripts.

5. **Memory Considerations**:
   - SML with 500 draws on 10K observations can use several GiB
   - Laplace method is memory-efficient even with large samples
   - Use `inference = "none"` if you only need model comparison

## Integration with devtools

These scripts use `devtools::load_all()` to test the development version of rpbnb. For production use, change to:
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
