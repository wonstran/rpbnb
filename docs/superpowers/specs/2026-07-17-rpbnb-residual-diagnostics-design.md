# Residual diagnostics for rpbnb — design

**Date:** 2026-07-17
**Status:** Approved (brainstorming)
**Scope:** Residual extraction, diagnostic plots, and formal residual checks for
both fitted model classes, `bnb_fit` (fixed-coefficient bivariate NB2:
Famoye/Sarmanov, discrete copula, independence) and `rpbnb_fit`
(random-parameter bivariate NB2).

## Problem

`rpbnb` fits **only count models** (bivariate NB2 margins). It has goodness-of-fit
(`bnb_gof()`), marginal effects, and elasticities, but no residual-based model
checking. Users need standard GLM residual diagnostics — checks for normality,
constant variance, independence, outliers, and misspecification, via
residual-vs-fitted / QQ / histogram / scale-location plots — adapted correctly
to the count-model setting.

**The count-model subtlety (drives the whole design):** for a correctly specified
count model, *raw* and *Pearson* residuals are skewed and discrete even under the
truth, so a QQ plot / histogram / normality test of them is misleading.
**Randomized quantile residuals** (RQR; Dunn & Smyth 1996) are approximately
N(0,1) under a correct model, which is exactly what makes normality-style checks
meaningful for a GLM. Therefore all normality-style diagnostics target RQR;
Pearson and deviance residuals are still provided and used on the
non-normality panels.

## Residual definitions

Each margin (y1, y2) is NB2 with fitted mean `mu_i` and dispersion `m`
(`size = r = 1/m`), NB2 variance `V(mu) = mu + m*mu^2`.

- **response**: `y_i - mu_i`
- **pearson**: `(y_i - mu_i) / sqrt(V_i)`
- **deviance**: `sign(y_i - mu_i) * sqrt(2 * [ y_i*log(y_i/mu_i)
  - (y_i + r)*log((y_i + r)/(mu_i + r)) ])`, with the `y_i*log(y_i/mu_i)` term
  taken as 0 when `y_i = 0`.
- **quantile (RQR)**: `qnorm(u_i)`, `u_i ~ Uniform(F(y_i - 1), F(y_i))`,
  `F(k) = pnbinom(k, size = r, mu = mu_i)` (with `F(-1) = 0`). Under a correct
  model `u_i ~ Uniform(0,1)` and RQR `~ N(0,1)`. Randomization is made
  reproducible by a `seed` argument.

### Class asymmetry (deliberate, documented)

- **`bnb_fit`**: all four types are exact NB2 — the per-margin distribution is
  `NB2(mu_i, m)` regardless of the dependence structure (Famoye/copula couples
  the margins but each margin is still NB2). `mu_i` is `fit$mu1`/`fit$mu2`;
  `m` from `exp(fit$coef[["log_m1"]])` / `log_m2`.
- **`rpbnb_fit`**: the per-margin distribution is a **mixture** over the
  random-coefficient draws, `NB2(mu_ir, m)` averaged over draws `r`, where
  `mu_ir = pmin(exp(x_i'b + XR_i %*% dev_r), RP_PRED_CAP)` reusing the SAME
  stored draws (`fit$rp_meta$Z1/Z2`) and `rand_realize()` machinery as
  `.rp_integrated_mu()` in `R/methods.R` (so residuals are consistent with
  `predict()`):
  - **quantile (RQR)** uses the exact mixture CDF
    `Fbar(k) = mean_r pnbinom(k, size = r, mu = mu_ir)`. This is the headline,
    correct residual for RP models.
  - **pearson** uses the exact mixture marginal moments from the draws:
    marginal mean `mu_i = mean_r mu_ir` (== `fit$mu1`/`mu2`), marginal variance
    `Vbar_i = mean_r (mu_ir + m*mu_ir^2) + var_r(mu_ir)` (law of total variance:
    mean of the per-draw NB2 variances plus the variance of the per-draw means).
    Pearson residual `= (y_i - mu_i) / sqrt(Vbar_i)`.
  - **response** = `y_i - mu_i`.
  - **deviance** is NOT defined for `rpbnb_fit` (no clean single-observation
    saturated deviance for a mixture); `type = "deviance"` errors with an
    explanatory message pointing to `quantile`/`pearson`.
  - Lognormal analytic-`Inf` rows (a lognormal random coefficient with
    `sign_j * x_ij > 0`, per `.rp_inf_rows()`): `mu_ir -> Inf`, so `Fbar -> `
    the degenerate limit. Mirror `predict()`'s behaviour — such rows produce a
    warning and their residuals are `NA` (not a cap-dependent finite value).

## API

### Residual extraction — `residuals()` S3 methods

Standard `stats::residuals` generic, one method per class:

```r
residuals(object,
          type   = c("quantile", "pearson", "deviance", "response"),
          margin = c("both", "y1", "y2"),
          seed   = NULL, ...)
```

- Returns a numeric vector for a single margin, or a two-column data frame
  (`y1`, `y2`) for `margin = "both"`.
- `type = "quantile"` is the default (the recommended residual for counts).
- `seed` seeds the RQR randomization (ignored for non-quantile types).
- `residuals.rpbnb_fit(type = "deviance")` errors (see class asymmetry).

### Diagnostic plots — `plot()` S3 methods

Base `graphics` (no new heavy dependency), one method per class:

```r
plot(x,
     margin     = c("both", "y1", "y2"),
     which      = 1:4,
     resid_type = "quantile",
     seed       = NULL, ...)
```

Four panels per margin (`par(mfrow = c(2, 2))`, restored on exit):
1. **Residuals vs fitted** — `resid_type` residual vs `mu_hat`, lowess trend
   line. Nonlinearity / link misspecification.
2. **QQ plot** — RQR vs N(0,1) quantiles + reference line. The normality panel;
   ALWAYS uses RQR regardless of `resid_type`.
3. **Histogram** — RQR with an overlaid N(0,1) density. ALWAYS RQR.
4. **Scale-location** — `sqrt(|resid|)` (of `resid_type`) vs `mu_hat`, lowess
   line. Constant-variance / NB2 variance-function adequacy.

`which` selects a subset of panels. `resid_type` affects panels 1 and 4 only
(2 and 3 are normality panels and always use RQR). `margin = "both"` draws y1's
set then y2's. The method restores the caller's `par()` via `on.exit()`.

### Formal checks — `bnb_residual_checks()`

One function for both classes (dispatches internally on class), returning an
invisible structured list with a `print` method (mirrors `bnb_gof()` style):

```r
bnb_residual_checks(fit,
                    seed = NULL, outlier_z = 3, digits = 4,
                    print_output = TRUE)
```

Per margin, one diagnostic per checklist item:

| Check | Diagnostic |
|---|---|
| Normality | Shapiro-Wilk on RQR when `n <= 5000`, else Kolmogorov-Smirnov of RQR vs `pnorm`. Reports statistic + p-value. |
| Constant variance | NB2 dispersion statistic `sum(pearson^2) / (n - k)` (`k` = number of mean coefficients for that margin); ~1 good, >1 over-, <1 under-dispersed. |
| Independence | Cross-margin RQR correlation between y1 and y2 (Pearson and Spearman). After the Famoye lambda / copula captures the association this should be ~0; nonzero flags residual dependence the structure missed. Classical serial autocorrelation is not computed (cross-sectional data, no time index). |
| Outliers | Count and row-indices where `|RQR| > outlier_z` (default 3) per margin; also `max(|pearson|)`. |
| Misspecification | Composite reasoned flag, raised if any of: normality p < 0.05; dispersion statistic outside `[0.5, 2]`; a material residuals-vs-fitted lowess trend; nonzero cross-margin residual correlation. Emits a short message naming which signals fired, not just a boolean. |

Return list carries the per-margin residuals, the test objects, the dispersion
statistics, the cross-margin correlations, the outlier indices, and the
misspecification verdict/messages.

Because RQR is randomized, `seed` fixes the realization; the print output notes
that a different seed gives a slightly different realization and that severe
verdicts should reproduce across seeds.

## Internal helpers

- `.nb_marginal_cdf(y, mu, r)` — `pnbinom` wrapper returning `F(y)` and `F(y-1)`.
- `.rqr(y, Flo, Fhi, seed)` — Dunn-Smyth randomized quantile residual from CDF
  corners; shared by both classes.
- `.rp_mixture_cdf(fit, eq, y)` — per-observation mixture CDF corners
  `Fbar(y)`, `Fbar(y-1)` for `rpbnb_fit`, built from the stored draws exactly as
  `.rp_integrated_mu()` builds the mean (reuses `rand_realize`,
  `RP_PRED_CAP`, `.rp_inf_rows`).
- `.rp_mixture_var(fit, eq)` — per-observation mixture marginal variance
  `Vbar_i` via the law of total variance over the draws.
- `.nb2_deviance_resid(y, mu, r)`, `.nb2_pearson_resid(y, mu, r)` — closed-form
  NB2 residuals for `bnb_fit`.

## Files

- **Create** `R/residuals.R` — `residuals.bnb_fit`, `residuals.rpbnb_fit`, and
  the residual/mixture helpers above.
- **Create** `R/residual_plots.R` — `plot.bnb_fit`, `plot.rpbnb_fit`.
- **Modify** `R/diagnostics.R` — add `bnb_residual_checks()` and its `print`
  method, next to `bnb_gof()`.
- **Modify** `DESCRIPTION` — add `graphics` to `Imports`.
- **Modify** `NAMESPACE`, `man/*.Rd` — regenerated by roxygen2 (`residuals`/`plot`
  S3 methods registered via `@exportS3Method`/`@export`; `bnb_residual_checks`
  exported).
- **Create** `tests/testthat/test-residuals.R` — fast fixture tier + slow tier.
- **Modify** `NEWS.md` — changelog entry.
- **Modify** `vignettes/rpbnb-intro.Rmd` — a short residual-diagnostics note.

## Testing

- **Fast tier** (fixture-based, no fitting; extend/mirror `make_rp_fixture()` and
  build a tiny synthetic `bnb_fit`):
  - Pearson formula matches `(y-mu)/sqrt(mu + m*mu^2)`.
  - Deviance residual sign matches `sign(y-mu)` and is 0 (to tolerance) when
    `y = mu`; the `y = 0` branch uses the finite limit.
  - RQR lies in `(qnorm(F(y-1)), qnorm(F(y)))` and is reproducible for a fixed
    `seed`; differs across seeds.
  - `rpbnb_fit` mixture CDF corners and mixture variance match an independent
    brute-force recompute from the fixture draws; `residuals(type="quantile")`
    on a fully-fixed equation reduces to the plain NB2 RQR.
  - `residuals.rpbnb_fit(type = "deviance")` errors.
  - `bnb_residual_checks()` returns finite statistics; planted outliers appear
    at the expected indices; the dispersion statistic is finite and positive.
  - `plot()` runs to `pdf(NULL)` / null device without error for both classes
    and restores `par()`.
- **Slow tier** (`RPBNB_RUN_SLOW=1`): on a real fit from a well-specified DGP the
  RQR normality test does not reject at 0.01; on a deliberately misspecified fit
  (e.g. omit a strong covariate) the misspecification flag fires.

## Out of scope

- `ggplot2` output (base graphics only, to avoid a heavy dependency).
- Influence measures (Cook's distance / leverage) — not cleanly defined for the
  MSL random-parameter fits; deferred.
- Serial-autocorrelation independence tests — undefined without a time/order
  index the models do not carry.
- Deviance residuals for `rpbnb_fit` — not cleanly defined for a mixture.
