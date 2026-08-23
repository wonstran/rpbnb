# JAX Engine (SML, all families) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a JAX re-implementation of the rpbnb TMB objective for all five dependence families under simulated maximum likelihood, reachable from R, and prove it agrees with the TMB engine on value and gradient.

**Architecture:** A Python package under `inst/python/rpbnb_jax/` holds the likelihood as pure JAX. R reaches it through reticulate: `.make_rpbnb_jax_object()` returns an object exposing `$par`, `$fn`, `$gr`, and `$env$last.par.best` — the same five-item surface `fit_rpbnb_tmb()` already consumes from TMB — so the existing `nlminb` wiring, restart loop, and `.rpbnb_inference()` work unchanged. The `map=` mechanism becomes an index-scatter into a template vector, so `jax.grad` differentiates only the free coordinates.

**Scope:** `est_method = 0` (SML) only. The Laplace path, `u1`/`u2` latents, `TMB::tmbprofile`, and `parallel_accumulator` are explicitly out of scope. `fit_rpbnb_tmb()` is not modified by this plan.

**Tech Stack:** JAX 0.11.1 (CPU, `cp314` wheel), Python 3.14, reticulate, R 4.5.2, testthat.

---

## Non-Negotiable Rules

These three rules cause silent wrong answers if broken. Every task depends on them.

**Rule 1 — float64 everywhere.** JAX defaults to float32. The entire numerical design of `src/rpbnb_tmb.cpp` rests on double precision: 1e-300 floors, the 1.1e-16 spacing at 1.0 that `gauss_corner_quantiles()` exists to work around, and cell probabilities down to 1e-136. In float32 all of it collapses. `jax.config.update("jax_enable_x64", True)` must run **before any other jax import side effect**, and it lives at the top of `inst/python/rpbnb_jax/__init__.py`.

**Rule 2 — the double-`where` discipline.** `jax.grad` propagates `NaN` backward through the *un-taken* branch of `jnp.where`. `jnp.where(x > 0, jnp.log(x), 0.0)` returns a correct value and a `NaN` gradient at `x <= 0`. The fix is to sanitize the input first, then select:

```python
x_safe = jnp.where(x > 0, x, 1.0)
out = jnp.where(x > 0, jnp.log(x_safe), 0.0)
```

This is the exact analogue of the repeated note in the C++ that "both CondExp branches are evaluated, so the exact branch gets a floored argument even where it is not the one selected" (`src/rpbnb_tmb.cpp:318`, `:495`, `:542`). Every `CondExp` in the C++ that floors its argument must become a double-`where` here. There are roughly fifteen.

**Rule 3 — the C++ is the specification, comments included.** Each kernel header in `src/rpbnb_tmb.cpp` records a measured failure that the current form repairs. Do not "simplify" a kernel back to its textbook form. In particular: never take a second difference of four corner CDFs, never form `1 - F`, never seed a count recursion in linear space, and never clip a Frank cell probability at 1e-300.

---

## File Structure

| Path | Responsibility |
|---|---|
| `inst/python/rpbnb_jax/__init__.py` | x64 config (env var, `config.update`, and a hard import-time post-condition); cross-language constants |
| `inst/python/tests/conftest.py` | puts `inst/python` on `sys.path` so pytest runs from the repo root |
| `.Rbuildignore` | exclude `.venv-jax/` and `__pycache__` from the built tarball |
| `inst/python/rpbnb_jax/margins.py` | NB2 / Poisson log-pmf, and the CDF-pair triple (`log_cdf_y`, `log_cdf_ym1`, `log_pmf_y`) |
| `inst/python/rpbnb_jax/frank.py` | `frank_log_cell_prob` |
| `inst/python/rpbnb_jax/clayton.py` | `clayton_cell_prob` |
| `inst/python/rpbnb_jax/gaussian.py` | `gauss_corner_quantiles`, `gaussian_cell_prob` |
| `inst/python/rpbnb_jax/packing.py` | flat free-vector ↔ parameter dict, the `map=` analogue |
| `inst/python/rpbnb_jax/objective.py` | `build_objective(data, kmax, obs_chunk)` → jitted `(value, grad)` |
| `inst/python/tests/` | pytest unit tests for the kernels |
| `R/jax_helpers.R` | `.rpbnb_jax_available()`, `.make_rpbnb_jax_object()` |
| `tests/testthat/test-jax-parity.R` | value + gradient parity against TMB |
| `tools/jax-setup.R` | one-shot venv + dependency installer |
| `.gitignore` | add `.venv-jax/` |

`inst/python/` is chosen so the Python sources ship with the installed package and resolve through `system.file("python", package = "rpbnb")`.

---

## Known Design Decisions

**The count recursion is vectorized, not sequential.** TMB loops `k = 1..y` per observation (`src/rpbnb_tmb.cpp:170`). JAX needs static shapes, so the NB2 log-mass over a static grid `k = 0..KMAX` is written as a closed-form cumulative sum:

```
log P(Y = k) = r*log_p + k*log_q + sum_{j=1}^{k} [log(r + j - 1) - log(j)]
```

The bracketed sum depends only on `r`, which is a scalar shared by every observation, so `cumsum` runs once per margin per evaluation rather than once per observation. `KMAX = max(Y)` is a **static Python int** fixed at object-construction time.

**Memory, and why `obs_chunk` exists.** The copula path materializes an array of shape `(n_chunk, R, KMAX+1)`. On the truck workload (`n = 3487`, `R = 300`, `KMAX = 266`) the unchunked array is 2.2 GB. Observations are processed in blocks via `jax.lax.map`; `obs_chunk = 256` gives 164 MB. Famoye and independence use the closed-form log-pmf and never build this array, so they are unaffected.

**Cost model shift.** TMB pays O(y) per observation (~12 terms on truck data); this pays O(KMAX) for every observation (~266). That is ~20x more arithmetic, fully vectorized, against TMB's sequential tape. Whether that trades well is one of the questions the benchmark in Task 7 answers. If it does not, the mitigation is bucketing observations by count magnitude — **not** in this plan.

**No tape, so no atomic.** `REGISTER_ATOMIC(gauss_cell_vec)` (`src/rpbnb_tmb.cpp:781`), `parallel_accumulator`, `TMB::config(tape.parallel=)`, and the `TAPE_CALIBRATION` workload guard all exist to manage TMB tape size. XLA fuses the quadrature with no tape at all, so none of them port. The Gaussian single-thread SIGSEGV cap in `.resolve_gaussian_threads()` (`R/tmb_helpers.R:32`) likewise has no analogue.

**The real NaN source is an input-dependent `-inf`, not the mask.** Measured in Task 3. A masked-out *literal* `-inf` differentiates cleanly; a masked-out `-inf` that carries a dependence on the differentiated input gives `NaN` regardless of whether it is masked by `where=`, `jnp.where`, or a select to `-1e30`. So the remedy is always to floor the *input* (Rule 2), never to sanitise the reduction. `mu` and `r` are floored at `1e-300` before any `log()`: `mu = 0` and, worse, a subnormal `mu` that XLA flushes to zero both give `log(mu) = -inf` and then `0 * -inf = NaN` in the **value**, not merely the gradient.

Also measured: `jax.scipy.special.logsumexp` in JAX 0.11.1 has **no `initial=` parameter** (signature `(a, axis, b, keepdims, return_sign, where)`), and out-of-bounds `jnp.take_along_axis` returns `NaN` rather than clamping — so an unguarded count above `KMAX` poisons every parameter's gradient through the sum-over-observations reduction rather than quietly returning the wrong mass.

**Accuracy note for later parity work.** At `mu = 1e-8, r = 200` the `log_q = log(mu) - log(r + mu)` form beats `scipy.stats.nbinom.logpmf` — relative error 4e-18 against a `longdouble` reference, versus 4.5e-9 for scipy, which forms `1 - p`. Same failure class as the TMB `eta`-floor divergence above. A parity test against scipy in that regime must not tighten `rtol` past ~1e-8.

**JAX builtins replace the stability shims.** `stable_expm1`/`stable_log1p` (`src/rpbnb_tmb.cpp:81-97`) exist only because CppAD lacks `expm1`/`log1p`. JAX has both natively with correct derivatives, so they become `jnp.expm1`/`jnp.log1p`. Likewise `log_add_exp` → `jnp.logaddexp`, `qnorm` → `jax.scipy.special.ndtri`, `pnorm` → `jax.scipy.stats.norm.cdf`, `pgamma(mu, shape)` → `jax.scipy.special.gammainc(shape, mu)`.

**Known divergence: the NB2 margin at the `eta` floor. The JAX port is the accurate one; do not "fix" it to match.**

Measured on 2026-08-23. `nb2_eta_floor()` (`src/rpbnb_tmb.cpp:1024-1029`) places `log(m) + eta` at exactly `-35.0`. The template then calls `dnbinom2(y, mu, mu + m*mu^2)`, and TMB's `dnbinom2` (`TMB/include/lgamma.hpp:130-136`) forms `log_var_minus_mu = log(var - mu)` — recovering `m*mu^2` by float subtraction from a sum that has already rounded it — and hands `dnbinom_robust` a size of `mu^2/(var - mu)`.

At the floor that subtraction retains **1.5 significant bits**. The recovered size is 1.866 against a true `1/m` of 1.667 (12% relative error), worth about 0.10 nats per observation, in the gradient as well as the value. Measured error in the recovered size against `log(m) + eta`:

| `log(m) + eta` | bits left in `var - mu` | size rel. error |
|---|---|---|
| -35.0 (the floor) | 1.5 | 1.2e-01 |
| -32 | 5.8 | 1.3e-04 |
| -29 | 10.2 | 6.2e-04 |
| -25 | 15.9 | 4.2e-06 |
| -22 | 20.3 | 8.8e-08 |

The C++ comment identifies the right threshold (`-36.04 = log(2^-52)`, where the increment vanishes entirely) but sets the floor only one nat above it. `margins.py` computes `r = 1/m` directly and never forms the difference, so it is exact — which is why a saturating-floor parity case fails by ~2.5 nats. That is TMB's error, not the port's.

Consequences for this plan: parity fixtures must stay out of the saturated-floor regime (the `eta` **ceiling** agrees to 4.5e-12 and is fine); and Task 7's end-to-end fit comparison must treat a floor-saturating fit as expected-to-diverge rather than as a port defect.

The fix on the TMB side is small and removes the need for this floor entirely: call `dnbinom_robust(y, log_mu, log(m) + 2*log_mu)` directly, giving `size = exp(-log m) = 1/m` exactly with no subtraction anywhere. Verified accurate to 5e-15 at every `eta` tested. Filed separately; out of scope here.

**Zero-count safety comes free.** `nb2_cdf_pair` leaves `log_cdf_ym1` at `log P(Y = 0)` for `y = 0` rather than `-inf` (`src/rpbnb_tmb.cpp:157`). Under `vmap` all Clayton branches evaluate for every observation, and that convention is exactly what keeps the unused branch finite. Do not "fix" it to `-inf`.

---

## Task 1: Environment and package skeleton

**Files:**
- Create: `tools/jax-setup.R`
- Create: `inst/python/rpbnb_jax/__init__.py`
- Create: `inst/python/tests/test_smoke.py`
- Modify: `.gitignore`

- [ ] **Step 1: Add the venv to `.gitignore`**

Append to `.gitignore`, under the `# renv` block:

```
# JAX experiment virtualenv
.venv-jax/
```

- [ ] **Step 2: Write the setup script**

Create `tools/jax-setup.R`:

```r
# One-shot setup for the JAX engine experiment (branch: jax-engine).
# Creates a project-local virtualenv so nothing is installed into the
# user's global Python. Run once:  Rscript tools/jax-setup.R
if (!requireNamespace("reticulate", quietly = TRUE)) {
  install.packages("reticulate", repos = "https://cloud.r-project.org")
}
venv <- normalizePath(file.path(getwd(), ".venv-jax"), mustWork = FALSE)
if (!dir.exists(venv)) {
  reticulate::virtualenv_create(venv)
}
reticulate::virtualenv_install(venv, packages = c("jax", "numpy", "scipy", "pytest"))
reticulate::use_virtualenv(venv, required = TRUE)
jax <- reticulate::import("jax")
cat("jax", jax$`__version__`, "devices:",
    paste(vapply(jax$devices(), function(d) d$device_kind, character(1)),
          collapse = ", "), "\n")
```

- [ ] **Step 3: Run the setup script**

Run: `Rscript tools/jax-setup.R`
Expected: final line reads `jax 0.11.1 devices: cpu` (patch version may differ).

- [ ] **Step 4: Write the package init with the x64 switch**

Create `inst/python/rpbnb_jax/__init__.py`:

```python
"""JAX re-implementation of the rpbnb SML objective.

Mirrors src/rpbnb_tmb.cpp with est_method = 0. See that file's kernel
headers for why each formula takes the shape it does; every one of them
records a measured failure of the textbook alternative.
"""

import jax

# MUST run before any array is created. The whole numerical design assumes
# double precision -- 1e-300 floors, the 1.1e-16 spacing at 1.0, cell
# probabilities down to 1e-136. float32 destroys all of it.
jax.config.update("jax_enable_x64", True)

FAM_INDEP = -1
FAM_FAMOYE = 0
FAM_FRANK = 1
FAM_GAUSSIAN = 2
FAM_CLAYTON = 3

DIST_NORMAL = 0
DIST_LOGNORMAL = 1
DIST_UNIFORM = 2
DIST_TRIANGULAR = 3

# Must match FRANK_THETA_MAX in src/rpbnb_tmb.cpp:37 and R/tmb_utilities.R:10.
FRANK_THETA_MAX = 35.0

# log(1e15); the shared ceiling on every linear predictor.
ETA_CEILING = 34.538776394910684

__all__ = [
    "FAM_INDEP", "FAM_FAMOYE", "FAM_FRANK", "FAM_GAUSSIAN", "FAM_CLAYTON",
    "DIST_NORMAL", "DIST_LOGNORMAL", "DIST_UNIFORM", "DIST_TRIANGULAR",
    "FRANK_THETA_MAX", "ETA_CEILING",
]
```

- [ ] **Step 5: Write the smoke test**

Create `inst/python/tests/test_smoke.py`:

```python
import jax.numpy as jnp
import rpbnb_jax  # noqa: F401  (imported for its x64 side effect)


def test_x64_is_enabled():
    assert jnp.zeros(1).dtype == jnp.float64


def test_grad_is_float64():
    import jax
    g = jax.grad(lambda x: jnp.sum(x ** 2))(jnp.ones(3))
    assert g.dtype == jnp.float64
```

- [ ] **Step 6: Run the smoke test**

Run: `.venv-jax/Scripts/python.exe -m pytest inst/python/tests/test_smoke.py -v`
(Set `PYTHONPATH=inst/python` first, or run from `inst/python`.)
Expected: 2 passed.

- [ ] **Step 7: Commit**

```bash
git add .gitignore tools/jax-setup.R inst/python/rpbnb_jax/__init__.py inst/python/tests/test_smoke.py
git commit -m "feat(jax): project-local venv, package skeleton, x64 enforcement"
```

---

## Task 2: Independence and Famoye objective

The closed-form-margin families. No count recursion, so this task establishes the whole object-construction and packing machinery against the simplest likelihood.

**Files:**
- Create: `inst/python/rpbnb_jax/margins.py`
- Create: `inst/python/rpbnb_jax/packing.py`
- Create: `inst/python/rpbnb_jax/objective.py`
- Create: `inst/python/tests/test_margins.py`

- [ ] **Step 1: Write the failing margin test**

Create `inst/python/tests/test_margins.py`:

```python
import jax.numpy as jnp
import numpy as np
from scipy.stats import nbinom, poisson

from rpbnb_jax.margins import log_dnbinom2, log_dpois


def test_log_dnbinom2_matches_scipy():
    # TMB's dnbinom2(y, mu, mu + m*mu^2) has size r = 1/m, prob = r/(r+mu).
    mu, m = 3.7, 0.45
    r = 1.0 / m
    y = np.arange(0, 40)
    want = nbinom.logpmf(y, n=r, p=r / (r + mu))
    got = np.asarray(log_dnbinom2(jnp.asarray(y, dtype=jnp.float64), mu, m))
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=0)


def test_log_dpois_matches_scipy():
    mu = 2.25
    y = np.arange(0, 40)
    want = poisson.logpmf(y, mu)
    got = np.asarray(log_dpois(jnp.asarray(y, dtype=jnp.float64), mu))
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=0)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv-jax/Scripts/python.exe -m pytest inst/python/tests/test_margins.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rpbnb_jax.margins'`

- [ ] **Step 3: Implement the closed-form margins**

Create `inst/python/rpbnb_jax/margins.py`:

```python
"""Marginal count distributions.

Closed-form log masses for the Famoye/independence path, and the log-space
CDF triple the copula families need. Mirrors src/rpbnb_tmb.cpp.
"""

import jax.numpy as jnp
from jax.scipy.special import gammaln


def log_dnbinom2(y, mu, m):
    """log dnbinom2(y, mu, mu + m*mu^2), i.e. NB2 with size r = 1/m."""
    r = 1.0 / m
    log_p = jnp.log(r) - jnp.log(r + mu)   # log(r / (r + mu))
    log_q = jnp.log(mu) - jnp.log(r + mu)  # log(mu / (r + mu))
    return (gammaln(y + r) - gammaln(r) - gammaln(y + 1.0)
            + r * log_p + y * log_q)


def log_dpois(y, mu):
    return y * jnp.log(mu) - mu - gammaln(y + 1.0)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv-jax/Scripts/python.exe -m pytest inst/python/tests/test_margins.py -v`
Expected: 2 passed.

- [ ] **Step 5: Implement parameter packing**

Create `inst/python/rpbnb_jax/packing.py`:

```python
"""Flat free-parameter vector <-> parameter dict.

This is the analogue of TMB's map= argument. R pins parameters by name
(R/fit_rpbnb_tmb.R:499-523); here the pinned values live in a fixed
template and the free coordinates are scattered into it, so jax.grad
differentiates only with respect to the free vector.

Layout matches par_names in R/fit_rpbnb_tmb.R:348-352 and the template's
declaration order in src/rpbnb_tmb.cpp:977-983, which coincide:
    beta1 (k1), beta2 (k2), log_sd1 (q1), log_sd2 (q2),
    log_m1, log_m2, z_dep
z_dep is always present in the template; when family < 0 R pins it, so it
simply never appears in the free vector.
"""

import jax.numpy as jnp


class Layout:
    def __init__(self, k1, k2, q1, q2, template, free_idx):
        self.k1, self.k2, self.q1, self.q2 = k1, k2, q1, q2
        self.template = jnp.asarray(template, dtype=jnp.float64)
        self.free_idx = jnp.asarray(free_idx, dtype=jnp.int32)
        self.n_total = int(self.template.shape[0])
        self.n_free = int(self.free_idx.shape[0])
        assert self.n_total == k1 + k2 + q1 + q2 + 3

    def unpack(self, free_vec):
        full = self.template.at[self.free_idx].set(free_vec)
        k1, k2, q1, q2 = self.k1, self.k2, self.q1, self.q2
        a = k1
        b = a + k2
        c = b + q1
        d = c + q2
        return {
            "beta1": full[0:a],
            "beta2": full[a:b],
            "log_sd1": full[b:c],
            "log_sd2": full[c:d],
            "log_m1": full[d],
            "log_m2": full[d + 1],
            "z_dep": full[d + 2],
        }
```

- [ ] **Step 6: Implement the independence/Famoye objective**

Create `inst/python/rpbnb_jax/objective.py`:

```python
"""The SML negative log-likelihood, mirroring
objective_function::operator() in src/rpbnb_tmb.cpp:951 with est_method = 0.
"""

import functools

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp, ndtri

from . import (ETA_CEILING, FAM_CLAYTON, FAM_FAMOYE, FAM_FRANK, FAM_GAUSSIAN,
               FAM_INDEP, FRANK_THETA_MAX, DIST_LOGNORMAL, DIST_NORMAL,
               DIST_TRIANGULAR, DIST_UNIFORM)
from .margins import log_dnbinom2, log_dpois
from .packing import Layout

FAMOYE_D = 1.0 - jnp.exp(-1.0)


def _u_to_base(u, dist_code):
    """Inverse CDF of the mixing distribution. dist_code is a static int."""
    if dist_code in (DIST_NORMAL, DIST_LOGNORMAL):
        return ndtri(u)
    if dist_code == DIST_TRIANGULAR:
        return jnp.where(u < 0.5, -1.0 + jnp.sqrt(2.0 * u),
                         1.0 - jnp.sqrt(2.0 * (1.0 - u)))
    return u  # DIST_UNIFORM


def _compute_dev(b, s, base, dist_code, sign_code):
    """Deviation added to the linear predictor. dist_code/sign_code static."""
    if dist_code == DIST_NORMAL:
        return s * base
    if dist_code == DIST_LOGNORMAL:
        return sign_code * jnp.exp(b + s * base) - b
    if dist_code == DIST_UNIFORM:
        return s * (2.0 * base - 1.0)
    return s * base  # DIST_TRIANGULAR


def _deviations(beta, log_sd, rand_idx, Z, dist, sign):
    """(R, q) matrix of per-draw deviations. Mirrors src/rpbnb_tmb.cpp:1114."""
    q = len(rand_idx)
    if q == 0:
        return jnp.zeros((Z.shape[0], 0))
    sd = jnp.exp(jnp.clip(log_sd, -20.0, 20.0))
    cols = []
    for j in range(q):  # q is small and dist/sign are static, so unroll
        base = _u_to_base(Z[:, j], int(dist[j]))
        cols.append(_compute_dev(beta[int(rand_idx[j])], sd[j], base,
                                 int(dist[j]), int(sign[j])))
    return jnp.stack(cols, axis=1)


def _eta(X, beta, rand_idx, dev):
    """(n, R) linear predictors. Mirrors src/rpbnb_tmb.cpp:1159-1183."""
    xb = X @ beta                       # (n,)
    eta = xb[:, None] + jnp.zeros((1, dev.shape[0]))
    for j in range(len(rand_idx)):
        col = int(rand_idx[j])
        eta = eta + X[:, col][:, None] * dev[None, :, j][0][None, :]
    return eta


def _eta_floor(log_m_clamped, is_pois):
    """src/rpbnb_tmb.cpp:1024-1031."""
    if is_pois:
        return -35.0
    return -35.0 - jnp.minimum(log_m_clamped, 0.0)


def _log_margin(y, mu, m, is_pois):
    if is_pois:
        return log_dpois(y, mu)
    return log_dnbinom2(y, mu, m)


def _famoye_c(mu, m, is_pois):
    """src/rpbnb_tmb.cpp:1198-1203."""
    if is_pois:
        return jnp.exp(-FAMOYE_D * mu)
    return jnp.exp(-jnp.log1p(FAMOYE_D * m * mu) / m)


def build_objective(data, layout, obs_chunk=256):
    """Return a jitted f(free_vec) -> (nll, grad)."""
    family = int(data["family"])
    pois1 = bool(int(data["pois1"]))
    pois2 = bool(int(data["pois2"]))
    Y1 = jnp.asarray(data["Y1"], dtype=jnp.float64)
    Y2 = jnp.asarray(data["Y2"], dtype=jnp.float64)
    X1 = jnp.asarray(data["X1"], dtype=jnp.float64)
    X2 = jnp.asarray(data["X2"], dtype=jnp.float64)
    Z1 = jnp.asarray(data["Z1"], dtype=jnp.float64)
    Z2 = jnp.asarray(data["Z2"], dtype=jnp.float64)
    rand_idx1 = [int(v) for v in data["rand_idx1"]]
    rand_idx2 = [int(v) for v in data["rand_idx2"]]
    dist1 = [int(v) for v in data["dist1"]]
    dist2 = [int(v) for v in data["dist2"]]
    sign1 = [int(v) for v in data["sign1"]]
    sign2 = [int(v) for v in data["sign2"]]
    lamLo = float(data["lamLo"])
    lamHi = float(data["lamHi"])
    n_draws = int(Z1.shape[0]) if (rand_idx1 or rand_idx2) else 1

    def nll(free_vec):
        p = layout.unpack(free_vec)
        log_m1 = jnp.clip(p["log_m1"], -20.0, 20.0)
        log_m2 = jnp.clip(p["log_m2"], -20.0, 20.0)
        m1 = jnp.exp(log_m1)
        m2 = jnp.exp(log_m2)

        dev1 = _deviations(p["beta1"], p["log_sd1"], rand_idx1, Z1,
                           dist1, sign1)
        dev2 = _deviations(p["beta2"], p["log_sd2"], rand_idx2, Z2,
                           dist2, sign2)
        eta1 = _eta(X1, p["beta1"], rand_idx1, dev1)
        eta2 = _eta(X2, p["beta2"], rand_idx2, dev2)
        mu1 = jnp.exp(jnp.clip(eta1, _eta_floor(log_m1, pois1), ETA_CEILING))
        mu2 = jnp.exp(jnp.clip(eta2, _eta_floor(log_m2, pois2), ETA_CEILING))

        y1 = Y1[:, None]
        y2 = Y2[:, None]
        lm1 = _log_margin(y1, mu1, m1, pois1)
        lm2 = _log_margin(y2, mu2, m2, pois2)

        if family == FAM_INDEP:
            log_draw = lm1 + lm2
        elif family == FAM_FAMOYE:
            sig = jax.nn.sigmoid(p["z_dep"])
            eps = 1e-6
            lam = lamLo + (lamHi - lamLo) * (eps + (1.0 - 2.0 * eps) * sig)
            c1 = _famoye_c(mu1, m1, pois1)
            c2 = _famoye_c(mu2, m2, pois2)
            dep = 1.0 + lam * (jnp.exp(-y1) - c1) * (jnp.exp(-y2) - c2)
            bad = dep <= 0.0
            # Double-where: log() must never see the non-positive value, even
            # though that branch is discarded. src/rpbnb_tmb.cpp:1217-1224.
            safe_dep = jnp.where(bad, 1e-300, dep)
            log_draw = (lm1 + lm2 + jnp.log(safe_dep)
                        - jnp.where(bad, 1e10, 0.0))
        else:
            raise NotImplementedError(f"family {family} lands in a later task")

        # log-sum-exp over draws minus log R. src/rpbnb_tmb.cpp:1317-1327.
        return -jnp.sum(logsumexp(log_draw, axis=1) - jnp.log(float(n_draws)))

    return jax.jit(jax.value_and_grad(nll))
```

- [ ] **Step 7: Write the R bridge**

Create `R/jax_helpers.R`:

```r
#' Is the JAX engine importable?
#' @keywords internal
#' @noRd
.rpbnb_jax_available <- function() {
  if (!requireNamespace("reticulate", quietly = TRUE)) return(FALSE)
  venv <- file.path(getwd(), ".venv-jax")
  if (dir.exists(venv)) {
    try(reticulate::use_virtualenv(venv, required = FALSE), silent = TRUE)
  }
  isTRUE(reticulate::py_module_available("jax"))
}

#' Construct a TMB-shaped objective backed by JAX
#'
#' Returns a list exposing `par`, `fn`, `gr` and `env$last.par.best` -- the
#' surface `fit_rpbnb_tmb()` consumes from `TMB::MakeADFun()`.
#' @keywords internal
#' @noRd
.make_rpbnb_jax_object <- function(data, start, free, obs_chunk = 256L) {
  py_dir <- system.file("python", package = "rpbnb")
  if (!nzchar(py_dir)) py_dir <- file.path(getwd(), "inst", "python")
  reticulate::py_run_string(sprintf(
    "import sys; p = r'%s'\nif p not in sys.path: sys.path.insert(0, p)",
    py_dir))
  rj <- reticulate::import("rpbnb_jax", delay_load = FALSE)

  k1 <- ncol(data$X1); k2 <- ncol(data$X2)
  q1 <- length(data$rand_idx1); q2 <- length(data$rand_idx2)
  layout <- rj$packing$Layout(
    k1 = as.integer(k1), k2 = as.integer(k2),
    q1 = as.integer(q1), q2 = as.integer(q2),
    template = as.numeric(start),
    free_idx = as.integer(which(free) - 1L)
  )
  fg <- rj$objective$build_objective(data, layout,
                                     obs_chunk = as.integer(obs_chunk))

  env <- new.env(parent = emptyenv())
  env$last.par.best <- NULL
  env$best_value <- Inf
  cache <- new.env(parent = emptyenv())
  cache$par <- NULL

  evaluate <- function(par) {
    if (!is.null(cache$par) && identical(cache$par, par)) return(cache$out)
    res <- fg(as.numeric(par))
    out <- list(value = as.numeric(res[[1]]), grad = as.numeric(res[[2]]))
    cache$par <- par
    cache$out <- out
    if (is.finite(out$value) && out$value < env$best_value) {
      env$best_value <- out$value
      env$last.par.best <- par
    }
    out
  }

  list(
    par = start[free],
    fn = function(par = NULL) evaluate(par)$value,
    gr = function(par = NULL) evaluate(par)$grad,
    env = env
  )
}
```

- [ ] **Step 8: Write the parity test for independence and Famoye**

Create `tests/testthat/test-jax-parity.R`:

```r
# Value and gradient parity between the TMB engine and the JAX engine.
# Skipped unless the project-local .venv-jax is present (tools/jax-setup.R).

jax_fixture <- function(family_code, n = 60L, draws = 16L, seed = 11L) {
  set.seed(seed)
  x <- rnorm(n)
  X1 <- cbind(`(Intercept)` = 1, x = x)
  X2 <- cbind(`(Intercept)` = 1, x = x)
  Y1 <- rpois(n, exp(0.4 + 0.2 * x))
  Y2 <- rpois(n, exp(0.3 - 0.1 * x))
  Z <- .tmb_halton_uniform(draws, 2L, burn = 30L)
  list(
    data = .build_tmb_data(
      Y1 = as.numeric(Y1), Y2 = as.numeric(Y2), X1 = X1, X2 = X2,
      rand_idx1 = 2L, rand_idx2 = 2L,
      Z1 = Z[, 1, drop = FALSE], Z2 = Z[, 2, drop = FALSE],
      dist1 = 0L, dist2 = 0L, sign1 = 1L, sign2 = 1L,
      family_code = family_code, pois1 = FALSE, pois2 = FALSE,
      lamLo = -0.9, lamHi = 0.9, est_method = 0L
    ),
    k1 = 2L, k2 = 2L, q1 = 1L, q2 = 1L
  )
}

# Template order: b1(k1), b2(k2), log_sd1(q1), log_sd2(q2), log_m1, log_m2, z_dep
jax_start <- function(z_dep = 0.3) {
  c(0.4, 0.2, 0.3, -0.1, log(0.25), log(0.30), log(0.5), log(0.6), z_dep)
}

expect_jax_parity <- function(family_code, pars, tol = 1e-8) {
  fx <- jax_fixture(family_code)
  start <- jax_start()
  free <- rep(TRUE, length(start))
  if (family_code < 0L) free[length(start)] <- FALSE  # z_dep pinned

  tmb <- .make_rpbnb_tmb_object(
    data = fx$data,
    parameters = list(
      beta1 = start[1:2], beta2 = start[3:4],
      log_sd1 = start[5], log_sd2 = start[6],
      log_m1 = start[7], log_m2 = start[8],
      z_dep = start[9],
      u1 = matrix(0, length(fx$data$Y1), 1L),
      u2 = matrix(0, length(fx$data$Y1), 1L)
    ),
    map = c(
      if (family_code < 0L) list(z_dep = factor(NA)),
      list(u1 = factor(rep(NA_integer_, length(fx$data$Y1))),
           u2 = factor(rep(NA_integer_, length(fx$data$Y1))))
    ),
    random = NULL, silent = TRUE, n_cores = 1L, max_threads = 1L
  )$obj
  jx <- .make_rpbnb_jax_object(fx$data, start, free)

  for (p in pars) {
    pf <- p[free]
    expect_equal(jx$fn(pf), tmb$fn(pf), tolerance = tol)
    expect_equal(jx$gr(pf), as.numeric(tmb$gr(pf)), tolerance = tol)
  }
}

test_that("JAX matches TMB for independence", {
  skip_if_not(.rpbnb_jax_available(), "jax not installed")
  expect_jax_parity(-1L, list(jax_start(), jax_start() + 0.15))
})

test_that("JAX matches TMB for Famoye", {
  skip_if_not(.rpbnb_jax_available(), "jax not installed")
  expect_jax_parity(0L, list(jax_start(0.3), jax_start(-0.8)))
})
```

- [ ] **Step 9: Run the parity test**

Run: `Rscript -e "devtools::load_all('.'); testthat::test_file('tests/testthat/test-jax-parity.R')"`
Expected: 2 tests pass (8 expectations). If they skip, `.venv-jax` was not created — rerun Task 1 Step 3.

- [ ] **Step 10: Commit**

```bash
git add inst/python/rpbnb_jax/margins.py inst/python/rpbnb_jax/packing.py inst/python/rpbnb_jax/objective.py inst/python/tests/test_margins.py R/jax_helpers.R tests/testthat/test-jax-parity.R
git commit -m "feat(jax): independence and Famoye SML objective with TMB parity"
```

---

## Task 3: Count CDF triples

The copula families need `(cdf_y, cdf_ym1, pmf_y)` in both linear and log space. This is the vectorized replacement for `nb2_cdf_pair()` (`src/rpbnb_tmb.cpp:160`) and `pois_cdf_pair()` (`:208`).

**Files:**
- Modify: `inst/python/rpbnb_jax/margins.py`
- Modify: `inst/python/tests/test_margins.py`

- [ ] **Step 1: Write the failing test**

Append to `inst/python/tests/test_margins.py`:

```python
from rpbnb_jax.margins import nb2_cdf_triple, pois_cdf_triple


def test_nb2_cdf_triple_matches_scipy():
    mu, m = 2.4, 0.6
    r = 1.0 / m
    kmax = 30
    y = np.array([0, 1, 5, 12, 30])
    lc, lcm, lp = nb2_cdf_triple(jnp.asarray(y), mu, r, kmax)
    p = r / (r + mu)
    np.testing.assert_allclose(np.exp(lc), nbinom.cdf(y, r, p), rtol=1e-11)
    np.testing.assert_allclose(np.exp(lp), nbinom.pmf(y, r, p), rtol=1e-11)
    # y = 0 leaves log_cdf_ym1 at log P(Y = 0) by convention; callers gate
    # on the observed count. src/rpbnb_tmb.cpp:157.
    np.testing.assert_allclose(np.exp(lcm[1:]), nbinom.cdf(y[1:] - 1, r, p),
                               rtol=1e-11)
    np.testing.assert_allclose(np.exp(lcm[0]), nbinom.pmf(0, r, p), rtol=1e-11)


def test_pois_cdf_triple_matches_scipy():
    mu, kmax = 3.1, 30
    y = np.array([0, 1, 5, 12, 30])
    lc, lcm, lp = pois_cdf_triple(jnp.asarray(y), mu, kmax)
    np.testing.assert_allclose(np.exp(lc), poisson.cdf(y, mu), rtol=1e-11)
    np.testing.assert_allclose(np.exp(lp), poisson.pmf(y, mu), rtol=1e-11)


def test_deep_tail_mass_survives_where_linear_space_underflows():
    # The failure nb2_cdf_pair()'s header records: a linear-space seed
    # underflows and zeroes the whole recursion. src/rpbnb_tmb.cpp:128-135.
    mu = r = 2000.0
    lc, lcm, lp = nb2_cdf_triple(jnp.asarray([2000]), mu, r, 2000)
    assert np.isfinite(float(lp[0]))
    np.testing.assert_allclose(np.exp(float(lp[0])),
                               nbinom.pmf(2000, r, r / (r + mu)), rtol=1e-8)


def test_gradient_is_finite_at_a_saturated_cdf():
    import jax
    def f(mu):
        lc, lcm, lp = nb2_cdf_triple(jnp.asarray([0, 70, 266]), mu, 2.0, 266)
        return jnp.sum(lp)
    g = jax.grad(f)(1.0)
    assert np.isfinite(float(g))
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv-jax/Scripts/python.exe -m pytest inst/python/tests/test_margins.py -v`
Expected: FAIL with `ImportError: cannot import name 'nb2_cdf_triple'`

- [ ] **Step 3: Implement the triples**

Append to `inst/python/rpbnb_jax/margins.py`:

```python
def _triple_from_grid(log_pmf_all, y, kmax):
    """Reduce a (..., kmax+1) log-mass grid to the (cdf_y, cdf_ym1, pmf_y)
    triple TMB's *_cdf_pair() returns, in log space.

    y is an integer array broadcastable against log_pmf_all[..., 0].
    Convention matches src/rpbnb_tmb.cpp:157 -- for y = 0 the "cdf_ym1" slot
    holds log P(Y = 0), not -inf; every caller gates on the observed count.
    """
    ks = jnp.arange(kmax + 1)
    yb = jnp.asarray(y)[..., None]
    log_cdf_y = logsumexp(log_pmf_all, axis=-1, where=(ks <= yb))
    # An all-false row would give a -inf VALUE, so the y = 0 row gets a true
    # mask and the P(Y = 0) convention of src/rpbnb_tmb.cpp:157. Note this is
    # a value concern, not a gradient one: `where=` masking was measured
    # against sanitising to -1e30 and gives identical values and gradients
    # wherever the mask has any true entry.
    mask_m1 = jnp.where(yb == 0, ks == 0, ks <= yb - 1)
    log_cdf_ym1 = logsumexp(log_pmf_all, axis=-1, where=mask_m1)
    log_pmf_y = jnp.take_along_axis(log_pmf_all, yb, axis=-1)[..., 0]
    return log_cdf_y, log_cdf_ym1, log_pmf_y


def nb2_cdf_triple(y, mu, r, kmax):
    """Log-space (cdf_y, cdf_ym1, pmf_y) for NB2 with size r.

    Vectorized replacement for nb2_cdf_pair() (src/rpbnb_tmb.cpp:160). The
    per-k increment log(r + j - 1) - log(j) depends only on r, which is a
    scalar shared by every observation, so the cumsum runs once rather than
    once per observation.
    """
    ks = jnp.arange(kmax + 1, dtype=jnp.float64)
    log_p = jnp.log(r) - jnp.log(r + mu)
    log_q = jnp.log(mu) - jnp.log(r + mu)
    js = jnp.arange(1, kmax + 1, dtype=jnp.float64)
    inc = jnp.log(r + js - 1.0) - jnp.log(js)
    cum = jnp.concatenate([jnp.zeros(1, dtype=jnp.float64), jnp.cumsum(inc)])
    log_pmf_all = (r * log_p)[..., None] + ks * log_q[..., None] + cum
    return _triple_from_grid(log_pmf_all, y, kmax)


def pois_cdf_triple(y, mu, kmax):
    """Poisson analogue of nb2_cdf_triple (src/rpbnb_tmb.cpp:208)."""
    ks = jnp.arange(kmax + 1, dtype=jnp.float64)
    cum = jnp.concatenate([
        jnp.zeros(1, dtype=jnp.float64),
        jnp.cumsum(-jnp.log(jnp.arange(1, kmax + 1, dtype=jnp.float64))),
    ])
    log_pmf_all = (-mu)[..., None] + ks * jnp.log(mu)[..., None] + cum
    return _triple_from_grid(log_pmf_all, y, kmax)
```

Add `from jax.scipy.special import logsumexp` to the imports at the top of `margins.py`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv-jax/Scripts/python.exe -m pytest inst/python/tests/test_margins.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add inst/python/rpbnb_jax/margins.py inst/python/tests/test_margins.py
git commit -m "feat(jax): vectorized log-space count CDF triples"
```

---

## Task 4: Frank copula

**Files:**
- Create: `inst/python/rpbnb_jax/frank.py`
- Create: `inst/python/tests/test_frank.py`
- Modify: `inst/python/rpbnb_jax/objective.py`
- Modify: `tests/testthat/test-jax-parity.R`

- [ ] **Step 1: Write the failing kernel test**

Create `inst/python/tests/test_frank.py`:

```python
import jax
import jax.numpy as jnp
import numpy as np

from rpbnb_jax.frank import frank_log_cell_prob


def _frank_C(u, v, th):
    return -np.log1p(np.expm1(-th * u) * np.expm1(-th * v)
                     / np.expm1(-th)) / th


def test_matches_second_difference_where_that_is_still_accurate():
    th = 2.5
    a, am = 0.62, 0.41
    b, bm = 0.55, 0.30
    want = (_frank_C(a, b, th) - _frank_C(am, b, th)
            - _frank_C(a, bm, th) + _frank_C(am, bm, th))
    got = float(jnp.exp(frank_log_cell_prob(
        a, am, jnp.log(a - am), b, bm, jnp.log(b - bm), th)))
    np.testing.assert_allclose(got, want, rtol=1e-9)


def test_near_independence_branch_is_continuous():
    a, am, b, bm = 0.62, 0.41, 0.55, 0.30
    lpa, lpb = jnp.log(a - am), jnp.log(b - bm)
    lo = float(frank_log_cell_prob(a, am, lpa, b, bm, lpb, 9e-6))
    hi = float(frank_log_cell_prob(a, am, lpa, b, bm, lpb, 1.1e-5))
    np.testing.assert_allclose(lo, hi, rtol=1e-6)


def test_deep_tail_returns_a_finite_log_not_a_floor():
    # The truck observation from src/rpbnb_tmb.cpp:258-264: a cell whose
    # probability is 1e-300. The log form must not clip it to -690.776.
    val = float(frank_log_cell_prob(1.0, 1.0 - 1e-300, -687.8,
                                    0.6, 0.4, jnp.log(0.2), 5.0))
    assert np.isfinite(val)
    assert val < -600.0


def test_gradient_is_finite_in_the_deep_tail():
    def f(th):
        return frank_log_cell_prob(1.0, 1.0 - 1e-300, -687.8,
                                   0.6, 0.4, jnp.log(0.2), th)
    assert np.isfinite(float(jax.grad(f)(5.0)))
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv-jax/Scripts/python.exe -m pytest inst/python/tests/test_frank.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rpbnb_jax.frank'`

- [ ] **Step 3: Port the kernel**

Create `inst/python/rpbnb_jax/frank.py`:

```python
"""Frank's log cell probability.

Line-for-line port of frank_log_cell_prob() (src/rpbnb_tmb.cpp:305). Read
that function's header before changing anything here: the telescoped form,
the log-space masses, and the identity used for M each repair a specific
measured failure of the naive second difference.
"""

import jax.numpy as jnp


def frank_log_cell_prob(a, am, log_pmf_a, b, bm, log_pmf_b, th):
    signed_eps = jnp.where(th >= 0.0, 1e-5, -1e-5)
    safe_th = jnp.where(jnp.abs(th) < 1e-5, signed_eps, th)
    abs_th = jnp.abs(safe_th)
    log_abs_th = jnp.log(abs_th)

    def log_abs_delta(um, log_pmf):
        # log|A(u) - A(u')| = -th*u' + log|expm1(-th*pmf)|, mass in logs.
        log_x = log_abs_th + log_pmf
        x = jnp.exp(log_x)
        x_safe = jnp.maximum(x, 1e-300)
        exact = jnp.where(safe_th > 0.0,
                          jnp.log(-jnp.expm1(-x_safe)),
                          jnp.log(jnp.expm1(x_safe)))
        return -safe_th * um + jnp.where(x < 1e-8, log_x, exact)

    def log_M(u, v):
        t1 = jnp.exp(-safe_th * u) * (-jnp.expm1(-safe_th * v))
        t2 = jnp.exp(-safe_th * v) * (-jnp.expm1(-safe_th * (1.0 - v)))
        return jnp.log((t1 + t2) / (-jnp.expm1(-safe_th)))

    L = (log_abs_delta(am, log_pmf_a) + log_abs_delta(bm, log_pmf_b)
         - jnp.log(jnp.abs(jnp.expm1(-safe_th)))
         - log_M(am, b) - log_M(a, bm))

    # th > 0 branch.
    L_neg = jnp.minimum(L, -1e-15)
    small_pos = L_neg < -30.0
    arg_pos = jnp.maximum(-jnp.log1p(-jnp.exp(L_neg)), 1e-300)
    arg_pos = jnp.where(small_pos, 1.0, arg_pos)   # double-where sanitation
    pos_branch = jnp.where(small_pos, L_neg, jnp.log(arg_pos))

    # th < 0 branch: three regimes, since L runs from far below zero to ~+35.
    L_cap = jnp.minimum(L, 30.0)
    big = L > 30.0
    small_neg = L < -30.0
    mid_arg = jnp.maximum(jnp.log1p(jnp.exp(L_cap)), 1e-300)
    mid_arg = jnp.where(big | small_neg, 1.0, mid_arg)
    big_arg = jnp.where(big, jnp.maximum(L, 1e-300), 1.0)
    neg_branch = jnp.where(
        big, jnp.log(big_arg),
        jnp.where(small_neg, L, jnp.log(mid_arg)))

    regular = jnp.where(safe_th > 0.0, pos_branch, neg_branch) - log_abs_th

    near_independence = (
        log_pmf_a + log_pmf_b
        + jnp.log1p(th * (1.0 - a - am) * (1.0 - b - bm) / 2.0))

    return jnp.where(jnp.abs(th) < 1e-5, near_independence, regular)
```

- [ ] **Step 4: Run the kernel tests**

Run: `.venv-jax/Scripts/python.exe -m pytest inst/python/tests/test_frank.py -v`
Expected: 4 passed.

- [ ] **Step 5: Wire Frank into the objective**

In `inst/python/rpbnb_jax/objective.py`, add to the imports:

```python
from .frank import frank_log_cell_prob
from .margins import nb2_cdf_triple, pois_cdf_triple
```

Add `kmax` to `build_objective`'s signature — `def build_objective(data, layout, kmax=None, obs_chunk=256):` — and just after the `Y2 = ...` line:

```python
    if kmax is None:
        kmax = int(max(int(Y1.max()), int(Y2.max())))
    kmax = int(kmax)
```

Add this helper above `build_objective`:

```python
def _margin_triple(y, mu, m, is_pois, kmax):
    """Log-space CDF triple for one margin, selecting the exact-Poisson form
    the same way src/rpbnb_tmb.cpp:1244-1253 does (a static flag, not a
    parameter-dependent branch)."""
    if is_pois:
        return pois_cdf_triple(y, mu, kmax)
    return nb2_cdf_triple(y, mu, 1.0 / m, kmax)
```

Then replace the `raise NotImplementedError` branch with:

```python
        else:
            y1i = jnp.asarray(data["Y1"], dtype=jnp.int32)[:, None]
            y2i = jnp.asarray(data["Y2"], dtype=jnp.int32)[:, None]
            la1, la1m, lpmf1 = _margin_triple(y1i, mu1, m1, pois1, kmax)
            lb1, lb1m, lpmf2 = _margin_triple(y2i, mu2, m2, pois2, kmax)
            a1, a1m = jnp.exp(la1), jnp.exp(la1m)
            b1, b1m = jnp.exp(lb1), jnp.exp(lb1m)

            if family == FAM_FRANK:
                # A smooth bounded link keeps exp(-theta*u) finite while
                # retaining theta = 0 and unit derivative at independence.
                theta = FRANK_THETA_MAX * jnp.tanh(p["z_dep"] / FRANK_THETA_MAX)
                # Frank is the one family returning a LOG cell probability,
                # and so the one family that skips the 1e-300 floor below.
                # src/rpbnb_tmb.cpp:1268-1272.
                log_draw = frank_log_cell_prob(a1, a1m, lpmf1,
                                               b1, b1m, lpmf2, theta)
            else:
                raise NotImplementedError(
                    f"family {family} lands in a later task")
```

- [ ] **Step 6: Add the Frank parity test**

Append to `tests/testthat/test-jax-parity.R`:

```r
test_that("JAX matches TMB for the Frank copula", {
  skip_if_not(.rpbnb_jax_available(), "jax not installed")
  expect_jax_parity(1L, list(jax_start(0.1), jax_start(2.0), jax_start(-1.5)))
})
```

- [ ] **Step 7: Run the parity test**

Run: `Rscript -e "devtools::load_all('.'); testthat::test_file('tests/testthat/test-jax-parity.R')"`
Expected: 3 tests pass.

- [ ] **Step 8: Commit**

```bash
git add inst/python/rpbnb_jax/frank.py inst/python/tests/test_frank.py inst/python/rpbnb_jax/objective.py tests/testthat/test-jax-parity.R
git commit -m "feat(jax): Frank copula cell probability with TMB parity"
```

---

## Task 5: Clayton copula

**Files:**
- Create: `inst/python/rpbnb_jax/clayton.py`
- Create: `inst/python/tests/test_clayton.py`
- Modify: `inst/python/rpbnb_jax/objective.py`
- Modify: `tests/testthat/test-jax-parity.R`

- [ ] **Step 1: Write the failing kernel test**

Create `inst/python/tests/test_clayton.py`:

```python
import jax
import jax.numpy as jnp
import numpy as np

from rpbnb_jax.clayton import clayton_cell_prob


def _clayton_C(u, v, th):
    return (u ** -th + v ** -th - 1.0) ** (-1.0 / th)


def _cell(a, am, b, bm, th):
    return (_clayton_C(a, b, th) - _clayton_C(am, b, th)
            - _clayton_C(a, bm, th) + _clayton_C(am, bm, th))


def test_matches_second_difference_in_the_benign_interior():
    th = 1.7
    a, am, b, bm = 0.62, 0.41, 0.55, 0.30
    got = float(clayton_cell_prob(
        jnp.log(a), jnp.log(am), jnp.log(a - am),
        jnp.log(b), jnp.log(bm), jnp.log(b - bm), th, False, False))
    np.testing.assert_allclose(got, _cell(a, am, b, bm, th), rtol=1e-9)


def test_axis_branch_matches_a_first_difference():
    th = 1.7
    a, am, b = 0.62, 0.41, 0.55
    want = _clayton_C(a, b, th) - _clayton_C(am, b, th)
    got = float(clayton_cell_prob(
        jnp.log(a), jnp.log(am), jnp.log(a - am),
        jnp.log(b), jnp.log(b), jnp.log(b), th, False, True))
    np.testing.assert_allclose(got, want, rtol=1e-9)


def test_strong_dependence_approaches_the_comonotonic_limit():
    # src/rpbnb_tmb.cpp:425-432: an earlier cap drove this to 2.4e-10 at
    # z_dep = 20 instead of approaching P(Y = 1) = 0.29630.
    a, am = 0.55, 0.26
    got = float(clayton_cell_prob(
        jnp.log(a), jnp.log(am), jnp.log(a - am),
        jnp.log(a), jnp.log(am), jnp.log(a - am),
        np.exp(20.0), False, False))
    np.testing.assert_allclose(got, a - am, rtol=1e-6)


def test_probability_is_positive_by_construction_in_the_deep_tail():
    got = float(clayton_cell_prob(
        jnp.asarray(-1e-16), jnp.asarray(-1e-15), jnp.asarray(-300.0),
        jnp.asarray(-1e-16), jnp.asarray(-1e-15), jnp.asarray(-290.0),
        3.0, False, False))
    assert got >= 0.0
    assert np.isfinite(got)


def test_gradient_is_finite_at_a_zero_count_cell():
    def f(th):
        return clayton_cell_prob(
            jnp.log(0.31), jnp.log(0.31), jnp.log(0.31),
            jnp.log(0.44), jnp.log(0.22), jnp.log(0.22), th, True, False)
    assert np.isfinite(float(jax.grad(f)(1.7)))
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv-jax/Scripts/python.exe -m pytest inst/python/tests/test_clayton.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rpbnb_jax.clayton'`

- [ ] **Step 3: Port the kernel**

Create `inst/python/rpbnb_jax/clayton.py`:

```python
"""Clayton's cell probability.

Port of clayton_cell_prob() (src/rpbnb_tmb.cpp:454). x and y are carried as
LOGARITHMS throughout because theta reaches 4.85e8 and a^-theta overflows a
double; see that header, and note in particular why du must not be computed
as log(1+x+y) - log1p(x) - log1p(y) on the small-w side.

y1_zero / y2_zero are STATIC Python bools when called on whole arrays with a
uniform zero pattern, but under vmap over observations they arrive as arrays.
Both are supported: pass bools for the scalar tests, arrays for the objective.
The zero branches are safe to evaluate unconditionally because *_cdf_triple
leaves log_cdf_ym1 at log P(Y = 0) rather than -inf (src/rpbnb_tmb.cpp:157).
"""

import jax.numpy as jnp


def clayton_cell_prob(log_a, log_am, log_pmf_a,
                      log_b, log_bm, log_pmf_b, th, y1_zero, y2_zero):
    k = -1.0 / th

    # log s00, s00 = a^-th + b^-th - 1 >= 1.
    La = -th * log_a
    Lb = -th * log_b
    M = jnp.maximum(La, Lb)
    log_s00 = M + jnp.log(jnp.exp(La - M) + jnp.exp(Lb - M) - jnp.exp(-M))
    C00 = jnp.exp(k * log_s00)

    def log_ratio(log_um, log_pmf_u):
        # The ratio log(u/u') is built from the MARGINAL MASS, never from
        # log_u - log_um: once the CDF saturates that difference is exactly
        # zero. src/rpbnb_tmb.cpp:470-484.
        t = log_pmf_u - log_um
        LR = jnp.logaddexp(0.0, t)
        LR_safe = jnp.maximum(LR, 1e-300)
        log_LR = jnp.where(t < -30.0, t, jnp.log(LR_safe))
        S = th * LR
        log_S = jnp.log(th) + log_LR
        S_safe = jnp.maximum(S, 1e-300)
        log_bracket = jnp.where(S < 1e-8, log_S,
                                jnp.log(-jnp.expm1(-S_safe)))
        return -th * log_um + log_bracket - log_s00

    log_x = log_ratio(log_am, log_pmf_a)
    log_y = log_ratio(log_bm, log_pmf_b)
    L1x = jnp.logaddexp(0.0, log_x)
    L1y = jnp.logaddexp(0.0, log_y)

    # Interior cell.
    u1 = k * jnp.logaddexp(0.0, log_y - L1x)
    u2 = k * L1y
    log_w = log_x + log_y - L1x - L1y          # log of xy/((1+x)(1+y)) < 0
    Lsum = jnp.logaddexp(L1x, log_y)           # log(1 + x + y)
    small_w = log_w < -0.7
    w_safe = jnp.exp(jnp.minimum(log_w, -0.7))
    du = k * jnp.where(small_w, jnp.log1p(-w_safe), Lsum - L1x - L1y)
    ex = jnp.expm1(k * L1x)
    interior = C00 * (jnp.exp(u2) * jnp.expm1(du) + ex * jnp.expm1(u1))

    # Axis branches: the second difference degenerates to a first difference.
    only_y2_zero = -C00 * jnp.expm1(k * L1x)
    only_y1_zero = -C00 * jnp.expm1(k * L1y)

    z1 = jnp.asarray(y1_zero)
    z2 = jnp.asarray(y2_zero)
    return jnp.where(
        z1 & z2, C00,
        jnp.where(z2, only_y2_zero,
                  jnp.where(z1, only_y1_zero, interior)))
```

- [ ] **Step 4: Run the kernel tests**

Run: `.venv-jax/Scripts/python.exe -m pytest inst/python/tests/test_clayton.py -v`
Expected: 5 passed.

- [ ] **Step 5: Wire Clayton into the objective**

In `inst/python/rpbnb_jax/objective.py`, add `from .clayton import clayton_cell_prob` to the imports, and replace the inner `raise NotImplementedError` with:

```python
            elif family == FAM_CLAYTON:
                theta = jnp.exp(jnp.clip(p["z_dep"], -20.0, 20.0))
                p_obs = clayton_cell_prob(la1, la1m, lpmf1, lb1, lb1m, lpmf2,
                                          theta, y1i == 0, y2i == 0)
                log_draw = jnp.log(jnp.maximum(p_obs, 1e-300))
            else:
                raise NotImplementedError(
                    f"family {family} lands in a later task")
```

- [ ] **Step 6: Add the Clayton parity test**

Append to `tests/testthat/test-jax-parity.R`:

```r
test_that("JAX matches TMB for the Clayton copula", {
  skip_if_not(.rpbnb_jax_available(), "jax not installed")
  expect_jax_parity(3L, list(jax_start(0.0), jax_start(1.2), jax_start(-1.0)))
})
```

- [ ] **Step 7: Run the parity test**

Run: `Rscript -e "devtools::load_all('.'); testthat::test_file('tests/testthat/test-jax-parity.R')"`
Expected: 4 tests pass.

- [ ] **Step 8: Commit**

```bash
git add inst/python/rpbnb_jax/clayton.py inst/python/tests/test_clayton.py inst/python/rpbnb_jax/objective.py tests/testthat/test-jax-parity.R
git commit -m "feat(jax): Clayton copula cell probability with TMB parity"
```

---

## Task 6: Gaussian copula

The heaviest kernel, and the one where JAX should gain most: TMB needs `REGISTER_ATOMIC` to keep the 16-point strip integral off the tape, and XLA needs nothing.

**Files:**
- Create: `inst/python/rpbnb_jax/gaussian.py`
- Create: `inst/python/tests/test_gaussian.py`
- Modify: `inst/python/rpbnb_jax/objective.py`
- Modify: `tests/testthat/test-jax-parity.R`

- [ ] **Step 1: Write the failing kernel test**

Create `inst/python/tests/test_gaussian.py`:

```python
import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import multivariate_normal, norm

from rpbnb_jax.gaussian import gauss_corner_quantiles, gaussian_cell_prob


def _bvn_cell(qa, qam, qb, qbm, rho):
    cov = [[1.0, rho], [rho, 1.0]]
    cdf = lambda x, y: multivariate_normal.cdf([x, y], mean=[0, 0], cov=cov)
    return cdf(qa, qb) - cdf(qam, qb) - cdf(qa, qbm) + cdf(qam, qbm)


def test_strip_integral_matches_a_bivariate_normal_rectangle():
    for rho in (-0.5, 0.0, 0.3, 0.8):
        got = float(gaussian_cell_prob(0.8, -0.4, 0.6, -0.9, rho))
        np.testing.assert_allclose(got, _bvn_cell(0.8, -0.4, 0.6, -0.9, rho),
                                   rtol=2e-8, atol=1e-12)


def test_extreme_rho_keeps_an_ordinary_cell():
    # src/rpbnb_tmb.cpp:594-601: a single rule over the whole interval
    # returned 1.7e-25 for a cell whose true probability is 0.0401.
    got = float(gaussian_cell_prob(1.06, -7.94, 1.10, -7.90, 0.9999))
    assert got > 1e-3


def test_probability_is_non_negative_by_construction():
    for rho in (-0.999, -0.9, 0.0, 0.9, 0.999):
        assert float(gaussian_cell_prob(-2.0, -2.4, 3.1, 3.0, rho)) >= 0.0


def test_corner_quantiles_carry_exactly_the_marginal_mass():
    # The invariant from src/rpbnb_tmb.cpp:826-829: the strip must carry the
    # marginal mass, Phi(q(y)) - Phi(q(y-1)) == P(Y = y).
    F_y, pmf = 1.0 - 1e-13, 4.0e-14
    sf = 1e-13
    q_y, q_ym = gauss_corner_quantiles(F_y, F_y - pmf, pmf, sf,
                                       y_zero=False, use_sf=True)
    got = norm.cdf(float(q_y)) - norm.cdf(float(q_ym))
    np.testing.assert_allclose(got, pmf, rtol=1e-9)


def test_zero_count_uses_the_historical_sentinel():
    q_y, q_ym = gauss_corner_quantiles(0.31, 0.31, 0.31, 0.69,
                                       y_zero=True, use_sf=True)
    np.testing.assert_allclose(float(q_ym), -7.941345, rtol=1e-6)


def test_gradient_in_rho_is_finite_at_extreme_dependence():
    g = jax.grad(lambda r: gaussian_cell_prob(0.8, -0.4, 0.6, -0.9, r))(0.9999)
    assert np.isfinite(float(g))
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv-jax/Scripts/python.exe -m pytest inst/python/tests/test_gaussian.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rpbnb_jax.gaussian'`

- [ ] **Step 3: Port the kernel**

Create `inst/python/rpbnb_jax/gaussian.py`:

```python
"""Gaussian copula: corner quantiles and the strip integral.

Ports gauss_corner_quantiles() (src/rpbnb_tmb.cpp:914) and
gaussian_cell_prob() (:672). The panel layout is not a matter of taste --
read the header at :592 before changing GAUSS_EDGE or the phi cuts.

The REGISTER_ATOMIC wrapper at :775 has no analogue here. It exists purely
to keep ~1,500 tape operations per observation-draw off TMB's tape; XLA
fuses the quadrature with no tape at all.
"""

import jax.numpy as jnp
from jax.scipy.special import ndtri
from jax.scipy.stats import norm

# 16-point Gauss-Legendre, positive half of the symmetric node set.
_GX = jnp.array([
    0.09501250983763769, 0.28160355077925908, 0.45801677765722731,
    0.61787624440264388, 0.75540440835500322, 0.86563120238783220,
    0.94457502307323249, 0.98940093499165027])
_GW = jnp.array([
    0.18945061045506834, 0.18260341504492425, 0.16915651939500245,
    0.14959598881657588, 0.12462897125553447, 0.09515851168249304,
    0.06225352393864833, 0.02715245941175411])
# Signed nodes/weights, so the panel loop is one vectorized axis of length 16.
_NODES = jnp.concatenate([-_GX[::-1], _GX])
_WEIGHTS = jnp.concatenate([_GW[::-1], _GW])

GAUSS_EDGE = 4.0
_PHI_CUTS = jnp.array([-2.0, 2.0])


def gauss_corner_quantiles(F_y, F_ym, pmf, sf, y_zero, use_sf):
    """Normal quantiles of one margin's two cell corners, each taken from
    whichever tail is still representable. src/rpbnb_tmb.cpp:914.

    use_sf is STATIC (it is the pois flag): for an NB2 margin the survival
    call must not appear at all -- its mere presence cost a finite gradient,
    see :896-906.
    """
    TINY = 1e-300
    ZERO_Q = -7.941345          # qnorm(1e-15), the historical floor
    NEAR_ONE = 1.0 - 1e-10      # only past here has F lost its tail
    NEAR_ONE_P = 1.0 - 1e-16    # keeps ndtri's argument below 1

    if not use_sf:
        clamp = lambda p: ndtri(jnp.clip(p, 1e-15, 1.0 - 1e-15))
        return clamp(F_y), clamp(F_ym)

    upper = F_y > NEAR_ONE
    # S(y-1) = S(y) + P(Y = y): an addition of positives, never a 1 - F.
    p_y = jnp.where(upper, jnp.maximum(sf, TINY), jnp.maximum(F_y, TINY))
    p_ym = jnp.where(upper,
                     jnp.clip(sf + pmf, TINY, NEAR_ONE_P),
                     jnp.maximum(F_ym, TINY))
    sgn = jnp.where(upper, -1.0, 1.0)

    q_y = sgn * ndtri(p_y)
    q_ym = sgn * ndtri(p_ym)
    return q_y, jnp.where(jnp.asarray(y_zero), ZERO_Q, q_ym)


def gaussian_cell_prob(qa, qam, qb, qbm, rho):
    """One strip integral over the conditioning normal, never a second
    difference of four corner CDFs. src/rpbnb_tmb.cpp:672."""
    sig2 = jnp.maximum(1.0 - rho * rho, 1e-12)
    sig = jnp.sqrt(sig2)

    rho_abs = jnp.abs(rho)
    rho_faf = jnp.maximum(rho_abs, 1e-6)
    rho_sgn = jnp.where(rho < 0.0, -rho_faf, rho_faf)

    edge = sig / rho_faf                 # transition width, in z
    c1 = qb / rho_sgn
    c2 = qbm / rho_sgn
    clo = jnp.minimum(c1, c2)
    chi = jnp.maximum(c1, c2)

    lo = qam
    hi = jnp.maximum(qa, qam)

    # Interior cuts: four edge brackets plus the two cuts on phi's own scale.
    # They must be SORTED before the monotone pass, or an out-of-order fixed
    # cut drags an edge bracket with it. jnp.sort is differentiable.
    v = jnp.stack([
        clo - GAUSS_EDGE * edge, clo + GAUSS_EDGE * edge,
        chi - GAUSS_EDGE * edge, chi + GAUSS_EDGE * edge,
        jnp.broadcast_to(_PHI_CUTS[0], jnp.shape(clo)),
        jnp.broadcast_to(_PHI_CUTS[1], jnp.shape(clo)),
    ], axis=-1)
    v = jnp.sort(v, axis=-1)

    # Clamp into [lo, hi], then force non-decreasing. The union of the panels
    # is exactly [q(a'), q(a)] however many collapse -- never a sub-interval.
    v = jnp.clip(v, lo[..., None] if jnp.ndim(lo) else lo,
                 hi[..., None] if jnp.ndim(hi) else hi)
    v = jax_cummax(v)
    cuts = jnp.concatenate([jnp.asarray(lo)[..., None], v,
                            jnp.asarray(hi)[..., None]], axis=-1)
    cuts = jax_cummax(cuts)

    left = cuts[..., :-1]
    right = cuts[..., 1:]
    half = (right - left) / 2.0
    mid = (right + left) / 2.0

    z = mid[..., None] + half[..., None] * _NODES      # (..., panel, node)
    A = (qb[..., None, None] - rho[..., None, None] * z) / sig[..., None, None]
    B = (qbm[..., None, None] - rho[..., None, None] * z) / sig[..., None, None]
    # Take the difference on whichever tail is not saturated. Applied to the
    # ARGUMENTS, not to two finished pnorm results: selecting afterwards would
    # double the pnorm count.
    flip = (A + B) > 0.0
    Au = jnp.where(flip, -B, A)
    Bu = jnp.where(flip, -A, B)
    integrand = norm.pdf(z) * (norm.cdf(Au) - norm.cdf(Bu))
    panel = half * jnp.sum(_WEIGHTS * integrand, axis=-1)
    return jnp.sum(panel, axis=-1)


def jax_cummax(x):
    """Running maximum along the last axis."""
    return jnp.maximum.accumulate(x, axis=-1)
```

Note: `qa`/`qb`/`rho` may be scalars in the unit tests and arrays in the objective. Broadcast them with `jnp.asarray(...)` at the top of `gaussian_cell_prob` if the scalar path errors on `[..., None]`; the tests in Step 1 exercise both shapes.

- [ ] **Step 4: Run the kernel tests**

Run: `.venv-jax/Scripts/python.exe -m pytest inst/python/tests/test_gaussian.py -v`
Expected: 6 passed.

- [ ] **Step 5: Wire Gaussian into the objective**

In `inst/python/rpbnb_jax/objective.py`, add:

```python
from jax.scipy.special import gammainc
from .gaussian import gauss_corner_quantiles, gaussian_cell_prob
```

and replace the final `raise NotImplementedError` with:

```python
            else:  # FAM_GAUSSIAN
                rho = jnp.tanh(p["z_dep"])
                pmf1 = jnp.exp(lpmf1)
                pmf2 = jnp.exp(lpmf2)
                # Upper-tail survival from the margin's own special function,
                # never as 1 - CDF. Poisson margins only: the NB2 form
                # (pbeta, shape r = 1/m running to 1e8) must stay off the
                # graph entirely. src/rpbnb_tmb.cpp:1282-1286.
                sf1 = (gammainc(y1i + 1.0, mu1) if pois1 else jnp.zeros_like(mu1))
                sf2 = (gammainc(y2i + 1.0, mu2) if pois2 else jnp.zeros_like(mu2))
                qa, qam = gauss_corner_quantiles(a1, a1m, pmf1, sf1,
                                                 y1i == 0, pois1)
                qb, qbm = gauss_corner_quantiles(b1, b1m, pmf2, sf2,
                                                 y2i == 0, pois2)
                p_obs = gaussian_cell_prob(qa, qam, qb, qbm, rho)
                log_draw = jnp.log(jnp.maximum(p_obs, 1e-300))
```

- [ ] **Step 6: Add the Gaussian parity test**

Append to `tests/testthat/test-jax-parity.R`:

```r
test_that("JAX matches TMB for the Gaussian copula", {
  skip_if_not(.rpbnb_jax_available(), "jax not installed")
  # The strip integral is quadrature, so parity is to its accuracy, not to
  # machine precision. src/rpbnb_tmb.cpp:52-54 measures 7e-7 worst case.
  expect_jax_parity(2L, list(jax_start(0.0), jax_start(0.9), jax_start(-1.4)),
                    tol = 1e-6)
})
```

- [ ] **Step 7: Run the full parity suite**

Run: `Rscript -e "devtools::load_all('.'); testthat::test_file('tests/testthat/test-jax-parity.R')"`
Expected: 5 tests pass, covering all five families.

- [ ] **Step 8: Commit**

```bash
git add inst/python/rpbnb_jax/gaussian.py inst/python/tests/test_gaussian.py inst/python/rpbnb_jax/objective.py tests/testthat/test-jax-parity.R
git commit -m "feat(jax): Gaussian copula strip integral with TMB parity"
```

---

## Task 7: End-to-end fit and benchmark

Parity on `fn`/`gr` at fixed parameters is the correctness result. This task answers whether it is worth anything.

**Files:**
- Create: `inst/benchmark_jax.R`
- Modify: `tests/testthat/test-jax-parity.R`

- [ ] **Step 1: Write the failing end-to-end test**

Append to `tests/testthat/test-jax-parity.R`:

```r
test_that("nlminb converges to the same optimum through either engine", {
  skip_if_not(.rpbnb_jax_available(), "jax not installed")
  fx <- jax_fixture(1L)
  start <- jax_start(0.1)
  free <- rep(TRUE, length(start))
  jx <- .make_rpbnb_jax_object(fx$data, start, free)
  opt <- stats::nlminb(start = jx$par, objective = jx$fn, gradient = jx$gr,
                       control = list(iter.max = 200L, eval.max = 400L))
  expect_lt(opt$objective, jx$fn(jx$par))
  expect_lt(max(abs(jx$gr(opt$par))), 1e-3)
  expect_false(is.null(jx$env$last.par.best))
})
```

- [ ] **Step 2: Run it**

Run: `Rscript -e "devtools::load_all('.'); testthat::test_file('tests/testthat/test-jax-parity.R')"`
Expected: 6 tests pass. A failure here means the gradient is right at the fixture points but wrong somewhere the optimizer visits — bisect by comparing `jx$gr` against `numDeriv::grad(jx$fn, par)` along the failing path.

- [ ] **Step 3: Write the benchmark script**

Create `inst/benchmark_jax.R`:

```r
# TMB vs JAX: objective and gradient timing across families and draw counts.
#   Rscript inst/benchmark_jax.R
# Reports median seconds per fn+gr pair. JAX times exclude the first call,
# which pays XLA compilation.
devtools::load_all(".")
stopifnot(.rpbnb_jax_available())

bench_one <- function(family_code, n, draws) {
  set.seed(7)
  x <- rnorm(n)
  X <- cbind(`(Intercept)` = 1, x = x)
  Z <- .tmb_halton_uniform(draws, 2L, burn = 300L)
  data <- .build_tmb_data(
    Y1 = as.numeric(rpois(n, exp(0.4 + 0.2 * x))),
    Y2 = as.numeric(rpois(n, exp(0.3 - 0.1 * x))),
    X1 = X, X2 = X, rand_idx1 = 2L, rand_idx2 = 2L,
    Z1 = Z[, 1, drop = FALSE], Z2 = Z[, 2, drop = FALSE],
    dist1 = 0L, dist2 = 0L, sign1 = 1L, sign2 = 1L,
    family_code = family_code, pois1 = FALSE, pois2 = FALSE,
    lamLo = -0.9, lamHi = 0.9, est_method = 0L)
  start <- c(0.4, 0.2, 0.3, -0.1, log(0.25), log(0.30),
             log(0.5), log(0.6), 0.3)
  free <- rep(TRUE, length(start))

  tmb <- .make_rpbnb_tmb_object(
    data = data,
    parameters = list(beta1 = start[1:2], beta2 = start[3:4],
                      log_sd1 = start[5], log_sd2 = start[6],
                      log_m1 = start[7], log_m2 = start[8],
                      z_dep = start[9],
                      u1 = matrix(0, n, 1L), u2 = matrix(0, n, 1L)),
    map = list(u1 = factor(rep(NA_integer_, n)),
               u2 = factor(rep(NA_integer_, n))),
    random = NULL, silent = TRUE, n_cores = 1L, max_threads = 1L)$obj
  jx <- .make_rpbnb_jax_object(data, start, free)

  invisible(jx$fn(start)); invisible(jx$gr(start))  # warm up XLA
  timeit <- function(o) {
    ts <- replicate(5, system.time({ o$fn(start + 1e-6); o$gr(start + 2e-6) })[["elapsed"]])
    median(ts)
  }
  data.frame(family = family_code, n = n, draws = draws,
             tmb = timeit(tmb), jax = timeit(jx))
}

grid <- expand.grid(family = c(-1L, 0L, 1L, 2L, 3L), draws = c(50L, 200L))
res <- do.call(rbind, Map(function(f, d) bench_one(f, 1000L, d),
                          grid$family, grid$draws))
res$speedup <- res$tmb / res$jax
print(res, row.names = FALSE)
```

- [ ] **Step 4: Run the benchmark**

Run: `Rscript inst/benchmark_jax.R`
Expected: a 10-row table. Record the numbers — the Gaussian rows are the ones to read first, since that family is where TMB's tape costs the most.

- [ ] **Step 5: Commit**

```bash
git add inst/benchmark_jax.R tests/testthat/test-jax-parity.R
git commit -m "feat(jax): end-to-end optimizer test and TMB benchmark"
```

---

## Self-Review Notes

**Spec coverage.** All five families are covered (Task 2 independence/Famoye, Task 4 Frank, Task 5 Clayton, Task 6 Gaussian). SML only, per scope — no task touches Laplace, `u1`/`u2`, or `tmbprofile`. Validation is value-plus-gradient against TMB on fixtures, per the chosen approach, in `tests/testthat/test-jax-parity.R`.

**Known gaps, deliberately left.**
- `obs_chunk` is threaded through `build_objective`'s signature and the R bridge but is not yet used to chunk anything; the fixtures are small enough not to need it. Wire it into a `jax.lax.map` when the truck-scale benchmark first runs out of memory, and expect that to be during Task 7 Step 4 at `draws = 200`.
- `.make_rpbnb_jax_object()` is a standalone entry point. Nothing routes `rpbnb(engine = "jax")` to it; `fit_rpbnb_tmb()` is untouched. That wiring is a follow-up, and should not be attempted before the benchmark says the port is worth keeping.
- The Famoye `lamLo`/`lamHi` box is read from `data` as TMB reads it — frozen at the starting values. The R-side `famoye_support_bounds()` machinery is not reimplemented, so a JAX-only fit path would need it.
