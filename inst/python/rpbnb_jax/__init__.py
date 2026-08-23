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

# Must match FRANK_THETA_MAX in src/rpbnb_tmb.cpp:37 and R/utilities.R.
FRANK_THETA_MAX = 35.0

# log(1e15); the shared ceiling on every linear predictor.
ETA_CEILING = 34.538776394910684

__all__ = [
    "FAM_INDEP", "FAM_FAMOYE", "FAM_FRANK", "FAM_GAUSSIAN", "FAM_CLAYTON",
    "DIST_NORMAL", "DIST_LOGNORMAL", "DIST_UNIFORM", "DIST_TRIANGULAR",
    "FRANK_THETA_MAX", "ETA_CEILING",
]
