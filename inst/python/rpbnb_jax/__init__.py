"""JAX re-implementation of the rpbnb SML objective.

Mirrors src/rpbnb_tmb.cpp with est_method = 0. See that file's kernel
headers for why each formula takes the shape it does; every one of them
records a measured failure of the textbook alternative.
"""

import os

# Set before jax is imported anywhere in the process. jax reads this at import
# time, so it is immune to the import-ordering problem that the config.update()
# below has. Verified: this variable alone yields float64 with no other call.
# setdefault, not assignment, so an explicit setting in the environment is
# still visible here rather than being clobbered -- though config.update()
# below will in practice still force x64 on afterwards.
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax  # noqa: E402  (must follow the environment setup above)
import jax.numpy as jnp  # noqa: E402

# Belt and braces: this covers the case where jax was already imported by
# someone else before the environment variable could take effect. The whole
# numerical design assumes double precision -- 1e-300 floors, the 1.1e-16
# spacing at 1.0, cell probabilities down to 1e-136. float32 destroys all of
# it.
jax.config.update("jax_enable_x64", True)

# Post-condition. A silently-float32 process produces plausible, wrong numbers
# everywhere downstream, so this raises rather than warns.
if jnp.zeros(()).dtype != jnp.float64:  # pragma: no cover
    raise RuntimeError(
        "jax_enable_x64 did not take effect; every downstream result would "
        "be silently wrong. Something imported jax and created an array "
        "before this module."
    )

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

# log(1e15); the shared ceiling on every linear predictor. In the C++ this is
# an unnamed repeated literal with no symbol to grep for -- it appears as
# Type(34.538776394910684) in the mu1 and mu2 clamps at
# src/rpbnb_tmb.cpp:1186 and :1188.
ETA_CEILING = 34.538776394910684

__all__ = [
    "FAM_INDEP", "FAM_FAMOYE", "FAM_FRANK", "FAM_GAUSSIAN", "FAM_CLAYTON",
    "DIST_NORMAL", "DIST_LOGNORMAL", "DIST_UNIFORM", "DIST_TRIANGULAR",
    "FRANK_THETA_MAX", "ETA_CEILING",
]
