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
import numpy as np


class Layout:
    def __init__(self, k1, k2, q1, q2, template, free_idx):
        self.k1, self.k2, self.q1, self.q2 = (
            int(k1), int(k2), int(q1), int(q2))
        # np.atleast_1d first: reticulate hands a length-1 R vector across as
        # a bare Python scalar, and jnp.asarray(3.0) is a 0-d array whose
        # .shape[0] does not exist. Every array-shaped argument that can be
        # length 1 in a real fit gets this treatment.
        self.template = jnp.asarray(np.atleast_1d(template),
                                    dtype=jnp.float64)
        self.free_idx = jnp.asarray(
            np.atleast_1d(free_idx).astype(np.int32), dtype=jnp.int32)
        self.n_total = int(self.template.shape[0])
        self.n_free = int(self.free_idx.shape[0])
        if self.n_total != self.k1 + self.k2 + self.q1 + self.q2 + 3:
            raise ValueError(
                "template has {} entries but k1+k2+q1+q2+3 = {}".format(
                    self.n_total,
                    self.k1 + self.k2 + self.q1 + self.q2 + 3))

    def unpack(self, free_vec):
        # .at[idx].set() BROADCASTS, so a length-1 free_vec would set every
        # free coordinate to that one scalar and return a finite, wrong
        # objective -- silently. Length 3 against 9 free raises; only the
        # length-1 case is silent, and length 1 is exactly what a scalar
        # argument from R collapses to. n_free exists for this check.
        free_vec = jnp.asarray(free_vec)
        if free_vec.shape != (self.n_free,):
            raise ValueError(
                "free_vec must have shape ({},), got {}".format(
                    self.n_free, tuple(free_vec.shape)))
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
