# Copula math: CDFs, partial derivatives, Kendall's tau. Internal.
# All functions are vectorized over (u, v); theta/rho is scalar.

# ---- Copula CDFs ----

#' Frank copula CDF: C(u,v;θ), θ ∈ ℝ \ {0}; limit at θ=0 is u*v.
#' @keywords internal
#' @noRd
frank_cdf <- function(u, v, theta) {
  out <- numeric(max(length(u), length(v)))
  ok  <- (u > 0) & (v > 0)
  if (any(ok)) {
    ui <- if (length(u) == 1L) rep(u, sum(ok)) else u[ok]
    vi <- if (length(v) == 1L) rep(v, sum(ok)) else v[ok]
    if (abs(theta) < 1e-10) {
      out[ok] <- ui * vi
    } else {
      et  <- exp(-theta)
      out[ok] <- -log(1 + (exp(-theta * ui) - 1) * (exp(-theta * vi) - 1) / (et - 1)) / theta
    }
  }
  out
}

#' Gaussian (normal) copula CDF: C(u,v;ρ) via pbivnorm.
#' @keywords internal
#' @noRd
normal_cdf <- function(u, v, rho) {
  out <- numeric(max(length(u), length(v)))
  ok  <- (u > 0) & (v > 0)
  if (any(ok)) {
    ui <- pmin(pmax(if (length(u) == 1L) rep(u, sum(ok)) else u[ok], 1e-15), 1 - 1e-15)
    vi <- pmin(pmax(if (length(v) == 1L) rep(v, sum(ok)) else v[ok], 1e-15), 1 - 1e-15)
    out[ok] <- pbivnorm::pbivnorm(qnorm(ui), qnorm(vi), rho)
  }
  out
}

#' Kimeldorf-Sampson (Clayton) copula CDF: C(u,v;θ), θ > 0.
#' @keywords internal
#' @noRd
kimeldorf_cdf <- function(u, v, theta) {
  out <- numeric(max(length(u), length(v)))
  ok  <- (u > 0) & (v > 0)
  if (any(ok)) {
    ui <- if (length(u) == 1L) rep(u, sum(ok)) else u[ok]
    vi <- if (length(v) == 1L) rep(v, sum(ok)) else v[ok]
    if (theta < 1e-10) {
      out[ok] <- ui * vi
    } else {
      out[ok] <- pmax(ui^(-theta) + vi^(-theta) - 1, 0)^(-1 / theta)
    }
  }
  out
}

# ---- Copula partial derivatives (vectorized) ----

#' dC/du  — same for all three families; dC/dv = dC/du with args swapped (symmetry).
#' When v=0: C(u,0)=0 identically → dC/du = 0. Guards enforce this.
#' @keywords internal
#' @noRd
frank_du <- function(u, v, theta) {
  out <- numeric(max(length(u), length(v)))
  ok  <- (u > 0) & (v > 0)
  if (any(ok)) {
    ui <- if (length(u) == 1L) rep(u, sum(ok)) else u[ok]
    vi <- if (length(v) == 1L) rep(v, sum(ok)) else v[ok]
    if (abs(theta) < 1e-10) {
      out[ok] <- vi
    } else {
      et  <- exp(-theta)
      etu <- exp(-theta * ui)
      etv <- exp(-theta * vi)
      out[ok] <- etu * (etv - 1) / ((et - 1) + (etu - 1) * (etv - 1))
    }
  }
  out
}

#' @keywords internal
#' @noRd
normal_du <- function(u, v, rho) {
  out <- numeric(max(length(u), length(v)))
  ok  <- (u > 0) & (v > 0)
  if (any(ok)) {
    ui <- pmin(pmax(if (length(u) == 1L) rep(u, sum(ok)) else u[ok], 1e-15), 1 - 1e-15)
    vi <- pmin(pmax(if (length(v) == 1L) rep(v, sum(ok)) else v[ok], 1e-15), 1 - 1e-15)
    qu <- qnorm(ui); qv <- qnorm(vi)
    out[ok] <- pnorm((qv - rho * qu) / sqrt(1 - rho^2))
  }
  out
}

#' @keywords internal
#' @noRd
kimeldorf_du <- function(u, v, theta) {
  out <- numeric(max(length(u), length(v)))
  ok  <- (u > 0) & (v > 0)
  if (any(ok)) {
    ui <- if (length(u) == 1L) rep(u, sum(ok)) else u[ok]
    vi <- if (length(v) == 1L) rep(v, sum(ok)) else v[ok]
    if (theta < 1e-10) {
      out[ok] <- vi
    } else {
      inner <- pmax(ui^(-theta) + vi^(-theta) - 1, 1e-300)
      out[ok] <- inner^(-(1 / theta) - 1) * ui^(-(theta + 1))
    }
  }
  out
}

# dispatch helpers used by the gradient
.cop_du <- function(u, v, theta, family) {
  switch(family,
    frank     = frank_du(u, v, theta),
    normal    = normal_du(u, v, theta),
    kimeldorf = kimeldorf_du(u, v, theta)
  )
}
.cop_dv <- function(u, v, theta, family) .cop_du(v, u, theta, family)   # symmetry

# ---- Copula partial derivatives wrt theta ----

#' dC/dtheta for Frank copula
#' @keywords internal
#' @noRd
frank_dtheta <- function(u, v, theta) {
  out <- numeric(max(length(u), length(v)))
  ok  <- (u > 0) & (v > 0)
  if (any(ok)) {
    ui <- if (length(u) == 1L) rep(u, sum(ok)) else u[ok]
    vi <- if (length(v) == 1L) rep(v, sum(ok)) else v[ok]
    if (abs(theta) < 1e-10) {
      out[ok] <- ui * vi * (ui - 1) * (vi - 1) / 2
    } else {
      et  <- exp(-theta)
      etu <- exp(-theta * ui)
      etv <- exp(-theta * vi)
      A   <- (etu - 1) * (etv - 1) / (et - 1)
      C   <- -log(1 + A) / theta
      dA  <- ((-ui * etu) * (etv - 1) * (et - 1) +
              (etu - 1) * (-vi * etv) * (et - 1) -
              (etu - 1) * (etv - 1) * (-et)) / (et - 1)^2
      out[ok] <- -dA / (theta * (1 + A)) - C / theta
    }
  }
  out
}

#' dC/drho for Gaussian copula
#' @keywords internal
#' @noRd
normal_drho <- function(u, v, rho) {
  out <- numeric(max(length(u), length(v)))
  ok  <- (u > 0) & (v > 0)
  if (any(ok)) {
    ui <- pmin(pmax(if (length(u) == 1L) rep(u, sum(ok)) else u[ok], 1e-15), 1 - 1e-15)
    vi <- pmin(pmax(if (length(v) == 1L) rep(v, sum(ok)) else v[ok], 1e-15), 1 - 1e-15)
    qu <- qnorm(ui); qv <- qnorm(vi)
    r2 <- 1 - rho^2
    out[ok] <- exp(-(qu^2 - 2 * rho * qu * qv + qv^2) / (2 * r2)) / (2 * pi * sqrt(r2))
  }
  out
}

#' dC/dtheta for Kimeldorf-Sampson (Clayton) copula; 0 at boundary (u=0 or v=0)
#' @keywords internal
#' @noRd
kimeldorf_dtheta <- function(u, v, theta) {
  out <- numeric(max(length(u), length(v)))
  ok  <- (u > 0) & (v > 0)
  if (any(ok)) {
    ui <- if (length(u) == 1L) rep(u, sum(ok)) else u[ok]
    vi <- if (length(v) == 1L) rep(v, sum(ok)) else v[ok]
    if (theta < 1e-10) {
      out[ok] <- ui * vi * (log(ui) + log(vi)) / 2
    } else {
      inner      <- pmax(ui^(-theta) + vi^(-theta) - 1, 1e-300)
      C_ok       <- inner^(-1 / theta)
      d_inner_dt <- -(ui^(-theta) * log(ui) + vi^(-theta) * log(vi))
      out[ok]    <- C_ok * (log(inner) / theta^2 - d_inner_dt / (theta * inner))
    }
  }
  out
}

.cop_dtheta <- function(u, v, theta, family) {
  switch(family,
    frank     = frank_dtheta(u, v, theta),
    normal    = normal_drho(u, v, theta),
    kimeldorf = kimeldorf_dtheta(u, v, theta)
  )
}

# ---- Kendall's tau ----

#' Kendall's tau for Frank copula (numerical via Debye function)
#' @keywords internal
#' @noRd
frank_tau <- function(theta) {
  if (abs(theta) < 1e-10) return(0)
  debye1 <- function(th) {
    if (abs(th) < 1e-10) return(1)
    integrate(function(t) t / (exp(t) - 1), 0, th, rel.tol = 1e-8)$value / th
  }
  1 - 4 / theta * (1 - debye1(theta))
}

#' Kendall's tau for Gaussian copula
#' @keywords internal
#' @noRd
normal_tau <- function(rho) (2 / pi) * asin(rho)

#' Kendall's tau for Kimeldorf-Sampson (Clayton) copula
#' @keywords internal
#' @noRd
kimeldorf_tau <- function(theta) theta / (theta + 2)

#' Copula tau and its derivative wrt z_theta (for delta-method SE)
#' @keywords internal
#' @noRd
copula_tau_and_deriv <- function(family, z_theta) {
  if (family == "frank") {
    theta   <- z_theta
    tau     <- frank_tau(theta)
    eps     <- 1e-6
    dtau_dz <- (frank_tau(theta + eps) - frank_tau(theta - eps)) / (2 * eps)
  } else if (family == "normal") {
    rho      <- tanh(z_theta)
    tau      <- normal_tau(rho)
    dtau_dz  <- (2 / pi) / sqrt(1 - rho^2) * (1 - rho^2)
  } else {
    theta    <- exp(z_theta)
    tau      <- kimeldorf_tau(theta)
    dtau_dz  <- 2 / (theta + 2)^2 * theta
  }
  list(tau = tau, dtau_dz = dtau_dz)
}

#' Recover native copula parameter from z_theta
#' @keywords internal
#' @noRd
z_to_native <- function(family, z) {
  switch(family, frank = z, normal = tanh(z), kimeldorf = exp(z))
}

#' d(native)/dz
#' @keywords internal
#' @noRd
dnative_dz <- function(family, z) {
  switch(family, frank = 1, normal = { rho <- tanh(z); 1 - rho^2 }, kimeldorf = exp(z))
}
