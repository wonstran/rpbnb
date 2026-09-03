# ===============================
# Bivariate Negative Binomial (Famoye) — 3D joint pmf surfaces
# ===============================

# ---- Packages ----
req <- c("dplyr", "plotly", "patchwork")  # patchwork used only to combine htmlwidgets gracefully
to_install <- req[!req %in% installed.packages()[, "Package"]]
if (length(to_install)) install.packages(to_install, quiet = TRUE)
library(dplyr)
library(plotly)

# ---- Core model helpers ----
d_const <- 1 - exp(-1)

c_func <- function(mu, m) {
  # c = (1 + d * mu * m)^(-1/m)
  (1 + d_const * mu * m)^(-1/m)
}

dnbinom_mu_m <- function(y, mu, m) {
  # NB with mean mu and dispersion m (size = 1/m; prob = 1/(1 + m*mu))
  size <- 1/m
  prob <- 1 / (1 + m * mu)
  dnbinom(y, size = size, prob = prob)
}

bnbr_joint_pmf_point <- function(y1, y2, mu1, mu2, m1, m2, lambda) {
  c1 <- c_func(mu1, m1)
  c2 <- c_func(mu2, m2)
  dnbinom_mu_m(y1, mu1, m1) * dnbinom_mu_m(y2, mu2, m2) *
    (1 + lambda * (exp(-y1) - c1) * (exp(-y2) - c2))
}

# ---- Grid-safe lambda bounds so 1 + lambda*(...)*(...) >= 0 over the plotting grid ----
lambda_bounds_on_grid <- function(mu1, mu2, m1, m2, y_max1, y_max2) {
  c1 <- c_func(mu1, m1); c2 <- c_func(mu2, m2)
  y1 <- 0:y_max1; y2 <- 0:y_max2
  P  <- outer(exp(-y1) - c1, exp(-y2) - c2, FUN = "*")
  max_pos <- ifelse(any(P > 0), max(P[P > 0]), NA_real_)
  min_neg <- ifelse(any(P < 0), min(P[P < 0]), NA_real_)
  c(lower = if (is.na(max_pos)) -Inf else -1 / max_pos,
    upper = if (is.na(min_neg))  Inf else -1 / min_neg)
}

clip_lambda_to_grid <- function(lambda, mu1, mu2, m1, m2, y_max1, y_max2, shrink = 0.98) {
  b <- lambda_bounds_on_grid(mu1, mu2, m1, m2, y_max1, y_max2)
  if (!is.infinite(b["lower"]) && lambda < b["lower"]) return(shrink * b["lower"])
  if (!is.infinite(b["upper"]) && lambda > b["upper"]) return(shrink * b["upper"])
  lambda
}

# ---- Build joint pmf on a lattice ----
bnbr_joint_grid <- function(mu1, mu2, m1, m2, lambda, y_max1 = 30, y_max2 = 30,
                            ensure_nonneg = TRUE) {
  lam <- if (ensure_nonneg) clip_lambda_to_grid(lambda, mu1, mu2, m1, m2, y_max1, y_max2) else lambda
  y1 <- 0:y_max1; y2 <- 0:y_max2
  G  <- expand.grid(y1 = y1, y2 = y2)  # NOTE: y1 varies slowly; y2 fast
  G$p <- mapply(function(a, b) bnbr_joint_pmf_point(a, b, mu1, mu2, m1, m2, lam), G$y1, G$y2)
  G$p[G$p < 0] <- 0  # numerical guard
  attr(G, "lambda_used") <- lam
  G
}

# ---- Convert grid (long) to Z matrix for 3D plotting ----
grid_to_matrix <- function(G) {
  y1 <- sort(unique(G$y1))
  y2 <- sort(unique(G$y2))
  # For each column (fixed y1), collect probabilities across increasing y2
  Z <- sapply(y1, function(a1) G$p[G$y1 == a1][order(G$y2)])
  list(x = y1, y = y2, z = Z)
}

# ---- Plotly surface ----
plot_surface_plotly <- function(G, title = "BNB joint pmf (surface)") {
  M <- grid_to_matrix(G)
  lam_used <- attr(G, "lambda_used")
  s <- sum(G$p)  # mass inside the truncated window
  subtitle <- paste0("lambda used = ", signif(lam_used, 4), " | mass on grid = ", signif(s, 5))
  
  plot_ly(
    x = M$x, y = M$y, z = M$z,
    type = "surface", showscale = TRUE
  ) |>
    layout(
      title = list(text = paste0(title, "<br><sup>", subtitle, "</sup>")),
      scene = list(
        xaxis = list(title = "y1"),
        yaxis = list(title = "y2"),
        zaxis = list(title = "P(Y1=y1, Y2=y2)")
      )
    )
}

# ---- Base graphics (static) alternative using persp ----
plot_surface_persp <- function(G, theta = 40, phi = 25, main = "BNB joint pmf (persp)") {
  M <- grid_to_matrix(G)
  persp(x = M$x, y = M$y, z = M$z,
        theta = theta, phi = phi,
        ticktype = "detailed",
        xlab = "y1", ylab = "y2", zlab = "pmf",
        col = "lightblue", shade = 0.3, main = main)
}

# ===============================
# Demo: plot 3D surfaces for several lambdas
# ===============================
mu1 <- 6;   mu2 <- 10
m1  <- 0.4; m2  <- 0.6
ymax1 <- 30; ymax2 <- 30

lams <- c(-0.6, 0, 0.6)

surfaces <- lapply(lams, function(lam) {
  G <- bnbr_joint_grid(mu1, mu2, m1, m2, lambda = lam, y_max1 = ymax1, y_max2 = ymax2)
  plot_surface_plotly(G, title = sprintf("BNB surface — lambda requested = %.2f", lam))
})

# Display the three interactive surfaces in a single row (scroll horizontally if needed)
# In some R environments, htmlwidgets are best viewed one-by-one:
surfaces[[1]]
surfaces[[2]]
surfaces[[3]]

# --- If you prefer a static base-graphics 3D (persp) view for one lambda:
# G0 <- bnbr_joint_grid(mu1, mu2, m1, m2, lambda = lams[1], y_max1 = ymax1, y_max2 = ymax2)
# plot_surface_persp(G0, main = sprintf("persp view (lambda=%.2f)", lams[1]))
