skip_if_not(exists("pbivnorm_cpp", mode = "function"), "copula C++ not compiled")
skip_if_not_installed("pbivnorm")

test_that("pbivnorm_cpp matches pbivnorm over a grid", {
  g  <- seq(-3, 3, by = 0.5)
  gr <- expand.grid(h = g, k = g)
  for (rho in c(-0.9, -0.5, -0.2, 0, 0.2, 0.5, 0.9)) {
    ref <- pbivnorm::pbivnorm(gr$h, gr$k, rho)
    got <- pbivnorm_cpp(gr$h, gr$k, rho)
    expect_equal(got, ref, tolerance = 1e-9, info = paste("rho =", rho))
  }
})

cpp_grad_case <- function(rc1, rc2, par, fam, d1 = NULL, d2 = NULL,
                          n = 60, R = 96, seed = 5) {
  set.seed(seed)
  x1 <- rnorm(n); x2 <- rnorm(n)
  X1 <- cbind(`(Intercept)` = 1, x1 = x1, x2 = x2); X2 <- X1
  y1 <- rnbinom(n, mu = exp(0.4 + 0.2 * x1), size = 2)
  y2 <- rnbinom(n, mu = exp(0.3 - 0.1 * x2), size = 2)
  ri1 <- match(rc1, colnames(X1)); ri2 <- match(rc2, colnames(X2))
  q1 <- length(ri1); q2 <- length(ri2)
  XR1 <- if (q1) X1[, ri1, drop = FALSE] else NULL
  XR2 <- if (q2) X2[, ri2, drop = FALSE] else NULL
  if (is.null(d1)) d1 <- rep("normal", q1); if (is.null(d2)) d2 <- rep("normal", q2)
  s1 <- rep(1, q1); s2 <- rep(1, q2)
  set.seed(77); Z <- halton_uniform(R, q1 + q2, burn = 40)
  Z1 <- if (q1) Z[, seq_len(q1), drop = FALSE] else matrix(0, R, 0)
  Z2 <- if (q2) Z[, (q1 + 1):(q1 + q2), drop = FALSE] else matrix(0, R, 0)
  r_r <- bnbr_rp_copula_ll_grad(par, y1, y2, X1, X2, XR1, XR2, ri1, ri2, Z1, Z2,
                                fam, d1, d2, s1, s2, want_scores = TRUE)
  r_c <- bnbr_rp_copula_ll_grad_cpp(par, y1, y2, X1, X2, XR1, XR2, ri1, ri2, Z1, Z2,
                                    fam, d1, d2, s1, s2, want_scores = TRUE, n_threads = 2L)
  list(r = r_r, c = r_c)
}

test_that("C++ copula value+gradient+scores match the R oracle (all families)", {
  par <- c(0.4, 0.2, 0.0, 0.3, 0.0, -0.1, log(0.25), log(0.2), log(0.5), log(0.6), 0.3)
  for (fam in c("frank", "normal", "kimeldorf")) {
    o <- cpp_grad_case(c("x1"), c("x2"), par, fam)
    expect_equal(as.numeric(o$c), as.numeric(o$r), tolerance = 1e-7, info = fam)
    expect_equal(attr(o$c, "gradient"), attr(o$r, "gradient"), tolerance = 1e-7,
                 ignore_attr = TRUE, info = fam)
    expect_equal(attr(o$c, "scores"), attr(o$r, "scores"), tolerance = 1e-7,
                 ignore_attr = TRUE, info = fam)
  }
})

test_that("C++ copula matches R oracle (q=0 and lognormal)", {
  o0 <- cpp_grad_case(character(0), character(0),
                      c(0.4,0.2,0.0,0.3,0.0,-0.1,log(0.5),log(0.6),0.2), "normal")
  expect_equal(attr(o0$c,"gradient"), attr(o0$r,"gradient"), tolerance=1e-7, ignore_attr=TRUE)
  ol <- cpp_grad_case(c("x1"), c("x1"),
                      c(0.4,0.2,0.0,0.3,0.0,-0.1,log(0.2),log(0.2),log(0.5),log(0.6),0.25),
                      "frank", d1="lognormal", d2="normal")
  expect_equal(attr(ol$c,"gradient"), attr(ol$r,"gradient"), tolerance=1e-7, ignore_attr=TRUE)
})

test_that("fit_rpbnb copula path uses C++ and recovers the parameter", {
  skip_if_not(rpbnb_copula_cpp_available(), "copula C++ not compiled")
  sim <- simulate_rpbnb_copula(
    n = 800, beta1 = c("(Intercept)" = 0.3, x1 = 0.2),
    beta2 = c("(Intercept)" = 0.2, x1 = -0.1),
    random_1 = list(x1 = list(sd = 0.3)),
    dispersion = c(m1 = 0.5, m2 = 0.6),
    copula = copula("frank", par = 4), seed = 7)
  fit <- fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim$data,
                   random_1 = "x1", random_2 = "x1", dependence = copula("frank"),
                   draws = 120, seed = 1,
                   control = rpbnb_control(se_method = "opg", n_cores = 2))
  expect_equal(fit$coef[["z_theta"]], 4, tolerance = 1.2)  # frank theta native scale
  expect_true(is.finite(fit$se[["z_theta"]]))
})
