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
