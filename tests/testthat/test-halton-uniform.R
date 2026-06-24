test_that("halton_uniform returns rotated uniforms in (0,1)", {
  set.seed(11)
  U <- halton_uniform(50, 3, burn = 20)
  expect_equal(dim(U), c(50, 3))
  expect_true(all(U > 0 & U < 1))
})

test_that("halton_uniform handles zero dimension", {
  expect_equal(dim(halton_uniform(10, 0)), c(10L, 0L))
})

test_that("halton_normal equals qnorm of halton_uniform with same seed", {
  set.seed(7); Z <- halton_normal(40, 2, burn = 30)
  set.seed(7); U <- halton_uniform(40, 2, burn = 30)
  expect_equal(Z, stats::qnorm(U))
})

test_that("halton_uniform is reproducible for a fixed seed", {
  set.seed(3); a <- halton_uniform(25, 2)
  set.seed(3); b <- halton_uniform(25, 2)
  expect_identical(a, b)
})
