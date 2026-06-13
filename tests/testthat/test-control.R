test_that("rpbnb_control returns documented defaults", {
  ctl <- rpbnb_control()
  expect_s3_class(ctl, "rpbnb_control")
  expect_equal(ctl$method, "BFGS")
  expect_equal(ctl$iterlim, 300L)
  expect_equal(ctl$draws_hessian, 100L)
  expect_equal(ctl$n_cores, 1L)
  expect_true(ctl$compute_se)
})

test_that("rpbnb_control validates inputs", {
  expect_error(rpbnb_control(n_cores = 0), "n_cores")
  expect_error(rpbnb_control(method = "NOPE"), "method")
  expect_error(rpbnb_control(draws_hessian = -1), "draws_hessian")
  expect_error(rpbnb_control(iterlim = 0), "iterlim")
  expect_error(rpbnb_control(reltol = 0), "reltol")
})

test_that("rpbnb_control overrides take effect", {
  ctl <- rpbnb_control(method = "NR", iterlim = 50, compute_se = FALSE)
  expect_equal(ctl$method, "NR")
  expect_equal(ctl$iterlim, 50L)
  expect_false(ctl$compute_se)
})
