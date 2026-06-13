make_diag_fit <- function() {
  set.seed(31)
  d <- data.frame(x = rnorm(500), z = rbinom(500, 1, 0.4))
  d$y1 <- rnbinom(500, size = 2, mu = exp(0.3 + 0.2 * d$x + 0.1 * d$z))
  d$y2 <- rnbinom(500, size = 2, mu = exp(0.1 - 0.1 * d$x))
  fit_bnb(y1 ~ x + z, y2 ~ x, data = d, dependence = "famoye")
}

test_that("bnb_gof returns finite AIC/BIC and pseudo-R2 in [0,1]", {
  g <- bnb_gof(make_diag_fit(), print_output = FALSE)
  expect_true(is.finite(g$AIC) && is.finite(g$BIC))
  expect_true(all(g$pseudoR2 >= 0 & g$pseudoR2 <= 1, na.rm = TRUE))
})

test_that("bnb_marginal_effects returns a row per requested variable", {
  me <- bnb_marginal_effects(make_diag_fit(), which = "y1", type = "AME",
                             print_output = FALSE)
  expect_true(all(c("x", "z") %in% me$Name))
  expect_true(all(is.finite(me$Estimate)))
})

test_that("bnb_elasticities runs for both margins", {
  el <- bnb_elasticities(make_diag_fit(), which = "both", type = "AME",
                         print_output = FALSE)
  expect_true(is.list(el) && all(c("y1", "y2") %in% names(el)))
})
