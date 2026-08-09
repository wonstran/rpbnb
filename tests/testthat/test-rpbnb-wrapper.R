# rpbnb() is a dispatcher, not an estimator. Everything here is validation --
# no model is fitted, so this stays in the fast tier.

toy <- function(n = 30L) {
  set.seed(11)
  data.frame(y1 = rnbinom(n, mu = 2, size = 2),
             y2 = rnbinom(n, mu = 2, size = 2),
             x1 = rnorm(n))
}

test_that("engine-specific arguments are rejected by the other engine", {
  d <- toy()
  # tmb-only arguments under engine = "classic"
  expect_error(
    rpbnb(y1 ~ x1, y2 ~ x1, data = d, engine = "classic", inference = "none"),
    'only accepted by engine = "tmb"'
  )
  expect_error(
    rpbnb(y1 ~ x1, y2 ~ x1, data = d, engine = "classic", keep = "full"),
    'only accepted by engine = "tmb"'
  )
  # classic-only arguments under engine = "tmb"
  expect_error(
    rpbnb(y1 ~ x1, y2 ~ x1, data = d, engine = "tmb", draw_type = "halton"),
    'only accepted by engine = "classic"'
  )
  expect_error(
    rpbnb(y1 ~ x1, y2 ~ x1, data = d, engine = "tmb", .opt_draws = 10),
    'only accepted by engine = "classic"'
  )
})

test_that("unknown argument names are an error, not a silent no-op", {
  d <- toy()
  # The whole point of the wrapper: a typo must not vanish into `...`.
  expect_error(
    rpbnb(y1 ~ x1, y2 ~ x1, data = d, engine = "classic", drawtype = "halton"),
    "not an argument of either engine"
  )
  expect_error(
    rpbnb(y1 ~ x1, y2 ~ x1, data = d, engine = "tmb", scaling = list(x1 = 1)),
    "not an argument of either engine"
  )
})

test_that("unnamed extra arguments are rejected", {
  d <- toy()
  # An unnamed argument only reaches `...` once every preceding formal is
  # supplied by name -- otherwise R positionally binds it to the first unfilled
  # formal (here `random_1`, which then fails with "random name(s) not found").
  # So fill them all, or this asserts the wrong error.
  expect_error(
    rpbnb(formula_1 = y1 ~ x1, formula_2 = y2 ~ x1, data = d, engine = "classic",
          random_1 = NULL, random_2 = NULL, draws = 400, seed = 1234,
          start = NULL, dependence = "famoye",
          poisson_1 = FALSE, poisson_2 = FALSE,
          standardize = FALSE, continuous_vars = NULL,
          boundary_tests = FALSE, control = NULL,
          "halton"),
    "must be named"
  )
})

test_that("control objects are engine-typed and never translated", {
  d <- toy()
  expect_error(
    rpbnb(y1 ~ x1, y2 ~ x1, data = d, engine = "classic",
          control = rpbnb_tmb_control()),
    "needs a `rpbnb_control` object"
  )
  expect_error(
    rpbnb(y1 ~ x1, y2 ~ x1, data = d, engine = "tmb",
          control = rpbnb_control()),
    "needs a `rpbnb_tmb_control` object"
  )
})

test_that("dependence structures unsupported by an engine are rejected", {
  d <- toy()
  expect_error(
    rpbnb(y1 ~ x1, y2 ~ x1, data = d, engine = "classic",
          dependence = "independence"),
    'engine = "classic" does not implement'
  )
  # copula(par=) is simulation-only; rpbnb() rejects it for BOTH engines even
  # though fit_rpbnb() would accept and ignore it.
  for (eng in c("classic", "tmb")) {
    expect_error(
      rpbnb(y1 ~ x1, y2 ~ x1, data = d, engine = eng,
            dependence = copula("frank", par = 2)),
      "simulation-only"
    )
  }
})

test_that("rpbnb() returns exactly what the underlying fitter returns", {
  skip_on_cran()
  skip_slow()
  d <- toy(60L)
  ctl <- rpbnb_control(compute_se = FALSE, print_level = 0)

  a <- rpbnb(y1 ~ x1, y2 ~ x1, data = d, engine = "classic",
             draws = 30, seed = 7, control = ctl)
  b <- fit_rpbnb(y1 ~ x1, y2 ~ x1, data = d,
                 draws = 30, seed = 7, control = ctl)
  expect_s3_class(a, "rpbnb_fit")
  expect_equal(coef(a), coef(b))
  expect_equal(as.numeric(logLik(a)), as.numeric(logLik(b)))

  p <- rpbnb(y1 ~ x1, y2 ~ x1, data = d, engine = "tmb",
             draws = 30, seed = 7, inference = "none")
  q <- fit_rpbnb_tmb(y1 ~ x1, y2 ~ x1, data = d,
                     draws = 30, seed = 7, inference = "none")
  expect_s3_class(p, "rpbnb_tmb_fit")
  expect_equal(coef(p), coef(q))
  expect_equal(as.numeric(logLik(p)), as.numeric(logLik(q)))
})
