# Cross-engine agreement checks: the TMB engine against the Rcpp engine.
#
# Before the merge these lived in rpbnb.tmb and compared against `rpbnb` as a
# Suggests dependency, which made the result a function of whether an optional
# package happened to be installed. Both engines now ship in one package, so the
# comparison is intra-package and unconditional.
#
# The data deliberately comes from inst/extdata/rwm1984_bnb.csv rather than from
# rwm1984_clean. That dataset is not schema-stable across releases: in rpbnb
# 0.2.1 it loaded as a 3,874-by-1 data frame whose single column was named for
# the comma-joined header, so
#   Variable(s) not found in data: docvis, outwork, hospvis
# aborted both fits before either likelihood was compared. Feeding both engines
# a file we control keeps this a real comparison and makes the schema our own
# problem -- hence the explicit column assertion below.
rwm1984_bnb_data <- function() {
  path <- system.file("extdata", "rwm1984_bnb.csv",
                      package = "rpbnb", mustWork = TRUE)
  d <- utils::read.csv(path)
  needed <- c("docvis", "hospvis", "outwork")
  missing <- setdiff(needed, names(d))
  if (length(missing)) {
    stop("rwm1984_bnb.csv is missing required column(s): ",
         paste(missing, collapse = ", "))
  }
  d
}

test_that("TMB engine logLik matches fit_bnb (independence)", {
  skip_on_cran()
  d <- rwm1984_bnb_data()

  fit_cpp <- fit_bnb(docvis ~ outwork, hospvis ~ outwork,
                     data = d, dependence = "independence")
  fit_tmb <- fit_rpbnb_tmb(docvis ~ outwork, hospvis ~ outwork,
                           data = d, dependence = "independence",
                           draws = 1)  # no random coefficients, draws = 1 is fine

  expect_lt(abs(logLik(fit_cpp) - logLik(fit_tmb)), 0.01)
})

test_that("TMB engine logLik matches fit_bnb (famoye)", {
  skip_on_cran()
  d <- rwm1984_bnb_data()

  fit_cpp <- fit_bnb(docvis ~ outwork, hospvis ~ outwork,
                     data = d, dependence = "famoye",
                     control = rpbnb_control(compute_se = FALSE))
  fit_tmb <- fit_rpbnb_tmb(docvis ~ outwork, hospvis ~ outwork,
                           data = d, dependence = "famoye")

  # Looser tolerance: different optimizers (maxLik BFGS vs nlminb).
  expect_lt(abs(logLik(fit_cpp) - logLik(fit_tmb)), 0.1)
})

test_that("both random-parameter engines agree on the same model", {
  skip_on_cran()
  skip_slow()
  d <- rwm1984_bnb_data()

  args <- list(formula_1 = docvis ~ outwork, formula_2 = hospvis ~ outwork,
               data = d, random_1 = "outwork", draws = 200, seed = 4321)

  fit_cpp <- do.call(fit_rpbnb, c(args,
    list(control = rpbnb_control(compute_se = FALSE, print_level = 0))))
  fit_tmb <- do.call(fit_rpbnb_tmb, c(args, list(inference = "none")))

  # Tolerances are set from a measured run, not from a round number. The two
  # engines use different optimizers (maxLik BFGS vs nlminb + restart polish)
  # AND different Halton generators (halton_uniform() via randtoolbox vs
  # .tmb_halton_uniform()'s radical inverse), so exact agreement is not
  # expected and a tight tolerance would just be flaky. If either of those
  # changes, re-measure rather than widening blindly.
  expect_lt(abs(as.numeric(logLik(fit_cpp)) - as.numeric(logLik(fit_tmb))), 2)

  shared <- intersect(names(coef(fit_cpp)), names(coef(fit_tmb)))
  expect_gt(length(shared), 0L)
  expect_lt(max(abs(coef(fit_cpp)[shared] - coef(fit_tmb)[shared])), 0.2)
})

test_that("the Famoye admissible interval is one model property across engines", {
  skip_on_cran()
  skip_slow()
  # The admissible lambda set belongs to the model, not to whatever rule
  # approximates its random-coefficient integral. Both engines must therefore
  # agree, and neither may move with the draw count. Before this was shared, the
  # Rcpp engine reduced bounds over its optimization draws and returned
  # [-1.4242, 1.0995] at 50 draws against [-1.0796, 1.0668] at 800, while the
  # TMB engine had already switched to the support bound -- so the DEFAULT
  # engine still had a draw-dependent parameter space.
  sim <- simulate_rpbnb(
    n = 300,
    beta1 = c("(Intercept)" = 0.2, x1 = 0.4),
    beta2 = c("(Intercept)" = 0.1, x1 = -0.3),
    random_1 = list(x1 = list(sd = 0.3)),
    random_2 = list(x1 = list(sd = 0.3)),
    dispersion = c(m1 = 0.5, m2 = 0.5), seed = 77
  )
  ctl <- rpbnb_control(compute_se = FALSE, print_level = 0)

  cpp_50  <- fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim$data, random_1 = "x1",
                       random_2 = "x1", draws = 50,  seed = 3, control = ctl)
  cpp_400 <- fit_rpbnb(y1 ~ x1, y2 ~ x1, data = sim$data, random_1 = "x1",
                       random_2 = "x1", draws = 400, seed = 3, control = ctl)
  tmb_50  <- fit_rpbnb_tmb(y1 ~ x1, y2 ~ x1, data = sim$data, random_1 = "x1",
                           random_2 = "x1", dependence = "famoye",
                           draws = 50L, seed = 3, inference = "none",
                           control = rpbnb_tmb_control(iterlim = 20L))

  # Two normal coefficients with non-zero loadings: c1 and c2 each span (0,1),
  # so the support bound is exactly [-1, 1] and parameter-independent.
  expect_equal(unname(cpp_50$bounds), c(-1, 1))
  expect_equal(unname(cpp_400$bounds), c(-1, 1))
  expect_equal(unname(tmb_50$lambda_bounds), c(-1, 1))

  # Draw-independence within an engine, and agreement across engines.
  expect_equal(cpp_50$bounds, cpp_400$bounds)
  expect_equal(unname(cpp_50$bounds), unname(tmb_50$lambda_bounds))
})
