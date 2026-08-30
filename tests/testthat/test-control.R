test_that("rpbnb_control returns documented defaults", {
  ctl <- rpbnb_control()
  expect_s3_class(ctl, "rpbnb_control")
  expect_equal(ctl$method, "BFGS")
  expect_equal(ctl$draws_hessian, 100L)
  expect_equal(ctl$n_cores, 1L)
  expect_true(ctl$compute_se)
  expect_equal(ctl$hessian, "numeric")
  # TMB-side fields are on the same object since the two constructors merged.
  expect_equal(ctl$gradtol, 1e-5)
  expect_equal(ctl$restarts, 10L)
  expect_false(ctl$parallel_tape)
  # iterlim/print_level are the two fields whose default differs by estimator,
  # so they stay NULL until .resolve_control() knows which one is running.
  expect_null(ctl$iterlim)
  expect_null(ctl$print_level)
  expect_equal(rpbnb:::.resolve_control(ctl, "classic")$iterlim, 300L)
  expect_equal(rpbnb:::.resolve_control(ctl, "tmb")$iterlim, 500L)
})

test_that("rpbnb_control accepts hessian = 'analytic' and validates it", {
  expect_equal(rpbnb_control(hessian = "analytic")$hessian, "analytic")
  expect_equal(rpbnb_control(hessian = "numeric")$hessian, "numeric")
  expect_error(rpbnb_control(hessian = "bogus"), "hessian")
})

test_that("rpbnb_control validates inputs", {
  expect_error(rpbnb_control(n_cores = 0), "n_cores")
  expect_error(rpbnb_control(method = "NOPE"), "method")
  expect_error(rpbnb_control(draws_hessian = -1), "draws_hessian")
  expect_error(rpbnb_control(iterlim = 0), "iterlim")
  expect_error(rpbnb_control(reltol = 0), "reltol")
})

test_that("rpbnb_control rejects unimplemented optimizer methods", {
  # Only BFGS is wired through to the fitters; NR/BHHH/NM were advertised but
  # ignored, so they are no longer accepted (P2b).
  expect_error(rpbnb_control(method = "NR"), "BFGS")
  expect_error(rpbnb_control(method = "BHHH"), "BFGS")
  expect_error(rpbnb_control(method = "NM"), "BFGS")
  expect_equal(rpbnb_control(method = "BFGS")$method, "BFGS")
})

test_that("rpbnb_control overrides take effect", {
  ctl <- rpbnb_control(iterlim = 50, compute_se = FALSE)
  expect_equal(ctl$iterlim, 50L)
  expect_false(ctl$compute_se)
})

test_that("rpbnb_tmb_control() is an alias returning the same object", {
  expect_s3_class(rpbnb_tmb_control(), "rpbnb_control")
  expect_s3_class(rpbnb_control(), "rpbnb_tmb_control")
  expect_identical(unclass(rpbnb_tmb_control(n_cores = 3L, gradtol = 1e-7)),
                   unclass(rpbnb_control(n_cores = 3L, gradtol = 1e-7)))
  # The alias must forward only what the CALLER wrote, not its own signature:
  # otherwise every field would be reported as an ignored setting downstream.
  expect_length(attr(rpbnb_tmb_control(), "supplied"), 0L)
  expect_setequal(attr(rpbnb_tmb_control(n_cores = 3L), "supplied"), "n_cores")
  # Validation still fires through the alias.
  expect_error(rpbnb_tmb_control(gradtol = 0), "gradtol")
  expect_error(rpbnb_tmb_control(max_threads = 0), "max_threads")
})

test_that("the ignored-settings note names the estimator and the fields", {
  fake <- list(control_ignored = c("se_method", "draws_hessian"),
               control_engine = "tmb")
  out <- capture.output(rpbnb:::.print_control_ignored(fake))
  expect_length(out, 1L)
  expect_match(out, "Control settings ignored")
  expect_match(out, "TMB engine")
  expect_match(out, "se_method, draws_hessian")
  # Nothing ignored -> nothing printed, which is the common case.
  expect_length(
    capture.output(rpbnb:::.print_control_ignored(
      list(control_ignored = character(0), control_engine = "classic"))),
    0L)
})

test_that("print() shows only the fields the named estimator reads", {
  ctl <- rpbnb_control(n_cores = 4L, gradtol = 1e-5, tape_chunks = 4L)

  tmb <- capture.output(print(ctl, engine = "tmb", method = "sml"))
  # The header states the estimator outright, optimizer included: `method` is
  # two different things in this package, and naming nlminb here is what keeps
  # a maxLik "BFGS" from being read as the TMB engine's optimizer.
  expect_true(any(grepl("^  engine +tmb", tmb)))
  expect_true(any(grepl("^  method +sml   \\(estimator\\)", tmb)))
  expect_true(any(grepl("^  optimizer +nlminb", tmb)))
  # The maxLik-only knobs are absent entirely -- printing a `method BFGS` under
  # a TMB fit is the confusion this narrowing exists to remove. (The header's
  # own `method sml` row is the estimator, not control$method.)
  expect_false(any(grepl("^  method +BFGS", tmb)))
  expect_false(any(grepl("^  se_method ", tmb)))
  expect_false(any(grepl("^  hess_eps ", tmb)))
  expect_true(any(grepl("^  gradtol ", tmb)))
  expect_true(any(grepl("^  tape_chunks ", tmb)))

  # The classic engine reads `method` (its maxLik optimizer) and not the TMB
  # tape knobs, and has no sml/laplace estimator row.
  classic <- capture.output(print(ctl, engine = "classic"))
  expect_true(any(grepl("^  engine +classic", classic)))
  expect_true(any(grepl("^  method +BFGS   \\(maxLik optimizer\\)", classic)))
  expect_true(any(grepl("^  se_method ", classic)))
  expect_false(any(grepl("^  optimizer ", classic)))

  # An unresolved object has no estimator to narrow by, so it prints in full.
  full <- capture.output(print(ctl))
  expect_false(any(grepl("^  engine ", full)))
  expect_true(any(grepl("no estimator named", full)))
  for (f in rpbnb:::.CONTROL_ALL_FIELDS) {
    expect_true(any(grepl(paste0("^  ", f, " "), full)))
  }
})

test_that("print() never hides a supplied setting the estimator ignores", {
  # se_method is maxLik-only; supplying it and printing for the TMB engine must
  # show it AND mark it, never drop it -- a silently absent setting would look
  # honored, which is the hazard the whole ignored-report design guards against.
  ctl <- rpbnb_control(se_method = "opg", n_cores = 4L)
  out <- capture.output(print(ctl, engine = "tmb", method = "sml"))
  se_row <- grep("^  se_method ", out, value = TRUE)
  expect_length(se_row, 1L)
  expect_match(se_row, "(ignored here)", fixed = TRUE)

  # tape_chunks is TMB-only but METHOD-dependent: read under sml, not under
  # laplace (which has no draw dimension to chunk).
  ctl2 <- rpbnb_control(tape_chunks = 4L)
  sml <- grep("^  tape_chunks ", capture.output(
    print(ctl2, engine = "tmb", method = "sml")), value = TRUE)
  lap <- grep("^  tape_chunks ", capture.output(
    print(ctl2, engine = "tmb", method = "laplace")), value = TRUE)
  expect_false(grepl("ignored here", sml, fixed = TRUE))
  expect_true(grepl("(ignored here)", lap, fixed = TRUE))

  expect_error(print(ctl, engine = "tmb", method = "bogus"), "should be one of")
  expect_error(print(ctl, engine = "nosuchengine"), "should be one of")

  # sml/laplace belongs to the TMB fitter alone; asking for it under another
  # engine is a mistaken mental model, so it is named rather than rendered.
  expect_warning(out <- capture.output(print(ctl, engine = "classic",
                                             method = "sml")),
                 "TMB-engine argument")
  expect_false(any(grepl("\\(estimator\\)", out)))
})

test_that("print(draws =) reports the simulation size and the chunk layout", {
  ctl <- rpbnb_control(tape_chunks = 10L, n_cores = 4L)

  # Under SML `draws` defines the likelihood, and (draws, tape_chunks) together
  # set the tape's draw dimension -- the quantity peak memory scales with.
  sml <- capture.output(print(ctl, engine = "tmb", method = "sml", draws = 1000))
  expect_true(any(grepl("^  draws +1000   \\(simulated likelihood\\)", sml)))
  expect_true(any(grepl("^  tape_chunks +10   \\(100 draws per chunk\\)", sml)))

  # Under Laplace the latent is integrated out analytically: `draws` leaves the
  # likelihood, and tape_chunks is not read at all.
  lap <- capture.output(print(ctl, engine = "tmb", method = "laplace",
                              draws = 1000))
  expect_true(any(grepl("Halton grid only", lap, fixed = TRUE)))
  expect_true(any(grepl("^  tape_chunks +10   \\(ignored here\\)", lap)))

  # The classic engine simulates too, but has no tape to chunk.
  classic <- capture.output(print(ctl, engine = "classic", draws = 500))
  expect_true(any(grepl("^  draws +500   \\(simulated likelihood\\)", classic)))

  # Omitting draws leaves the printout exactly as it was.
  bare <- capture.output(print(ctl, engine = "tmb", method = "sml"))
  expect_false(any(grepl("^  draws ", bare)))
  expect_true(any(grepl("^  tape_chunks +10$", bare)))

  expect_error(print(ctl, engine = "tmb", draws = -5), "positive whole number")
  expect_error(print(ctl, engine = "tmb", draws = 2.5), "positive whole number")
  # fit_bnb() has fixed coefficients: no integral, so no draws to take.
  expect_warning(bnb <- capture.output(print(ctl, engine = "bnb", draws = 100)),
                 "does not apply to engine")
  expect_false(any(grepl("^  draws ", bnb)))
})

test_that("a resolved control prints for the estimator it was resolved for", {
  ctl <- rpbnb:::.resolve_control(rpbnb_control(n_cores = 4L), "tmb")
  out <- capture.output(print(ctl))
  expect_true(any(grepl("^  engine +tmb", out)))
  expect_true(any(grepl("^  optimizer +nlminb", out)))
  expect_false(any(grepl("^  method +BFGS", out)))
  # An explicit engine argument overrides the attribute.
  expect_true(any(grepl("^  method +BFGS",
                        capture.output(print(ctl, engine = "classic")))))
})

test_that("a classic fit records and prints the settings it did not read", {
  d <- data.frame(y1 = c(0L, 1L, 2L, 1L, 0L, 3L, 1L, 2L),
                  y2 = c(1L, 0L, 2L, 1L, 3L, 1L, 0L, 2L),
                  x1 = c(-1, 0, 1, 0.5, -0.5, 0.2, -0.2, 0.8))
  fit <- fit_bnb(y1 ~ x1, y2 ~ x1, data = d, dependence = "independence",
                 control = rpbnb_control(print_level = 0L, gradtol = 1e-7,
                                         se_method = "opg"))
  expect_setequal(fit$control_ignored, c("gradtol", "se_method"))
  expect_identical(fit$control_engine, "bnb")
  expect_true(any(grepl("Control settings ignored",
                        capture.output(print(fit)), fixed = TRUE)))
  expect_true(any(grepl("Control settings ignored",
                        capture.output(print(summary(fit))), fixed = TRUE)))
})
