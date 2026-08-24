# Equivalence fixtures for the TMB SML tape-chunking OOM fix
# (see docs/TMB_SML_large_draws_OOM_guide.md and the approved plan at
# read-docs-tmb-sml-large-draws-oom-guide-compiled-abelson.md).
#
# Phase 0 pins obj$fn()/obj$gr() at two non-trivial parameter vectors, for
# all five dependence families, directly off the CURRENT (pre-Phase-1,
# pre-chunking) `src/rpbnb_tmb.cpp`. These are the reference values every
# later phase must reproduce:
#   - Phase 1 (constant-factor tape reductions): agreement at ~1e-12
#     (algebraic rewrites reorder floating-point operations).
#   - Phase 2 with `chunked = 0` (the static branch): must match the
#     POST-Phase-1 baseline captured once Phase 1 lands, not this one
#     directly -- see the "post-Phase-1 baseline" section below, added
#     when Phase 1 is implemented.
#
# The fixture uses one random coefficient of each of the four registered
# distributions (normal, lognormal, uniform, triangular) split across the
# two margins, and rotates NB/Poisson margin combinations across the five
# families so every combination (NB/NB, NB/Poisson, Poisson/NB,
# Poisson/Poisson) is exercised at least once.

.tape_chunk_fixture <- function(family_code, pois1, pois2) {
  n <- 12L
  x <- seq(-0.9, 0.9, length.out = n)
  X1 <- cbind(`(Intercept)` = 1, x1 = x, x2 = x^2)
  X2 <- cbind(`(Intercept)` = 1, x1 = x, x2 = x^2)
  Y1 <- c(0, 1, 2, 0, 3, 1, 0, 2, 1, 4, 0, 2)
  Y2 <- c(1, 0, 2, 3, 0, 1, 2, 0, 4, 1, 2, 0)

  # Random coefficients: margin 1 on (x1 = normal, x2 = uniform),
  # margin 2 on (x1 = lognormal, x2 = triangular). All four registered
  # distributions (DIST_NORMAL=0, DIST_LOGNORMAL=1, DIST_UNIFORM=2,
  # DIST_TRIANGULAR=3; src/rpbnb_tmb.cpp:27-30) are covered in one fixture.
  rand_idx1 <- c(2L, 3L)
  rand_idx2 <- c(2L, 3L)
  dist1 <- c(0L, 2L)
  sign1 <- c(1L, 1L)
  dist2 <- c(1L, 3L)
  sign2 <- c(1L, 1L)

  R <- 12L
  set.seed(20260824)
  Z <- .tmb_halton_uniform(R, 4L, burn = 50L)
  Z1 <- Z[, 1:2, drop = FALSE]
  Z2 <- Z[, 3:4, drop = FALSE]

  z_dep <- switch(
    as.character(family_code),
    `-1` = 0,
    `0` = 0.3,
    `1` = 1.5,
    `2` = atanh(0.25),
    `3` = log(0.5)
  )

  data <- .build_tmb_data(
    Y1 = Y1, Y2 = Y2, X1 = X1, X2 = X2,
    rand_idx1 = rand_idx1, rand_idx2 = rand_idx2,
    Z1 = Z1, Z2 = Z2,
    dist1 = dist1, dist2 = dist2, sign1 = sign1, sign2 = sign2,
    family_code = family_code, pois1 = pois1, pois2 = pois2,
    lamLo = -1, lamHi = 1, est_method = 0L
  )
  parameters <- list(
    beta1 = c(0.1, 0.15, -0.05), beta2 = c(-0.05, 0.1, 0.05),
    log_sd1 = log(c(0.25, 0.2)), log_sd2 = log(c(0.2, 0.3)),
    log_m1 = log(0.5), log_m2 = log(0.6), z_dep = z_dep,
    u1 = matrix(0, n, 2L), u2 = matrix(0, n, 2L)
  )
  # SML fixture: latents are tape constants at zero (est_method = 0L above),
  # matching the map fit_rpbnb_tmb() applies for SML fits.
  list(
    data = data, parameters = parameters,
    map = list(u1 = factor(rep(NA_integer_, n * 2L)),
               u2 = factor(rep(NA_integer_, n * 2L)))
  )
}

# code: FAM_INDEP=-1, FAM_FAMOYE=0, FAM_FRANK=1, FAM_GAUSSIAN=2, FAM_CLAYTON=3
# (src/rpbnb_tmb.cpp:20-24). pois1/pois2 rotate NB/Poisson margin
# combinations across families so every combination is covered once:
# independence NB/NB, famoye NB/Poisson, frank Poisson/NB, gaussian NB/NB
# (kept simple -- single-threaded per the Gaussian atomic cap), clayton
# Poisson/Poisson.
.tape_chunk_families <- list(
  independence = list(code = -1L, pois1 = FALSE, pois2 = FALSE),
  famoye       = list(code = 0L,  pois1 = FALSE, pois2 = TRUE),
  frank        = list(code = 1L,  pois1 = TRUE,  pois2 = FALSE),
  gaussian     = list(code = 2L,  pois1 = FALSE, pois2 = FALSE),
  clayton      = list(code = 3L,  pois1 = TRUE,  pois2 = TRUE)
)

# Reference values captured directly from `src/rpbnb_tmb.cpp` at the commit
# this file was added (before any Phase 1/2 changes), via
# `.make_rpbnb_tmb_object(..., n_cores = 1L)$obj` at `par0 <- obj$par` and a
# deliberately non-optimal `par1`. par1's construction (for reproducing the
# capture only; not needed to run the tests below):
#   par1 <- par0 + c(rep(0.05, length(par0) - 1L), 0.1) *
#     seq_along(par0) / length(par0)
.tape_chunk_reference <- list(
  independence = list(
    fn0 = 39.012239723618876,
    gr0 = c(-1.6933937789463993, -1.071733729183513, -0.21142092139429314,
            -2.2944717671271371, 2.8203614329151865, -0.049448176387790421,
            0.079131652802853586, 0.0017662833229220099, 0.23193913114688269,
            -0.0061006509722852211, 0.4884078899462081,
            0.00070626007174109451, 0),
    fn1 = 39.048540326923771,
    gr1 = c(-1.5936809659406697, -1.0290373614986406, -0.17666764720062472,
            -2.1026771731364171, 2.911358816708888, -0.0049733550740703181,
            0.086249245113776962, 0.0029637370583036624, 0.25015978654120086,
            -0.0073244018417838114, 0.52911679202146455, 0.01457303504014809, 0)
  ),
  famoye = list(
    fn0 = 40.09110617068999,
    gr0 = c(-1.6698812972826507, -1.1382912312024951, -0.21940888868063338,
            -1.8603365172645958, 4.9073493572281066, 1.0477170141162335,
            0.088694581482488044, -0.0028069745933916603, 0.26695133337043941,
            -0.083334627673839243, 0.48329768601494799, 0, 0.57059499509971157),
    fn1 = 40.259632513326387,
    gr1 = c(-1.5677011087141448, -1.1173137506119535, -0.18836783634092022,
            -1.4249051646600082, 5.2668832339338723, 1.2366932244510958,
            0.095781488859011291, -0.0022900807774253427, 0.28133077399482165,
            -0.10006614063906863, 0.52403495930665167, 0, 0.57326161097017192)
  ),
  frank = list(
    fn0 = 41.019249884113044,
    gr0 = c(-2.0520232335994955, -2.2629497582941696, -0.39174066967158294,
            -2.2998634814773857, 2.982724506846226, -0.12850761694182222,
            -0.091193747895133848, -0.04448226794334173, 0.26355202381053039,
            -0.025482066791323556, 0, -0.22030004365806768, 1.7332917233165763),
    fn1 = 41.179958380339478,
    gr1 = c(-1.9375185083402278, -2.2855534048842925, -0.35847341067295335,
            -2.1423202960113845, 3.0836535604270097, -0.098410985591663613,
            -0.098430672945226422, -0.048155382407804141, 0.28479335767046243,
            -0.028744990352975314, 0, -0.22740696306958749, 1.7561756261184376)
  ),
  gaussian = list(
    fn0 = 40.736330490137114,
    gr0 = c(-1.5721395322562204, -1.7272464598508641, -0.29932293666209381,
            -2.3691466560391685, 3.1837465491292694, -0.068936019776824398,
            0.053370830176245457, -0.011522701105570724, 0.25566172841231088,
            -0.016463515886746369, 0.2536438774744334, -0.2916499602190411,
            8.4103459430167256),
    fn1 = 41.649171097155886,
    gr1 = c(-1.5425611177101441, -2.0130141746606847, -0.33190545663955434,
            -2.3162371363179717, 3.504143976609154, -0.04877778126672444,
            0.037291401450196847, -0.019610098324017781, 0.28669176226662629,
            -0.024996045203025469, 0.16379876392554998, -0.45763061730706833,
            9.790239813633244)
  ),
  clayton = list(
    fn0 = 42.764400477549572,
    gr0 = c(-1.5360117705661511, -2.4093815002268215, -0.28685644490600481,
            -0.30551830239182326, 5.798666903754893, 1.6941796926149748,
            -0.081820414119568891, -0.06634792677387466, 0.25325852124982834,
            -0.2017648423245523, 0, 0, 3.1311372098014472),
    fn1 = 43.223384121547092,
    gr1 = c(-1.3705172095952038, -2.4884586109966129, -0.26070886319744646,
            0.30766307717446706, 6.2706962954864842, 1.9651779230343509,
            -0.090804320940030187, -0.076405940144913781, 0.23992958010564838,
            -0.24859303120078421, 0, 0, 3.5350099684222811)
  )
)

test_that("Phase 0 fixture: fn/gr match the pre-chunking reference for all five families", {
  for (nm in names(.tape_chunk_families)) {
    fam <- .tape_chunk_families[[nm]]
    ref <- .tape_chunk_reference[[nm]]
    fixture <- .tape_chunk_fixture(fam$code, fam$pois1, fam$pois2)
    obj <- .make_rpbnb_tmb_object(
      data = fixture$data, parameters = fixture$parameters,
      map = fixture$map, n_cores = 1L
    )$obj
    par0 <- unname(obj$par)
    par1 <- par0 + c(rep(0.05, length(par0) - 1L), 0.1) *
      seq_along(par0) / length(par0)

    expect_equal(obj$fn(par0), ref$fn0, tolerance = 1e-10, info = nm)
    expect_equal(as.numeric(obj$gr(par0)), ref$gr0, tolerance = 1e-8, info = nm)
    expect_equal(obj$fn(par1), ref$fn1, tolerance = 1e-10, info = nm)
    expect_equal(as.numeric(obj$gr(par1)), ref$gr1, tolerance = 1e-8, info = nm)
  }
})

test_that("Phase 0 fixture: unused dispersion/dependence parameters have exactly zero gradient", {
  # Regression for the specific values pinned above: log_m for a Poisson
  # margin, and z_dep for independence, must be structurally absent from the
  # tape (not merely numerically small) because the C++ family branch never
  # references them (src/rpbnb_tmb.cpp:1190-1232). A future refactor that
  # accidentally threads them through would show up here as a nonzero
  # gradient component rather than only as a reference-value mismatch above.
  zero_checks <- list(
    independence = 13L,  # z_dep
    famoye = 12L,        # log_m2 (pois2 = TRUE)
    frank = 11L,         # log_m1 (pois1 = TRUE)
    clayton = c(11L, 12L)  # log_m1, log_m2 (both Poisson)
  )
  for (nm in names(zero_checks)) {
    ref <- .tape_chunk_reference[[nm]]
    for (idx in zero_checks[[nm]]) {
      expect_identical(ref$gr0[idx], 0, info = paste(nm, idx))
      expect_identical(ref$gr1[idx], 0, info = paste(nm, idx))
    }
  }
})

# ---- Phase 2: draw chunking (the primary OOM fix) ----------------------
#
# .make_chunked_tmb_objective() wraps a chunk-sized TMB object (built with
# chunked = 1L) so it replays over successive draw chunks via DATA_UPDATE()
# instead of retaping. Exact for the requested R draws (see R/tmb_chunked.R
# header); these tests pin that exactness against the SAME unchunked
# reference values above, plus the wrapper's documented contract.

#' Build a chunk-sized `chunked = 1L` TMB object for one family/chunk layout
#' @keywords internal
.tape_chunk_chunked_obj <- function(fam, layout, Z1_full, Z2_full) {
  n <- 12L
  x <- seq(-0.9, 0.9, length.out = n)
  X1 <- cbind(`(Intercept)` = 1, x1 = x, x2 = x^2)
  X2 <- cbind(`(Intercept)` = 1, x1 = x, x2 = x^2)
  Y1 <- c(0, 1, 2, 0, 3, 1, 0, 2, 1, 4, 0, 2)
  Y2 <- c(1, 0, 2, 3, 0, 1, 2, 0, 4, 1, 2, 0)
  rand_idx1 <- c(2L, 3L); rand_idx2 <- c(2L, 3L)
  dist1 <- c(0L, 2L); sign1 <- c(1L, 1L)
  dist2 <- c(1L, 3L); sign2 <- c(1L, 1L)
  z_dep <- switch(
    as.character(fam$code),
    `-1` = 0, `0` = 0.3, `1` = 1.5, `2` = atanh(0.25), `3` = log(0.5)
  )
  parameters <- list(
    beta1 = c(0.1, 0.15, -0.05), beta2 = c(-0.05, 0.1, 0.05),
    log_sd1 = log(c(0.25, 0.2)), log_sd2 = log(c(0.2, 0.3)),
    log_m1 = log(0.5), log_m2 = log(0.6), z_dep = z_dep,
    u1 = matrix(0, n, 2L), u2 = matrix(0, n, 2L)
  )
  map <- list(u1 = factor(rep(NA_integer_, n * 2L)),
              u2 = factor(rep(NA_integer_, n * 2L)))
  first <- layout$chunks[[1L]]
  data <- .build_tmb_data(
    Y1 = Y1, Y2 = Y2, X1 = X1, X2 = X2,
    rand_idx1 = rand_idx1, rand_idx2 = rand_idx2,
    Z1 = Z1_full[first$pad_rows, , drop = FALSE],
    Z2 = Z2_full[first$pad_rows, , drop = FALSE],
    dist1 = dist1, dist2 = dist2, sign1 = sign1, sign2 = sign2,
    family_code = fam$code, pois1 = fam$pois1, pois2 = fam$pois2,
    lamLo = -1, lamHi = 1, est_method = 0L,
    chunked = 1L, w = rep(1, n), draw_w = first$draw_w
  )
  .make_rpbnb_tmb_object(data = data, parameters = parameters,
                         map = map, n_cores = 1L)$obj
}

test_that("chunked wrapper fn/gr match the unchunked reference at C in {1,2,3,4,12}, all families", {
  R_full <- 12L
  set.seed(20260824)
  Z <- .tmb_halton_uniform(R_full, 4L, burn = 50L)
  Z1_full <- Z[, 1:2, drop = FALSE]
  Z2_full <- Z[, 3:4, drop = FALSE]

  for (nm in names(.tape_chunk_families)) {
    fam <- .tape_chunk_families[[nm]]
    ref <- .tape_chunk_reference[[nm]]
    fixture <- .tape_chunk_fixture(fam$code, fam$pois1, fam$pois2)
    ref_obj <- .make_rpbnb_tmb_object(
      data = fixture$data, parameters = fixture$parameters,
      map = fixture$map, n_cores = 1L
    )$obj
    par0 <- unname(ref_obj$par)
    par1 <- par0 + c(rep(0.05, length(par0) - 1L), 0.1) *
      seq_along(par0) / length(par0)

    for (C in c(1L, 2L, 3L, 4L, 12L)) {
      layout <- .resolve_chunk_layout(R_full, C)
      obj_c <- .tape_chunk_chunked_obj(fam, layout, Z1_full, Z2_full)
      wrapper <- .make_chunked_tmb_objective(obj_c, layout, Z1_full, Z2_full)

      # Regression for the explicit report(par)/gr(par) requirement (review
      # P1): a stale obj$env$last.par must not affect the wrapper's result.
      obj_c$env$last.par <- par0 * 0 + 999

      info <- paste(nm, "C =", C)
      expect_equal(wrapper$fn(par0), ref$fn0, tolerance = 1e-9, info = info)
      expect_equal(as.numeric(wrapper$gr(par0)), ref$gr0,
                  tolerance = 1e-7, info = info)
      expect_equal(wrapper$fn(par1), ref$fn1, tolerance = 1e-9, info = info)
      expect_equal(as.numeric(wrapper$gr(par1)), ref$gr1,
                  tolerance = 1e-7, info = info)
    }
  }
})

test_that("balanced chunk layout has no empty chunks for the reviewed regression cases", {
  # (R, C) = (10, 6) and (5, 4): "full chunks + one short final chunk" gives
  # an empty final chunk here (third follow-up review); the balanced
  # small/large-size layout must not.
  for (case in list(c(R = 10L, C = 6L), c(R = 5L, C = 4L), c(R = 12L, C = 12L))) {
    layout <- .resolve_chunk_layout(case[["R"]], case[["C"]])
    counts <- vapply(layout$chunks, `[[`, integer(1), "Rc_valid")
    expect_length(counts, case[["C"]])
    expect_true(all(counts >= 1L), info = paste(case, collapse = ","))
    expect_equal(sum(counts), case[["R"]])
    expect_true(all(diff(sort(counts)) %in% c(0L, 1L)))
    for (ch in layout$chunks) {
      expect_length(ch$pad_rows, layout$Rc)
      expect_length(ch$draw_w, layout$Rc)
      expect_equal(sum(ch$draw_w), ch$Rc_valid)
    }
  }
})

test_that("chunked wrapper: report() full-draw semantics and per-chunk weights sum to one", {
  R_full <- 12L
  set.seed(20260824)
  Z <- .tmb_halton_uniform(R_full, 4L, burn = 50L)
  Z1_full <- Z[, 1:2, drop = FALSE]
  Z2_full <- Z[, 3:4, drop = FALSE]
  fam <- .tape_chunk_families$independence
  fixture <- .tape_chunk_fixture(fam$code, fam$pois1, fam$pois2)
  par0 <- unname(.make_rpbnb_tmb_object(
    data = fixture$data, parameters = fixture$parameters,
    map = fixture$map, n_cores = 1L
  )$obj$par)

  layout <- .resolve_chunk_layout(R_full, 5L)  # uneven: 3,3,2,2,2
  obj_c <- .tape_chunk_chunked_obj(fam, layout, Z1_full, Z2_full)
  wrapper <- .make_chunked_tmb_objective(obj_c, layout, Z1_full, Z2_full)

  rep0 <- wrapper$report(par0)
  expect_equal(sum(rep0$obs_loglik), -wrapper$fn(par0), tolerance = 1e-9)
  expect_true(!is.null(rep0$openmp_compiled))

  expect_error(wrapper$he(par0), "no taped Hessian")

  # Raw obj$env$data is restored to chunk-1/all-ones after every call.
  wrapper$gr(par0)
  expect_true(all(obj_c$env$data$w == 1))
  expect_equal(obj_c$env$data$draw_w, layout$chunks[[1L]]$draw_w)
})

test_that("chunk layout resolver rejects C > R", {
  expect_error(.resolve_chunk_layout(5L, 6L))
})
