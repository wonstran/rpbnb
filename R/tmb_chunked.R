# Draw-chunked TMB objective for large-R SML fits (the primary fix for the
# TMB+SML large-draw OOM described in docs/TMB_SML_large_draws_OOM_guide.md).
#
# Rather than a single [n x R] tape, one TMB object of a fixed, smaller draw
# count (`Rc`) is built once and REPLAYED over successive draw chunks via
# TMB's DATA_UPDATE() (src/rpbnb_tmb.cpp's `chunked == 1` branch), which lets
# R change `Z1`/`Z2`/`w`/`draw_w` between fn()/gr() calls without retaping.
# Peak memory becomes O(n * Rc) instead of O(n * R).
#
# The combination is EXACT for the user's requested R draws (no rounding):
# with valid draw counts Rc_c summing to R,
#
#   log S_i = logsumexp_c(log S_ic + log Rc_c) - log R
#   d/dtheta log S_i = sum_c w_ic * d/dtheta log S_ic,
#   w_ic = Rc_c * S_ic / sum_c(Rc_c * S_ic)
#
# where log S_ic is the per-observation, per-chunk mean log-likelihood the
# template's `report(par)$obs_loglik` returns. See the reviewed plan
# (comments/review_*.md / comments/response_*.md) for the full derivation
# and the design rounds that arrived at this file's contract.

#' Balanced draw-chunk layout: R draws into C chunks, no empty chunks
#'
#' Splits `R` draws into `C` chunks whose sizes differ by at most one
#' (`floor(R/C)` or `floor(R/C) + 1`), then pads every chunk shorter than the
#' tape size `Rc = ceiling(R/C)` by duplicating one of its OWN valid draws,
#' masked with `draw_w = 0` on the padded rows. Because `C <= R` (validated
#' by the caller), every chunk has at least one valid draw -- sequential
#' "full chunks + one short final chunk" filling does not have this
#' guarantee (e.g. R = 10, C = 6 gives chunk sizes 2,2,2,2,2,0), which is
#' why this function assigns sizes directly instead.
#'
#' @param R Total requested draws (positive whole number).
#' @param C Number of chunks (positive whole number, `<= R`).
#' @return A list with `R`, `C`, `Rc` (the fixed tape size every chunk is
#'   padded/truncated to), and `chunks`: a length-`C` list of
#'   `list(valid_rows, pad_rows, draw_w, Rc_valid)`, where `valid_rows`/
#'   `pad_rows` are 1-based row indices into the full `R`-row Halton draw
#'   matrix (`pad_rows` has length `Rc`, `valid_rows` has length `Rc_valid`).
#' @keywords internal
#' @noRd
.resolve_chunk_layout <- function(R, C) {
  R <- as.integer(R); C <- as.integer(C)
  stopifnot(C >= 1L, C <= R)
  small <- R %/% C
  n_large <- R %% C
  large <- small + 1L
  sizes <- c(rep(large, n_large), rep(small, C - n_large))
  Rc <- max(sizes)
  ends <- cumsum(sizes)
  starts <- ends - sizes + 1L

  chunks <- vector("list", C)
  for (c in seq_len(C)) {
    valid_rows <- starts[c]:ends[c]
    k <- length(valid_rows)
    if (k < Rc) {
      pad_rows <- c(valid_rows, rep(valid_rows[1L], Rc - k))
      draw_w <- c(rep(1, k), rep(0, Rc - k))
    } else {
      pad_rows <- valid_rows
      draw_w <- rep(1, Rc)
    }
    chunks[[c]] <- list(valid_rows = valid_rows, pad_rows = pad_rows,
                        draw_w = draw_w, Rc_valid = k)
  }
  list(R = R, C = C, Rc = Rc, chunks = chunks)
}

#' Build a chunked full-objective wrapper over a chunk-sized TMB object
#'
#' `obj` must have been constructed (via `.make_rpbnb_tmb_object()`) with
#' `chunked = 1L` and draw matrices `Z1`/`Z2` sized to `layout$Rc` rows --
#' i.e. the DATA list `obj` was built from already contains chunk 1's rows
#' (`Z1_full[layout$chunks[[1]]$pad_rows, ]`, etc.) and `draw_w` set to
#' chunk 1's mask. This function does not build the TMB object itself; it
#' only wraps an already-built one.
#'
#' Returned wrapper contract (a `MakeADFun`-*like* but NOT MakeADFun-*shaped*
#' object -- see the class marker below):
#'   - `par`: the fixed-effect parameter vector (snapshot of `obj$par` at
#'     construction; matches `stats::nlminb(start = wrapper$par, ...)`
#'     usage).
#'   - `fn(par)`/`gr(par)`: the FULL R-draw objective/gradient, exact (see
#'     header). `fn`/`gr` always pass `par` explicitly to `obj$report()`/
#'     `obj$gr()` -- never relying on TMB's internal `last.par` default --
#'     so a stale `obj$env$last.par` cannot corrupt a result.
#'   - `report(par)`: FULL-draw semantics. `obs_loglik` is the aggregated
#'     per-observation log-likelihood (`sum(report(par)$obs_loglik) ==
#'     -fn(par)`), and every other (parameter-only) reported field is taken
#'     from one chunk verbatim, since those fields do not vary by chunk.
#'     Never returns chunk-local values.
#'   - `he(par)`: errors -- no taped Hessian exists for this objective; the
#'     package's own finite-difference path (`stats::optimHess()`, already
#'     used by `.rpbnb_inference()`) is the supported alternative.
#'   - `env`: a WRAPPER-OWNED environment (not `obj$env`), holding
#'     `last.par.best`/`value.best` (mirroring TMB's own bookkeeping, read by
#'     the NA/NaN nlminb-recovery path in [fit_rpbnb_tmb()]) and `random`
#'     (always empty -- chunked SML has no random effects).
#'
#' `obj`'s own `Z1`/`Z2`/`w`/`draw_w` are mutated in place while `fn`/`gr`/
#' `report` run (that is the whole point of DATA_UPDATE -- see
#' src/rpbnb_tmb.cpp), and restored to chunk 1 / all-ones on every exit path
#' (including on error, via `on.exit`), so a `saveRDS()`/`readRDS()` round
#' trip of the retained raw object always retapes into the same, documented
#' state, never whichever chunk happened to run last.
#'
#' Memory: wrapper state is O(n), never O(n * C). `fn(par)` streams across
#' chunks, folding each into a running (max, rescaled-sum) log-sum-exp
#' accumulator -- no per-chunk column is retained. `gr(par)` re-runs
#' `report(par)` per chunk (a second serial pass; see the plan's Phase 3
#' wall-time benchmark) against the `fn(par)` pass's cached O(n) denominator
#' to recover each chunk's weight.
#'
#' @param obj The chunk-sized TMB object (`chunked = 1L`), as built by
#'   `.make_rpbnb_tmb_object()`.
#' @param layout A layout from `.resolve_chunk_layout()`.
#' @param Z1_full,Z2_full The FULL `R`-row Halton draw matrices (`R x q1`,
#'   `R x q2`) this layout's chunks are carved out of.
#' @return A list with class `"rpbnb_chunked_objective"`; see the contract
#'   above.
#' @keywords internal
#' @noRd
.make_chunked_tmb_objective <- function(obj, layout, Z1_full, Z2_full) {
  n <- length(obj$env$data$Y1)
  logR <- log(layout$R)

  wrapper_env <- new.env(parent = emptyenv())
  wrapper_env$last.par.best <- NULL
  wrapper_env$value.best <- Inf
  wrapper_env$random <- integer(0)

  cache <- new.env(parent = emptyenv())
  cache$par <- NULL
  cache$logS_i <- NULL
  cache$fn <- NULL

  .load_chunk <- function(ch) {
    obj$env$data$Z1 <- Z1_full[ch$pad_rows, , drop = FALSE]
    obj$env$data$Z2 <- Z2_full[ch$pad_rows, , drop = FALSE]
    obj$env$data$draw_w <- ch$draw_w
  }
  .reset_raw_data <- function() {
    .load_chunk(layout$chunks[[1L]])
    obj$env$data$w <- rep(1, n)
  }
  # NOTE: on.exit() must be called directly inside fn()/gr()/report()
  # themselves -- it registers in the CALLING function's own frame, not the
  # caller's caller's, so a one-line `on_exit_reset <- function()
  # on.exit(...)` helper (an earlier version of this file) fires as soon as
  # that helper itself returns, not when fn()/gr()/report() do. Every exit
  # path of fn()/gr()/report() restores this, including error exits, so a
  # retained `obj` is never left mutated mid-chunk -- see the three
  # `on.exit(.reset_raw_data(), add = TRUE)` calls below.

  # Pass 1 (streamed, O(n) state): fold each chunk's contribution into a
  # running log-sum-exp accumulator instead of retaining an n x C matrix.
  # See the header derivation; log S_i = m + log(s) - log(R) once every
  # chunk has been folded in.
  .pass1 <- function(par) {
    if (!is.null(cache$par) && identical(par, cache$par)) return(invisible(NULL))
    m <- rep(-Inf, n)
    s <- rep(0, n)
    for (ch in layout$chunks) {
      .load_chunk(ch)
      logS_ic <- obj$report(par)$obs_loglik + log(ch$Rc_valid)
      new_m <- pmax(m, logS_ic)
      s <- s * exp(m - new_m) + exp(logS_ic - new_m)
      m <- new_m
    }
    cache$par <- par
    cache$logS_i <- m + log(s) - logR
    cache$fn <- -sum(cache$logS_i)
    invisible(NULL)
  }

  fn <- function(par) {
    on.exit(.reset_raw_data(), add = TRUE)
    .pass1(par)
    val <- cache$fn
    if (is.finite(val) && val < wrapper_env$value.best) {
      wrapper_env$value.best <- val
      wrapper_env$last.par.best <- par
    }
    val
  }

  gr <- function(par) {
    on.exit(.reset_raw_data(), add = TRUE)
    .pass1(par)
    logS_i <- cache$logS_i
    log_denom <- logS_i + logR  # = m + log(s) from the pass-1 accumulator
    gr_total <- NULL
    for (ch in layout$chunks) {
      .load_chunk(ch)
      logS_ic <- obj$report(par)$obs_loglik + log(ch$Rc_valid)
      obj$env$data$w <- exp(logS_ic - log_denom)
      g <- as.numeric(obj$gr(par))
      gr_total <- if (is.null(gr_total)) g else gr_total + g
    }
    matrix(gr_total, nrow = 1L, dimnames = list(NULL, names(par)))
  }

  he <- function(par) {
    stop(
      "he() is not available for a draw-chunked TMB objective: no taped ",
      "Hessian exists for the chunked SML likelihood (see R/tmb_chunked.R). ",
      "Use the package's own finite-difference Hessian path instead -- ",
      "stats::optimHess(par, wrapper$fn, wrapper$gr) -- which is what ",
      ".rpbnb_inference() already calls for every SML fit.",
      call. = FALSE
    )
  }

  report <- function(par) {
    on.exit(.reset_raw_data(), add = TRUE)
    .pass1(par)
    .load_chunk(layout$chunks[[1L]])
    rep1 <- obj$report(par)
    rep1$obs_loglik <- cache$logS_i
    rep1
  }

  .reset_raw_data()
  structure(
    list(par = obj$par, fn = fn, gr = gr, he = he, report = report,
        env = wrapper_env, layout = layout),
    class = "rpbnb_chunked_objective"
  )
}
