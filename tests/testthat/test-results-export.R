test_that("truck results writer creates the timestamped Markdown report", {
  results_dir <- file.path(tempdir(), "truck-results")
  unlink(results_dir, recursive = TRUE)

  output_path <- rpbnb:::.write_truck_results_markdown(
    model_summary = c("Summary: fitted model", "Log-likelihood: -12.5"),
    marginal_effects = c(
      "--- Marginal effects (equation 1, AME) ---",
      "x  1.25"
    ),
    elasticities = c(
      "--- Elasticities (equation 1, AME) ---",
      "x  0.50"
    ),
    results_dir = results_dir,
    timestamp = as.POSIXct("2026-07-27 14:35:09", tz = "UTC")
  )

  expect_identical(
    output_path,
    file.path(results_dir, "results_2026-07-27-143509.md")
  )
  expect_true(file.exists(output_path))

  report <- readLines(output_path, warn = FALSE)
  expect_identical(report, c(
    "# RP-BNB truck model results",
    "",
    "Generated: 2026-07-27 14:35:09 UTC",
    "",
    "## Model fit summary",
    "",
    "```text",
    "Summary: fitted model",
    "Log-likelihood: -12.5",
    "```",
    "",
    "## Average marginal effects (AME)",
    "",
    "```text",
    "--- Marginal effects (equation 1, AME) ---",
    "x  1.25",
    "```",
    "",
    "## Elasticities / semi-elasticities (AME)",
    "",
    "```text",
    "--- Elasticities (equation 1, AME) ---",
    "x  0.50",
    "```"
  ))
})

test_that("shipped truck examples remain valid R syntax", {
  # system.file() rather than test_path("..", "..", "inst", ...): installation
  # flattens inst/, so the relative path resolves only in a source tree and
  # turns this into a release-check failure. system.file() resolves the source
  # inst/ under devtools::load_all() as well, so the assertion stays live in
  # both development and installed-package testing.
  # Discovered rather than listed. A hand-written list silently shrinks its own
  # coverage: this test previously named two of the four shipped truck scripts
  # while claiming in its title to cover them all, so a syntax regression in
  # tmb_truck_rpbnb_diff_famoye_laplace.R or tmb_truck_rpbnb_diff_frank_laplace.R
  # would have gone unnoticed. The pattern is anchored and narrow so it cannot
  # start sweeping up unrelated files.
  #
  # The `tmb_` prefix dates from the rpbnb.tmb merge: every script carried over
  # from that package took the prefix so the engine a script targets is legible
  # from its filename. The discovery guard below is what caught the rename --
  # it failed on zero matches rather than passing over an empty set.
  root <- system.file(package = "rpbnb", mustWork = TRUE)
  scripts <- list.files(root, pattern = "^tmb_truck_[A-Za-z0-9_]+\\.R$")
  # Guard the discovery itself: if the pattern or the layout ever stops
  # matching, this fails loudly instead of vacuously passing over zero files.
  expect_gte(length(scripts), 4L)
  expect_true(all(c("tmb_truck_rpbnb_diff_famoye_dense.R",
                    "tmb_truck_rpbnb_diff_famoye_laplace.R",
                    "tmb_truck_rpbnb_diff_frank_laplace.R",
                    "tmb_truck_rpbnb_diff_kimeldorf_laplace.R") %in% scripts))

  for (nm in scripts) {
    script <- system.file(nm, package = "rpbnb", mustWork = TRUE)
    # parse() raises on invalid syntax, and its own message names the file, so
    # a syntax error identifies itself; `label` additionally names the file on
    # the (unlikely) path where parse succeeds but returns something odd.
    # expect_no_error() is not used here because it forbids extra arguments.
    expect_true(is.expression(parse(file = script)), label = nm)
  }
})

test_that("dependence and method add a Model information section", {
  results_dir <- file.path(tempdir(), "truck-results-info")
  unlink(results_dir, recursive = TRUE)

  output_path <- rpbnb:::.write_truck_results_markdown(
    model_summary = "Summary: fitted model",
    marginal_effects = "x  1.25",
    elasticities = "x  0.50",
    dependence = copula("normal"),
    method = "laplace",
    results_dir = results_dir,
    timestamp = as.POSIXct("2026-07-27 14:35:09", tz = "UTC")
  )

  report <- readLines(output_path, warn = FALSE)
  expect_identical(report[1:8], c(
    "# RP-BNB truck model results",
    "",
    "Generated: 2026-07-27 14:35:09 UTC",
    "",
    "## Model information",
    "",
    "- Dependence: Gaussian copula (normal)",
    "- Method: laplace"
  ))
  # The section is additive: everything the three-argument form wrote is still
  # there, in the same order.
  expect_true(all(c("## Model fit summary",
                    "## Average marginal effects (AME)",
                    "## Elasticities / semi-elasticities (AME)") %in% report))
})

test_that("dependence labels stay readable for every structure", {
  label <- rpbnb:::.dependence_label
  expect_identical(label(copula("normal")), "Gaussian copula (normal)")
  expect_identical(label(copula("frank")), "Frank copula (frank)")
  expect_identical(label(copula("kimeldorf")), "Clayton copula (kimeldorf)")
  expect_identical(label("famoye"), "Famoye/Sarmanov")
  expect_identical(label("independence"), "independence")
})

test_that("an exhausted suffix search refuses to write rather than overwrite", {
  # The collision guard appends -2, -3, ... to a second-precision stamp. If the
  # bounded search ran out, an earlier version fell through with output_path
  # still at the unsuffixed base name and writeLines() then overwrote the very
  # file the guard exists to protect -- silently, since the caller only sees the
  # returned path. Refusing is the only safe outcome: there is no free name.
  dir <- file.path(tempdir(), "results-exhausted")
  unlink(dir, recursive = TRUE)
  dir.create(dir, recursive = TRUE, showWarnings = FALSE)

  ts <- as.POSIXct("2026-01-01 00:00:00", tz = "")
  stamp <- format(ts, "%Y-%m-%d-%H%M%S")
  base <- file.path(dir, sprintf("results_%s.md", stamp))
  writeLines("SENTINEL", base)
  for (i in 2:1000L) {
    writeLines("x", file.path(dir, sprintf("results_%s-%d.md", stamp, i)))
  }

  # The content arguments are supplied even though the call cannot reach the
  # write: without them this would error on a missing argument instead, and
  # expect_error() would pass for the wrong reason.
  expect_error(
    rpbnb:::.write_truck_results_markdown(
      model_summary = "Summary", marginal_effects = "x 1", elasticities = "x 2",
      results_dir = dir, timestamp = ts),
    "Refusing to write"
  )
  # The guard must not have consumed the file it was protecting.
  expect_identical(readLines(base)[1], "SENTINEL")
})

test_that("a same-second collision is suffixed, not overwritten", {
  dir <- file.path(tempdir(), "results-collide-one")
  unlink(dir, recursive = TRUE)
  dir.create(dir, recursive = TRUE, showWarnings = FALSE)

  ts <- as.POSIXct("2026-01-01 00:00:00", tz = "")
  stamp <- format(ts, "%Y-%m-%d-%H%M%S")
  base <- file.path(dir, sprintf("results_%s.md", stamp))
  writeLines("SENTINEL", base)

  expect_warning(
    out <- rpbnb:::.write_truck_results_markdown(
      model_summary = "Summary", marginal_effects = "x 1", elasticities = "x 2",
      results_dir = dir, timestamp = ts),
    "already exists"
  )
  expect_identical(basename(out), sprintf("results_%s-2.md", stamp))
  expect_true(file.exists(out))
  expect_identical(readLines(base)[1], "SENTINEL")
})
