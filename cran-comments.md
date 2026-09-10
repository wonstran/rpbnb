## Test environments

* Local: Windows 11 x64, R 4.6.1 (x86_64-w64-mingw32), `R CMD check --as-cran`
  with `_R_CHECK_DONTTEST_EXAMPLES_=TRUE` (matches CRAN's own check behavior).
* GitHub Actions (`.github/workflows/release-binaries.yaml`): ubuntu-latest,
  macos-latest, windows-latest, R release -- build, install, and (on Linux)
  verify a default source install is optimized (`-O2`, OpenMP available).
* GitHub Actions (`.github/workflows/tests.yaml`): ubuntu-latest, R release --
  full testthat suite.

## R CMD check results

0 errors | 0 warnings | 2 notes (win-builder: Windows 2 notes, Debian 1 note)

* checking CRAN incoming feasibility ... NOTE
  New submission

  Possibly misspelled words in DESCRIPTION: Famoye, Sarmanov

  This is the package's first submission to CRAN. Both words are correctly
  spelled: they are the surnames of the statisticians the bivariate
  dependence structure is named after (the Famoye/Sarmanov bivariate
  negative binomial). The software names previously flagged here ('OpenMP',
  'Rcpp', 'TMB') are now in single quotes.

* checking compilation flags used ... NOTE
  Compilation used the following non-portable flag(s): '-Wa,-mbig-obj'

  Required on Windows only (`src/Makevars.win`): one translation unit
  (`src/rpbnb_tmb.cpp`, which builds the package's TMB automatic-
  differentiation template) exceeds the COFF object format's 32767-section
  limit without it. Does not apply to, or affect, any other platform.

## Changes since the previous pretest

An earlier upload of this version also flagged two issues, both fixed:

* Invalid file URIs in README.md ("ref/rpbnb_0.4.9.pdf",
  "docs/TMB_SML_large_draws_OOM_guide.md") -- both pointed at repo-relative
  paths excluded from the built tarball (`ref/`, `docs/` are in
  `.Rbuildignore`); changed to absolute GitHub URLs.
* "Non-standard file/directory found at top level: 'cran-comments.md'" --
  this file is now itself listed in `.Rbuildignore`.

## Additional notes for reviewers

* **Multithreading.** The package uses OpenMP (via Rcpp and TMB) for optional
  multithreaded likelihood evaluation. `n_cores` defaults to 1 everywhere
  (`rpbnb_control()`), and no example, test, or vignette requests more than
  the default. Tests that exercise multithreading explicitly (`n_cores > 1`,
  or realized thread count > 1) are gated with `testthat::skip_on_cran()`.
* **`SystemRequirements: GNU make`** is needed to compile the bundled TMB
  template (`src/rpbnb_tmb.cpp`), consistent with other TMB-based packages
  already on CRAN.
* Examples that fit a real model are wrapped in `\donttest{}` where a single
  run exceeds ~5 seconds; the rest run unconditionally as smoke tests.

## Downstream dependencies

None -- this is the package's first release.
