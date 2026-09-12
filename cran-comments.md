## Test environments

* Local: Windows 11 x64, R 4.6.1 (x86_64-w64-mingw32), `R CMD check --as-cran`
  with `_R_CHECK_DONTTEST_EXAMPLES_=TRUE` (matches CRAN's own check behavior).
* Previous CRAN pretests (2026-09-10): Windows Server 2022 and Debian
  forky/sid, R-devel. Both passed tests, examples, vignettes, and manuals.
* Configured CI (not rerun for this local resubmission): release builds on
  Ubuntu, macOS, and Windows; fast tests on Ubuntu for pushes/PRs, with all
  test tiers on the weekly schedule or manual dispatch.

## R CMD check results

Local verification on 2026-09-12, using the rebuilt source tarball:

* Full `R CMD check --as-cran`, including `--run-donttest` examples, tests,
  vignette rebuilding, and PDF/HTML manuals: 0 errors, 0 warnings, 2 notes.
  The notes were "New submission" and a local Pandoc PATH issue when
  checking README/NEWS. All examples, tests, vignettes, and manuals passed.
* After adding the installed Pandoc to PATH, a follow-up `--as-cran` check
  with `--no-tests --no-examples --no-vignettes --no-manual` completed with
  0 errors, 0 warnings, 1 note ("New submission"). Top-level file checking
  and compilation-flags checking both report OK. The tarball was unchanged.
* Testthat: 1,664 passes, 0 failures, 7 warnings, 182 skips under the
  package's existing CRAN/slow-test guards and installed-file availability
  check. For comparison, the previous CRAN Windows test log had the same
  pass/skip counts, 0 failures, and 8 warnings; its test check also passed.
* The installed build reports both optimization and OpenMP enabled.

The previous CRAN pretests (2026-09-10) reported 0 errors, 0 warnings,
2 notes on Windows and 1 note on Debian. This resubmission addresses them
as follows.

* checking CRAN incoming feasibility ... NOTE
  New submission

  Possibly misspelled words in DESCRIPTION: Famoye, Sarmanov

  This is the package's first submission to CRAN. Both words are correctly
  spelled: they are the surnames of the statisticians the bivariate
  dependence structure is named after (the Famoye/Sarmanov bivariate
  negative binomial). The software names previously flagged here ('OpenMP',
  'Rcpp', 'TMB') are now in single quotes.

* The Windows-only compilation-flags NOTE has been addressed by removing
  `-Wa,-mbig-obj` from `src/Makevars.win`. A clean source build with Rtools45
  GCC 14.3.0 succeeds without it. The TMB object has 2,852 sections and uses
  standard `pe-x86-64` COFF format. Our previous comment that the current
  template required the extended object format was incorrect. R's standard
  optimization flags and OpenMP compile/link flags remain in use.

## Changes since the previous pretest

This is a resubmission of version 0.4.9, which has not yet been accepted
on CRAN. Relative to the 2026-09-10 pretests, software names in DESCRIPTION
are now single-quoted and the unnecessary Windows assembler flag is removed.

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
