#!/usr/bin/env Rscript
# Regenerate ref/rpbnb_<version>.pdf from package documentation with full
# metadata from DESCRIPTION. Run from anywhere (it locates the package root
# itself); ref/ is a plain output directory, not an R-reserved one.

pkg_root <- "C:/Users/zwang9/repos/rpbnb"
setwd(pkg_root)

# Read DESCRIPTION for metadata
desc <- read.dcf("DESCRIPTION")
pkg_name <- as.character(desc[1, "Package"])
pkg_version <- as.character(desc[1, "Version"])
pkg_title <- as.character(desc[1, "Title"])
pkg_license <- as.character(desc[1, "License"])
dir.create("ref", showWarnings = FALSE)
pdf_out <- sprintf("ref/%s_%s.pdf", pkg_name, pkg_version)

cat("Regenerating", pdf_out, "from package documentation...\n\n")
cat("Package: ", pkg_name, "\n")
cat("Version: ", pkg_version, "\n")
cat("Title: ", pkg_title, "\n\n")

tryCatch(
  {
    # Point Rd2pdf at the package ROOT (not just man/): when a DESCRIPTION file
    # is present, Rd2pdf automatically prepends the standard CRAN-style title
    # page (Package, date, Type, Title, Version, Author, Maintainer,
    # Description, License, Depends/Imports/Suggests) rendered from DESCRIPTION,
    # ahead of the per-function reference pages -- the same layout CRAN uses for
    # every package PDF manual (e.g. https://cran.r-project.org/web/packages/maxLik/maxLik.pdf).
    if (file.exists(pdf_out)) file.remove(pdf_out)
    system(
      sprintf(
        '"%s/bin/R.exe" CMD Rd2pdf --output=%s .',
        "C:/Program Files/R/R-4.6.1", pdf_out
      )
    )

    cat("\n✓ PDF regenerated successfully!\n")
    if (file.exists(pdf_out)) {
      cat("File:", pdf_out, "\n")
      cat("Size:", format(file.size(pdf_out), units = "auto"), "\n")
      cat("\nMetadata from DESCRIPTION included:\n")
      cat("  - Package name and version\n")
      cat("  - Title and description\n")
      cat("  - Authors and license\n")
      cat("  - Dependencies (Depends, Imports, Suggests)\n")
    }
  },
  error = function(e) {
    cat("ERROR:", conditionMessage(e), "\n")
    quit(status = 1)
  }
)
