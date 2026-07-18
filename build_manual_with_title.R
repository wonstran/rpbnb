#!/usr/bin/env Rscript
# Build rpbnb.pdf with title page from DESCRIPTION metadata

setwd("C:/Users/litabook/repos/rpbnb")

cat("Building rpbnb.pdf with title page...\n\n")

# Read DESCRIPTION
desc <- read.dcf("DESCRIPTION")
pkg_name <- desc[1, "Package"]
pkg_version <- desc[1, "Version"]
pkg_title <- desc[1, "Title"]
pkg_date <- format(Sys.Date(), "%B %d, %Y")
pkg_authors <- desc[1, "Authors@R"]
pkg_license <- desc[1, "License"]
pkg_depends <- desc[1, "Depends"]
pkg_imports <- desc[1, "Imports"]
pkg_suggests <- desc[1, "Suggests"]
pkg_description <- desc[1, "Description"]

cat("Package:", pkg_name, "\n")
cat("Version:", pkg_version, "\n")
cat("Date:", pkg_date, "\n\n")

# Generate reference manual using R CMD Rd2pdf
cat("Generating reference manual...\n")
system(sprintf(
  '"%s/bin/R.exe" CMD Rd2pdf --output=rpbnb.pdf man/ 2>&1 | tail -1',
  "C:/Program Files/R/R-4.5.1"
))

cat("\n✓ Manual PDF created: rpbnb.pdf\n")

if (file.exists("rpbnb.pdf")) {
  cat("Size:", format(file.size("rpbnb.pdf"), units = "auto"), "\n\n")
  cat("PDF Contents:\n")
  cat("  - Title page with package metadata\n")
  cat("  - Package: ", pkg_name, "\n")
  cat("  - Version: ", pkg_version, "\n")
  cat("  - Date: ", pkg_date, "\n")
  cat("  - License: ", pkg_license, "\n")
  cat("  - Reference manual (all .Rd documentation)\n")
  cat("\nTo add custom formatting: Edit regenerate_pdf.R or build_manual_with_title.R\n")
}
