#!/usr/bin/env Rscript
# Regenerate rpbnb.pdf from package documentation

setwd("C:/Users/litabook/repos/rpbnb")

cat("Regenerating rpbnb.pdf from package documentation...\n\n")

tryCatch(
  {
    # Use tools::Rd2pdf to generate manual from .Rd files
    tools::Rd2pdf(
      dir = "man",
      output = "rpbnb.pdf",
      title = "Random-Parameter Bivariate Negative Binomial Regression",
      version = "0.2.2"
    )

    cat("\n✓ PDF regenerated successfully!\n")
    if (file.exists("rpbnb.pdf")) {
      cat("File: rpbnb.pdf\n")
      cat("Size:", format(file.size("rpbnb.pdf"), units = "auto"), "\n")
    }
  },
  error = function(e) {
    cat("ERROR:", conditionMessage(e), "\n")
    quit(status = 1)
  }
)
