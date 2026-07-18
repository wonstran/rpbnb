#!/usr/bin/env Rscript
# Build rpbnb.pdf with formatted title page + reference manual

setwd("C:/Users/litabook/repos/rpbnb")

cat("Building rpbnb.pdf with formatted title page...\n\n")

# Read DESCRIPTION
desc <- read.dcf("DESCRIPTION")
pkg_name <- desc[1, "Package"]
pkg_version <- desc[1, "Version"]
pkg_title <- desc[1, "Title"]
pkg_date <- format(Sys.Date(), "%B %d, %Y")
pkg_authors <- desc[1, "Authors@R"]
pkg_license <- desc[1, "License"]
pkg_depends <- trimws(desc[1, "Depends"])
pkg_imports <- trimws(desc[1, "Imports"])
pkg_suggests <- trimws(desc[1, "Suggests"])
pkg_description <- trimws(desc[1, "Description"])

# Clean up multiline fields
pkg_depends <- gsub("\n\\s+", ", ", pkg_depends)
pkg_imports <- gsub("\n\\s+", ", ", pkg_imports)
pkg_suggests <- gsub("\n\\s+", ", ", pkg_suggests)
pkg_description <- gsub("\n\\s+", " ", pkg_description)

cat("Package: ", pkg_name, "\n")
cat("Version: ", pkg_version, "\n")
cat("Date: ", pkg_date, "\n\n")

# Create LaTeX title page
title_tex <- "title_page.tex"
cat("Creating LaTeX title page...\n")

title_content <- sprintf(
"\\documentclass[11pt,a4paper]{article}
\\usepackage[margin=1in]{geometry}

\\pagestyle{empty}
\\pagenumbering{gobble}

\\begin{document}

\\begin{center}
  \\vspace*{0.5in}

  {\\fontsize{28}{34}\\selectfont \\textbf{Package '%s'}}

  \\vspace{0.3in}

  {\\fontsize{18}{22}\\selectfont An R Package}

  \\vspace{0.5in}

  {\\fontsize{12}{14}\\selectfont
    May %s\\\\
    Version %s
  }

  \\vspace{0.8in}

  {\\textbf{\\large Title}}

  {\\large %s}

  \\vspace{0.6in}

  {\\textbf{\\large Depends}}

  {%s}

  \\vspace{0.3in}

  {\\textbf{Imports}}

  {%s}

  \\vspace{0.3in}

  {\\textbf{Suggests}}

  {%s}

  \\vspace{0.6in}

  {\\textbf{\\large Description}}

  {\\fontsize{11}{13}\\selectfont %s}

  \\vspace{0.8in}

  {\\textbf{License}}

  {%s}

  \\vspace{1in}

  {Author: %s}

\\end{center}

\\end{document}
",
  pkg_name,
  format(Sys.Date(), "%d, %Y"),
  pkg_version,
  pkg_title,
  pkg_depends,
  pkg_imports,
  pkg_suggests,
  pkg_description,
  pkg_license,
  "Zhenyu Wang"
)

writeLines(title_content, title_tex)

tryCatch(
  {
    # Compile LaTeX to PDF
    cat("Compiling title page with pdflatex...\n")
    ret <- system(sprintf(
      'pdflatex -interaction=nonstopmode -output-directory=. "%s" > nul 2>&1',
      title_tex
    ))

    if (ret != 0 || !file.exists("title_page.pdf")) {
      stop("pdflatex compilation failed")
    }

    # Generate reference manual
    cat("Generating reference manual...\n")
    ret <- system(sprintf(
      '"%s/bin/R.exe" CMD Rd2pdf --output=manual.pdf man/ > nul 2>&1',
      "C:/Program Files/R/R-4.5.1"
    ))

    if (ret != 0 || !file.exists("manual.pdf")) {
      stop("Reference manual generation failed")
    }

    # Merge PDFs using qpdf
    cat("Merging title page with reference manual...\n")
    ret <- system(sprintf(
      'qpdf --empty --pages title_page.pdf 1 manual.pdf -- rpbnb.pdf 2>nul'
    ))

    if (ret != 0) {
      stop("PDF merge failed")
    }

    # Cleanup temporary files
    file.remove("title_page.pdf", "manual.pdf", "title_page.tex",
                "title_page.aux", "title_page.log", "Rd2.pdf",
                "Rd2.log", "Rd2.aux")

    if (!file.exists("rpbnb.pdf")) {
      stop("Final PDF not created")
    }

    cat("\n✓ SUCCESS! rpbnb.pdf created with formatted title page\n\n")
    cat("File: rpbnb.pdf\n")
    cat("Size:", format(file.size("rpbnb.pdf"), units = "auto"), "\n")
    cat("\nContents:\n")
    cat("  Page 1: Formatted title page with package metadata\n")
    cat("  Pages 2+: Complete reference manual\n\n")
    cat("Title page includes:\n")
    cat("  ✓ Package name: ", pkg_name, "\n")
    cat("  ✓ Version: ", pkg_version, "\n")
    cat("  ✓ Date: ", pkg_date, "\n")
    cat("  ✓ Title: ", pkg_title, "\n")
    cat("  ✓ Depends, Imports, Suggests\n")
    cat("  ✓ Description\n")
    cat("  ✓ License and Author\n")
  },
  error = function(e) {
    cat("ERROR:", conditionMessage(e), "\n")
    cat("Cleaning up temporary files...\n")
    file.remove(c("title_page.pdf", "manual.pdf", "title_page.tex",
                  "title_page.aux", "title_page.log", "Rd2.pdf",
                  "Rd2.log", "Rd2.aux"), showWarnings = FALSE)
    quit(status = 1)
  }
)
