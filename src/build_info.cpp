// Compile-time build characteristics of this package's shared object.
//
// WHY: rpbnb's cost is almost entirely compiled likelihood evaluation, so an
// unoptimized build is not a curiosity -- it is roughly a 2x slowdown on every
// fit, and nothing in R's output says which one you have. A source install
// picks up R's own CXXFLAGS (-O2 on every standard platform), but a stray
// -O0 in the user's ~/.R/Makevars, or a package built by a tool that injects
// debug flags (pkgbuild::compile_dll(debug = TRUE) does exactly this), silently
// produces the slow one. This reports what was actually compiled so the
// question is answerable instead of guessed at.
//
// This file is deliberately Rcpp-only and TMB-free: src/rpbnb_tmb.cpp must
// remain the single translation unit that includes TMB.hpp (see its header).
// Every object in this shared library is compiled from one CXXFLAGS setting,
// so what this unit reports holds for the TMB template too.
#include <Rcpp.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// [[Rcpp::export]]
Rcpp::List rpbnb_build_flags_cpp() {
  // __OPTIMIZE__ is defined by GCC and clang at -O1 and above, and left
  // undefined at -O0. It is the compiler's own account of what it did, not an
  // inference from flags R reports, so it stays correct however the package
  // was built.
  bool optimized =
#ifdef __OPTIMIZE__
      true;
#else
      false;
#endif

  bool openmp =
#ifdef _OPENMP
      true;
#else
      false;
#endif

  int max_threads =
#ifdef _OPENMP
      omp_get_max_threads();
#else
      1;
#endif

  // NDEBUG off means assertions are live: another marker of a debug build, and
  // a further (smaller) cost on top of missing optimization.
  bool assertions =
#ifdef NDEBUG
      false;
#else
      true;
#endif

  std::string compiler = "unknown";
#if defined(__clang__)
  compiler = "clang " + std::to_string(__clang_major__) + "." +
             std::to_string(__clang_minor__);
#elif defined(__GNUC__)
  compiler = "gcc " + std::to_string(__GNUC__) + "." +
             std::to_string(__GNUC_MINOR__);
#endif

  return Rcpp::List::create(
      Rcpp::Named("optimized") = optimized,
      Rcpp::Named("openmp") = openmp,
      Rcpp::Named("openmp_max_threads") = max_threads,
      Rcpp::Named("assertions_enabled") = assertions,
      Rcpp::Named("compiler") = compiler);
}
