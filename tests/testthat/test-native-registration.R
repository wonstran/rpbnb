# Guard on the hand-written R_init_rpbnb in src/rpbnb_tmb.cpp.
#
# This package hosts two engines in ONE shared library: the Rcpp/OpenMP kernels
# and the TMB template. Both would otherwise define R_init_rpbnb, so the table
# is written by hand and Rcpp::compileAttributes() stands down on detecting it.
# The cost of that arrangement is that the table is maintained manually. These
# assertions turn a missed entry into a hard failure here rather than a
# "function not available" at first call, or a silent fallback.

test_that("R_init_rpbnb registers both engines' native routines", {
  reg <- getDLLRegisteredRoutines("rpbnb")$.Call
  nm <- names(reg)

  rcpp_entries <- c(
    "_rpbnb_pbivnorm_cpp", "_rpbnb_rpbnb_copula_ll_grad_cpp",
    "_rpbnb_get_num_threads", "_rpbnb_set_rcpp_parallel_threads",
    "_rpbnb_rpbnb_openmp_enabled", "_rpbnb_rpbnb_ll_grad_cpp"
  )
  # TMB_CALLDEFS, from TMB/include/tmb_core.hpp.
  tmb_entries <- c(
    "MakeADFunObject", "FreeADFunObject", "InfoADFunObject", "tmbad_print",
    "EvalADFunObject", "TransformADFunObject", "MakeDoubleFunObject",
    "EvalDoubleFunObject", "getParameterOrder", "MakeADGradObject",
    "MakeADHessObject2", "usingAtomics", "getFramework", "getSetGlobalPtr",
    "TMBconfig"
  )

  expect_true(all(rcpp_entries %in% nm))
  expect_true(all(tmb_entries %in% nm))

  # Adding an // [[Rcpp::export]] without adding a row to rpbnbCallEntries[]
  # in src/rpbnb_tmb.cpp fails HERE.
  expect_equal(length(reg), length(rcpp_entries) + length(tmb_entries))
})

test_that("TMB resolves its symbols against DLL = \"rpbnb\"", {
  # TMB's R code uses .Call("Name", ..., PACKAGE = DLL) with a character symbol.
  # R checks the registered table before falling back to dlsym, which our init
  # disables via R_useDynamicSymbols(FALSE) -- so this only works because
  # TMB_CALLDEFS is in the table above.
  expect_silent(TMB::config(DLL = "rpbnb"))
  expect_type(TMB::openmp(max = TRUE, DLL = "rpbnb"), "integer")
})
