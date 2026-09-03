# Report how this installation of rpbnb was compiled

Nearly all of this package's running time is compiled likelihood
evaluation, so whether its shared object was built with optimization is
a performance fact worth roughly a factor of two on every fit – and
nothing in R's own output tells you which build you have. This reports
what the compiler actually did.

## Usage

``` r
rpbnb_build_info()
```

## Value

A list, invisibly when printed:

- `optimized`:

  `TRUE` when compiled at `-O1` or above, read from the compiler's own
  `__OPTIMIZE__` macro rather than inferred.

- `openmp`:

  `TRUE` when OpenMP is available, so `n_cores > 1` can do anything. A
  build without it fits single-threaded whatever `control$n_cores` says.

- `openmp_max_threads`:

  The OpenMP runtime's own thread ceiling.

- `assertions_enabled`:

  `TRUE` when `NDEBUG` is unset, a further marker of a debug build.

- `compiler`:

  Compiler and version that built the shared object.

## Details

A source install (`install.packages(type = "source")`, `R CMD INSTALL`,
`remotes::install_github()`) compiles with R's own `CXXFLAGS`, which is
`-O2` on every standard platform, so the optimized build is what you get
by default. The package does not – and by CRAN policy must not –
override those flags. Two things do produce a slow build: a `-O0` (or
`-Og`) entry in the user's `~/.R/Makevars`, which applies to every
package compiled on that machine, and development helpers that inject
debug flags of their own, notably `pkgbuild::compile_dll(debug = TRUE)`
– its default – which
[`devtools::load_all()`](https://devtools.r-lib.org/reference/load_all.html)
uses when recompiling changed sources.

## Examples

``` r
rpbnb_build_info()
#> rpbnb build
#>   optimized      TRUE
#>   openmp         TRUE
#>   max threads    1
#>   assertions     FALSE
#>   compiler       gcc 14.3
```
