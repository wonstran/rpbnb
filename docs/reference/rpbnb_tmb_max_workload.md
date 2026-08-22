# Compute a TMB workload budget from a memory figure

Companion to [`rpbnb_tmb_control()`](rpbnb_tmb_control.md)'s
`max_workload`: rather than picking a weighted-observation-draw count
directly, state a memory budget and let this function do the arithmetic
`TAPE_CALIBRATION` implies.

## Usage

``` r
rpbnb_tmb_max_workload(budget_gib = NULL, fraction = 0.8)
```

## Arguments

- budget_gib:

  Optional memory budget in GiB, stated explicitly. When supplied, used
  as-is – `fraction` does not apply, because a number you state yourself
  is not second-guessed with a discount. When omitted (the default),
  available memory is auto-detected and `fraction` of it is budgeted
  instead.

- fraction:

  Of auto-detected available memory, the fraction to actually budget;
  one number in `(0, 1]`. Available memory fluctuates and competes with
  other processes, so budgeting all of it risks the guard passing a fit
  that then exhausts memory anyway. Ignored when `budget_gib` is
  supplied.

## Value

One positive numeric workload value, on the same scale as
[`rpbnb_tmb_control()`](rpbnb_tmb_control.md)'s `max_workload`.

## Details

With `budget_gib` omitted, this is also
[`rpbnb_tmb_control()`](rpbnb_tmb_control.md)'s own default: every fit
that doesn't set `max_workload` explicitly already goes through this
function.

## Examples

``` r
rpbnb_tmb_max_workload(budget_gib = 16)
#> [1] 1e+06
if (FALSE) { # \dontrun{
ctrl <- rpbnb_tmb_control(max_workload = rpbnb_tmb_max_workload())
} # }
```
