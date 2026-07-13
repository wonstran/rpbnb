# Analytic observed-information Hessian for the RP-BNB simulated likelihood

Reference: Famoye (2010), *On the bivariate negative binomial regression model*,
J. Appl. Stat. 37(6), Appendix A1–A20. This document records the derivation used
by `bnbr_rp_hessian()` (R/rpbnb_hessian.R), validated against `numDeriv::hessian`.

## Simulated likelihood

For observation `i`, `L_i = log( (1/R) Σ_r f_ir )` where `f_ir` is the Famoye BNB
joint density (paper Eq. 4/6) evaluated at the per-draw means
`μ_{t,ir} = exp(η_{t,ir})`, `η_{t,ir} = x_i'β_t + Σ_j XR_{t,ij} dev_{t,rj}`.

Parameter order `θ`: `β1 (k1), β2 (k2), log_sd1 (q1), log_sd2 (q2), log_m1, log_m2, z_lambda`.

## Mixture (Louis) Hessian

With `w_ir = f_ir / Σ_r f_ir`, per-draw score `g_ir = ∂log f_ir/∂θ`, and
`s_i = Σ_r w_ir g_ir`:

    ∂²L_i/∂θ∂θ' = Σ_r w_ir ( H_ir + g_ir g_ir' ) - s_i s_i'
    H_ir = ∂²log f_ir/∂θ∂θ'      (per-draw log-density Hessian)

Total Hessian `H = Σ_i ∂²L_i`; observed information `= -H`; covariance `= (-H)^{-1}`.
Assembled as three matrices:

    H = term_H + term_gg - term_ss
    term_gg = Σ_r G_r' diag(w_·r) G_r     (G_r = n×npar per-draw scores)
    term_ss = S'S                          (S = n×npar per-obs scores)
    term_H  = Σ_r (design-contracted per-draw density Hessian)

## Reduced per-draw derivatives in the (η1, η2, m1, m2, λ) basis

Let `d = 1-e^{-1}`, `r_t = 1/m_t`, `G_t = 1+d m_t μ_t`, `c_t = G_t^{-1/m_t}`,
`k_t = e^{-y_t} - c_t`, `dep = 1 + λ k1 k2`, `invD = 1/dep`,
`pen1 = λ k2 invD`, `pen2 = λ k1 invD`, `w_t = (y_t-μ_t)/(1+m_t μ_t)`.

c-derivatives (famoye_core.R):
`rf_t = ∂c_t/∂η_t = -d c_t μ_t/G_t`, `dcm_t = ∂c_t/∂m_t = dct_dm`,
`b_t = ∂²c_t/∂η_t² = d2ct_dbeta2_factor`, `e_t = ∂²c_t/∂η_t∂m_t = d2ct_dmdbeta_factor`,
`d2cm_t = ∂²c_t/∂m_t² = d2ct_dm2`.

NB dispersion pieces via digamma/trigamma (cleaner than the paper's Σ_j sum,
which equals ψ'(r)-ψ'(r+y)): with `A_t = S_t + log r_t + 1 - log(r_t+μ_t) - (r_t+y_t)/(r_t+μ_t)`,
`S_t = ψ(y_t+r_t)-ψ(r_t)`, `B_t = ψ'(y_t+r_t)-ψ'(r_t) + 1/r_t - 1/(r_t+μ_t) + (y_t-μ_t)/(r_t+μ_t)²`:

    nb_dm_t  = -r_t² A_t
    nb_d2m_t =  r_t⁴ B_t + 2 r_t³ A_t

First derivatives (natural λ, m):
    dη_t  = w_t - pen_t rf_t
    dm_tn = nb_dm_t - pen_t dcm_t
    dλn   = k1 k2 invD

Second derivatives:
    Hη1η1 = -μ1(1+m1 y1)/(1+m1 μ1)²  - λ²k2² invD² rf1²      - pen1 b1
    Hη1m1 =  μ1(μ1-y1)/(1+m1 μ1)²    - λ²k2² invD² rf1 dcm1  - pen1 e1
    Hm1m1 =  nb_d2m1                  - λ²k2² invD² dcm1²      - pen1 d2cm1
    Hη1λ  = -k2 invD² rf1     Hm1λ = -k2 invD² dcm1     Hλλ = -(k1 k2)² invD²
    (eq-2 analogues swap 1↔2 and k2↔k1)
    cross-equation (dep only):
    Hη1η2 = λ invD² rf1 rf2   Hη1m2 = λ invD² rf1 dcm2
    Hη2m1 = λ invD² rf2 dcm1  Hm1m2 = λ invD² dcm1 dcm2

## Chain rule to θ

Design multipliers per draw: `D_t = ∂η_t/∂β_t` (X with random cols scaled by
`dloc = ∂coef/∂b`), `M_t = ∂η_t/∂log_sd_t` (XR scaled by `dscale = ∂coef/∂log_s`).
`m_t = exp(log_m_t)` ⇒ `∂m/∂log_m = m`, `∂²m/∂log_m² = m`.
`λ = lam(z)` ⇒ `∂λ/∂z = dlam_dz`, `∂²λ/∂z² = d2lam_dz2`.

Second-order design (only random columns; `base` = draw variate):
`d2_bb = ∂²coef/∂b²`, `d2_bs = ∂²coef/∂b∂log_s`, `d2_ss = ∂²coef/∂log_s²`.
- normal/uniform/triangular: `(0, 0, dscale)`
- lognormal: `(coef, dscale, dscale·(1+s·base))`

Per-draw H_ir blocks (contracted, weighted by `w`):
- ββ: `D_a' diag(w Hηaηb) D_b`  (+ diag `Σ w dη_a XR d2_bb` on random-col β diagonal)
- β·logsd (same eq): `D' diag(w Hηη) M` (+ `Σ w dη XR d2_bs` on same random col)
- logsd·logsd: `M' diag(w Hηη) M` (+ diag `Σ w dη XR d2_ss`)
- η·logm: `D' (w Hηm m)`;  logm·logm: `Σ w (Hmm m² + dm_n m)`
- η·z: `D' (w Hηλ dlam_dz)`;  z·z: `Σ w (Hλλ dlam_dz² + dλn d2lam_dz2)`
- logm·z: `Σ w (Hmλ m dlam_dz)`;  logm1·logm2: `Σ w Hm1m2 m1 m2`

Per-draw scores `G_r` (columns, unweighted): `β_t: D_t dη_t`, `logsd_t: M_t dη_t`,
`logm_t: m_t dm_tn`, `z: dλn dlam_dz`.
