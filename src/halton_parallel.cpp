// Multithreaded RP-BNB simulated log-likelihood + analytic gradient.
//
// This is a math-identical port of the R reference `bnbr_rp_ll_and_grad`
// (R/rpbnb_likelihood.R). The per-draw distribution transforms (rand_realize)
// are computed in R and passed in as the dev/dloc/dscale matrices, so this file
// only ports the numerically hot triple loop (Pass 1 / Pass 2 / gradient) and
// parallelises it over the R simulation draws with OpenMP.
//
// Verified against the pure-R implementation in tests/testthat.

#include <Rcpp.h>
#include <cmath>
#include <vector>
#ifdef _OPENMP
  #include <omp.h>
#endif

using namespace Rcpp;

// ---- thread helpers -------------------------------------------------------

// Report the number of processors available to OpenMP. This is independent of
// any prior omp_set_num_threads() call (which a single-threaded fit issues), so
// rpbnb_threads() keeps reporting the machine's core count.
// [[Rcpp::export]]
int get_num_threads() {
#ifdef _OPENMP
  return omp_get_num_procs();
#else
  return 1;
#endif
}

// [[Rcpp::export]]
void set_rcpp_parallel_threads(int n_threads) {
#ifdef _OPENMP
  if (n_threads > 0) omp_set_num_threads(n_threads);
#else
  (void) n_threads;
#endif
}

// [[Rcpp::export]]
bool rpbnb_openmp_enabled() {
#ifdef _OPENMP
  return true;
#else
  return false;
#endif
}

// ---- core math (mirrors famoye_core.R) ------------------------------------

static inline double d_const_cpp() { return 1.0 - std::exp(-1.0); }

static inline double c_val_cpp(double mu, double m, double d) {
  return std::pow(1.0 + d * m * mu, -1.0 / m);
}

static inline double nb_logpmf_cpp(double y, double mu, double r) {
  double p = r / (r + mu);
  return std::lgamma(y + r) - std::lgamma(r) - std::lgamma(y + 1.0)
         + r * std::log(p) + y * std::log1p(-p);
}

// ---------------------------------------------------------------------------
// Full simulated LL + gradient. When want_grad == 0 only the value is computed
// (used by the numeric-Hessian objective). When use_fixed != 0 the caller
// supplies frozen lambda bounds (lamLo_in / lamHi_in) instead of rebuilding
// them from Pass 1 (also for the Hessian, to match bnbr_rp_ll_fixed_bounds).
//
// Parameter order of the returned gradient matches the R reference:
//   beta1 (k1), beta2 (k2), log_sd1 (q1), log_sd2 (q2), log_m1, log_m2, z_lambda
//
// rand_idx1 / rand_idx2 are 0-based column indices into X1 / X2.
// [[Rcpp::export]]
List rpbnb_ll_grad_cpp(
    NumericVector y1, NumericVector y2,
    NumericMatrix X1, NumericMatrix X2,
    NumericMatrix XR1, NumericMatrix XR2,
    IntegerVector rand_idx1, IntegerVector rand_idx2,
    NumericMatrix dev1, NumericMatrix dev2,
    NumericMatrix dloc1, NumericMatrix dloc2,
    NumericMatrix dscale1, NumericMatrix dscale2,
    NumericVector xb1, NumericVector xb2,
    NumericVector S1, NumericVector S2,
    double m1, double m2, double zlam,
    int want_grad, int use_fixed, double lamLo_in, double lamHi_in,
    int want_scores, int num_threads) {

#ifdef _OPENMP
  if (num_threads > 0) omp_set_num_threads(num_threads);
#endif

  const int n  = y1.size();
  const int k1 = X1.ncol();
  const int k2 = X2.ncol();
  const int q1 = rand_idx1.size();
  const int q2 = rand_idx2.size();
  const int R  = (q1 + q2 > 0) ? dev1.nrow() : 1;

  const double d  = d_const_cpp();
  const double r1 = 1.0 / m1;
  const double r2 = 1.0 / m2;
  const double log_m1_v = std::log(m1);
  const double log_m2_v = std::log(m2);
  const int npar = k1 + k2 + q1 + q2 + 3;

  // Raw pointers (column-major) for lock-free reads inside parallel regions.
  const double* pX1  = X1.begin();
  const double* pX2  = X2.begin();
  const double* pXR1 = XR1.begin();
  const double* pXR2 = XR2.begin();
  const double* pdev1 = dev1.begin();
  const double* pdev2 = dev2.begin();
  const double* pdloc1 = dloc1.begin();
  const double* pdloc2 = dloc2.begin();
  const double* pdsc1 = dscale1.begin();
  const double* pdsc2 = dscale2.begin();
  const double* py1 = y1.begin();
  const double* py2 = y2.begin();
  const double* pxb1 = xb1.begin();
  const double* pxb2 = xb2.begin();
  const double* pS1 = S1.begin();
  const double* pS2 = S2.begin();
  const int* pri1 = rand_idx1.begin();
  const int* pri2 = rand_idx2.begin();

  // Precompute exp(-y) once.
  std::vector<double> ey1(n), ey2(n);
  for (int i = 0; i < n; i++) { ey1[i] = std::exp(-py1[i]); ey2[i] = std::exp(-py2[i]); }

  // Storage for per-draw mu / c (n x R, column-major).
  std::vector<double> mu1(static_cast<size_t>(n) * R);
  std::vector<double> mu2(static_cast<size_t>(n) * R);
  std::vector<double> c1(static_cast<size_t>(n) * R);
  std::vector<double> c2(static_cast<size_t>(n) * R);
  std::vector<double> lamLo_r(R), lamHi_r(R);

  // ---- Pass 1: mu, c, per-draw bounds -------------------------------------
  #pragma omp parallel for schedule(static)
  for (int r = 0; r < R; r++) {
    const size_t off = static_cast<size_t>(n) * r;
    double lo = -std::numeric_limits<double>::infinity(); // max of lam_min
    double hi =  std::numeric_limits<double>::infinity(); // min of lam_max
    for (int i = 0; i < n; i++) {
      double eta1 = pxb1[i];
      for (int j = 0; j < q1; j++)
        eta1 += pXR1[i + static_cast<size_t>(j) * n] * pdev1[r + static_cast<size_t>(j) * R];
      double eta2 = pxb2[i];
      for (int j = 0; j < q2; j++)
        eta2 += pXR2[i + static_cast<size_t>(j) * n] * pdev2[r + static_cast<size_t>(j) * R];

      double m1i = std::exp(eta1); if (m1i > 1e15) m1i = 1e15;
      double m2i = std::exp(eta2); if (m2i > 1e15) m2i = 1e15;
      double c1i = c_val_cpp(m1i, m1, d);
      double c2i = c_val_cpp(m2i, m2, d);

      mu1[off + i] = m1i; mu2[off + i] = m2i;
      c1[off + i]  = c1i; c2[off + i]  = c2i;

      double lam_min = -1.0 / ((1.0 - c1i) * (1.0 - c2i));
      double denom_max = std::max(c1i * (1.0 - c2i), c2i * (1.0 - c1i));
      double lam_max = 1.0 / denom_max;
      if (lam_min > lo) lo = lam_min;
      if (lam_max < hi) hi = lam_max;
    }
    lamLo_r[r] = lo; lamHi_r[r] = hi;
  }

  // Reduce bounds across draws (or use frozen bounds).
  double lamLo, lamHi;
  if (use_fixed) {
    lamLo = lamLo_in; lamHi = lamHi_in;
  } else {
    lamLo = -std::numeric_limits<double>::infinity();
    lamHi =  std::numeric_limits<double>::infinity();
    for (int r = 0; r < R; r++) {
      if (lamLo_r[r] > lamLo) lamLo = lamLo_r[r];
      if (lamHi_r[r] < lamHi) lamHi = lamHi_r[r];
    }
  }

  const double eps = 1e-6;
  const double sig = 1.0 / (1.0 + std::exp(-zlam));

  if (!use_fixed && !(lamLo < lamHi && std::isfinite(lamLo) && std::isfinite(lamHi))) {
    NumericVector grad(npar); // zeros
    return List::create(Named("value") = -1e50, Named("gradient") = grad,
                        Named("lamLo") = lamLo, Named("lamHi") = lamHi,
                        Named("scores") = NumericMatrix(0, 0));
  }

  const double lam     = lamLo + (lamHi - lamLo) * (eps + (1.0 - 2.0 * eps) * sig);
  const double dlam_dz = (lamHi - lamLo) * (1.0 - 2.0 * eps) * sig * (1.0 - sig);

  // ---- Pass 2: LL matrix (n x R) ------------------------------------------
  std::vector<double> LL(static_cast<size_t>(n) * R);
  #pragma omp parallel for schedule(static)
  for (int r = 0; r < R; r++) {
    const size_t off = static_cast<size_t>(n) * r;
    for (int i = 0; i < n; i++) {
      double lnb1 = nb_logpmf_cpp(py1[i], mu1[off + i], r1);
      double lnb2 = nb_logpmf_cpp(py2[i], mu2[off + i], r2);
      double dep = 1.0 + lam * (ey1[i] - c1[off + i]) * (ey2[i] - c2[off + i]);
      if (dep < 1e-300) dep = 1e-300;
      LL[off + i] = lnb1 + lnb2 + std::log(dep);
    }
  }

  // ---- row log-sum-exp -> value, weights W --------------------------------
  const double logR = std::log((double) R);
  const bool need_W = (want_grad || want_scores);
  double value = 0.0;
  std::vector<double> W;
  if (need_W) W.resize(static_cast<size_t>(n) * R);

  #pragma omp parallel for schedule(static) reduction(+:value)
  for (int i = 0; i < n; i++) {
    double mx = -std::numeric_limits<double>::infinity();
    for (int r = 0; r < R; r++) {
      double v = LL[static_cast<size_t>(n) * r + i];
      if (v > mx) mx = v;
    }
    double s = 0.0;
    for (int r = 0; r < R; r++)
      s += std::exp(LL[static_cast<size_t>(n) * r + i] - mx);
    double lse = mx + std::log(s);
    value += (lse - logR);
    if (need_W) {
      for (int r = 0; r < R; r++)
        W[static_cast<size_t>(n) * r + i] = std::exp(LL[static_cast<size_t>(n) * r + i] - lse);
    }
  }

  if (!want_grad && !want_scores) {
    NumericVector grad(npar);
    return List::create(Named("value") = value, Named("gradient") = grad,
                        Named("lamLo") = lamLo, Named("lamHi") = lamHi,
                        Named("scores") = NumericMatrix(0, 0));
  }

  // ---- gradient loop over draws (reduction over threads) ------------------
  std::vector<double> g(npar, 0.0);   // shared accumulator
  double g_logm1 = 0.0, g_logm2 = 0.0, g_z = 0.0;
  const double r1v = r1, r2v = r2;

  #pragma omp parallel
  {
    std::vector<double> gloc(npar, 0.0);
    double lm1 = 0.0, lm2 = 0.0, lz = 0.0;

    #pragma omp for schedule(static) nowait
    for (int r = 0; r < R; r++) {
      const size_t off = static_cast<size_t>(n) * r;
      for (int i = 0; i < n; i++) {
        double mu1i = mu1[off + i], mu2i = mu2[off + i];
        double c1i = c1[off + i],  c2i = c2[off + i];
        double wir = W[off + i];

        double k1v = ey1[i] - c1i;
        double k2v = ey2[i] - c2i;
        double dep = 1.0 + lam * (k1v * k2v);
        if (dep < 1e-300) dep = 1e-300;
        double inv_dep = 1.0 / dep;
        double pen1 = lam * k2v * inv_dep;
        double pen2 = lam * k1v * inv_dep;

        double w1 = (py1[i] - mu1i) / (1.0 + m1 * mu1i);
        double w2 = (py2[i] - mu2i) / (1.0 + m2 * mu2i);

        double denom1 = 1.0 + d * m1 * mu1i;
        double denom2 = 1.0 + d * m2 * mu2i;
        double rf1 = -(d * c1i * mu1i) / denom1;   // dc/dbeta row factor eq1
        double rf2 = -(d * c2i * mu2i) / denom2;

        // Shared per-obs factor: score contribution scaled by W.
        double A1 = wir * (w1 - rf1 * pen1);
        double A2 = wir * (w2 - rf2 * pen2);

        // ---- beta gradients (Xeff = X with random cols scaled by dloc) ----
        // Non-random columns: dloc = 1. Random column jj at rand_idx[jj]:
        // Xeff = X * dloc. Build a per-column dloc multiplier lazily.
        for (int col = 0; col < k1; col++) {
          double xval = pX1[i + static_cast<size_t>(col) * n];
          gloc[col] += A1 * xval;
        }
        for (int jj = 0; jj < q1; jj++) {
          int col = pri1[jj];
          double dl = pdloc1[r + static_cast<size_t>(jj) * R];
          // correct the col already added with dloc = 1 -> multiply by (dl-1)... simpler: add (dl-1) share
          double xval = pX1[i + static_cast<size_t>(col) * n];
          gloc[col] += A1 * xval * (dl - 1.0);
        }
        for (int col = 0; col < k2; col++) {
          double xval = pX2[i + static_cast<size_t>(col) * n];
          gloc[k1 + col] += A2 * xval;
        }
        for (int jj = 0; jj < q2; jj++) {
          int col = pri2[jj];
          double dl = pdloc2[r + static_cast<size_t>(jj) * R];
          double xval = pX2[i + static_cast<size_t>(col) * n];
          gloc[k1 + col] += A2 * xval * (dl - 1.0);
        }

        // ---- dispersion gradients -----------------------------------------
        // dc/dm eq1
        double term_c1 = (1.0 / m1) * ((1.0 / m1) * std::log(denom1) - (d * mu1i) / denom1);
        double dc1_dm1 = term_c1 * c1i;
        double term_c2 = (1.0 / m2) * ((1.0 / m2) * std::log(denom2) - (d * mu2i) / denom2);
        double dc2_dm2 = term_c2 * c2i;

        double term_m1 = r1v * r1v * log_m1_v
                       + r1v * r1v * (std::log(mu1i + r1v) - 1.0)
                       + r1v * r1v * (py1[i] + r1v) / (mu1i + r1v)
                       - r1v * r1v * pS1[i]
                       - (lam * k2v * inv_dep) * dc1_dm1;
        double term_m2 = r2v * r2v * log_m2_v
                       + r2v * r2v * (std::log(mu2i + r2v) - 1.0)
                       + r2v * r2v * (py2[i] + r2v) / (mu2i + r2v)
                       - r2v * r2v * pS2[i]
                       - (lam * k1v * inv_dep) * dc2_dm2;
        lm1 += wir * (m1 * term_m1);
        lm2 += wir * (m2 * term_m2);

        // ---- dependence gradient ------------------------------------------
        lz += wir * ((k1v * k2v) * inv_dep * dlam_dz);

        // ---- log_sd gradients ---------------------------------------------
        // score_logsd[j] = dscale[r,j] * XR[i,j] * (w - rf*pen), weighted by W.
        for (int j = 0; j < q1; j++) {
          double xr = pXR1[i + static_cast<size_t>(j) * n];
          double ds = pdsc1[r + static_cast<size_t>(j) * R];
          gloc[k1 + k2 + j] += A1 * xr * ds;
        }
        for (int j = 0; j < q2; j++) {
          double xr = pXR2[i + static_cast<size_t>(j) * n];
          double ds = pdsc2[r + static_cast<size_t>(j) * R];
          gloc[k1 + k2 + q1 + j] += A2 * xr * ds;
        }
      }
    }

    #pragma omp critical
    {
      for (int p = 0; p < npar; p++) g[p] += gloc[p];
      g_logm1 += lm1; g_logm2 += lm2; g_z += lz;
    }
  }

  NumericVector grad(npar);
  for (int p = 0; p < k1 + k2 + q1 + q2; p++) grad[p] = g[p];
  grad[k1 + k2 + q1 + q2]     = g_logm1;
  grad[k1 + k2 + q1 + q2 + 1] = g_logm2;
  grad[k1 + k2 + q1 + q2 + 2] = g_z;

  // ---- per-observation scores for the BHHH / OPG covariance ---------------
  // s_i[theta] = sum_r W_ir * dlogP_ir/dtheta.  The summed gradient above is
  // sum_i s_i; here we keep the rows separate so R can form the OPG
  // information matrix crossprod(scores) = sum_i s_i s_i'. Parallelised over
  // observations (each thread owns whole rows) so writes never race.
  NumericMatrix scores(want_scores ? n : 0, want_scores ? npar : 0);
  if (want_scores) {
    const int off_lm1 = k1 + k2 + q1 + q2;
    double* psc = scores.begin();   // column-major: element (i, col) = psc[i + col*n]
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
      for (int r = 0; r < R; r++) {
        const size_t off = static_cast<size_t>(n) * r;
        double mu1i = mu1[off + i], mu2i = mu2[off + i];
        double c1i = c1[off + i],  c2i = c2[off + i];
        double wir = W[off + i];

        double k1v = ey1[i] - c1i;
        double k2v = ey2[i] - c2i;
        double dep = 1.0 + lam * (k1v * k2v);
        if (dep < 1e-300) dep = 1e-300;
        double inv_dep = 1.0 / dep;
        double pen1 = lam * k2v * inv_dep;
        double pen2 = lam * k1v * inv_dep;

        double w1 = (py1[i] - mu1i) / (1.0 + m1 * mu1i);
        double w2 = (py2[i] - mu2i) / (1.0 + m2 * mu2i);

        double denom1 = 1.0 + d * m1 * mu1i;
        double denom2 = 1.0 + d * m2 * mu2i;
        double rf1 = -(d * c1i * mu1i) / denom1;
        double rf2 = -(d * c2i * mu2i) / denom2;
        double A1 = wir * (w1 - rf1 * pen1);
        double A2 = wir * (w2 - rf2 * pen2);

        // beta1 (Xeff = X * dloc on random columns)
        for (int col = 0; col < k1; col++)
          psc[i + static_cast<size_t>(col) * n] += A1 * pX1[i + static_cast<size_t>(col) * n];
        for (int jj = 0; jj < q1; jj++) {
          int col = pri1[jj];
          double dl = pdloc1[r + static_cast<size_t>(jj) * R];
          psc[i + static_cast<size_t>(col) * n] += A1 * pX1[i + static_cast<size_t>(col) * n] * (dl - 1.0);
        }
        // beta2
        for (int col = 0; col < k2; col++)
          psc[i + static_cast<size_t>(k1 + col) * n] += A2 * pX2[i + static_cast<size_t>(col) * n];
        for (int jj = 0; jj < q2; jj++) {
          int col = pri2[jj];
          double dl = pdloc2[r + static_cast<size_t>(jj) * R];
          psc[i + static_cast<size_t>(k1 + col) * n] += A2 * pX2[i + static_cast<size_t>(col) * n] * (dl - 1.0);
        }
        // log_sd1 / log_sd2
        for (int j = 0; j < q1; j++)
          psc[i + static_cast<size_t>(k1 + k2 + j) * n] +=
            A1 * pXR1[i + static_cast<size_t>(j) * n] * pdsc1[r + static_cast<size_t>(j) * R];
        for (int j = 0; j < q2; j++)
          psc[i + static_cast<size_t>(k1 + k2 + q1 + j) * n] +=
            A2 * pXR2[i + static_cast<size_t>(j) * n] * pdsc2[r + static_cast<size_t>(j) * R];

        // log_m1, log_m2
        double term_c1 = (1.0 / m1) * ((1.0 / m1) * std::log(denom1) - (d * mu1i) / denom1);
        double dc1_dm1 = term_c1 * c1i;
        double term_c2 = (1.0 / m2) * ((1.0 / m2) * std::log(denom2) - (d * mu2i) / denom2);
        double dc2_dm2 = term_c2 * c2i;
        double term_m1 = r1v * r1v * log_m1_v
                       + r1v * r1v * (std::log(mu1i + r1v) - 1.0)
                       + r1v * r1v * (py1[i] + r1v) / (mu1i + r1v)
                       - r1v * r1v * pS1[i]
                       - (lam * k2v * inv_dep) * dc1_dm1;
        double term_m2 = r2v * r2v * log_m2_v
                       + r2v * r2v * (std::log(mu2i + r2v) - 1.0)
                       + r2v * r2v * (py2[i] + r2v) / (mu2i + r2v)
                       - r2v * r2v * pS2[i]
                       - (lam * k1v * inv_dep) * dc2_dm2;
        psc[i + static_cast<size_t>(off_lm1) * n]     += wir * (m1 * term_m1);
        psc[i + static_cast<size_t>(off_lm1 + 1) * n] += wir * (m2 * term_m2);
        psc[i + static_cast<size_t>(off_lm1 + 2) * n] += wir * ((k1v * k2v) * inv_dep * dlam_dz);
      }
    }
  }

  return List::create(Named("value") = value, Named("gradient") = grad,
                      Named("lamLo") = lamLo, Named("lamHi") = lamHi,
                      Named("scores") = scores);
}
