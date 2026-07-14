// Multithreaded (OpenMP) C++ core for the copula RP-BNB likelihood + gradient.
// Bivariate-normal CDF via Genz's tvpack bvnu (Drezner-Wesolowsky).
#include <Rcpp.h>
#include <cmath>
#include <vector>
#ifdef _OPENMP
  #include <omp.h>
#endif
using namespace Rcpp;

// Genz tvpack bvnu: P(X > dh, Y > dk) for standard bivariate normal, corr r.
static double bvnu(double dh, double dk, double r) {
  const double TWOPI = 6.283185307179586476925286766559;
  if (!R_finite(dh) && dh > 0) return 0.0;                 // dh = +Inf
  if (!R_finite(dk) && dk > 0) return 0.0;                 // dk = +Inf
  if (!R_finite(dh) && dh < 0)                              // dh = -Inf
    return (!R_finite(dk) && dk < 0) ? 1.0 : R::pnorm(-dk, 0.0, 1.0, 1, 0);
  if (!R_finite(dk) && dk < 0) return R::pnorm(-dh, 0.0, 1.0, 1, 0);
  if (r == 0.0) return R::pnorm(-dh,0.0,1.0,1,0) * R::pnorm(-dk,0.0,1.0,1,0);

  static const double w6[3]  = {0.1713244923791705,0.3607615730481386,0.4679139345726904};
  static const double x6[3]  = {0.9324695142031522,0.6612093864662647,0.2386191860831970};
  static const double w12[6] = {.04717533638651177,.1069393259953183,.1600783285433464,.2031674267230659,.2334925365383547,.2491470458134029};
  static const double x12[6] = {.9815606342467191,.9041172563704750,.7699026741943050,.5873179542866171,.3678314989981802,.1252334085114692};
  static const double w20[10]= {.01761400713915212,.04060142980038694,.06267204833410906,.08327674157670475,.1019301198172404,.1181945319615184,.1316886384491766,.1420961093183821,.1491729864726037,.1527533871307259};
  static const double x20[10]= {.9931285991850949,.9639719272779138,.9122344282513259,.8391169718222188,.7463319064601508,.6360536807265150,.5108670019508271,.3737060887154196,.2277858511416451,.07652652113349733};
  const double *w, *x; int ng;
  double ar = std::fabs(r);
  if (ar < 0.3)       { ng = 3;  w = w6;  x = x6;  }
  else if (ar < 0.75) { ng = 6;  w = w12; x = x12; }
  else                { ng = 10; w = w20; x = x20; }

  double h = dh, k = dk, hk = h * k, bvn = 0.0;
  if (ar < 0.925) {
    double hs = (h*h + k*k) / 2.0, asr = std::asin(r) / 2.0;
    for (int i = 0; i < ng; i++)
      for (int is = -1; is <= 1; is += 2) {
        double sn = std::sin(asr * (is * x[i] + 1.0));
        bvn += w[i] * std::exp((sn*hk - hs) / (1.0 - sn*sn));
      }
    bvn = bvn * asr / TWOPI + R::pnorm(-h,0.0,1.0,1,0) * R::pnorm(-k,0.0,1.0,1,0);
  } else {
    if (r < 0) { k = -k; hk = -hk; }
    if (ar < 1.0) {
      double as = (1.0 - r) * (1.0 + r), a = std::sqrt(as), bs = (h - k)*(h - k);
      double c = (4.0 - hk) / 8.0, d = (12.0 - hk) / 16.0;
      double asr = -(bs/as + hk) / 2.0;
      if (asr > -100.0)
        bvn = a * std::exp(asr) * (1.0 - c*(bs - as)*(1.0 - d*bs/5.0)/3.0 + c*d*as*as/5.0);
      if (hk > -100.0) {
        double b = std::sqrt(bs);
        bvn -= std::exp(-hk/2.0) * std::sqrt(TWOPI) * R::pnorm(-b/a,0.0,1.0,1,0) * b
               * (1.0 - c*bs*(1.0 - d*bs/5.0)/3.0);
      }
      a = a / 2.0;
      for (int i = 0; i < ng; i++)
        for (int is = -1; is <= 1; is += 2) {
          double xs = a * (is * x[i] + 1.0); xs = xs * xs;
          double rs = std::sqrt(1.0 - xs);
          double asr2 = -(bs/xs + hk) / 2.0;
          if (asr2 > -100.0)
            bvn += a * w[i] * std::exp(asr2)
                   * (std::exp(-hk*xs/(2.0*(1.0+rs)*(1.0+rs)))/rs - (1.0 + c*xs*(1.0 + d*xs)));
        }
      bvn = -bvn / TWOPI;
    }
    if (r > 0) bvn += R::pnorm(-std::fmax(h,k),0.0,1.0,1,0);
    else { bvn = -bvn; if (k > h) bvn += R::pnorm(k,0.0,1.0,1,0) - R::pnorm(h,0.0,1.0,1,0); }
  }
  if (bvn < 0.0) bvn = 0.0; if (bvn > 1.0) bvn = 1.0;
  return bvn;
}

// Standard bivariate-normal CDF P(X<=h, Y<=k) = bvnu(-h,-k,rho).
static inline double bvncdf(double h, double k, double rho) {
  return bvnu(-h, -k, rho);
}

// [[Rcpp::export]]
NumericVector pbivnorm_cpp(NumericVector h, NumericVector k, double rho) {
  int n = h.size();
  NumericVector out(n);
  for (int i = 0; i < n; i++) out[i] = bvncdf(h[i], k[i], rho);
  return out;
}

static const double DNEG = 1e-300;

// ---- copula CDF C(u,v; theta) by family (fam: 0=frank,1=normal,2=kimeldorf) ----
static double cop_cdf(double u, double v, double th, int fam) {
  if (!(u > 0.0 && v > 0.0)) return 0.0;
  if (fam == 0) { // frank
    if (std::fabs(th) < 1e-10) return u * v;
    double et = std::exp(-th);
    return -std::log(1.0 + (std::exp(-th*u)-1.0)*(std::exp(-th*v)-1.0)/(et-1.0)) / th;
  } else if (fam == 1) { // normal (rho = th)
    double ui = std::fmin(std::fmax(u,1e-15),1.0-1e-15);
    double vi = std::fmin(std::fmax(v,1e-15),1.0-1e-15);
    return bvncdf(R::qnorm(ui,0.0,1.0,1,0), R::qnorm(vi,0.0,1.0,1,0), th);
  } else { // kimeldorf (Clayton, th>0)
    if (th < 1e-10) return u * v;
    double inner = std::pow(u,-th) + std::pow(v,-th) - 1.0;
    if (inner < 0.0) inner = 0.0;
    return std::pow(inner, -1.0/th);
  }
}

// dC/du (dC/dv = cop_du with u,v swapped)
static double cop_du(double u, double v, double th, int fam) {
  if (!(u > 0.0 && v > 0.0)) return 0.0;
  if (fam == 0) {
    if (std::fabs(th) < 1e-10) return v;
    double et=std::exp(-th), etu=std::exp(-th*u), etv=std::exp(-th*v);
    return etu*(etv-1.0)/((et-1.0)+(etu-1.0)*(etv-1.0));
  } else if (fam == 1) {
    double ui = std::fmin(std::fmax(u,1e-15),1.0-1e-15);
    double vi = std::fmin(std::fmax(v,1e-15),1.0-1e-15);
    double qu=R::qnorm(ui,0.0,1.0,1,0), qv=R::qnorm(vi,0.0,1.0,1,0);
    return R::pnorm((qv - th*qu)/std::sqrt(1.0-th*th),0.0,1.0,1,0);
  } else {
    if (th < 1e-10) return v;
    double inner=std::pow(u,-th)+std::pow(v,-th)-1.0; if (inner<DNEG) inner=DNEG;
    return std::pow(inner,-(1.0/th)-1.0)*std::pow(u,-(th+1.0));
  }
}
static inline double cop_dv(double u, double v, double th, int fam) { return cop_du(v,u,th,fam); }

// dC/dtheta
static double cop_dtheta(double u, double v, double th, int fam) {
  if (!(u > 0.0 && v > 0.0)) return 0.0;
  if (fam == 0) {
    if (std::fabs(th) < 1e-10) return u*v*(u-1.0)*(v-1.0)/2.0;
    double et=std::exp(-th), etu=std::exp(-th*u), etv=std::exp(-th*v);
    double A=(etu-1.0)*(etv-1.0)/(et-1.0), C=-std::log(1.0+A)/th;
    double dA=(((-u*etu)*(etv-1.0)*(et-1.0)) + ((etu-1.0)*(-v*etv)*(et-1.0))
              - ((etu-1.0)*(etv-1.0)*(-et)))/((et-1.0)*(et-1.0));
    return -dA/(th*(1.0+A)) - C/th;
  } else if (fam == 1) { // dC/drho = bivariate normal density
    double ui=std::fmin(std::fmax(u,1e-15),1.0-1e-15), vi=std::fmin(std::fmax(v,1e-15),1.0-1e-15);
    double qu=R::qnorm(ui,0.0,1.0,1,0), qv=R::qnorm(vi,0.0,1.0,1,0), r2=1.0-th*th;
    return std::exp(-(qu*qu-2.0*th*qu*qv+qv*qv)/(2.0*r2))/(2.0*M_PI*std::sqrt(r2));
  } else {
    if (th < 1e-10) return u*v*(std::log(u)+std::log(v))/2.0;
    double inner=std::pow(u,-th)+std::pow(v,-th)-1.0; if (inner<DNEG) inner=DNEG;
    double Cok=std::pow(inner,-1.0/th);
    double dinner=-(std::pow(u,-th)*std::log(u)+std::pow(v,-th)*std::log(v));
    return Cok*(std::log(inner)/(th*th) - dinner/(th*inner));
  }
}

// dnb_cdf_dr: dF(y;mu,r)/dr (0 if y<0 or r not finite)
static double dnb_cdf_dr(int y, double mu, double r) {
  if (y < 0 || !R_finite(r)) return 0.0;
  double s = 0.0, lr = std::log(r/(r+mu));
  for (int kk = 0; kk <= y; kk++) {
    double pk = R::dnbinom_mu((double)kk, r, mu, 0);
    double wk = R::digamma(kk + r) - R::digamma(r) + lr + 1.0 - (r + kk)/(r + mu);
    s += pk * wk;
  }
  return s;
}

// [[Rcpp::export]]
List rpbnb_copula_ll_grad_cpp(
    NumericVector y1, NumericVector y2,
    NumericMatrix X1, NumericMatrix X2, NumericMatrix XR1, NumericMatrix XR2,
    IntegerVector rand_idx1, IntegerVector rand_idx2,
    NumericMatrix dev1, NumericMatrix dev2,
    NumericMatrix dloc1, NumericMatrix dloc2,
    NumericMatrix dscale1, NumericMatrix dscale2,
    NumericVector xb1, NumericVector xb2,
    double r1, double r2, double theta, double dth_dz,
    int family_code, int want_scores, int num_threads) {
#ifdef _OPENMP
  if (num_threads > 0) omp_set_num_threads(num_threads);
#endif
  const int n = y1.size(), k1 = X1.ncol(), k2 = X2.ncol();
  const int q1 = rand_idx1.size(), q2 = rand_idx2.size();
  const int R = (q1 + q2 > 0) ? dev1.nrow() : 1;
  const int npar = k1 + k2 + q1 + q2 + 3;
  const double* pX1=X1.begin(); const double* pX2=X2.begin();
  const double* pXR1=XR1.begin(); const double* pXR2=XR2.begin();
  const double* pdev1=dev1.begin(); const double* pdev2=dev2.begin();
  const double* pdl1=dloc1.begin(); const double* pdl2=dloc2.begin();
  const double* pds1=dscale1.begin(); const double* pds2=dscale2.begin();
  const double* py1=y1.begin(); const double* py2=y2.begin();
  const double* pxb1=xb1.begin(); const double* pxb2=xb2.begin();
  const int* pri1=rand_idx1.begin(); const int* pri2=rand_idx2.begin();

  // Pass 1: per-draw mu (stored) and LL matrix.
  std::vector<double> mu1(static_cast<size_t>(n)*R), mu2(static_cast<size_t>(n)*R);
  std::vector<double> LL(static_cast<size_t>(n)*R);
  #pragma omp parallel for schedule(static)
  for (int r = 0; r < R; r++) {
    size_t off = static_cast<size_t>(n)*r;
    for (int i = 0; i < n; i++) {
      double e1 = pxb1[i]; for (int j=0;j<q1;j++) e1 += pXR1[i+(size_t)j*n]*pdev1[r+(size_t)j*R];
      double e2 = pxb2[i]; for (int j=0;j<q2;j++) e2 += pXR2[i+(size_t)j*n]*pdev2[r+(size_t)j*R];
      double m1i = std::exp(e1); if (m1i<1e-300) m1i=1e-300; if (m1i>1e15) m1i=1e15;
      double m2i = std::exp(e2); if (m2i<1e-300) m2i=1e-300; if (m2i>1e15) m2i=1e15;
      mu1[off+i]=m1i; mu2[off+i]=m2i;
      double a = R::pnbinom(py1[i], r1, r1/(r1+m1i), 1, 0);
      double am= (py1[i]>0) ? R::pnbinom(py1[i]-1, r1, r1/(r1+m1i), 1, 0) : 0.0;
      double b = R::pnbinom(py2[i], r2, r2/(r2+m2i), 1, 0);
      double bm= (py2[i]>0) ? R::pnbinom(py2[i]-1, r2, r2/(r2+m2i), 1, 0) : 0.0;
      bool ok = R_finite(a)&&R_finite(am)&&R_finite(b)&&R_finite(bm);
      double p = cop_cdf(a,b,theta,family_code) - cop_cdf(am,b,theta,family_code)
               - cop_cdf(a,bm,theta,family_code) + cop_cdf(am,bm,theta,family_code);
      ok = ok && R_finite(p);
      LL[off+i] = ok ? std::log(std::fmax(p,1e-300)) : R_NegInf;
    }
  }

  // row log-sum-exp -> value, weights
  const double logR = std::log((double)R);
  double value = 0.0;
  std::vector<double> W(static_cast<size_t>(n)*R);
  #pragma omp parallel for schedule(static) reduction(+:value)
  for (int i = 0; i < n; i++) {
    double mx = R_NegInf;
    for (int r=0;r<R;r++){ double v=LL[(size_t)n*r+i]; if (v>mx) mx=v; }
    double s=0.0; for (int r=0;r<R;r++) s += std::exp(LL[(size_t)n*r+i]-mx);
    double lse = mx + std::log(s);
    value += (lse - logR);
    for (int r=0;r<R;r++){ double w=std::exp(LL[(size_t)n*r+i]-lse); W[(size_t)n*r+i]=R_finite(w)?w:0.0; }
  }

  // Pass 2: gradient (+ scores) via per-draw score scalars, contracted with design.
  const int iL1 = k1+k2, iL2 = k1+k2+q1, im1 = k1+k2+q1+q2, im2 = im1+1, iz = im1+2;
  std::vector<double> grad(npar, 0.0);
  NumericMatrix scores(want_scores?n:0, want_scores?npar:0);
  double* psc = want_scores ? scores.begin() : nullptr;

  // Parallelize over OBSERVATIONS (each thread owns whole rows) so both the
  // thread-local gradient reduction AND the per-obs score writes are race-free.
  #pragma omp parallel
  {
    std::vector<double> gloc(npar, 0.0);
    #pragma omp for schedule(static) nowait
    for (int i = 0; i < n; i++) {
      for (int r = 0; r < R; r++) {
        size_t off = static_cast<size_t>(n)*r;
        double m1i=mu1[off+i], m2i=mu2[off+i], wv=W[off+i];
        double a = R::pnbinom(py1[i], r1, r1/(r1+m1i), 1, 0);
        double am= (py1[i]>0)?R::pnbinom(py1[i]-1, r1, r1/(r1+m1i), 1, 0):0.0;
        double b = R::pnbinom(py2[i], r2, r2/(r2+m2i), 1, 0);
        double bm= (py2[i]>0)?R::pnbinom(py2[i]-1, r2, r2/(r2+m2i), 1, 0):0.0;
        bool ok = R_finite(a)&&R_finite(am)&&R_finite(b)&&R_finite(bm);
        double p = cop_cdf(a,b,theta,family_code)-cop_cdf(am,b,theta,family_code)
                 - cop_cdf(a,bm,theta,family_code)+cop_cdf(am,bm,theta,family_code);
        ok = ok && R_finite(p);
        // Raw pmf underflowed (extreme count -> both NB CDF corners round to 1 ->
        // rectangle cancels to ~0): there the log-lik is a clamped constant so its
        // gradient is 0, but numerator/floor would be a huge *finite* score that
        // R_finite cannot reject. Flag it for the score mask (mirrors R's
        // .copula_pmf `underflow`); the value path keeps the finite log-floor.
        bool underflow = !(p > DNEG);
        p = std::fmax(p, DNEG);

        double cu_ab=cop_du(a,b,theta,family_code),  cu_amb=cop_du(am,b,theta,family_code);
        double cu_abm=cop_du(a,bm,theta,family_code), cu_ambm=cop_du(am,bm,theta,family_code);
        double cv_ab=cop_dv(a,b,theta,family_code),  cv_amb=cop_dv(am,b,theta,family_code);
        double cv_abm=cop_dv(a,bm,theta,family_code), cv_ambm=cop_dv(am,bm,theta,family_code);
        double ct = cop_dtheta(a,b,theta,family_code)-cop_dtheta(am,b,theta,family_code)
                  - cop_dtheta(a,bm,theta,family_code)+cop_dtheta(am,bm,theta,family_code);

        double da_dmu1 = -(py1[i]+1.0)*R::dnbinom_mu(py1[i]+1.0, r1, m1i, 0)/m1i;
        double dam_dmu1= (py1[i]>0)? -py1[i]*R::dnbinom_mu(py1[i], r1, m1i, 0)/m1i : 0.0;
        double du_a = cu_ab - cu_abm, du_am = -cu_amb + cu_ambm;
        double s_eta1 = (du_a*da_dmu1*m1i + du_am*dam_dmu1*m1i)/p;

        double db_dmu2 = -(py2[i]+1.0)*R::dnbinom_mu(py2[i]+1.0, r2, m2i, 0)/m2i;
        double dbm_dmu2= (py2[i]>0)? -py2[i]*R::dnbinom_mu(py2[i], r2, m2i, 0)/m2i : 0.0;
        double dv_b = cv_ab - cv_amb, dv_bm = -cv_abm + cv_ambm;
        double s_eta2 = (dv_b*db_dmu2*m2i + dv_bm*dbm_dmu2*m2i)/p;

        double da_dr1 = dnb_cdf_dr((int)py1[i], m1i, r1);
        double dam_dr1= (py1[i]>0)? dnb_cdf_dr((int)py1[i]-1, m1i, r1):0.0;
        double s_logm1 = (-r1)*(du_a*da_dr1 + du_am*dam_dr1)/p;
        double db_dr2 = dnb_cdf_dr((int)py2[i], m2i, r2);
        double dbm_dr2= (py2[i]>0)? dnb_cdf_dr((int)py2[i]-1, m2i, r2):0.0;
        double s_logm2 = (-r2)*(dv_b*db_dr2 + dv_bm*dbm_dr2)/p;
        double s_z = ct*dth_dz/p;

        bool bad = !ok || underflow || !R_finite(s_eta1)||!R_finite(s_eta2)||!R_finite(s_logm1)
                   ||!R_finite(s_logm2)||!R_finite(s_z);
        if (bad){ s_eta1=0;s_eta2=0;s_logm1=0;s_logm2=0;s_z=0; }

        double A1 = wv*s_eta1, A2 = wv*s_eta2;
        // beta1 (Xeff via dloc)
        for (int col=0; col<k1; col++) gloc[col] += A1*pX1[i+(size_t)col*n];
        for (int jj=0; jj<q1; jj++){ int col=pri1[jj]; double dl=pdl1[r+(size_t)jj*R];
          gloc[col] += A1*pX1[i+(size_t)col*n]*(dl-1.0); }
        for (int col=0; col<k2; col++) gloc[k1+col] += A2*pX2[i+(size_t)col*n];
        for (int jj=0; jj<q2; jj++){ int col=pri2[jj]; double dl=pdl2[r+(size_t)jj*R];
          gloc[k1+col] += A2*pX2[i+(size_t)col*n]*(dl-1.0); }
        // log_sd (M via dscale)
        for (int j=0;j<q1;j++) gloc[iL1+j] += A1*pXR1[i+(size_t)j*n]*pds1[r+(size_t)j*R];
        for (int j=0;j<q2;j++) gloc[iL2+j] += A2*pXR2[i+(size_t)j*n]*pds2[r+(size_t)j*R];
        gloc[im1] += wv*s_logm1; gloc[im2] += wv*s_logm2; gloc[iz] += wv*s_z;

        if (want_scores) {
          for (int col=0; col<k1; col++) psc[i+(size_t)col*n] += A1*pX1[i+(size_t)col*n];
          for (int jj=0; jj<q1; jj++){ int col=pri1[jj]; double dl=pdl1[r+(size_t)jj*R];
            psc[i+(size_t)col*n] += A1*pX1[i+(size_t)col*n]*(dl-1.0); }
          for (int col=0; col<k2; col++) psc[i+(size_t)(k1+col)*n] += A2*pX2[i+(size_t)col*n];
          for (int jj=0; jj<q2; jj++){ int col=pri2[jj]; double dl=pdl2[r+(size_t)jj*R];
            psc[i+(size_t)(k1+col)*n] += A2*pX2[i+(size_t)col*n]*(dl-1.0); }
          for (int j=0;j<q1;j++) psc[i+(size_t)(iL1+j)*n] += A1*pXR1[i+(size_t)j*n]*pds1[r+(size_t)j*R];
          for (int j=0;j<q2;j++) psc[i+(size_t)(iL2+j)*n] += A2*pXR2[i+(size_t)j*n]*pds2[r+(size_t)j*R];
          psc[i+(size_t)im1*n] += wv*s_logm1; psc[i+(size_t)im2*n] += wv*s_logm2; psc[i+(size_t)iz*n] += wv*s_z;
        }
      }
    }
    #pragma omp critical
    { for (int p=0;p<npar;p++) grad[p]+=gloc[p]; }
  }

  NumericVector g(npar); for (int p=0;p<npar;p++) g[p]=grad[p];
  return List::create(Named("value")=value, Named("gradient")=g, Named("scores")=scores);
}
