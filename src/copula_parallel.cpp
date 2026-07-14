// Multithreaded (OpenMP) C++ core for the copula RP-BNB likelihood + gradient.
// Bivariate-normal CDF via Genz's tvpack bvnu (Drezner-Wesolowsky).
#include <Rcpp.h>
#include <cmath>
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
