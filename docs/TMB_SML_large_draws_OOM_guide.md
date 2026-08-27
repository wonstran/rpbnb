# TMB 做 Simulated Maximum Likelihood（SML）时 Draws 很大导致 OOM：原因与解决方案

当 TMB 使用 SML、simulation draws 增加到 1000 或更多时，OOM 往往不是
draws 数据本身造成的，而是大量参数相关运算被记录到自动微分（AD）tape
中。对于 RP-NB / correlated RPBNB，这一点尤其明显。

## 1. 最重要：不要构造 `[N × R × K]` 的 `Type` 数组

避免：

``` cpp
array<Type> beta(N, R, K);
array<Type> eta(N, R);
matrix<Type> loglik_draw(N, R);
```

例如 N=50,000、R=1,000、K=8 时，`N×R×K = 400,000,000` 个逻辑元素。`Type`
还会产生 AD 运算记录，因此内存消耗远超过普通 double 数组。

更好的策略是逐 observation / draw 计算，只保留少量临时量。

## 2. Simulation draws 必须作为 DATA

推荐：

``` cpp
DATA_MATRIX(Z);
```

draw 本身不需要求导。尽量保持为普通数据，只在和参数发生运算时进入 `Type`
表达式：

``` cpp
double zij = Z(r, k);
Type beta = mu(k) + sd(k) * zij;
```

Halton / scrambled Halton、`qnorm()` 等都应尽量在 R/Python 中预先生成。

## 3. 不要储存全部 draw likelihood

避免：

``` cpp
matrix<Type> loglik_draw(N, R);
```

应该逐 observation 累积 simulation likelihood：

``` cpp
for (int i = 0; i < N; i++) {
    Type sumprob = 0;

    for (int r = 0; r < R; r++) {
        Type eta = ...;
        Type lp  = ...;
        sumprob += exp(lp);
    }

    nll -= log(sumprob / Type(R));
}
```

这避免显式 `[N,R]` 数组，但要注意：1000 个 draws
的参数相关运算仍然会进入 AD tape。

## 4. 使用稳定的 log-sum-exp

SML 中通常需要：

\[ `\log`{=tex}`\left[\frac{1}{R}\sum_r \exp(\ell_{ir})\right]`{=tex}.
\]

直接 `exp()` 后求和可能发生 overflow/underflow。可使用稳定的 pairwise
log-space accumulation，例如：

``` cpp
Type logspace_add(Type a, Type b) {
    Type m = CppAD::CondExpGt(a, b, a, b);
    return m + log(exp(a - m) + exp(b - m));
}
```

然后逐 draw 累积：

``` cpp
Type lse = lp0;

for (int r = 1; r < R; r++) {
    Type lp = ...;
    lse = logspace_add(lse, lp);
}

nll -= lse - log(Type(R));
```

## 5. Chunking 要谨慎

把 1000 draws 分成 10 × 100 在代码结构上有帮助，但如果所有 chunk
都仍然在同一个 TMB objective 中执行，AD tape 并不会因此自动缩小。

此外：

\[
`\log`{=tex}`\left`{=tex}(`\frac{1}{1000}`{=tex}`\sum`{=tex}\_{r=1}\^{1000}L_r`\right`{=tex})
\]

不能替换为 10 个 100-draw log-likelihood 的简单求和。因此在 R 层拆 chunk
时，需要正确组合每个 observation 的 probability sums 以及对应
derivatives。

## 6. TMB 的核心问题是 AD tape 长度

即使 C++ 代码只使用几个 scalar 临时变量：

``` cpp
for (r = 0; r < 1000; r++) {
    eta = ...;
    lp = ...;
}
```

只要 `eta`、NB likelihood、`exp()`、`log()`、`lgamma()`
等依赖参数，这些运算都会进入 AD tape。

因此 TMB 的 memory scaling 很大程度上与参数相关 AD operation
数量有关，而不仅是显式数组大小。

## 7. 把所有常数计算移出 TMB AD

可以预计算的东西尽量提前计算：

-   Halton / scrambled Halton draws
-   normal transforms (`qnorm`)
-   design-matrix transformations
-   categorical coding
-   exposure / offsets
-   固定常数
-   `lgamma(y + 1)` 等不依赖参数的项

例如 R 中：

``` r
logfact <- lgamma(y + 1)
```

TMB：

``` cpp
DATA_VECTOR(logfact);
```

而不是让这些常数运算进入复杂的 objective 表达式。

## 8. 检查 ADREPORT

不要：

``` cpp
ADREPORT(big_matrix);
ADREPORT(draw_specific_beta);
ADREPORT(draw_likelihood);
```

只对真正需要 delta-method SE 的少量最终参数/派生量使用 `ADREPORT()`。

## 9. OOM 时先降低 OpenMP threads

多线程可能复制 AD/workspace，导致 peak RAM 显著增加。

如果当前使用：

``` r
openmp(8)
```

建议先测试：

``` r
openmp(1)
```

如果单线程不再 OOM，则说明线程相关内存复制是重要原因之一。

## 10. RP-NB / correlated RPBNB 的推荐计算结构

对于

\[ `\beta`{=tex}\_r = `\mu `{=tex}+ Lz_r, \]

不要显式创建每个 observation × draw × coefficient 的 beta。

利用：

\[ `\eta`{=tex}\_{ir} = x_i\^`\top`{=tex}`\mu `{=tex}+
(x_i\^`\top `{=tex}L)z_r. \]

因此每个 observation 先计算一次：

``` cpp
Type base = ...;       // x_i' mu
vector<Type> xL(K);    // x_i' L
```

随后每个 draw：

``` cpp
Type eta = base;

for (int k = 0; k < K; k++)
    eta += xL(k) * Z(r,k);
```

这样活动对象主要只有：

-   `xL[K]`
-   `eta` scalar
-   `lp` scalar
-   `log_sum` scalar

而不是 `beta[N,R,K]`、`eta[N,R]`、`lik[N,R]`。

## 11. 什么时候考虑 JAX

如果 simulation integration 本身就是主要计算负担，尤其：

-   N 很大；
-   R = 1000--5000；
-   correlated random parameters；
-   每个 draw 都需要大量 NB likelihood 运算；

JAX/GPU 往往比继续扩大 TMB tape 更自然。

JAX 可以按 observation batch，例如 B=500--2000：

\[
\[B,R\]`\rightarrow `{=tex}`\text{GPU likelihood}`{=tex}`\rightarrow`{=tex}
`\text{logsumexp}`{=tex}`\rightarrow `{=tex}`\text{gradient}`{=tex}. \]

B=1000、R=1000 时，一个 FP64 `[B,R]` 主矩阵只有约 8 MB。

## 实际建议优先级

1.  删除 `[N,R]` / `[N,R,K]` 的 `Type` 对象。
2.  确保 simulation draws 是 `DATA_*`。
3.  用 `x_i'μ + (x_i'L)z_r`，不要构造完整 random-coefficient tensor。
4.  使用稳定 log-sum-exp。
5.  移出所有 parameter-independent calculations。
6.  删除不必要的 `ADREPORT()`。
7.  OOM 时测试 `openmp(1)`。
8.  再逐步增加 threads / draws，并监控 peak RAM。
9.  如果 R 经常达到 1000--5000 且 N 很大，考虑将 SML 核心迁移到
    JAX/GPU。

## TMB vs JAX 的经验性选择

  场景                                                 更适合
  ---------------------------------------------------- -------------------------
  R = 50--200                                          TMB
  R ≈ 500                                              TMB 可用，但应优化 tape
  R ≈ 1000                                             JAX 开始很有吸引力
  R = 2000--5000                                       JAX/GPU
  大 N + 大 R + correlated RP                          JAX/GPU
  大量 latent random effects + Laplace approximation   TMB

核心结论：**对于 TMB SML，OOM 首先应该解决 AD tape
和计算结构，而不是简单增加 RAM。**
