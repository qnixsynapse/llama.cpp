#!/usr/bin/env python3
"""
Compute optimal Lloyd-Max centroids for TurboQuant at block size d=64.

After Hadamard rotation, each coordinate follows the Beta distribution
from Lemma 1 of the TurboQuant paper (arXiv:2504.19874):

    f_X(x) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1 - x^2)^((d-3)/2)

with variance 1/d. The quantizer normalizes by the block's RMS, so the
effective distribution is the standardized version Y = X * sqrt(d) with
unit variance:

    f_Y(y) = (1/sqrt(d)) * f_X(y/sqrt(d))

We solve the Lloyd-Max optimization (Eq. 4 in the paper) for this exact
distribution at d=64, rather than using the N(0,1) approximation which
is only tight for large d.
"""

import numpy as np
from scipy import integrate, special

D = 64  # block size

def beta_pdf(x):
    """Exact Beta PDF from Lemma 1, for coordinate of random point on S^{d-1}."""
    if abs(x) >= 1.0:
        return 0.0
    coeff = special.gamma(D / 2.0) / (np.sqrt(np.pi) * special.gamma((D - 1) / 2.0))
    return coeff * (1.0 - x * x) ** ((D - 3) / 2.0)

def standardized_pdf(y):
    """PDF of Y = X * sqrt(d), which has unit variance."""
    x = y / np.sqrt(D)
    if abs(x) >= 1.0:
        return 0.0
    return beta_pdf(x) / np.sqrt(D)

def lloyd_max(num_levels, pdf, support_lo, support_hi, max_iter=1000, tol=1e-12):
    """
    Lloyd-Max algorithm for optimal scalar quantization.

    Finds centroids c_1 < c_2 < ... < c_K that minimize:
        sum_i integral_{b_{i-1}}^{b_i} (x - c_i)^2 * pdf(x) dx

    where boundaries b_i = (c_i + c_{i+1}) / 2.
    """
    # Initialize with uniform quantile spacing
    centroids = np.array([
        support_lo + (support_hi - support_lo) * (i + 0.5) / num_levels
        for i in range(num_levels)
    ])

    for iteration in range(max_iter):
        # Compute boundaries (midpoints between consecutive centroids)
        boundaries = [support_lo]
        for i in range(num_levels - 1):
            boundaries.append((centroids[i] + centroids[i + 1]) / 2.0)
        boundaries.append(support_hi)

        # Update centroids: c_i = E[X | b_{i-1} < X < b_i]
        new_centroids = np.zeros(num_levels)
        for i in range(num_levels):
            lo, hi = boundaries[i], boundaries[i + 1]

            numerator, _ = integrate.quad(lambda x: x * pdf(x), lo, hi)
            denominator, _ = integrate.quad(pdf, lo, hi)

            if denominator > 1e-15:
                new_centroids[i] = numerator / denominator
            else:
                new_centroids[i] = (lo + hi) / 2.0

        # Check convergence
        if np.max(np.abs(new_centroids - centroids)) < tol:
            print(f"  Converged after {iteration + 1} iterations")
            centroids = new_centroids
            break
        centroids = new_centroids
    else:
        print(f"  Warning: did not converge after {max_iter} iterations")

    return centroids

def compute_mse(centroids, pdf, support_lo, support_hi):
    """Compute MSE distortion for given centroids."""
    num_levels = len(centroids)
    boundaries = [support_lo]
    for i in range(num_levels - 1):
        boundaries.append((centroids[i] + centroids[i + 1]) / 2.0)
    boundaries.append(support_hi)

    mse = 0.0
    for i in range(num_levels):
        lo, hi = boundaries[i], boundaries[i + 1]
        c = centroids[i]
        val, _ = integrate.quad(lambda x: (x - c) ** 2 * pdf(x), lo, hi)
        mse += val
    return mse

if __name__ == "__main__":
    # Verify standardized PDF integrates to 1 and has unit variance
    norm, _ = integrate.quad(standardized_pdf, -np.sqrt(D), np.sqrt(D))
    var, _  = integrate.quad(lambda y: y**2 * standardized_pdf(y), -np.sqrt(D), np.sqrt(D))
    print(f"Block size d = {D}")
    print(f"Standardized PDF: integral = {norm:.6f}, variance = {var:.6f}")
    print()

    support = (-np.sqrt(D), np.sqrt(D))

    # Current N(0,1) centroids for comparison
    gauss_centroids_2bit = np.array([-1.5104, -0.4528, 0.4528, 1.5104])
    gauss_centroids_3bit = np.array([-2.1519, -1.3440, -0.7560, -0.2451,
                                      0.2451,  0.7560,  1.3440,  2.1519])

    print("=" * 60)
    print("2-bit (4 levels) — for TVQ3_0")
    print("=" * 60)
    print("\nComputing optimal centroids for Beta(d=64)...")
    opt_2bit = lloyd_max(4, standardized_pdf, *support)
    print(f"\n  Optimal (d=64):  {opt_2bit}")
    print(f"  Gaussian (d→∞):  {gauss_centroids_2bit}")

    mse_opt = compute_mse(opt_2bit, standardized_pdf, *support)
    mse_gauss = compute_mse(gauss_centroids_2bit, standardized_pdf, *support)
    print(f"\n  MSE with optimal centroids:  {mse_opt:.6f}")
    print(f"  MSE with Gaussian centroids: {mse_gauss:.6f}")
    print(f"  Improvement: {(1 - mse_opt/mse_gauss)*100:.2f}%")

    print()
    print("=" * 60)
    print("3-bit (8 levels) — for TVQ4_0")
    print("=" * 60)
    print("\nComputing optimal centroids for Beta(d=64)...")
    opt_3bit = lloyd_max(8, standardized_pdf, *support)
    print(f"\n  Optimal (d=64):  {opt_3bit}")
    print(f"  Gaussian (d→∞):  {gauss_centroids_3bit}")

    mse_opt = compute_mse(opt_3bit, standardized_pdf, *support)
    mse_gauss = compute_mse(gauss_centroids_3bit, standardized_pdf, *support)
    print(f"\n  MSE with optimal centroids:  {mse_opt:.6f}")
    print(f"  MSE with Gaussian centroids: {mse_gauss:.6f}")
    print(f"  Improvement: {(1 - mse_opt/mse_gauss)*100:.2f}%")

    print()
    print("=" * 60)
    print("C code (copy into ggml-quants.c)")
    print("=" * 60)
    print()
    print("// Lloyd-Max optimal centroids for Beta distribution at d=64")
    print("// (standardized to unit variance after Hadamard rotation)")
    print("// 2-bit (4 levels)")
    vals = ", ".join(f"{v:.4f}f" for v in opt_2bit)
    print(f"static const float tvq3_centroids[4] = {{{vals}}};")
    print("// 3-bit (8 levels)")
    vals_lo = ", ".join(f"{v:.4f}f" for v in opt_3bit[:4])
    vals_hi = ", ".join(f"{v:.4f}f" for v in opt_3bit[4:])
    print(f"static const float tvq4_centroids[8] = {{")
    print(f"    {vals_lo},")
    print(f"    {vals_hi}")
    print("};")
