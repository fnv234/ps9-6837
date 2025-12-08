#include "a9.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <functional>

using namespace std;

// === Basic operations ===

double dotIm(const Image& im1, const Image& im2) {
    double sum = 0.0;
    for (size_t i = 0; i < im1.data.size(); ++i) {
        sum += im1.data[i] * im2.data[i];
    }
    return sum;
}

Image convolve3(const Image& im, const Kernel& kernel) {
    Image result(im.height, im.width, im.channels);
    int ky_center = kernel.height / 2;
    int kx_center = kernel.width / 2;
    
    for (int c = 0; c < im.channels; ++c) {
        for (int y = 0; y < im.height; ++y) {
            for (int x = 0; x < im.width; ++x) {
                double sum = 0.0;
                for (int ky = 0; ky < kernel.height; ++ky) {
                    for (int kx = 0; kx < kernel.width; ++kx) {
                        int iy = y + ky - ky_center;
                        int ix = x + kx - kx_center;
                        
                        // Reflect boundary conditions
                        if (iy < 0) iy = -iy;
                        if (iy >= im.height) iy = 2 * im.height - iy - 2;
                        if (ix < 0) ix = -ix;
                        if (ix >= im.width) ix = 2 * im.width - ix - 2;
                        
                        sum += im(iy, ix, c) * kernel(ky, kx);
                    }
                }
                result(y, x, c) = sum;
            }
        }
    }
    return result;
}

// === Deconvolution with gradient descent ===

Image applyKernel(const Image& im, const Kernel& kernel) {
    return convolve3(im, kernel);
}

Image applyConjugatedKernel(const Image& im, const Kernel& kernel) {
    // M^T is the kernel flipped
    Kernel flipped(kernel.height, kernel.width);
    for (int y = 0; y < kernel.height; ++y) {
        for (int x = 0; x < kernel.width; ++x) {
            flipped(y, x) = kernel(kernel.height - 1 - y, kernel.width - 1 - x);
        }
    }
    return convolve3(im, flipped);
}

Image computeResidual(const Kernel& kernel, const Image& x, const Image& y) {
    Image mx = applyKernel(x, kernel);
    return subtractImages(y, mx);
}

double computeStepSize(const Image& r, const Kernel& kernel) {
    double r_dot_r = dotIm(r, r);
    Image Mr = applyKernel(r, kernel);
    Image MTMr = applyConjugatedKernel(Mr, kernel);
    double r_dot_MTMr = dotIm(r, MTMr);
    return r_dot_r / r_dot_MTMr;
}

Image deconvGradDescent(const Image& im_blur, const Kernel& kernel, int niter) {
    Image x(im_blur.height, im_blur.width, im_blur.channels);
    
    for (int iter = 0; iter < niter; ++iter) {
        // Compute residual: r = M^T(y - Mx)
        Image y_minus_Mx = computeResidual(kernel, x, im_blur);
        Image r = applyConjugatedKernel(y_minus_Mx, kernel);
        
        // Compute step size
        double alpha = computeStepSize(r, kernel);
        
        // Update x
        x = addImages(x, scaleImage(r, alpha));
        
        if ((iter + 1) % 5 == 0) {
            std::cout << "Iteration " << (iter + 1) << "/" << niter << std::endl;
        }
    }
    
    return x;
}

// === Deconvolution with conjugate gradient ===

double computeGradientStepSize(const Image& r, const Image& d, const Kernel& kernel) {
    double r_dot_r = dotIm(r, r);
    Image Ad = applyAMatrix(d, kernel);
    double d_dot_Ad = dotIm(d, Ad);
    return r_dot_r / d_dot_Ad;
}

double computeConjugateDirectionStepSize(const Image& old_r, const Image& new_r) {
    double new_r_dot_new_r = dotIm(new_r, new_r);
    double old_r_dot_old_r = dotIm(old_r, old_r);
    return new_r_dot_new_r / old_r_dot_old_r;
}

Image deconvCG(const Image& im_blur, const Kernel& kernel, int niter) {
    Image x(im_blur.height, im_blur.width, im_blur.channels);
    
    // r0 = d0 = b - Ax0 = M^T y - M^T M x0
    Image Ax = applyAMatrix(x, kernel);
    Image MTy = applyConjugatedKernel(im_blur, kernel);
    Image r = subtractImages(MTy, Ax);
    Image d = r.copy();
    
    for (int iter = 0; iter < niter; ++iter) {
        // alpha = (r_i · r_i) / (d_i · Ad_i)
        double alpha = computeGradientStepSize(r, d, kernel);
        
        // x_i+1 = x_i + alpha * d_i
        x = addImages(x, scaleImage(d, alpha));
        
        // r_i+1 = r_i - alpha * Ad_i
        Image Ad = applyAMatrix(d, kernel);
        Image old_r = r.copy();
        r = subtractImages(r, scaleImage(Ad, alpha));
        
        // beta = (r_i+1 · r_i+1) / (r_i · r_i)
        double beta = computeConjugateDirectionStepSize(old_r, r);
        
        // d_i+1 = r_i+1 + beta * d_i
        d = addImages(r, scaleImage(d, beta));
        
        if ((iter + 1) % 5 == 0) {
            std::cout << "CG Iteration " << (iter + 1) << "/" << niter << std::endl;
        }
    }
    
    return x;
}

// === Regularized deconvolution ===

Kernel laplacianKernel() {
    Kernel lap(3, 3);
    lap(0, 0) = 0;  lap(0, 1) = -1; lap(0, 2) = 0;
    lap(1, 0) = -1; lap(1, 1) = 4;  lap(1, 2) = -1;
    lap(2, 0) = 0;  lap(2, 1) = -1; lap(2, 2) = 0;
    return lap;
}

Image applyLaplacian(const Image& im) {
    Kernel lap = laplacianKernel();
    return convolve3(im, lap);
}

Image applyAMatrix(const Image& im, const Kernel& kernel) {
    Image Mx = applyKernel(im, kernel);
    return applyConjugatedKernel(Mx, kernel);
}

Image applyRegularizedOperator(const Image& im, const Kernel& kernel, double lamb) {
    Image Ax = applyAMatrix(im, kernel);
    Image Lx = applyLaplacian(im);
    return addImages(Ax, scaleImage(Lx, lamb));
}

double computeGradientStepSize_reg(const Image& grad, const Image& p, 
                                    const Kernel& kernel, double lamb) {
    double r_dot_r = dotIm(grad, grad);
    Image ALp = applyRegularizedOperator(p, kernel, lamb);
    double p_dot_ALp = dotIm(p, ALp);
    return r_dot_r / p_dot_ALp;
}

Image deconvCG_reg(const Image& im_blur, const Kernel& kernel, 
                   double lamb, int niter) {
    Image x(im_blur.height, im_blur.width, im_blur.channels);
    
    // r0 = d0 = b - (A + lambda*L)x0
    Image ALx = applyRegularizedOperator(x, kernel, lamb);
    Image MTy = applyConjugatedKernel(im_blur, kernel);
    Image r = subtractImages(MTy, ALx);
    Image d = r.copy();
    
    for (int iter = 0; iter < niter; ++iter) {
        // alpha = (r_i · r_i) / (d_i · (A+lambda*L)d_i)
        double alpha = computeGradientStepSize_reg(r, d, kernel, lamb);
        
        // x_i+1 = x_i + alpha * d_i
        x = addImages(x, scaleImage(d, alpha));
        
        // r_i+1 = r_i - alpha * (A+lambda*L)d_i
        Image ALd = applyRegularizedOperator(d, kernel, lamb);
        Image old_r = r.copy();
        r = subtractImages(r, scaleImage(ALd, alpha));
        
        // beta = (r_i+1 · r_i+1) / (r_i · r_i)
        double beta = computeConjugateDirectionStepSize(old_r, r);
        
        // d_i+1 = r_i+1 + beta * d_i
        d = addImages(r, scaleImage(d, beta));
        
        if ((iter + 1) % 5 == 0) {
            std::cout << "Regularized CG Iteration " << (iter + 1) << "/" << niter << std::endl;
        }
    }
    
    return x;
}

// === Image compositing ===

Image naiveComposite(const Image& bg, const Image& fg, const Image& mask, int y, int x) {
    Image result = bg.copy();
    
    for (int fy = 0; fy < fg.height; ++fy) {
        for (int fx = 0; fx < fg.width; ++fx) {
            int by = y + fy;
            int bx = x + fx;
            
            if (by >= 0 && by < bg.height && bx >= 0 && bx < bg.width) {
                double m = mask(fy, fx, 0);
                for (int c = 0; c < bg.channels; ++c) {
                    result(by, bx, c) = m * fg(fy, fx, c) + (1.0 - m) * bg(by, bx, c);
                }
            }
        }
    }
    
    return result;
}

Image Poisson(const Image& bg, const Image& fg, const Image& mask, int niter) {
    Image x = bg.copy();
    
    // b = Laplacian of source
    Image b = applyLaplacian(fg);
    
    for (int iter = 0; iter < niter; ++iter) {
        // r = b - Ax (masked)
        Image Ax = applyLaplacian(x);
        Image r = subtractImages(b, Ax);
        r = multiplyImages(r, mask);
        
        // alpha = (r · r) / (r · Ar)
        Image Ar = applyLaplacian(r);
        double r_dot_r = dotIm(r, r);
        double r_dot_Ar = dotIm(r, Ar);
        double alpha = r_dot_r / r_dot_Ar;
        
        // Update x only in masked region
        Image update = scaleImage(r, alpha);
        x = addImages(x, update);
        
        // Restore background values outside mask
        for (int y = 0; y < x.height; ++y) {
            for (int xi = 0; xi < x.width; ++xi) {
                if (mask(y, xi, 0) < 0.5) {
                    for (int c = 0; c < x.channels; ++c) {
                        x(y, xi, c) = bg(y, xi, c);
                    }
                }
            }
        }
        
        if ((iter + 1) % 50 == 0) {
            std::cout << "Poisson Iteration " << (iter + 1) << "/" << niter << std::endl;
        }
    }
    
    return x;
}

Image PoissonCG(const Image& bg, const Image& fg, const Image& mask, int niter) {
    Image x = bg.copy();
    
    // b = Laplacian of source
    Image b = applyLaplacian(fg);
    
    // r0 = d0 = b - Ax0 (masked)
    Image Ax = applyLaplacian(x);
    Image r = subtractImages(b, Ax);
    r = multiplyImages(r, mask);
    Image d = r.copy();
    
    for (int iter = 0; iter < niter; ++iter) {
        // alpha = (r · r) / (d · Ad)
        Image Ad = applyLaplacian(d);
        double r_dot_r = dotIm(r, r);
        double d_dot_Ad = dotIm(d, Ad);
        double alpha = r_dot_r / d_dot_Ad;
        
        // x = x + alpha * d
        x = addImages(x, scaleImage(d, alpha));
        
        // r_new = r - alpha * Ad (masked)
        Image old_r = r.copy();
        r = subtractImages(r, scaleImage(Ad, alpha));
        r = multiplyImages(r, mask);
        
        // beta = (r_new · r_new) / (r_old · r_old)
        double beta = computeConjugateDirectionStepSize(old_r, r);
        
        // d = r + beta * d (masked)
        d = addImages(r, scaleImage(d, beta));
        d = multiplyImages(d, mask);
        
        // Restore background values outside mask
        for (int y = 0; y < x.height; ++y) {
            for (int xi = 0; xi < x.width; ++xi) {
                if (mask(y, xi, 0) < 0.5) {
                    for (int c = 0; c < x.channels; ++c) {
                        x(y, xi, c) = bg(y, xi, c);
                    }
                }
            }
        }
        
        if ((iter + 1) % 50 == 0) {
            std::cout << "Poisson CG Iteration " << (iter + 1) << "/" << niter << std::endl;
        }
    }
    
    return x;
}

// === Helper functions ===

std::vector<double> horiGaussKernel(double sigma, int truncate) {
    int size = 2 * int(sigma * truncate) + 1;
    std::vector<double> kernel(size);
    int center = size / 2;
    double sum = 0.0;
    
    for (int i = 0; i < size; ++i) {
        double x = i - center;
        kernel[i] = std::exp(-x * x / (2.0 * sigma * sigma));
        sum += kernel[i];
    }
    
    for (int i = 0; i < size; ++i) {
        kernel[i] /= sum;
    }
    
    return kernel;
}

Kernel gauss2D(double sigma, int truncate) {
    std::vector<double> kernel1d = horiGaussKernel(sigma, truncate);
    int size = kernel1d.size();
    Kernel kernel2d(size, size);
    
    double sum = 0.0;
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            kernel2d(y, x) = kernel1d[y] * kernel1d[x];
            sum += kernel2d(y, x);
        }
    }
    
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            kernel2d(y, x) /= sum;
        }
    }
    
    return kernel2d;
}

Image addImages(const Image& im1, const Image& im2) {
    Image result(im1.height, im1.width, im1.channels);
    for (size_t i = 0; i < im1.data.size(); ++i) {
        result.data[i] = im1.data[i] + im2.data[i];
    }
    return result;
}

Image subtractImages(const Image& im1, const Image& im2) {
    Image result(im1.height, im1.width, im1.channels);
    for (size_t i = 0; i < im1.data.size(); ++i) {
        result.data[i] = im1.data[i] - im2.data[i];
    }
    return result;
}

Image scaleImage(const Image& im, double scale) {
    Image result(im.height, im.width, im.channels);
    for (size_t i = 0; i < im.data.size(); ++i) {
        result.data[i] = im.data[i] * scale;
    }
    return result;
}

Image multiplyImages(const Image& im1, const Image& im2) {
    Image result(im1.height, im1.width, im1.channels);
    for (int y = 0; y < im1.height; ++y) {
        for (int x = 0; x < im1.width; ++x) {
            for (int c = 0; c < im1.channels; ++c) {
                result(y, x, c) = im1(y, x, c) * im2(y, x, 0);
            }
        }
    }
    return result;
}

// Richardson-Lucy deconvolution
// Additional functionality - denoising

// === Helper elementwise ops ===

// Elementwise multiply (supports multi-channel)
Image elementwiseMultiply(const Image& a, const Image& b) {
    Image out(a.height, a.width, a.channels);
    for (int y = 0; y < a.height; ++y) {
        for (int x = 0; x < a.width; ++x) {
            for (int c = 0; c < a.channels; ++c) {
                out(y,x,c) = a(y,x,c) * b(y,x,c);
            }
        }
    }
    return out;
}

// Divide num / denom (per channel) with tiny eps to avoid div-by-zero.
// denom expected multi-channel or single-channel: if single-channel, broadcast.
Image divideImages(const Image& num, const Image& denom, double eps) {
    Image out(num.height, num.width, num.channels);
    bool denomIsSingleChannel = (denom.channels == 1);
    for (int y = 0; y < num.height; ++y) {
        for (int x = 0; x < num.width; ++x) {
            for (int c = 0; c < num.channels; ++c) {
                double d = denomIsSingleChannel ? denom(y,x,0) : denom(y,x,c);
                out(y,x,c) = num(y,x,c) / (d + eps);
            }
        }
    }
    return out;
}

// Clamp helper
static inline double clamp01(double v) {
    if (v < 0.0) return 0.0;
    if (v > 1.0) return 1.0;
    return v;
}

// === Richardson-Lucy deconvolution ===
// x_{k+1} = x_k * ( K^T ( y / (K x_k + eps) ) )
Image richardsonLucy(const Image& im_blur, const Kernel& kernel, int niter, double eps) {
    // initialize with blurred image (non-negative)
    Image x = im_blur.copy();

    for (int iter = 0; iter < niter; ++iter) {
        // K * x
        Image Kx = applyKernel(x, kernel);

        // ratio = y / (Kx + eps)
        Image ratio = divideImages(im_blur, addImages(Kx, scaleImage(Kx, 0.0) /* no-op to get same shape */), eps);
        // Note: addImages(Kx, scaleImage(Kx,0)) is just to ensure we pass an Image; kept for clarity.

        // correction = K^T (ratio)
        Image correction = applyConjugatedKernel(ratio, kernel);

        // x = x * correction (elementwise)
        x = elementwiseMultiply(x, correction);

        // Optional small safeguard: keep non-negative and avoid NaNs/Infs
        for (size_t i = 0; i < x.data.size(); ++i) {
            if (!isfinite(x.data[i]) || x.data[i] < 0.0) x.data[i] = 0.0;
        }

        if ((iter + 1) % 5 == 0) {
            std::cout << "Richardson-Lucy Iteration " << (iter + 1) << "/" << niter << std::endl;
        }
    }
    return x;
}

// === Gaussian denoising (separable implemented by gauss2D + convolve3) ===
Image gaussianDenoise(const Image& im, double sigma, int truncate) {
    Kernel k = gauss2D(sigma, truncate);
    return convolve3(im, k);
}

// === Perona-Malik anisotropic diffusion (simple explicit scheme).
// Runs per-channel independently.
Image anisotropicDiffusion(const Image& im, int niter, double kappa, double lambda) {
    Image u = im.copy();
    // 4-neighborhood finite differences
    for (int iter = 0; iter < niter; ++iter) {
        Image u_new = u.copy();
        for (int y = 0; y < u.height; ++y) {
            for (int x = 0; x < u.width; ++x) {
                for (int c = 0; c < u.channels; ++c) {
                    double center = u(y,x,c);
                    // neighbors with reflect BCs
                    int ym = (y == 0) ? 1 : y-1;
                    int yp = (y == u.height-1) ? u.height-2 : y+1;
                    int xm = (x == 0) ? 1 : x-1;
                    int xp = (x == u.width-1) ? u.width-2 : x+1;

                    double north = u(ym,x,c);
                    double south = u(yp,x,c);
                    double west  = u(y,xm,c);
                    double east  = u(y,xp,c);

                    double dn = north - center;
                    double ds = south - center;
                    double dw = west  - center;
                    double de = east  - center;

                    // Perona-Malik diffusivity (exponential)
                    double cN = std::exp(-(dn*dn) / (kappa*kappa));
                    double cS = std::exp(-(ds*ds) / (kappa*kappa));
                    double cW = std::exp(-(dw*dw) / (kappa*kappa));
                    double cE = std::exp(-(de*de) / (kappa*kappa));

                    double update = lambda * (cN*dn + cS*ds + cW*dw + cE*de);
                    u_new(y,x,c) = center + update;
                }
            }
        }
        u = std::move(u_new);
        if ((iter+1) % 10 == 0) {
            std::cout << "Anisotropic diffusion iteration " << (iter+1) << "/" << niter << std::endl;
        }
    }
    return u;
}

// === Simple Non-Local Means (slow but effective for small images)
// patchRadius: half patch size (e.g., 1 or 2), searchRadius: search window radius (e.g., 5), h: filtering parameter
Image nlMeansDenoise(const Image& im, int patchRadius, int searchRadius, double h) {
    Image out(im.height, im.width, im.channels);
    const int patchSize = 2 * patchRadius + 1;
    const int searchSize = 2 * searchRadius + 1;
    const double h2 = h * h;

    // precompute gaussian patch weights (spatial) to speed similarity
    std::vector<std::vector<double>> spatialWeight(patchSize, std::vector<double>(patchSize));
    int center = patchRadius;
    double spatSum = 0.0;
    double sigma_spat = patchRadius + 1.0;
    for (int py = 0; py < patchSize; ++py) {
        for (int px = 0; px < patchSize; ++px) {
            double dy = py - center;
            double dx = px - center;
            spatialWeight[py][px] = std::exp(-(dx*dx + dy*dy) / (2.0 * sigma_spat * sigma_spat));
            spatSum += spatialWeight[py][px];
        }
    }
    for (int py = 0; py < patchSize; ++py)
        for (int px = 0; px < patchSize; ++px)
            spatialWeight[py][px] /= spatSum;

    // For each pixel, compute NLM weighted average
    for (int y = 0; y < im.height; ++y) {
        for (int x = 0; x < im.width; ++x) {
            // accumulate weights per channel
            std::vector<double> accum(im.channels, 0.0);
            double wsum = 0.0;

            // reference patch centered at (y,x)
            for (int sy = std::max(0, y - searchRadius); sy <= std::min(im.height-1, y + searchRadius); ++sy) {
                for (int sx = std::max(0, x - searchRadius); sx <= std::min(im.width-1, x + searchRadius); ++sx) {
                    // compute squared patch distance
                    double dist2 = 0.0;
                    for (int py = -patchRadius; py <= patchRadius; ++py) {
                        int ry = y + py;
                        int syy = sy + py;
                        // reflect boundaries
                        if (ry < 0) ry = -ry;
                        if (ry >= im.height) ry = 2*im.height - ry - 2;
                        if (syy < 0) syy = -syy;
                        if (syy >= im.height) syy = 2*im.height - syy - 2;

                        for (int px = -patchRadius; px <= patchRadius; ++px) {
                            int rx = x + px;
                            int sxx = sx + px;
                            if (rx < 0) rx = -rx;
                            if (rx >= im.width) rx = 2*im.width - rx - 2;
                            if (sxx < 0) sxx = -sxx;
                            if (sxx >= im.width) sxx = 2*im.width - sxx - 2;

                            int pyy = py + patchRadius;
                            int pxx = px + patchRadius;
                            double sw = spatialWeight[pyy][pxx];
                            for (int c = 0; c < im.channels; ++c) {
                                double diff = im(ry,rx,c) - im(syy,sxx,c);
                                dist2 += sw * diff * diff;
                            }
                        }
                    }
                    double w = std::exp(-std::max(0.0, dist2) / h2);
                    wsum += w;
                    for (int c = 0; c < im.channels; ++c) {
                        accum[c] += w * im(sy, sx, c);
                    }
                }
            }
            if (wsum > 1e-12) {
                for (int c = 0; c < im.channels; ++c) {
                    out(y,x,c) = accum[c] / wsum;
                }
            } else {
                for (int c = 0; c < im.channels; ++c) out(y,x,c) = im(y,x,c);
            }
        }
    }
    return out;
}


// IGNORE FOR NOW
// EXTRA CREDIT - from Perez paper
// Perez poisson blending

Image CG_custom(
    const std::function<Image(const Image&)>& A,
    const Image& b,
    Image x,
    int niter
) {
    Image r = subtractImages(b, A(x));
    Image d = r.copy();
    double rsold = dotIm(r, r);
    const double tiny = 1e-12;

    for (int i = 0; i < niter; ++i) {
        Image Ad = A(d);
        double dDotAd = dotIm(d, Ad);
        
        // Check for numerical issues
        if (std::abs(dDotAd) < tiny || std::isnan(dDotAd) || std::isinf(dDotAd)) {
            std::cout << "CG_custom: numerical issue with d^T A d (" << dDotAd << "), stopping\n";
            break;
        }
        
        double alpha = rsold / dDotAd;
        
        // Check alpha for numerical issues
        if (std::isnan(alpha) || std::isinf(alpha)) {
            std::cout << "CG_custom: numerical issue with alpha (" << alpha << "), stopping\n";
            break;
        }

        x = addImages(x, scaleImage(d, alpha));

        Image r_new = subtractImages(r, scaleImage(Ad, alpha));
        double rsnew = dotIm(r_new, r_new);
        
        // Check for convergence or numerical issues
        if (rsnew < 1e-12 || std::isnan(rsnew) || std::isinf(rsnew)) {
            if (rsnew < 1e-12) {
                std::cout << "CG_custom: converged after " << (i + 1) << " iterations\n";
            }
            r = r_new;
            break;
        }

        double beta = rsnew / rsold;
        
        // Check beta for numerical issues
        if (std::isnan(beta) || std::isinf(beta)) {
            std::cout << "CG_custom: numerical issue with beta (" << beta << "), stopping\n";
            break;
        }
        
        d = addImages(r_new, scaleImage(d, beta));

        r = r_new;
        rsold = rsnew;

        if ((i + 1) % 10 == 0) {
            std::cout << "CG_custom iter " << (i + 1) << "/" << niter << " (res=" << sqrt(rsold) << ")\n";
        }
    }

    return x;
}

// Image poissonPerez(const Image& bg, const Image& fg, const Image& mask, int cgIters) {
//     int H = fg.height;
//     int W = fg.width;
//     int C = fg.channels;

//     auto inOmega = [&](int y, int x) -> bool {
//         return mask(y, x, 0) > 0.5;
//     };

//     const int nOff = 4;
//     const int dy[nOff] = {-1, 1, 0, 0};
//     const int dx[nOff] = {0, 0, -1, 1};

//     Image b(H, W, C);
//     for (size_t i = 0; i < b.data.size(); ++i) b.data[i] = 0.0;

//     for (int y = 0; y < H; ++y) {
//         for (int x = 0; x < W; ++x) {
//             if (!inOmega(y, x)) continue;

//             for (int k = 0; k < nOff; ++k) {
//                 int ny = y + dy[k];
//                 int nx = x + dx[k];

//                 if (ny < 0 || ny >= H || nx < 0 || nx >= W) {
//                     continue;
//                 }

//                 for (int c = 0; c < C; ++c) {
//                     double diff_f = bg(y, x, c) - bg(ny, nx, c); // f*_p - f*_q
//                     double diff_g = fg(y, x, c) - fg(ny, nx, c); // g_p - g_q

//                     double v_pq = (std::abs(diff_f) > std::abs(diff_g)) ? diff_f : diff_g;

//                     // add v_pq to b_p
//                     b(y, x, c) += v_pq;
//                 }

//                 if (!inOmega(ny, nx)) {
//                     for (int c = 0; c < C; ++c) {
//                         b(y, x, c) += bg(ny, nx, c);
//                     }
//                 }
//             }
//         }
//     }

  
//     auto A_op = [&](const Image& z) -> Image {
//         Image out(H, W, C);
//         for (size_t i = 0; i < out.data.size(); ++i) out.data[i] = 0.0;

//         for (int y = 0; y < H; ++y) {
//             for (int x = 0; x < W; ++x) {
//                 if (!inOmega(y, x)) continue; 

//                 int neighborCount = 0;
//                 for (int k = 0; k < nOff; ++k) {
//                     int ny = y + dy[k];
//                     int nx = x + dx[k];
//                     if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue;
//                     neighborCount++;
//                 }

//                 for (int c = 0; c < C; ++c) {
//                     double sumNeighborInside = 0.0;
//                     for (int k = 0; k < nOff; ++k) {
//                         int ny = y + dy[k];
//                         int nx = x + dx[k];
//                         if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue;
//                         if (inOmega(ny, nx)) {
//                             sumNeighborInside += z(ny, nx, c);
//                         }
//                     }
//                     out(y, x, c) = neighborCount * z(y, x, c) - sumNeighborInside;
//                 }
//             }
//         }
//         return out;
//     };

//     Image x0 = bg.copy();

//     for (int y = 0; y < H; ++y)
//         for (int x = 0; x < W; ++x)
//             if (!inOmega(y, x))
//                 for (int c = 0; c < C; ++c)
//                     b(y, x, c) = 0.0;

//     Image sol = CG_custom(A_op, b, x0, cgIters);

//     for (int y = 0; y < H; ++y) {
//         for (int x = 0; x < W; ++x) {
//             if (!inOmega(y, x)) {
//                 for (int c = 0; c < C; ++c) {
//                     sol(y, x, c) = bg(y, x, c);
//                 }
//             }
//         }
//     }

//     return sol;
// }

// // ================== IRLS for Lp Regularized Deconvolution (p = 1 or 0.8) ==================

// Image computeIRLSWeights(const Image& x, double p, double eps) {
//     Image Lx = applyLaplacian(x);
//     int H = x.height, W = x.width, C = x.channels;

//     Image Wimg(H, W, 1);

//     for (int y = 0; y < H; ++y) {
//         for (int xj = 0; xj < W; ++xj) {
//             double accum = 0.0;
//             for (int c = 0; c < C; ++c) {
//                 accum += std::abs(Lx(y,xj,c));
//             }
//             accum /= C;

//             Wimg(y,xj,0) = 1.0 / std::pow(accum + eps, 2.0 - p);
//         }
//     }

//     return Wimg;
// }

// Image deconvIRLS(const Image& y, const Kernel& kernel,
//                  double p, double lambda, int outerIters, int cgIters)
// {
//     // Start with initial guess (could be y or zeros)
//     Image x = y.copy();
//     const double eps = 1e-6;

//     for (int i = 0; i < outerIters; ++i) {
//         // Compute IRLS weights based on current solution
//         Image W = computeIRLSWeights(x, p, eps);

//         // Set up the regularized operator A = M^T M + lambda * W * L
//         auto A_op = [&](const Image& z)->Image {
//             Image MTMz = applyAMatrix(z, kernel);
//             Image Lz = applyLaplacian(z);
//             Image WLz = multiplyImages(Lz, W);
//             return addImages(MTMz, scaleImage(WLz, lambda));
//         };

//         // Right-hand side: M^T y (where y is the blurred image)
//         Image b = applyConjugatedKernel(y, kernel);

//         // Solve (M^T M + lambda * W * L) x = M^T y using CG
//         x = CG_custom(A_op, b, x, cgIters);
        
//         // Clamp values to valid range
//         for (size_t j = 0; j < x.data.size(); j++) {
//             x.data[j] = std::max(0.0, std::min(1.0, x.data[j]));
//         }
        
//         std::cout << "IRLS outer iteration " << i+1 << "/" << outerIters << std::endl;
//     }

//     return x;
// }
// inline double fwdX(const Image& im, int y, int x, int c) {
//     int xp = (x == im.width-1) ? x : x+1;
//     return im(y,xp,c) - im(y,x,c);
// }

// inline double bwdX(const Image& im, int y, int x, int c) {
//     int xm = (x == 0) ? x : x-1;
//     return im(y,x,c) - im(y,xm,c);
// }

// inline double fwdY(const Image& im, int y, int x, int c) {
//     int yp = (y == im.height-1) ? y : y+1;
//     return im(yp,x,c) - im(y,x,c);
// }

// // === RESIZE FUNCTION ===
Image resizeToMatch(const Image& source, int target_height, int target_width, bool is_mask) {
    Image result(target_height, target_width, source.channels);
    
    float scale_y = (float)source.height / target_height;
    float scale_x = (float)source.width / target_width;
    
    for (int y = 0; y < target_height; y++) {
        for (int x = 0; x < target_width; x++) {
            // Map to source coordinates
            float src_y = y * scale_y;
            float src_x = x * scale_x;
            
            // Nearest neighbor
            int src_y_int = (int)src_y;
            int src_x_int = (int)src_x;
            
            // Clamp to source bounds
            src_y_int = std::max(0, std::min(source.height - 1, src_y_int));
            src_x_int = std::max(0, std::min(source.width - 1, src_x_int));
            
            // Copy all channels
            for (int c = 0; c < source.channels; c++) {
                double val = source(src_y_int, src_x_int, c);
                
                // For masks, ensure binary values after resize
                if (is_mask) {
                    // Threshold at 0.5 to get clean mask
                    result(y, x, c) = (val > 0.5) ? 1.0 : 0.0;
                } else {
                    result(y, x, c) = val;
                }
            }
        }
    }
    return result;
}

// === PROPER GRADIENT COMPUTATION ===
void computeGradients(const Image& im, Image& gx, Image& gy) {
    for (int y = 0; y < im.height; y++) {
        for (int x = 0; x < im.width; x++) {
            for (int c = 0; c < im.channels; c++) {
                // Forward differences with boundary handling
                gx(y, x, c) = (x < im.width - 1) ? 
                    im(y, x + 1, c) - im(y, x, c) : 0;
                gy(y, x, c) = (y < im.height - 1) ? 
                    im(y + 1, x, c) - im(y, x, c) : 0;
            }
        }
    }
}

// inline double bwdY(const Image& im, int y, int x, int c) {
//     int ym = (y == 0) ? y : y-1;
//     return im(y,x,c) - im(ym,x,c);
// }
Image divergenceMixed(const Image& gx, const Image& gy) {
    int H = gx.height, W = gx.width, C = gx.channels;
    Image out(H, W, C);
}
// Image divergenceMixed(const Image& gx, const Image& gy) {
//     int H = gx.height, W = gx.width, C = gx.channels;
//     Image out(H, W, C);

//     for (int y = 0; y < H; ++y) {
//         for (int x = 0; x < W; ++x) {
//             for (int c = 0; c < C; ++c) {
//                 double divx = bwdX(gx,y,x,c);
//                 double divy = fwdY(gy,y,x,c);
//                 out(y,x,c) = divx + divy;
//             }
//         }
Image poissonPerez(const Image& bg, const Image& fg, const Image& mask)
{
    int H = bg.height, W = bg.width, C = bg.channels;
    Image out = bg;

    // ----------------------------------------------------
    // 1. Build Ω index map
    // ----------------------------------------------------
    std::vector<std::pair<int,int>> omega;
    std::vector<std::vector<int>> grid(H, std::vector<int>(W, -1));

    for (int y=0; y<H; y++)
        for (int x=0; x<W; x++)
            if (mask(y,x,0) > 0.5) {
                grid[y][x] = omega.size();
                omega.emplace_back(y,x);
            }

    int n = omega.size();
    if (n == 0) return out;

    // ----------------------------------------------------
    // 2. Precompute source Laplacian
    // ----------------------------------------------------
    Kernel lap(3,3);
    lap(0,1)=lap(1,0)=lap(1,2)=lap(2,1)=1;
    lap(1,1)=-4;

    Image lapSrc(fg.height, fg.width, fg.channels);

    for (int y = 0; y < fg.height; y++) {
        for (int x = 0; x < fg.width; x++) {
            for (int c = 0; c < fg.channels; c++) {
                double center = fg(y,x,c);
                double up    = (y > 0)            ? fg(y-1,x,c) : center;
                double down  = (y < fg.height-1)  ? fg(y+1,x,c) : center;
                double left  = (x > 0)            ? fg(y,x-1,c) : center;
                double right = (x < fg.width-1)   ? fg(y,x+1,c) : center;

                lapSrc(y,x,c) = 4*center - (up + down + left + right);
            }
        }
    }


    const int dy[4] = {-1,1,0,0};
    const int dx[4] = {0,0,-1,1};

    // ----------------------------------------------------
    // 3. Solve per channel
    // ----------------------------------------------------
    for (int c=0; c<C; c++) {

        std::vector<double> b(n, 0.0);
        std::vector<double> x(n, 0.0);

        // ---- Build RHS ----
        for (int i=0; i<n; i++) {
            int y = omega[i].first;
            int xp = omega[i].second;

            // Boundary conditions
            for (int k=0; k<4; k++) {
                int ny=y+dy[k], nx=xp+dx[k];
                if (ny>=0 && ny<H && nx>=0 && nx<W) {
                    if (grid[ny][nx] < 0)
                        b[i] += bg(ny,nx,c);
                } else {
                    // outside image = fixed background
                    b[i] += bg(y,xp,c);
                }

            }

            // Source Laplacian
            b[i] += lapSrc(y,xp,c);
        }

        // ---- Laplacian operator A·v ----
        auto A_op = [&](const std::vector<double>& v)
        {
            std::vector<double> outv(n, 0.0);
            for (int i=0; i<n; i++) {
                int y = omega[i].first;
                int xp = omega[i].second;
                outv[i] = 4 * v[i];
                for (int k=0; k<4; k++) {
                    int ny=y+dy[k], nx=xp+dx[k];
                    if (ny>=0 && ny<H && nx>=0 && nx<W) {
                        int j = grid[ny][nx];
                        if (j >= 0) outv[i] -= v[j];
                    }
                }
            }
            return outv;
        };

        // ----------------------------------------------------
        // 4. Conjugate Gradient
        // ----------------------------------------------------
        cgSolve(x, b, A_op, 1000, 1e-6);

        // ----------------------------------------------------
        // 5. Write back solution
        // ----------------------------------------------------
        for (int i=0; i<n; i++) {
            int y = omega[i].first;
            int xp = omega[i].second;
            out(y,xp,c) = std::min(1.0, std::max(0.0, x[i]));
        }
    }

    return out;
}
void cgSolve(
    std::vector<double>& x,
    const std::vector<double>& b,
    const std::function<std::vector<double>(const std::vector<double>&)>& A,
    int maxIter,
    double tol
)
{
    int n = x.size();
    std::vector<double> r(n), p(n), Ap(n);

    Ap = A(x);
    for (int i=0;i<n;i++) r[i] = b[i] - Ap[i];
    p = r;

    double rsold = inner_product(r.begin(), r.end(), r.begin(), 0.0);

    for (int k=0;k<maxIter;k++) {
        Ap = A(p);
        double alpha = rsold /
            inner_product(p.begin(), p.end(), Ap.begin(), 0.0);

        for (int i=0;i<n;i++) x[i] += alpha * p[i];
        for (int i=0;i<n;i++) r[i] -= alpha * Ap[i];

        double rsnew = inner_product(r.begin(), r.end(), r.begin(), 0.0);
        if (sqrt(rsnew) < tol) break;

        for (int i=0;i<n;i++) p[i] = r[i] + (rsnew / rsold) * p[i];
        rsold = rsnew;
    }
}
