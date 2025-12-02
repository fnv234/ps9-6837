#include "a9.h"
#include <cmath>
#include <algorithm>
#include <iostream>

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

Image deconvCG_reg_IRL1(const Image& im_blur, const Kernel& kernel, 
                        double lamb, int niter, int ir_iter) {
    Image x(im_blur.height, im_blur.width, im_blur.channels);
    Image weights(im_blur.height, im_blur.width, im_blur.channels);
    for (size_t i = 0; i < weights.data.size(); ++i) {
        weights.data[i] = 1.0; // initial uniform weights
    }

    for (int ir = 0; ir < ir_iter; ++ir) {
        // Solve weighted L2 problem: (A + lambda*W*L) x = b
        Image x_prev = x.copy();

        // r0 = d0 = b - (A + lambda*W*L)x0
        Image Lx = applyLaplacian(x);
        Image WLx = multiplyImages(Lx, weights);
        Image ALx = applyAMatrix(x, kernel);
        Image ALx_WLx = addImages(ALx, scaleImage(WLx, lamb));
        Image MTy = applyConjugatedKernel(im_blur, kernel);
        Image r = subtractImages(MTy, ALx_WLx);
        Image d = r.copy();

        for (int iter = 0; iter < niter; ++iter) {
            // Compute (A + lambda*W*L)d
            Image Ld = applyLaplacian(d);
            Image WLd = multiplyImages(Ld, weights);
            Image Ad = applyAMatrix(d, kernel);
            Image ALd_WLd = addImages(Ad, scaleImage(WLd, lamb));

            double r_dot_r = dotIm(r, r);
            double d_dot_Ad = dotIm(d, ALd_WLd);
            if (d_dot_Ad == 0 || r_dot_r < 1e-10) break;
            double alpha = r_dot_r / d_dot_Ad;

            x = addImages(x, scaleImage(d, alpha));

            Image old_r = r.copy();
            Image ALx_new_WLx = addImages(applyAMatrix(x, kernel), scaleImage(multiplyImages(applyLaplacian(x), weights), lamb));
            r = subtractImages(MTy, ALx_new_WLx);

            double beta = computeConjugateDirectionStepSize(old_r, r);
            d = addImages(r, scaleImage(d, beta));
        }

        // Update weights for L1: w = 1 / (|Lx| + eps)
        Image Lx_new = applyLaplacian(x);
        const double eps = 1e-6;
        for (size_t i = 0; i < weights.data.size(); ++i) {
            weights.data[i] = 1.0 / (std::abs(Lx_new.data[i]) + eps);
        }

        std::cout << "IR L1 iteration " << (ir + 1) << "/" << ir_iter << " completed." << std::endl;
    }
    return x;
}

Image deconvCG_reg_IRL08(const Image& im_blur, const Kernel& kernel, 
                         double lamb, int niter, int ir_iter) {
    Image x(im_blur.height, im_blur.width, im_blur.channels);
    Image weights(im_blur.height, im_blur.width, im_blur.channels);
    for (size_t i = 0; i < weights.data.size(); ++i) {
        weights.data[i] = 1.0;
    }

    for (int ir = 0; ir < ir_iter; ++ir) {
        Image x_prev = x.copy();

        Image Lx = applyLaplacian(x);
        Image WLx = multiplyImages(Lx, weights);
        Image ALx = applyAMatrix(x, kernel);
        Image ALx_WLx = addImages(ALx, scaleImage(WLx, lamb));
        Image MTy = applyConjugatedKernel(im_blur, kernel);
        Image r = subtractImages(MTy, ALx_WLx);
        Image d = r.copy();

        for (int iter = 0; iter < niter; ++iter) {
            Image Ld = applyLaplacian(d);
            Image WLd = multiplyImages(Ld, weights);
            Image Ad = applyAMatrix(d, kernel);
            Image ALd_WLd = addImages(Ad, scaleImage(WLd, lamb));

            double r_dot_r = dotIm(r, r);
            double d_dot_Ad = dotIm(d, ALd_WLd);
            if (d_dot_Ad == 0 || r_dot_r < 1e-10) break;
            double alpha = r_dot_r / d_dot_Ad;

            x = addImages(x, scaleImage(d, alpha));

            Image old_r = r.copy();
            Image ALx_new_WLx = addImages(applyAMatrix(x, kernel), scaleImage(multiplyImages(applyLaplacian(x), weights), lamb));
            r = subtractImages(MTy, ALx_new_WLx);

            double beta = computeConjugateDirectionStepSize(old_r, r);
            d = addImages(r, scaleImage(d, beta));
        }

        // Update weights for L0.8: w = 1 / (|Lx|^0.2 + eps)
        Image Lx_new = applyLaplacian(x);
        const double eps = 1e-6;
        const double p = 0.8;
        const double q = 1.0 - p; // 0.2
        for (size_t i = 0; i < weights.data.size(); ++i) {
            weights.data[i] = 1.0 / (std::pow(std::abs(Lx_new.data[i]), q) + eps);
        }

        std::cout << "IR L0.8 iteration " << (ir + 1) << "/" << ir_iter << " completed." << std::endl;
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
    // Initialize x with background everywhere
    Image x = bg.copy();
    
    // b = Laplacian of source - the guidance field
    Image b = applyLaplacian(fg);
    
    for (int iter = 0; iter < niter; ++iter) {
        // Compute Ax = Laplacian(x)
        Image Ax = applyLaplacian(x);
        
        // Residual: r = (b - Ax), masked to only inside region
        Image r = subtractImages(b, Ax);
        r = multiplyImages(r, mask);
        
        // Compute Ar = Laplacian(r) for step size
        Image Ar = applyLaplacian(r);
        
        // alpha = (r · r) / (r · Ar)
        double r_dot_r = dotIm(r, r);
        double r_dot_Ar = dotIm(r, Ar);
        if (r_dot_Ar == 0 || r_dot_r < 1e-10) {
            break;
        }
        double alpha = r_dot_r / r_dot_Ar;
        
        // Update x: x = x + alpha * r
        // Since r is already masked, this only updates inside mask
        Image update = scaleImage(r, alpha);
        x = addImages(x, update);
        
        // Enforce boundary: outside mask must equal bg
        // This is CRITICAL - pixels outside mask never change
        for (int y = 0; y < x.height; ++y) {
            for (int xi = 0; xi < x.width; ++xi) {
                double m = mask(y, xi, 0);
                for (int c = 0; c < x.channels; ++c) {
                    x(y, xi, c) = m * x(y, xi, c) + (1.0 - m) * bg(y, xi, c);
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
    // Initialize x with background everywhere
    Image x = bg.copy();
    
    // b = Laplacian of source - the guidance field
    Image b = applyLaplacian(fg);
    
    // Initial residual: r = (b - Ax), masked
    Image Ax = applyLaplacian(x);
    Image r = subtractImages(b, Ax);
    r = multiplyImages(r, mask);
    
    Image d = r.copy();
    
    for (int iter = 0; iter < niter; ++iter) {
        // Compute Ad = Laplacian(d)
        Image Ad = applyLaplacian(d);
        
        // alpha = (r · r) / (d · Ad)
        double r_dot_r = dotIm(r, r);
        double d_dot_Ad = dotIm(d, Ad);
        if (d_dot_Ad == 0 || r_dot_r < 1e-10) {
            break;
        }
        double alpha = r_dot_r / d_dot_Ad;
        
        // x = x + alpha * d
        x = addImages(x, scaleImage(d, alpha));
        
        // Enforce boundary: outside mask equals bg
        for (int y = 0; y < x.height; ++y) {
            for (int xi = 0; xi < x.width; ++xi) {
                double m = mask(y, xi, 0);
                for (int c = 0; c < x.channels; ++c) {
                    x(y, xi, c) = m * x(y, xi, c) + (1.0 - m) * bg(y, xi, c);
                }
            }
        }
        
        // r_new = r - alpha * Ad, then mask
        Image old_r = r.copy();
        r = subtractImages(r, scaleImage(Ad, alpha));
        r = multiplyImages(r, mask);
        
        // beta = (r_new · r_new) / (r_old · r_old)
        double beta = computeConjugateDirectionStepSize(old_r, r);
        
        // d = r + beta * d
        d = addImages(r, scaleImage(d, beta));
        
        if ((iter + 1) % 50 == 0) {
            std::cout << "Poisson CG Iteration " << (iter + 1) << "/" << niter << std::endl;
        }
    }
    
    return x;
}

// === Mask extraction ===

Image extractMaskFromImage(const Image& im, const std::vector<double>& bg_color, double threshold) {
    Image mask(im.height, im.width, 1);
    for (int y = 0; y < im.height; ++y) {
        for (int x = 0; x < im.width; ++x) {
            double dist = 0.0;
            for (int c = 0; c < 3; ++c) {
                double diff = im(y, x, c) - bg_color[c];
                dist += diff * diff;
            }
            dist = std::sqrt(dist);
            mask(y, x, 0) = (dist > threshold) ? 1.0 : 0.0;
        }
    }
    // Simple morphological cleanup: remove isolated pixels
    Image cleaned = mask.copy();
    for (int y = 1; y < im.height - 1; ++y) {
        for (int x = 1; x < im.width - 1; ++x) {
            int fg = 0;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (mask(y + dy, x + dx, 0) > 0.5) ++fg;
                }
            }
            // If fewer than 4 neighbors are foreground, zero it out
            cleaned(y, x, 0) = (fg >= 4) ? 1.0 : 0.0;
        }
    }
    return cleaned;
}

// === Simple utilities ===

Image downscale(const Image& im, double scale) {
    int new_h = static_cast<int>(im.height / scale);
    int new_w = static_cast<int>(im.width / scale);
    Image result(new_h, new_w, im.channels);
    for (int y = 0; y < new_h; ++y) {
        for (int x = 0; x < new_w; ++x) {
            int src_y = static_cast<int>(y * scale);
            int src_x = static_cast<int>(x * scale);
            for (int c = 0; c < im.channels; ++c) {
                result(y, x, c) = im(src_y, src_x, c);
            }
        }
    }
    return result;
}

Image blurMask(const Image& mask, double sigma) {
    Kernel gauss = gauss2D(sigma, 2);
    return convolve3(mask, gauss);
}

// === Fun composite ===

void createFunComposite() {
    // Load images
    Image bear = imread("Input/bear.png");
    Image bear_mask_raw = imread("Input/mask.png");
    Image sun = imread("Input/sun.png");
    Image sky = imread("Input/sky.png");
    Image galaxy = imread("Input/galaxy.png");

    // Prepare bear mask from provided mask.png (ensure single-channel and blur)
    Image mask_bear_raw = bear_mask_raw.copy();
    if (mask_bear_raw.channels > 1) {
        Image tmp(mask_bear_raw.height, mask_bear_raw.width, 1);
        for (int y = 0; y < mask_bear_raw.height; ++y)
            for (int x = 0; x < mask_bear_raw.width; ++x)
                tmp(y, x, 0) = mask_bear_raw(y, x, 0);
        mask_bear_raw = tmp;
    }
    // Binarize mask
    for (size_t i = 0; i < mask_bear_raw.data.size(); ++i) {
        mask_bear_raw.data[i] = (mask_bear_raw.data[i] > 0.5) ? 1.0 : 0.0;
    }
    Image mask_bear = blurMask(mask_bear_raw, 1.2);

    // Helper to composite bear onto a background using the same procedure as test_Poisson
    auto compositeBear = [&](const Image& bg, const std::string& out_name) {
        int bear_h = bear.height, bear_w = bear.width;
        int bear_y = (bg.height - bear_h) / 2, bear_x = (bg.width - bear_w) / 2;

        // Extract background region
        Image bg_region(bear_h, bear_w, 3);
        for (int i = 0; i < bear_h; ++i) {
            for (int j = 0; j < bear_w; ++j) {
                for (int c = 0; c < 3; ++c) {
                    bg_region(i, j, c) = bg(bear_y + i, bear_x + j, c);
                }
            }
        }

        // Apply log transform (as in test_Poisson)
        Image bg_log = bg_region.copy();
        Image fg_log = bear.copy();
        for (int i = 0; i < bear_h; ++i) {
            for (int j = 0; j < bear_w; ++j) {
                for (int c = 0; c < 3; ++c) {
                    bg_log(i, j, c) = std::log(std::max(bg_region(i, j, c), 1e-4)) + 3.0;
                    fg_log(i, j, c) = std::log(std::max(bear(i, j, c), 1e-4)) + 3.0;
                }
            }
        }

        Image tmp = Poisson(bg_log, fg_log, mask_bear, 200);

        // Exponentiate and copy back
        Image out = bg.copy();
        for (int i = 0; i < bear_h; ++i) {
            for (int j = 0; j < bear_w; ++j) {
                for (int c = 0; c < 3; ++c) {
                    out(bear_y + i, bear_x + j, c) = std::exp(tmp(i, j, c) - 3.0);
                }
            }
        }

        imwrite(out, out_name.c_str());
        std::cout << "Wrote " << out_name << std::endl;
    };

    // Generate composites
    compositeBear(sun, "Output/composite_bear_sun.png");
    compositeBear(sky, "Output/composite_bear_sky.png");
    compositeBear(galaxy, "Output/composite_bear_galaxy.png");
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
            double m = im2(y, x, 0);  // mask value (same for all channels)
            for (int c = 0; c < im1.channels; ++c) {
                result(y, x, c) = im1(y, x, c) * m;
            }
        }
    }
    return result;
}