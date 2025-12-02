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