#ifndef A9_H
#define A9_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <functional>

class Image {
public:
    int height, width, channels;
    std::vector<double> data;
    
    Image() : height(0), width(0), channels(0) {}
    Image(int h, int w, int c = 3) : height(h), width(w), channels(c) {
        data.resize(h * w * c, 0.0);
    }
    
    double& operator()(int y, int x, int c) {
        return data[(y * width + x) * channels + c];
    }
    
    const double& operator()(int y, int x, int c) const {
        return data[(y * width + x) * channels + c];
    }
    
    Image copy() const {
        Image result(height, width, channels);
        result.data = data;
        return result;
    }
};

class Kernel {
public:
    int height, width;
    std::vector<double> data;
    
    Kernel() : height(0), width(0) {}
    Kernel(int h, int w) : height(h), width(w) {
        data.resize(h * w, 0.0);
    }
    
    double& operator()(int y, int x) {
        return data[y * width + x];
    }
    
    const double& operator()(int y, int x) const {
        return data[y * width + x];
    }
};

double dotIm(const Image& im1, const Image& im2);
Image convolve3(const Image& im, const Kernel& kernel);

Image applyKernel(const Image& im, const Kernel& kernel);
Image applyConjugatedKernel(const Image& im, const Kernel& kernel);

Image computeResidual(const Kernel& kernel, const Image& x, const Image& y);
double computeStepSize(const Image& r, const Kernel& kernel);

Image deconvGradDescent(const Image& im_blur, const Kernel& kernel, int niter = 10);

double computeGradientStepSize(const Image& r, const Image& d, const Kernel& kernel);
double computeConjugateDirectionStepSize(const Image& old_r, const Image& new_r);

Image deconvCG(const Image& im_blur, const Kernel& kernel, int niter = 10);

Kernel laplacianKernel();
Image applyLaplacian(const Image& im);
Image applyAMatrix(const Image& im, const Kernel& kernel);
Image applyRegularizedOperator(const Image& im, const Kernel& kernel, double lamb);
double computeGradientStepSize_reg(const Image& grad, const Image& p, 
                                    const Kernel& kernel, double lamb);

Image deconvCG_reg(const Image& im_blur, const Kernel& kernel, 
                   double lamb = 0.05, int niter = 10);

Image naiveComposite(const Image& bg, const Image& fg, const Image& mask, int y, int x);
Image Poisson(const Image& bg, const Image& fg, const Image& mask, int niter = 200);
Image PoissonCG(const Image& bg, const Image& fg, const Image& mask, int niter = 200);

Kernel gauss2D(double sigma = 2.0, int truncate = 3);
std::vector<double> horiGaussKernel(double sigma, int truncate = 3);
Image addImages(const Image& im1, const Image& im2);
Image subtractImages(const Image& im1, const Image& im2);
Image scaleImage(const Image& im, double scale);
Image multiplyImages(const Image& im1, const Image& im2);
Image richardsonLucy(const Image& im_blur, const Kernel& kernel, int niter, double eps = 1e-7);
Image gaussianDenoise(const Image& im, double sigma, int truncate = 3);
Image anisotropicDiffusion(const Image& im, int niter, double kappa, double lambda = 0.25);
Image nlMeansDenoise(const Image& im, int patchRadius, int searchRadius, double h);
Image elementwiseMultiply(const Image& a, const Image& b);
Image divideImages(const Image& num, const Image& denom, double eps = 1e-10);
Image poissonPerez(
    const Image& bg,      // target/background image
    const Image& fg,      // source/foreground image (same size)
    const Image& mask     // binary mask Ω
);
// Image poissonPerez(const Image& bg, const Image& fg, const Image& mask, int cgIters);
Image CG_custom(const std::function<Image(const Image&)>& A, const Image& b, Image x, int niter);
Image deconvIRLS(const Image& y, const Kernel& kernel, double p, double lambda, int outerIters, int cgIters);
Image computeIRLSWeights(const Image& x, double p, double eps);
void computeGradients(const Image& im, Image& gx, Image& gy);
Image divergenceMixed(const Image& gx, const Image& gy);
void cgSolve(
    std::vector<double>& x,
    const std::vector<double>& b,
    const std::function<std::vector<double>(const std::vector<double>&)>& A,
    int maxIter,
    double tol
);
#endif // A9_H