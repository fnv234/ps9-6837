#include "a9.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cstdlib>
#include <ctime>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

Image imread(const std::string& filename) {
    int width, height, channels;
    unsigned char* data = stbi_load(filename.c_str(), &width, &height, &channels, 3);
    
    if (!data) {
        std::cerr << "Error: Cannot load image " << filename << std::endl;
        std::cerr << "Reason: " << stbi_failure_reason() << std::endl;
        return Image();
    }
    
    Image img(height, width, 3);
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < 3; ++c) {
                int idx = (y * width + x) * 3 + c;
                img(y, x, c) = data[idx] / 255.0;
            }
        }
    }
    
    stbi_image_free(data);
    std::cout << "Loaded " << filename << " (" << width << "x" << height << ")" << std::endl;
    return img;
}

void imwrite(const Image& img, const std::string& filename) {
    std::vector<unsigned char> buffer(img.width * img.height * 3);
    
    for (int y = 0; y < img.height; ++y) {
        for (int x = 0; x < img.width; ++x) {
            for (int c = 0; c < 3; ++c) {
                int idx = (y * img.width + x) * 3 + c;
                double val = img(y, x, c);
                val = std::max(0.0, std::min(1.0, val));
                buffer[idx] = static_cast<unsigned char>(val * 255);
            }
        }
    }
    
    std::string ext = filename.substr(filename.find_last_of(".") + 1);
    bool success = false;
    
    if (ext == "png") {
        success = stbi_write_png(filename.c_str(), img.width, img.height, 3, 
                                 buffer.data(), img.width * 3);
    } else if (ext == "jpg" || ext == "jpeg") {
        success = stbi_write_jpg(filename.c_str(), img.width, img.height, 3, 
                                 buffer.data(), 95);
    } else {
        std::cerr << "Unsupported format: " << ext << std::endl;
        return;
    }
    
    if (success) {
        std::cout << "Wrote " << filename << std::endl;
    } else {
        std::cerr << "Failed to write " << filename << std::endl;
    }
}

Image createTestImage(int height, int width) {
    Image img(height, width, 3);
    int square_size = 20;
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            bool is_white = ((y / square_size) + (x / square_size)) % 2 == 0;
            double val = is_white ? 1.0 : 0.2;
            for (int c = 0; c < 3; ++c) {
                img(y, x, c) = val;
            }
        }
    }
    
    return img;
}

Image createGradientImage(int height, int width) {
    Image img(height, width, 3);
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            double val = static_cast<double>(x) / width;
            for (int c = 0; c < 3; ++c) {
                img(y, x, c) = val;
            }
        }
    }
    
    return img;
}

Image addNoise(const Image& img, double noise_level) {
    Image noisy = img.copy();
    std::srand(42);
    
    for (size_t i = 0; i < noisy.data.size(); ++i) {
        double noise = (std::rand() / static_cast<double>(RAND_MAX)) - 0.5;
        noisy.data[i] += noise_level * noise;
    }
    
    return noisy;
}

Image createCircleMask(int height, int width) {
    Image mask(height, width, 3);
    int cy = height / 2;
    int cx = width / 2;
    int radius = std::min(height, width) / 3;
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int dy = y - cy;
            int dx = x - cx;
            double dist = std::sqrt(dy * dy + dx * dx);
            double val = dist < radius ? 1.0 : 0.0;
            for (int c = 0; c < 3; ++c) {
                mask(y, x, c) = val;
            }
        }
    }
    
    return mask;
}

void test_grad_descent() {
    std::cout << "\n=== Testing Gradient Descent Deconvolution ===" << std::endl;
    
    Image im = imread("Input/pru.png");
    if (im.height == 0) return;
    
    imwrite(im, "Output/pru_original.png");
    
    Kernel kernel = gauss2D(1.0);
    Image im_blur = convolve3(im, kernel);
    imwrite(im_blur, "Output/pru_blur.png");
    
    Image im_sharp = deconvGradDescent(im_blur, kernel, 10);
    imwrite(im_sharp, "Output/pru_sharp_gd.png");
    
    std::cout << "Gradient descent test complete!" << std::endl;
}

void test_conjugate_grad_descent() {
    std::cout << "\n=== Testing Conjugate Gradient Deconvolution ===" << std::endl;
    
    Image im = imread("Input/pru.png");
    if (im.height == 0) return;
    
    Kernel kernel = gauss2D(1.0);
    Image im_blur = convolve3(im, kernel);
    
    Image im_sharp = deconvCG(im_blur, kernel, 10);
    imwrite(im_sharp, "Output/pru_sharp_cg.png");
    
    std::cout << "Conjugate gradient test complete!" << std::endl;
}

void test_conjugate_grad_descent_reg() {
    std::cout << "\n=== Testing Regularized Conjugate Gradient ===" << std::endl;
    
    Image im = imread("Input/pru.png");
    if (im.height == 0) return;
    
    Kernel kernel = gauss2D(1.0);
    Image im_blur = convolve3(im, kernel);
    Image im_blur_noisy = addNoise(im_blur, 0.05);
    
    imwrite(im_blur_noisy, "Output/pru_blur_noise.png");
    
    Image im_sharp = deconvCG_reg(im_blur_noisy, kernel, 0.05, 10);
    imwrite(im_sharp, "Output/pru_sharp_cg_reg.png");
    
    Image im_sharp_wo_reg = deconvCG(im_blur_noisy, kernel, 10);
    imwrite(im_sharp_wo_reg, "Output/pru_sharp_cg_wo_reg.png");
    
    std::cout << "Regularized test complete!" << std::endl;
}

void test_naive_composite() {
    std::cout << "\n=== Testing Naive Composite ===" << std::endl;
    
    Image fg = imread("Input/bear.png");
    Image bg = imread("Input/waterpool.png");
    Image mask = imread("Input/mask.png");
    
    if (fg.height == 0 || bg.height == 0 || mask.height == 0) return;
    
    Image composite = naiveComposite(bg, fg, mask, 50, 1);
    imwrite(composite, "Output/naive_composite.png");
    
    std::cout << "Naive composite test complete!" << std::endl;
}

void test_Poisson() {
    std::cout << "\n=== Testing Poisson Editing (Gradient Descent) ===" << std::endl;
    
    int y = 50, x = 10;
    bool useLog = true;
    
    Image fg = imread("Input/bear.png");
    Image bg = imread("Input/waterpool.png");
    Image mask = imread("Input/mask.png");
    
    if (fg.height == 0 || bg.height == 0 || mask.height == 0) return;
    
    int h = fg.height, w = fg.width;
    
    for (int i = 0; i < mask.height; ++i) {
        for (int j = 0; j < mask.width; ++j) {
            double val = mask(i, j, 0) > 0.5 ? 1.0 : 0.0;
            mask(i, j, 0) = val;
            mask(i, j, 1) = val;
            mask(i, j, 2) = val;
        }
    }
    
    Image bg2(h, w, 3);
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            for (int c = 0; c < 3; ++c) {
                bg2(i, j, c) = bg(y + i, x + j, c);
            }
        }
    }
    
    Image bg3, fg3;
    if (useLog) {
        bg3 = bg2.copy();
        fg3 = fg.copy();
        for (int i = 0; i < h; ++i) {
            for (int j = 0; j < w; ++j) {
                for (int c = 0; c < 3; ++c) {
                    bg3(i, j, c) = std::log(std::max(bg2(i, j, c), 1e-4)) + 3.0;
                    fg3(i, j, c) = std::log(std::max(fg(i, j, c), 1e-4)) + 3.0;
                }
            }
        }
    } else {
        bg3 = bg2;
        fg3 = fg;
    }
    
    Image tmp = Poisson(bg3, fg3, mask, 200);
    
    Image out = bg.copy();
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            for (int c = 0; c < 3; ++c) {
                if (useLog) {
                    out(y + i, x + j, c) = std::exp(tmp(i, j, c) - 3.0);
                } else {
                    out(y + i, x + j, c) = tmp(i, j, c);
                }
            }
        }
    }
    
    imwrite(out, "Output/poisson.png");
    std::cout << "Poisson test complete!" << std::endl;
}

void test_PoissonCG() {
    std::cout << "\n=== Testing Poisson Editing (Conjugate Gradient) ===" << std::endl;
    
    int y = 50, x = 10;
    bool useLog = true;
    
    Image fg = imread("Input/bear.png");
    Image bg = imread("Input/waterpool.png");
    Image mask = imread("Input/mask.png");
    
    if (fg.height == 0 || bg.height == 0 || mask.height == 0) return;
    
    int h = fg.height, w = fg.width;
    
    for (int i = 0; i < mask.height; ++i) {
        for (int j = 0; j < mask.width; ++j) {
            double val = mask(i, j, 0) > 0.5 ? 1.0 : 0.0;
            mask(i, j, 0) = val;
            mask(i, j, 1) = val;
            mask(i, j, 2) = val;
        }
    }
    
    Image bg2(h, w, 3);
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            for (int c = 0; c < 3; ++c) {
                bg2(i, j, c) = bg(y + i, x + j, c);
            }
        }
    }
    
    Image bg3, fg3;
    if (useLog) {
        bg3 = bg2.copy();
        fg3 = fg.copy();
        for (int i = 0; i < h; ++i) {
            for (int j = 0; j < w; ++j) {
                for (int c = 0; c < 3; ++c) {
                    bg3(i, j, c) = std::log(std::max(bg2(i, j, c), 1e-4)) + 3.0;
                    fg3(i, j, c) = std::log(std::max(fg(i, j, c), 1e-4)) + 3.0;
                }
            }
        }
    } else {
        bg3 = bg2;
        fg3 = fg;
    }
    
    Image tmp = PoissonCG(bg3, fg3, mask, 200);
    
    Image out = bg.copy();
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            for (int c = 0; c < 3; ++c) {
                if (useLog) {
                    out(y + i, x + j, c) = std::exp(tmp(i, j, c) - 3.0);
                } else {
                    out(y + i, x + j, c) = tmp(i, j, c);
                }
            }
        }
    }
    
    imwrite(out, "Output/poisson_cg.png");
    std::cout << "Poisson CG test complete!" << std::endl;
}

void test_basic_operations() {
    std::cout << "\n=== Testing Basic Operations ===" << std::endl;
    
    Image im1 = imread("Input/pru.png");
    if (im1.height == 0) {
        std::cout << "Using synthetic test image" << std::endl;
        im1 = createTestImage(200, 200);
    }
    
    Kernel kernel = gauss2D(1.0);
    std::cout << "Gaussian kernel size: " << kernel.height << "x" << kernel.width << std::endl;
    
    Image blurred = convolve3(im1, kernel);
    imwrite(blurred, "Output/test_convolution.png");
    
    Kernel lap = laplacianKernel();
    Image laplacian_result = convolve3(im1, lap);
    imwrite(laplacian_result, "Output/test_laplacian.png");
    
    std::cout << "Basic operations test complete!" << std::endl;
}
void test_synthetic_deconvolution() {
    std::cout << "\n=== Test Case: Synthetic Checkerboard Deconvolution ===" << std::endl;

    Image synth = createTestImage(300, 300);
    imwrite(synth, "Output/synth_checker.png");

    Kernel kernel = gauss2D(1.0);
    Image blur = convolve3(synth, kernel);
    imwrite(blur, "Output/synth_checker_blur.png");

    Image blur_noisy = addNoise(blur, 0.05);
    imwrite(blur_noisy, "Output/synth_checker_blur_noise.png");

    Image sharp_gd = deconvGradDescent(blur, kernel, 10);
    imwrite(sharp_gd, "Output/synth_checker_gd.png");

    Image sharp_cg = deconvCG(blur, kernel, 10);
    imwrite(sharp_cg, "Output/synth_checker_cg.png");

    Image sharp_cg_reg = deconvCG_reg(blur_noisy, kernel, 0.05, 10);
    imwrite(sharp_cg_reg, "Output/synth_checker_cg_reg.png");

    std::cout << "Synthetic deconvolution test complete!" << std::endl;
}



void test_synthetic_poisson() {
    std::cout << "\n=== Test Case: Synthetic Gradient + Poisson Editing ===" << std::endl;

    Image bg = createGradientImage(300, 300);
    imwrite(bg, "Output/synth_grad.png");

    Image mask = createCircleMask(300, 300);
    imwrite(mask, "Output/synth_grad_mask.png");

    Image fg = createTestImage(300, 300);

    Image out_gd = Poisson(bg, fg, mask, 200);
    imwrite(out_gd, "Output/synth_grad_poisson.png");

    Image out_cg = PoissonCG(bg, fg, mask, 200);
    imwrite(out_cg, "Output/synth_grad_poisson_cg.png");

    std::cout << "Synthetic Poisson test complete!" << std::endl;
}


int main() {
    std::cout << "Starting A9 Tests" << std::endl;
    std::cout << "=================" << std::endl;

#ifdef _WIN32
    system("mkdir Output 2>nul");
#else
    system("mkdir -p Output");
#endif

    test_basic_operations();
    test_grad_descent();
    test_conjugate_grad_descent();
    test_conjugate_grad_descent_reg();
    test_naive_composite();
    test_Poisson();
    test_PoissonCG();

    test_synthetic_deconvolution();
    test_synthetic_poisson();

    // ==================== EXTRA CREDIT TESTS ====================
// === LOAD IMAGES ===
    Image sharp = imread("Input/bear.png");
    Image bg = imread("Input/waterpool.png");
    Image fg = imread("Input/bear.png");         // foreground reused
    Image mask = imread("Input/mask.png");

    // Create a blur kernel (Gaussian)
    Kernel blurK = gauss2D(2.0, 3);

    // === CREATE A TEST BLUR ===
    Image blurred = convolve3(sharp, blurK);
    imwrite(blurred, "Output/test_blur.png");
    std::cout << "Saved test_blur.png\n";


    // // ============================================================
    // // ============ IRLS: L1 DECONVOLUTION TEST ===================
    // // ============================================================
    // {
    //     std::cout << "\n=== IRLS L1 test ===\n";
    //     Image recL1 = deconvIRLS(
    //         blurred,     // y 
    //         blurK,       // kernel 
    //         1.0,         // p = 1 (L1)
    //         0.02,        // lambda 
    //         100,           // outer IRLS iterations 
    //         100           // inner CG iterations
    //     );

    //     imwrite(recL1, "Output/irls_L1.png");
    //     std::cout << "Saved irls_L1.png\n";
    // }

    // // ============================================================
    // // ============ IRLS: L0.8 DECONVOLUTION TEST =================
    // // ============================================================
    // {
    //     std::cout << "\n=== IRLS L0.8 test ===\n";
    //     Image recL08 = deconvIRLS(
    //         blurred, 
    //         blurK,
    //         0.8,        // L0.8 promotes even sharper detail
    //         0.02,
    //         100,
    //         100
    //     );

    //     imwrite(recL08, "Output/irls_L08.png");
    //     std::cout << "Saved irls_L08.png\n";
    // }


    // // ============================================================
    // // === Compare Richardson-Lucy to IRLS for sanity check =======
    // // ============================================================
    // {
    //     std::cout << "\n=== Richardson–Lucy test ===\n";
    //     Image rl = richardsonLucy(blurred, blurK, 20, 1e-6);
    //     imwrite(rl, "Output/richardson_lucy.png");
    //     std::cout << "Saved richardson_lucy.png\n";
    // }


    // // ============================================================
    // // ====== Test mixed forward/backward divergence ==============
    // // ============================================================
    // {
    //     std::cout << "\n=== Checking divergence correctness ===\n";

    //     // Make a synthetic gradient field:
    //     Image gx(sharp.height, sharp.width, sharp.channels);
    //     Image gy(sharp.height, sharp.width, sharp.channels);
    //     for (int y = 0; y < gx.height; ++y) {
    //         for (int x = 0; x < gx.width; ++x) {
    //             for (int c = 0; c < gx.channels; ++c) {
    //                 gx(y,x,c) = (x % 20 < 10) ? 1.0 : -1.0;
    //                 gy(y,x,c) = (y % 20 < 10) ? 1.0 : -1.0;
    //             }
    //         }
    //     }

    //     Image div = divergenceMixed(gx,gy);
    //     imwrite(div, "Output/test_divergence.png");
    //     std::cout << "Saved test_divergence.png\n";
    // }


    // ============================================================
    // === Standard Poisson vs Pérez Poisson blending =============
    // ============================================================
    {
        std::cout << "\n=== Poisson blending sanity test ===\n";

        Image pstd = PoissonCG(bg, fg, mask, 200);
        imwrite(pstd, "Output/poisson_standard.png");
        std::cout << "Saved poisson_standard.png\n";

        Image pperez = poissonPerez(bg, fg, mask);
        imwrite(pperez, "Output/poisson_perez.png");
        std::cout << "Saved poisson_perez.png\n";
    }


    // // ============================================================
    // // === Mixed gradient compositing (Perez et al. paper) ========
    // // ============================================================
    // {
    //     std::cout << "\n=== Mixed-gradients compositing test ===\n";

    //     // Make gradients of foreground
    //     Image gx(sharp.height, sharp.width, sharp.channels);
    //     Image gy(sharp.height, sharp.width, sharp.channels);
    //     computeGradients(fg, gx, gy);

    //     // Divergence → Poisson solve
    //     Image b = divergenceMixed(gx, gy);

    //     // Build masked operator
    //     auto A = [&](const Image& z)->Image {
    //         Image Lz = applyLaplacian(z);
    //         return multiplyImages(Lz, mask);
    //     };

    //     Image x0 = bg.copy();
    //     Image mixed = CG_custom(A, b, x0, 300);

    //     // copy outside mask
    //     for (int y = 0; y < mixed.height; ++y)
    //         for (int x = 0; x < mixed.width; ++x)
    //             if (mask(y,x,0) < 0.5)
    //                 for (int c = 0; c < mixed.channels; ++c)
    //                     mixed(y,x,c) = bg(y,x,c);

    //     imwrite(mixed, "Output/perez_mixed_gradient.png");
    //     std::cout << "Saved perez_mixed_gradient.png\n";
    // }


    std::cout << "\n=== All tests complete ===\n";
    return 0;
}
