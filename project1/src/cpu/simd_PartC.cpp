#include <iostream>
#include <chrono>
#include <immintrin.h>
#include "utils.hpp"  // Ensure utils.hpp is properly included

// Fast exponential approximation using AVX2
__m256 exp_approx(__m256 x) {
    __m256 res = _mm256_set1_ps(1.0f);  // Initialize result with the constant term of the series (0! = 1)
    __m256 fac = res;                   // Factorial starts at 1
    __m256 pow = res;                   // x^0 is 1
    __m256 x_pow = x;                   // x^1 is x

    // Add terms up to x^5/5!
    for (int n = 1; n <= 5; n++) {
        fac = _mm256_mul_ps(fac, _mm256_set1_ps((float)n)); // Factorial n!
        pow = _mm256_div_ps(x_pow, fac);                    // x^n / n!
        res = _mm256_add_ps(res, pow);                      // Summing up the terms
        x_pow = _mm256_mul_ps(x_pow, x);                    // x to the power of n+1
    }

    return res;
}

// Function to approximate the Gaussian function for intensity differences
inline __m256 gaussian_exp(__m256 x, float sigma) {
    __m256 inv_sigma = _mm256_set1_ps(-0.5f / (sigma * sigma));
    __m256 exponent = _mm256_mul_ps(_mm256_mul_ps(x, x), inv_sigma);
    return exp_approx(exponent); // Use the exp_approx function defined above
}

// Apply bilateral filter to a single color channel
void apply_bilateral_soa(const ColorValue* input_channel, ColorValue* output_channel, int width, int height, float sigma_s, float sigma_r) {
    int index, kernel_idx;
    __m256 center_pixel, kernel_pixel, diff, weight, accum;

    for (int row = 1; row < height - 1; ++row) {
        for (int col = 1; col < width - 1; ++col) {
            index = row * width + col;
            center_pixel = _mm256_set1_ps(input_channel[index]);
            accum = _mm256_setzero_ps();

            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    kernel_idx = (row + ky) * width + (col + kx);
                    kernel_pixel = _mm256_set1_ps(input_channel[kernel_idx]);
                    diff = _mm256_sub_ps(center_pixel, kernel_pixel);

                    weight = gaussian_exp(diff, sigma_r);  // Calculate weight using Gaussian of intensity difference
                    accum = _mm256_add_ps(accum, _mm256_mul_ps(kernel_pixel, weight));
                }
            }
            output_channel[index] = _mm256_cvtss_f32(_mm256_hadd_ps(accum, accum));  // Horizontal add and convert to scalar
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: ./bilateral_filter <input_jpeg_path> <output_jpeg_path>\n";
        return 1;
    }

    // Read input JPEG into SOA format
    JpegSOA input_jpeg = read_jpeg_soa(argv[1]);
    if (input_jpeg.r_values == nullptr) {
        std::cerr << "Failed to read input JPEG image.\n";
        return 1;
    }

    // Allocate output channels
    ColorValue* output_r = new ColorValue[input_jpeg.width * input_jpeg.height];
    ColorValue* output_g = new ColorValue[input_jpeg.width * input_jpeg.height];
    ColorValue* output_b = new ColorValue[input_jpeg.width * input_jpeg.height];

    // Start timer
    auto start_time = std::chrono::high_resolution_clock::now();

    // Apply bilateral filter to each channel
    apply_bilateral_soa(input_jpeg.r_values, output_r, input_jpeg.width, input_jpeg.height, 2.0f, 50.0f);
    apply_bilateral_soa(input_jpeg.g_values, output_g, input_jpeg.width, input_jpeg.height, 2.0f, 50.0f);
    apply_bilateral_soa(input_jpeg.b_values, output_b, input_jpeg.width, input_jpeg.height, 2.0f, 50.0f);

    // Stop timer
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Filtering completed in " << elapsed_time.count() << " milliseconds.\n";

    // Write output JPEG
    JpegSOA output_jpeg{output_r, output_g, output_b, input_jpeg.width, input_jpeg.height, input_jpeg.num_channels, input_jpeg.color_space};
    if (export_jpeg(output_jpeg, argv[2]) != 0) {
        std::cerr << "Failed to write output JPEG image.\n";
        return 1;
    }

    // Cleanup
    delete[] output_r;
    delete[] output_g;
    delete[] output_b;

    return 0;
}
