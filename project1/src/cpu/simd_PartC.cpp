// Created by Yang Yufan on 2023/9/16.
// Email: yufanyang1@link.cuhk.edu.cm
//
// SIMD (AVX2) implementation of bilateral filtering on a JPEG picture
//

#include <immintrin.h>  // For AVX2 and SSE instructions
#include <chrono>
#include <iostream>
#include <cstring>  // For memset

#include "../utils.hpp"

// Rough approximation of the exponential function using a polynomial
inline __m256 exp256_ps(__m256 x) {
    __m256 a = _mm256_set1_ps(1.0f);
    __m256 b = _mm256_set1_ps(1.0f / 6.0f);
    __m256 c = _mm256_set1_ps(1.0f / 24.0f);
    __m256 d = _mm256_set1_ps(1.0f / 120.0f);

    __m256 x2 = _mm256_mul_ps(x, x);
    __m256 x3 = _mm256_mul_ps(x2, x);
    __m256 x4 = _mm256_mul_ps(x2, x2);

    __m256 res;
    res = _mm256_add_ps(a, x);
    res = _mm256_add_ps(res, _mm256_mul_ps(b, x2));
    res = _mm256_add_ps(res, _mm256_mul_ps(c, x3));
    res = _mm256_add_ps(res, _mm256_mul_ps(d, x4));

    return res;
}

// Bilateral filter kernel using SIMD
__m256 bilateral_filter_kernel(unsigned char* buffer, int idx, int width, int height, int num_channels, float sigma_s, float sigma_r) {
    int radius = 1;
    float gauss_color_coeff = -0.5 / (sigma_r * sigma_r);
    float gauss_space_coeff = -0.5 / (sigma_s * sigma_s);

    __m256 weight_sum = _mm256_setzero_ps();
    __m256 pixel_sum = _mm256_setzero_ps();

    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            int new_idx = idx + (dy * width + dx) * num_channels;
            __m256 neighbor_value = _mm256_set1_ps((float)buffer[new_idx]);
            __m256 pixel_value = _mm256_set1_ps((float)buffer[idx]);

            __m256 range_diff = _mm256_sub_ps(pixel_value, neighbor_value);
            __m256 range_weight = _mm256_mul_ps(range_diff, range_diff);
            range_weight = _mm256_mul_ps(range_weight, _mm256_set1_ps(gauss_color_coeff));
            range_weight = exp256_ps(range_weight);

            __m256 space_weight = _mm256_set1_ps(dx * dx + dy * dy);
            space_weight = _mm256_mul_ps(space_weight, _mm256_set1_ps(gauss_space_coeff));
            space_weight = exp256_ps(space_weight);

            __m256 weight = _mm256_mul_ps(range_weight, space_weight);
            weight_sum = _mm256_add_ps(weight_sum, weight);
            __m256 weighted_value = _mm256_mul_ps(weight, neighbor_value);
            pixel_sum = _mm256_add_ps(pixel_sum, weighted_value);
        }
    }
    return _mm256_div_ps(pixel_sum, weight_sum);
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Invalid argument, should be: ./executable /path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }

    const char* input_filepath = argv[1];
    std::cout << "Input file from: " << input_filepath << "\n";
    auto input_jpeg = read_from_jpeg(input_filepath);
    if (input_jpeg.buffer == nullptr) {
        std::cerr << "Failed to read input JPEG image\n";
        return -1;
    }

    unsigned char* filteredImage = new unsigned char[input_jpeg.width * input_jpeg.height * input_jpeg.num_channels];
    memset(filteredImage, 0, input_jpeg.width * input_jpeg.height * input_jpeg.num_channels);

    auto start_time = std::chrono::high_resolution_clock::now(); // Start recording time

    // Apply bilateral filter using SIMD for each pixel
    for (int y = 1; y < input_jpeg.height - 1; y++) {
        for (int x = 1; x < input_jpeg.width - 1; x++) {
            for (int c = 0; c < input_jpeg.num_channels; c++) {
                int idx = (y * input_jpeg.width + x) * input_jpeg.num_channels + c;
                __m256 result = bilateral_filter_kernel(input_jpeg.buffer, idx, input_jpeg.width, input_jpeg.height, input_jpeg.num_channels, 2.0f, 50.0f);
                float output;

                // Extract the lower 128 bits of the result
                __m128 result_low = _mm256_castps256_ps128(result);

                // Store the first float from the result
                _mm_store_ss(&output, result_low);
                filteredImage[idx] = static_cast<unsigned char>(output);
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now(); // Stop recording time
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";

    const char* output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    if (!export_jpeg({filteredImage, input_jpeg.width, input_jpeg.height, input_jpeg.num_channels, input_jpeg.color_space}, output_filepath)) {
        std::cerr << "Failed to write output JPEG\n";
        delete[] filteredImage;
        return -1;
    }

    delete[] filteredImage;
    return 0;
}
