// Created by Liu Yuxuan on 2024/9/10
// Modified on Yang Yufan's simd_PartB.cpp on 2023/9/16
// Email: yufanyang1@link.cuhk.edu.cn
// Email: yuxuanliu1@link.cuhk.edu.cn
//
// SIMD (AVX2) implementation of Bilateral Filtering for JPEG picture

#include <memory.h>
#include <chrono>
#include <iostream>
#include <immintrin.h>
#include <cmath>

#include "../utils.hpp"

inline float gaussian(float x, float sigma) {
    return std::exp(-(x * x) / (2 * sigma * sigma));
}

inline __m256 exp_ps(__m256 x) {
    x = _mm256_min_ps(x, _mm256_set1_ps(88.3762626647949f));
    x = _mm256_max_ps(x, _mm256_set1_ps(-88.3762626647949f));
    __m256 fx = _mm256_mul_ps(x, _mm256_set1_ps(1.44269504088896341f));
    fx = _mm256_round_ps(fx, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 y = _mm256_sub_ps(x, _mm256_mul_ps(fx, _mm256_set1_ps(0.693359375f)));
    y = _mm256_sub_ps(y, _mm256_mul_ps(fx, _mm256_set1_ps(-2.12194440e-4f)));
    __m256 z = _mm256_add_ps(_mm256_set1_ps(1.0f), _mm256_mul_ps(y, _mm256_set1_ps(0.041944388f)));
    z = _mm256_add_ps(_mm256_set1_ps(1.0f), _mm256_mul_ps(z, y));
    z = _mm256_add_ps(_mm256_set1_ps(1.0f), _mm256_mul_ps(z, y));
    z = _mm256_add_ps(_mm256_set1_ps(1.0f), _mm256_mul_ps(z, y));
    return _mm256_mul_ps(z, _mm256_castsi256_ps(_mm256_cvttps_epi32(fx)));
}

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cerr << "Invalid argument, should be: ./executable /path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }
    // Read JPEG File
    const char* input_filepath = argv[1];
    std::cout << "Input file from: " << input_filepath << "\n";
    JpegSOA input_jpeg = read_jpeg_soa(input_filepath);
    if (input_jpeg.r_values == nullptr)
    {
        std::cerr << "Failed to read input JPEG image\n";
        return -1;
    }

    auto filteredImage = new unsigned char[input_jpeg.width * input_jpeg.height * input_jpeg.num_channels];

    float sigma_s = 10.0f; // Spatial sigma
    float sigma_r = 25.0f; // Range sigma

    auto start_time = std::chrono::high_resolution_clock::now();

    // Bilateral Filtering with SIMD (AVX2)
    for (int y = 1; y < input_jpeg.height - 1; ++y)
    {
        for (int x = 1; x < input_jpeg.width - 1; x += 8)
        {
            int idx = y * input_jpeg.width + x;

            __m256 w_sum_r = _mm256_setzero_ps();
            __m256 w_sum_g = _mm256_setzero_ps();
            __m256 w_sum_b = _mm256_setzero_ps();
            __m256 sum_r = _mm256_setzero_ps();
            __m256 sum_g = _mm256_setzero_ps();
            __m256 sum_b = _mm256_setzero_ps();

            for (int ky = -1; ky <= 1; ++ky)
            {
                for (int kx = -1; kx <= 1; ++kx)
                {
                    int neighbor_idx = (y + ky) * input_jpeg.width + (x + kx);

                    // Load neighboring pixel values as unsigned chars and convert to float
                    __m128i neighbor_r_chars = _mm_loadl_epi64((__m128i*)&input_jpeg.r_values[neighbor_idx]);
                    __m128i neighbor_g_chars = _mm_loadl_epi64((__m128i*)&input_jpeg.g_values[neighbor_idx]);
                    __m128i neighbor_b_chars = _mm_loadl_epi64((__m128i*)&input_jpeg.b_values[neighbor_idx]);

                    __m256 neighbor_r = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(neighbor_r_chars));
                    __m256 neighbor_g = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(neighbor_g_chars));
                    __m256 neighbor_b = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(neighbor_b_chars));

                    // Spatial weight
                    float w_s = gaussian(std::sqrt(kx * kx + ky * ky), sigma_s);
                    __m256 w_s_vec = _mm256_set1_ps(w_s);

                    // Range weight (R, G, B channels separately)
                    __m128i center_r_chars = _mm_loadl_epi64((__m128i*)&input_jpeg.r_values[idx]);
                    __m128i center_g_chars = _mm_loadl_epi64((__m128i*)&input_jpeg.g_values[idx]);
                    __m128i center_b_chars = _mm_loadl_epi64((__m128i*)&input_jpeg.b_values[idx]);

                    __m256 center_r = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(center_r_chars));
                    __m256 center_g = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(center_g_chars));
                    __m256 center_b = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(center_b_chars));

                    __m256 diff_r = _mm256_sub_ps(neighbor_r, center_r);
                    __m256 diff_g = _mm256_sub_ps(neighbor_g, center_g);
                    __m256 diff_b = _mm256_sub_ps(neighbor_b, center_b);

                    __m256 w_r_r = exp_ps(_mm256_div_ps(_mm256_mul_ps(diff_r, diff_r), _mm256_set1_ps(-2 * sigma_r * sigma_r)));
                    __m256 w_r_g = exp_ps(_mm256_div_ps(_mm256_mul_ps(diff_g, diff_g), _mm256_set1_ps(-2 * sigma_r * sigma_r)));
                    __m256 w_r_b = exp_ps(_mm256_div_ps(_mm256_mul_ps(diff_b, diff_b), _mm256_set1_ps(-2 * sigma_r * sigma_r)));

                    __m256 weight_r = _mm256_mul_ps(w_s_vec, w_r_r);
                    __m256 weight_g = _mm256_mul_ps(w_s_vec, w_r_g);
                    __m256 weight_b = _mm256_mul_ps(w_s_vec, w_r_b);

                    sum_r = _mm256_add_ps(_mm256_mul_ps(weight_r, neighbor_r), sum_r);
                    sum_g = _mm256_add_ps(_mm256_mul_ps(weight_g, neighbor_g), sum_g);
                    sum_b = _mm256_add_ps(_mm256_mul_ps(weight_b, neighbor_b), sum_b);

                    w_sum_r = _mm256_add_ps(w_sum_r, weight_r);
                    w_sum_g = _mm256_add_ps(w_sum_g, weight_g);
                    w_sum_b = _mm256_add_ps(w_sum_b, weight_b);
                }
            }

            // Normalize and store results
            __m256 result_r = _mm256_div_ps(sum_r, w_sum_r);
            __m256 result_g = _mm256_div_ps(sum_g, w_sum_g);
            __m256 result_b = _mm256_div_ps(sum_b, w_sum_b);

            __m256i result_r_int = _mm256_cvtps_epi32(result_r);
            __m256i result_g_int = _mm256_cvtps_epi32(result_g);
            __m256i result_b_int = _mm256_cvtps_epi32(result_b);

            __m128i result_r_chars = _mm256_castsi256_si128(result_r_int);
            __m128i result_g_chars = _mm256_castsi256_si128(result_g_int);
            __m128i result_b_chars = _mm256_castsi256_si128(result_b_int);

            _mm_storel_epi64((__m128i*)&filteredImage[idx * input_jpeg.num_channels], result_r_chars);
            _mm_storel_epi64((__m128i*)&filteredImage[idx * input_jpeg.num_channels + 1], result_g_chars);
            _mm_storel_epi64((__m128i*)&filteredImage[idx * input_jpeg.num_channels + 2], result_b_chars);
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // Save output JPEG image
    const char* output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    JPEGMeta output_jpeg{filteredImage, input_jpeg.width, input_jpeg.height, input_jpeg.num_channels, JCS_RGB};
    if (export_jpeg(output_jpeg, output_filepath))
    {
        std::cerr << "Failed to write output JPEG\n";
        return -1;
    }

    // Post-processing
    delete[] input_jpeg.r_values;
    delete[] input_jpeg.g_values;
    delete[] input_jpeg.b_values;
    delete[] filteredImage;

    std::cout << "Transformation Complete!\n";
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";

    return 0;
}