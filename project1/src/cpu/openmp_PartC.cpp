//
// Created by Zhang Na on 2023/10/6.
// Email: nazhang@link.cuhk.edu.cn
//
// OpenMP implementation of applying bilateral filtering to a color JPEG image
//

#include <iostream>
#include <chrono>
#include <omp.h> // OpenMP header for parallel programming
#include <cmath> // For exp function
#include "../utils.hpp"

// Function to perform the bilateral filter on a color image
void bilateral_filter(unsigned char* input, unsigned char* output, int width, int height, int num_channels, float sigma_s, float sigma_r) {
    int filter_radius = 3;
    float s_factor = -0.5 / (sigma_s * sigma_s);
    float r_factor = -0.5 / (sigma_r * sigma_r);

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int channel = 0; channel < num_channels; channel++) {
                float sum = 0.0f;
                float norm = 0.0f;

                for (int dy = -filter_radius; dy <= filter_radius; dy++) {
                    int ny = y + dy;
                    if (ny >= 0 && ny < height) {
                        for (int dx = -filter_radius; dx <= filter_radius; dx++) {
                            int nx = x + dx;
                            if (nx >= 0 && nx < width) {
                                int current_index = (y * width + x) * num_channels + channel;
                                int neighbor_index = (ny * width + nx) * num_channels + channel;
                                float spatial = dx * dx + dy * dy;
                                float range = input[current_index] - input[neighbor_index];
                                range *= range;
                                float weight = exp(spatial * s_factor + range * r_factor);

                                sum += input[neighbor_index] * weight;
                                norm += weight;
                            }
                        }
                    }
                }

                int output_index = (y * width + x) * num_channels + channel;
                output[output_index] = static_cast<unsigned char>(sum / norm); // Normalize the weighted sum
            }
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Invalid argument, should be: ./executable /path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }

    // Read input JPEG image
    const char* input_filepath = argv[1];
    std::cout << "Input file from: " << input_filepath << "\n";
    auto input_jpeg = read_from_jpeg(input_filepath);
    if (input_jpeg.buffer == NULL) {
        std::cerr << "Failed to read input JPEG image\n";
        return -1;
    }

    auto output_buffer = new unsigned char[input_jpeg.width * input_jpeg.height * input_jpeg.num_channels]; // Allocate buffer for the output image

    auto start_time = std::chrono::high_resolution_clock::now();

    // Apply the bilateral filter
    bilateral_filter(input_jpeg.buffer, output_buffer, input_jpeg.width, input_jpeg.height, input_jpeg.num_channels, 2.0f, 50.0f);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";

    // Save output JPEG image
    const char* output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    JPEGMeta output_jpeg{output_buffer, input_jpeg.width, input_jpeg.height, input_jpeg.num_channels, input_jpeg.color_space};
    if (!export_jpeg(output_jpeg, output_filepath)) {
        std::cerr << "Failed to save output JPEG image\n";
        return -1;
    }

    // Release the allocated memory
    delete[] input_jpeg.buffer;
    delete[] output_buffer;

    return 0;
}
