#include <memory.h>
#include <chrono>
#include <cmath>
#include <iostream>

#include "../utils.hpp"

// Function to apply bilateral filter for a specific pixel's color channel
float bilateral_filter(unsigned char* buffer, int index, int width, int height, int num_channels, float sigma_s, float sigma_r) {
    int filter_radius = 3;
    float s_factor = -0.5 / (sigma_s * sigma_s);
    float r_factor = -0.5 / (sigma_r * sigma_r);
    float norm = 0.0f, sum = 0.0f;

    int y = index / (width * num_channels);
    int x = (index % (width * num_channels)) / num_channels;

    for (int dy = -filter_radius; dy <= filter_radius; ++dy) {
        int ny = y + dy;
        if (ny >= 0 && ny < height) {
            for (int dx = -filter_radius; dx <= filter_radius; ++dx) {
                int nx = x + dx;
                if (nx >= 0 && nx < width) {
                    int current_index = (y * width + x) * num_channels + (index % num_channels);
                    int neighbor_index = (ny * width + nx) * num_channels + (index % num_channels);
                    float spatial = dx * dx + dy * dy;
                    float range = buffer[current_index] - buffer[neighbor_index];
                    float weight = exp(spatial * s_factor + range * range * r_factor);
                    sum += buffer[neighbor_index] * weight;
                    norm += weight;
                }
            }
        }
    }
    return sum / norm;
}



int main(int argc, char** argv)
{
    if (argc != 3) {
        std::cerr << "Invalid argument, should be: ./executable /path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }

    const char* input_filename = argv[1];
    std::cout << "Input file from: " << input_filename << "\n";
    auto input_jpeg = read_from_jpeg(input_filename);

    unsigned char* filteredImage = new unsigned char[input_jpeg.width * input_jpeg.height * input_jpeg.num_channels];
    memset(filteredImage, 0, input_jpeg.width * input_jpeg.height * input_jpeg.num_channels);

    float sigma_s = 2.0f;  // Spatial standard deviation for the filter
    float sigma_r = 50.0f; // Range standard deviation for the filter

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int y = 1; y < input_jpeg.height - 1; y++) {
        for (int x = 1; x < input_jpeg.width - 1; x++) {
            for (int c = 0; c < input_jpeg.num_channels; c++) {
                int index = (y * input_jpeg.width + x) * input_jpeg.num_channels + c;
                float sum = bilateral_filter(input_jpeg.buffer, index, input_jpeg.width, input_jpeg.height, input_jpeg.num_channels, sigma_s, sigma_r);
                filteredImage[index] = clamp_pixel_value(sum);
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    const char* output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    JPEGMeta output_jpeg{filteredImage, input_jpeg.width, input_jpeg.height, input_jpeg.num_channels, input_jpeg.color_space};
    if (export_jpeg(output_jpeg, output_filepath)) {
        std::cerr << "Failed to write output JPEG\n";
        delete[] input_jpeg.buffer;
        delete[] filteredImage;
        return -1;
    }

    delete[] input_jpeg.buffer;
    delete[] filteredImage;
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";
    return 0;
}
