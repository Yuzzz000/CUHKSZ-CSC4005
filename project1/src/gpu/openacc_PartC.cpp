#include <memory.h>
#include <cstring>
#include <chrono>
#include <cmath>
#include <iostream>
#include <openacc.h>

#include "../utils.hpp"

#define FILTER_RADIUS 2
#define SIGMA_S 10.0f
#define SIGMA_R 50.0f

#pragma acc routine seq
float gaussian(float x, float sigma) {
    return expf(- (x * x) / (2 * sigma * sigma));
}

#pragma acc routine seq
float acc_bilateral_filter(unsigned char* image_buffer, int x, int y, int width, int height, int num_channels) {
    float sum = 0.0f;
    float Wp = 0.0f;
    for (int dy = -FILTER_RADIUS; dy <= FILTER_RADIUS; dy++) {
        for (int dx = -FILTER_RADIUS; dx <= FILTER_RADIUS; dx++) {
            int nx = x + dx;
            int ny = y + dy;
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                int idx = (ny * width + nx) * num_channels;
                float range_kernel = gaussian(image_buffer[idx] - image_buffer[(y * width + x) * num_channels], SIGMA_R);
                float spatial_kernel = gaussian(sqrtf(dx * dx + dy * dy), SIGMA_S);
                float weight = range_kernel * spatial_kernel;
                sum += image_buffer[idx] * weight;
                Wp += weight;
            }
        }
    }
    return sum / Wp;
}

#pragma acc routine seq
unsigned char acc_clamp_pixel_value(float pixel) {
    return pixel > 255 ? 255 : pixel < 0 ? 0 : static_cast<unsigned char>(pixel);
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Invalid argument, should be: ./executable /path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }

    const char* input_filename = argv[1];
    std::cout << "Input file from: " << input_filename << "\n";
    auto input_jpeg = read_from_jpeg(input_filename);
    
    int width = input_jpeg.width;
    int height = input_jpeg.height;
    int num_channels = input_jpeg.num_channels;
    size_t buffer_size = width * height * num_channels;
    unsigned char* filteredImage = new unsigned char[buffer_size];
    unsigned char* buffer = new unsigned char[buffer_size];

    memcpy(buffer, input_jpeg.buffer, buffer_size);
    delete[] input_jpeg.buffer;

#pragma acc enter data copyin(buffer[0:buffer_size])
#pragma acc enter data create(filteredImage[0:buffer_size])

    auto start_time = std::chrono::high_resolution_clock::now();
#pragma acc parallel loop present(buffer[0:buffer_size], filteredImage[0:buffer_size])
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < num_channels; c++) {
                int idx = (y * width + x) * num_channels + c;
                float filtered_value = acc_bilateral_filter(buffer, x, y, width, height, num_channels);
                filteredImage[idx] = acc_clamp_pixel_value(filtered_value);
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
#pragma acc exit data copyout(filteredImage[0:buffer_size])

    const char* output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    JPEGMeta output_jpeg{filteredImage, width, height, num_channels, input_jpeg.color_space};
    if (!export_jpeg(output_jpeg, output_filepath)) {
        std::cerr << "Failed to write output JPEG\n";
        return -1;
    }

    delete[] buffer;
    delete[] filteredImage;
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";

    return 0;
}
