#include <iostream>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>
#include "utils.hpp"  // Make sure this path is correct

#define FILTER_RADIUS 2
#define SIGMA_S 10.0f
#define SIGMA_R 50.0f

__device__ float gaussian(float x, float sigma) {
    return expf(-0.5f * x * x / (sigma * sigma));
}

__device__ float d_bilateral_filter(unsigned char* image_buffer,
                                    int x, int y, int width, int height, int channels,
                                    float sigma_s, float sigma_r, int pixel_id) {
    float iFiltered = 0.0f;
    float wP = 0.0f;
    
    for (int dy = -FILTER_RADIUS; dy <= FILTER_RADIUS; dy++) {
        for (int dx = -FILTER_RADIUS; dx <= FILTER_RADIUS; dx++) {
            int nx = x + dx;
            int ny = y + dy;

            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                int neighbor_id = (ny * width + nx) * channels + pixel_id % channels;
                float dist = sqrtf(dx * dx + dy * dy);
                float colorDistance = fabs(float(image_buffer[neighbor_id]) - float(image_buffer[pixel_id]));

                float gs = gaussian(dist, sigma_s);
                float gr = gaussian(colorDistance, sigma_r);
                float weight = gs * gr;

                iFiltered += image_buffer[neighbor_id] * weight;
                wP += weight;
            }
        }
    }

    return wP > 0.0f ? iFiltered / wP : float(image_buffer[pixel_id]);
}

__device__ unsigned char d_clamp_pixel_value(float pixel) {
    return pixel > 255 ? 255 : pixel < 0 ? 0 : static_cast<unsigned char>(pixel);
}

__global__ void apply_bilateral_filter_kernel(unsigned char* input_buffer,
                                              unsigned char* filtered_image, 
                                              int width, int height, int num_channels,
                                              float sigma_s, float sigma_r) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
        for (int c = 0; c < num_channels; c++) {
            int idx = (y * width + x) * num_channels + c;
            float filtered_val = d_bilateral_filter(input_buffer, x, y, width, height, num_channels, sigma_s, sigma_r, idx);
            filtered_image[idx] = d_clamp_pixel_value(filtered_val);
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Invalid argument, should be: ./executable /path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }

    const char* input_filename = argv[1];
    const char* output_filename = argv[2];
    std::cout << "Input file from: " << input_filename << "\n";

    JPEGMeta input_jpeg = read_from_jpeg(input_filename);
    if (input_jpeg.buffer == nullptr) {
        std::cerr << "Failed to read input JPEG image\n";
        return -1;
    }

    int width = input_jpeg.width;
    int height = input_jpeg.height;
    int num_channels = input_jpeg.num_channels;
    size_t buffer_size = width * height * num_channels;

    unsigned char* d_input_buffer;
    unsigned char* d_filtered_image;
    cudaMalloc((void**)&d_input_buffer, buffer_size);
    cudaMalloc((void**)&d_filtered_image, buffer_size);

    cudaMemcpy(d_input_buffer, input_jpeg.buffer, buffer_size, cudaMemcpyHostToDevice);

    dim3 blockDim(32, 32);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                 (height + blockDim.y - 1) / blockDim.y);

    apply_bilateral_filter_kernel<<<gridDim, blockDim>>>(d_input_buffer, d_filtered_image, width, height, num_channels, SIGMA_S, SIGMA_R);

    unsigned char* filteredImage = new unsigned char[buffer_size];
    cudaMemcpy(filteredImage, d_filtered_image, buffer_size, cudaMemcpyDeviceToHost);

    // Wrapping filtered image data into JPEGMeta structure for export
    JPEGMeta output_jpeg{
        filteredImage,
        width,
        height,
        num_channels,
        input_jpeg.color_space
    };

    if (export_jpeg(output_jpeg, output_filename) != 0) {
        std::cerr << "Failed to write output JPEG\n";
        return -1;
    }

    cudaFree(d_input_buffer);
    cudaFree(d_filtered_image);
    delete[] input_jpeg.buffer;
    delete[] filteredImage;

    std::cout << "Filtering complete and image saved to " << output_filename << std::endl;
    return 0;
}
