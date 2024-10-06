//
// Created by Zhong Yebin on 2023/9/16.
// Email: yebinzhong@link.cuhk.edu.cn

#include <memory.h>
#include <cstring>
#include <chrono>
#include <cmath>
#include <iostream>
#include <openacc.h>

#include "../utils.hpp"

#pragma acc routine seq
ColorValue acc_clamp_pixel_value(float value)
{
    return value > 255 ? 255
           : value < 0 ? 0
                       : static_cast<unsigned char>(value);
}

#pragma acc routine seq
float acc_bilateral_filter(const ColorValue* channel, int row, int col, int width)
{
    const float sigma_space = 10.0f;
    const float sigma_color = 25.0f;
    const int filter_radius = 1;

    float sum = 0;
    float norm = 0;
    int index = row * width + col;
    float center_value = channel[index];

    for (int dy = -filter_radius; dy <= filter_radius; ++dy)
    {
        for (int dx = -filter_radius; dx <= filter_radius; ++dx)
        {
            int neighbor_row = row + dy;
            int neighbor_col = col + dx;
            int neighbor_index = neighbor_row * width + neighbor_col;

            float neighbor_value = channel[neighbor_index];

            float spatial_weight = expf(-(dx * dx + dy * dy) / (2 * sigma_space * sigma_space));
            float color_weight = expf(-(neighbor_value - center_value) * (neighbor_value - center_value) / (2 * sigma_color * sigma_color));

            float weight = spatial_weight * color_weight;

            sum += neighbor_value * weight;
            norm += weight;
        }
    }

    return sum / norm;
}

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cerr << "Invalid argument, should be: ./executable "
                     "/path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }

    // Read JPEG File
    const char* input_filename = argv[1];
    std::cout << "Input file from: " << input_filename << "\n";
    JpegSOA input_jpeg = read_jpeg_soa(input_filename);
    if (input_jpeg.r_values == nullptr)
    {
        std::cerr << "Failed to read input JPEG image\n";
        return -1;
    }

    int width = input_jpeg.width;
    int height = input_jpeg.height;
    int num_channels = input_jpeg.num_channels;
    ColorValue* output_r_values = new ColorValue[width * height];
    ColorValue* output_g_values = new ColorValue[width * height];
    ColorValue* output_b_values = new ColorValue[width * height];
    JpegSOA output_jpeg{
        output_r_values, output_g_values, output_b_values, width,
        height, num_channels, input_jpeg.color_space};

    size_t image_size = width * height;

    // Copy input_jpeg data to the device, including its members
#pragma acc enter data copyin(input_jpeg, input_jpeg.r_values[0:image_size], \
                              input_jpeg.g_values[0:image_size], \
                              input_jpeg.b_values[0:image_size])

    // Allocate memory on GPU for output arrays
#pragma acc enter data create(output_r_values[0:image_size], \
                              output_g_values[0:image_size], \
                              output_b_values[0:image_size])

    auto start_time = std::chrono::high_resolution_clock::now();

    // Parallel loop for filtering each channel
#pragma acc parallel loop collapse(2) present(input_jpeg, output_r_values, output_g_values, output_b_values)
    for (int row = 1; row < height - 1; ++row)
    {
        for (int col = 1; col < width - 1; ++col)
        {
            int index = row * width + col;
            output_r_values[index] = acc_clamp_pixel_value(
                acc_bilateral_filter(input_jpeg.r_values, row, col, width));
            output_g_values[index] = acc_clamp_pixel_value(
                acc_bilateral_filter(input_jpeg.g_values, row, col, width));
            output_b_values[index] = acc_clamp_pixel_value(
                acc_bilateral_filter(input_jpeg.b_values, row, col, width));
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

    // Copy data back from GPU to host
#pragma acc update self(output_r_values[0:image_size], \
                        output_g_values[0:image_size], \
                        output_b_values[0:image_size])

    // Free GPU memory
#pragma acc exit data delete(input_jpeg.r_values[0:image_size], \
                             input_jpeg.g_values[0:image_size], \
                             input_jpeg.b_values[0:image_size], \
                             output_r_values[0:image_size], \
                             output_g_values[0:image_size], \
                             output_b_values[0:image_size])

    // Save output JPEG image
    const char* output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    if (export_jpeg(output_jpeg, output_filepath))
    {
        std::cerr << "Failed to write output JPEG\n";
        return -1;
    }

    // Cleanup
    delete[] output_r_values;
    delete[] output_g_values;
    delete[] output_b_values;

    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count()
              << " milliseconds\n";
    return 0;
}
