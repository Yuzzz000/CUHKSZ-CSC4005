#include <memory.h>
#include <chrono>
#include <cmath>
#include <iostream>

#include "../utils.hpp"

// Define the bilateral filtering function
ColorValue bilateral_filter(ColorValue* values, int index, int width, int height, float sigma_s, float sigma_r) {
    int filter_radius = 3;
    float s_factor = -0.5 / (sigma_s * sigma_s);
    float r_factor = -0.5 / (sigma_r * sigma_r);
    float norm = 0.0f, sum = 0.0f;

    int row = index / width;
    int col = index % width;

    for (int dy = -filter_radius; dy <= filter_radius; ++dy) {
        int ny = row + dy;
        if (ny >= 0 && ny < height) {  // Check vertical boundaries
            for (int dx = -filter_radius; dx <= filter_radius; ++dx) {
                int nx = col + dx;
                if (nx >= 0 && nx < width) {  // Check horizontal boundaries
                    int neighbor_index = ny * width + nx;
                    float spatial = dx * dx + dy * dy;
                    float range = values[index] - values[neighbor_index];
                    float weight = exp(spatial * s_factor + range * range * r_factor);
                    sum += values[neighbor_index] * weight;
                    norm += weight;
                }
            }
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

    // Allocate memory for output channels
    ColorValue* filtered_r_values = new ColorValue[width * height];
    ColorValue* filtered_g_values = new ColorValue[width * height];
    ColorValue* filtered_b_values = new ColorValue[width * height];
    
    JpegSOA output_jpeg{filtered_r_values, filtered_g_values, filtered_b_values, width, height, num_channels, input_jpeg.color_space};

    float sigma_s = 2.0f;  // Spatial standard deviation for the filter
    float sigma_r = 50.0f; // Range standard deviation for the filter

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int channel = 0; channel < num_channels; ++channel)
    {
        ColorValue* channel_values = input_jpeg.get_channel(channel);
        for (int row = 1; row < height - 1; ++row)
        {
            for (int col = 1; col < width - 1; ++col)
            {
                int index = row * width + col;
                ColorValue filtered_value = bilateral_filter(channel_values, index, width, height, sigma_s, sigma_r);
                output_jpeg.set_value(channel, index, filtered_value);
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    const char* output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    if (export_jpeg(output_jpeg, output_filepath))
    {
        std::cerr << "Failed to write output JPEG\n";
        delete[] filtered_r_values;
        delete[] filtered_g_values;
        delete[] filtered_b_values;
        return -1;
    }

    // Cleanup
    delete[] filtered_r_values;
    delete[] filtered_g_values;
    delete[] filtered_b_values;
    
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";
    return 0;
}
