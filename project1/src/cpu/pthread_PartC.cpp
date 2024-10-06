//
// Created by Zhang Na on 2023/9/15.
// Email: nazhang@link.cuhk.edu.cn
//
// Pthread implementation of applying a bilateral filter to a JPEG image
//

#include <iostream>
#include <chrono>
#include <pthread.h>
#include <cmath>
#include "utils.hpp"

// Structure to pass data to each thread
struct ThreadData {
    unsigned char* input_buffer;
    unsigned char* output_buffer;
    int width;
    int height;
    int start_row;
    int end_row;
    float sigma_s;
    float sigma_r;
};

// Function to apply bilateral filtering for a portion of the image
void* apply_bilateral_filter(void* arg) {
    ThreadData* data = reinterpret_cast<ThreadData*>(arg);
    int filter_radius = 3;
    float s_factor = -0.5 / (data->sigma_s * data->sigma_s);
    float r_factor = -0.5 / (data->sigma_r * data->sigma_r);

    for (int y = data->start_row; y < data->end_row; y++) {
        for (int x = 0; x < data->width; x++) {
            for (int channel = 0; channel < 3; channel++) { // Process each color channel
                float sum = 0.0f;
                float norm = 0.0f;

                for (int dy = -filter_radius; dy <= filter_radius; dy++) {
                    int ny = y + dy;
                    if (ny >= 0 && ny < data->height) {
                        for (int dx = -filter_radius; dx <= filter_radius; dx++) {
                            int nx = x + dx;
                            if (nx >= 0 && nx < data->width) {
                                int current_index = ((y * data->width + x) * 3) + channel;
                                int neighbor_index = ((ny * data->width + nx) * 3) + channel;
                                float spatial = dx * dx + dy * dy;
                                float range = data->input_buffer[current_index] - data->input_buffer[neighbor_index];
                                range *= range;
                                float weight = exp((spatial * s_factor) + (range * r_factor));
                                
                                sum += data->input_buffer[neighbor_index] * weight;
                                norm += weight;
                            }
                        }
                    }
                }

                int output_index = ((y * data->width + x) * 3) + channel;
                data->output_buffer[output_index] = static_cast<unsigned char>(sum / norm);
            }
        }
    }

    return nullptr;
}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "Invalid argument, should be: ./executable /path/to/input/jpeg /path/to/output/jpeg num_threads sigma_s sigma_r\n";
        return -1;
    }

    int num_threads = std::stoi(argv[3]);
    float sigma_s = std::stof(argv[4]);
    float sigma_r = std::stof(argv[5]);

    // Read input JPEG
    const char* input_filepath = argv[1];
    std::cout << "Input file from: " << input_filepath << "\n";
    auto input_jpeg = read_from_jpeg(input_filepath);

    auto output_buffer = new unsigned char[input_jpeg.width * input_jpeg.height * 3]; // Allocate buffer for color image

    pthread_t threads[num_threads];
    ThreadData thread_data[num_threads];

    auto start_time = std::chrono::high_resolution_clock::now();

    int rows_per_thread = input_jpeg.height / num_threads;
    for (int i = 0; i < num_threads; i++) {
        thread_data[i].input_buffer = input_jpeg.buffer;
        thread_data[i].output_buffer = output_buffer;
        thread_data[i].width = input_jpeg.width;
        thread_data[i].height = input_jpeg.height;
        thread_data[i].start_row = i * rows_per_thread;
        thread_data[i].end_row = (i == num_threads - 1) ? input_jpeg.height : (i + 1) * rows_per_thread;
        thread_data[i].sigma_s = sigma_s;
        thread_data[i].sigma_r = sigma_r;

        pthread_create(&threads[i], nullptr, apply_bilateral_filter, &thread_data[i]);
    }

    // Wait for all threads to finish
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], nullptr);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // Write output JPEG
    const char* output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    JPEGMeta output_jpeg{output_buffer, input_jpeg.width, input_jpeg.height, 3, JCS_RGB};
    if (export_jpeg(output_jpeg, output_filepath)) {
        std::cerr << "Failed to write output JPEG\n";
        return -1;
    }

    // Release allocated memory
    delete[] input_jpeg.buffer;
    delete[] output_buffer;

    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";

    return 0;
}
