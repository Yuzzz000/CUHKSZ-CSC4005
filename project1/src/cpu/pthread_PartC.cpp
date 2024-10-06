#include <iostream>
#include <chrono>
#include <pthread.h>
#include <cmath>
#include "utils.hpp"  // 确保此头文件包含必要的JPEG读写功能及数据结构定义

struct ThreadData {
    unsigned char* input_buffer;
    unsigned char* output_buffer;
    int width;
    int height;
    int num_channels;
    int start_row;
    int end_row;
};

void* apply_bilateral_filter(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    int filter_radius = 3;
    float sigma_s = 25.0f;  // 默认空间标准差
    float sigma_r = 75.0f;  // 默认范围标准差
    float s_factor = -0.5 / (sigma_s * sigma_s);
    float r_factor = -0.5 / (sigma_r * sigma_r);

    for (int y = data->start_row; y < data->end_row; y++) {
        for (int x = 0; x < data->width; x++) {
            for (int channel = 0; channel < 3; channel++) {  // 假设处理的是RGB图像
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
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <input_path> <output_path> <num_threads>\n";
        return -1;
    }

    const char* input_filename = argv[1];
    const char* output_filename = argv[2];
    int num_threads = std::stoi(argv[3]);

    auto input_jpeg = read_from_jpeg(input_filename);
    unsigned char* filtered_image = new unsigned char[input_jpeg.width * input_jpeg.height * input_jpeg.num_channels];

    pthread_t* threads = new pthread_t[num_threads];
    ThreadData* thread_data = new ThreadData[num_threads];
    int rows_per_thread = input_jpeg.height / num_threads;

    for (int i = 0; i < num_threads; i++) {
        thread_data[i] = {
            input_jpeg.buffer,
            filtered_image,
            input_jpeg.width,
            input_jpeg.height,
            3,  // 假设是RGB图像
            i * rows_per_thread,
            (i == num_threads - 1) ? input_jpeg.height : (i + 1) * rows_per_thread
        };
        pthread_create(&threads[i], NULL, apply_bilateral_filter, &thread_data[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    JPEGMeta output_jpeg{filtered_image, input_jpeg.width, input_jpeg.height, input_jpeg.num_channels, JCS_RGB};
    if (!export_jpeg(output_jpeg, output_filename)) {
        std::cerr << "Failed to write output JPEG\n";
        return -1;
    }

    delete[] input_jpeg.buffer;
    delete[] filtered_image;
    delete[] threads;
    delete[] thread_data;

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - std::chrono::high_resolution_clock::now());
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";

    return 0;
}
