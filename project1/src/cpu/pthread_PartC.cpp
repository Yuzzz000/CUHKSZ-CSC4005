#include <memory.h>
#include <chrono>
#include <iostream>
#include <pthread.h>

#include "../utils.hpp"

struct ThreadData {
    ColorValue *inputChannel;
    ColorValue *outputChannel;
    int width;
    int height;
    int startRow;
    int endRow;
};

void *bilateral_filter_thread(void *arg) {
    ThreadData *data = (ThreadData *)arg;
    for (int row = data->startRow; row < data->endRow; ++row) {
        for (int col = 1; col < data->width - 1; ++col) {
            int index = row * data->width + col;
            ColorValue filteredValue = bilateral_filter(data->inputChannel, row, col, data->width);
            data->outputChannel[index] = filteredValue;
        }
    }
    return nullptr;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        std::cerr << "Invalid argument, should be: ./executable /path/to/input/jpeg /path/to/output/jpeg NUM_THREADS\n";
        return -1;
    }
    const char* input_filename = argv[1];
    std::cout << "Input file from: " << input_filename << "\n";
    JpegSOA input_jpeg = read_jpeg_soa(input_filename);
    if (input_jpeg.r_values == nullptr) {
        std::cerr << "Failed to read input JPEG image\n";
        return -1;
    }

    int NUM_THREADS = std::stoi(argv[3]);
    pthread_t threads[NUM_THREADS];
    ThreadData threadData[NUM_THREADS];
    int rowsPerThread = input_jpeg.height / NUM_THREADS;

    ColorValue *output_r_values = new ColorValue[input_jpeg.width * input_jpeg.height];

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < NUM_THREADS; i++) {
        threadData[i] = {
            input_jpeg.r_values,
            output_r_values,
            input_jpeg.width,
            input_jpeg.height,
            i * rowsPerThread,
            (i + 1) * rowsPerThread - (i == NUM_THREADS - 1 ? 0 : 1)
        };
        pthread_create(&threads[i], nullptr, bilateral_filter_thread, &threadData[i]);
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], nullptr);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";

    delete[] output_r_values;
    return 0;
}
