//
// Created by Yang Yufan on 2023/10/6.
// Email: yufanyang1@link.cuhk.edu.cn
//
// MPI implementation of applying bilateral filter to a JPEG image
//

#include <mpi.h> // MPI Header
#include <chrono>
#include <iostream>
#include <vector>
#include <cmath>

#include "utils.hpp" // Assume this handles JPEG read/write and provides necessary structs

#define MASTER 0
#define TAG_GATHER 0

// Function to apply bilateral filtering on a segment of the image
void apply_bilateral_filter(JPEGMeta& input_jpeg, unsigned char* output, int start, int end, int width, float sigma_color, float sigma_space) {
    int filter_radius = 3;
    for (int index = start; index < end; index++) {
        int x = index % width;
        int y = index / width;
        float sum_weights = 0.0;
        float filtered_value = 0.0;
        for (int dy = -filter_radius; dy <= filter_radius; dy++) {
            for (int dx = -filter_radius; dx <= filter_radius; dx++) {
                int nx = x + dx;
                int ny = y + dy;
                if (nx >= 0 && nx < width && ny >= 0 && ny < input_jpeg.height) {
                    int neighbor_index = ny * width + nx;
                    float distance = sqrtf(dx * dx + dy * dy);
                    float color_difference = input_jpeg.buffer[index] - input_jpeg.buffer[neighbor_index];
                    float space_weight = expf(-(distance * distance) / (2 * sigma_space * sigma_space));
                    float color_weight = expf(-(color_difference * color_difference) / (2 * sigma_color * sigma_color));
                    float weight = space_weight * color_weight;
                    filtered_value += input_jpeg.buffer[neighbor_index] * weight;
                    sum_weights += weight;
                }
            }
        }
        output[index - start] = static_cast<unsigned char>(filtered_value / sum_weights);
    }
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int numtasks, taskid;
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);

    if (argc != 3) {
        std::cerr << "Invalid argument, should be: ./executable /path/to/input/jpeg /path/to/output/jpeg\n";
        MPI_Finalize();
        return -1;
    }

    const char* input_filepath = argv[1];
    auto input_jpeg = read_from_jpeg(input_filepath);
    if (input_jpeg.buffer == NULL) {
        std::cerr << "Failed to read input JPEG image\n";
        MPI_Finalize();
        return -1;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    // Divide the image data among processes
    int total_pixels = input_jpeg.width * input_jpeg.height;
    int pixels_per_task = total_pixels / numtasks;
    int remaining_pixels = total_pixels % numtasks;
    std::vector<int> starts(numtasks), ends(numtasks);
    for (int i = 0; i < numtasks; i++) {
        starts[i] = i * pixels_per_task + std::min(i, remaining_pixels);
        ends[i] = starts[i] + pixels_per_task + (i < remaining_pixels ? 1 : 0);
    }

    // Allocate memory for local processing
    unsigned char* local_output = new unsigned char[ends[taskid] - starts[taskid]];

    // Apply the filter locally
    apply_bilateral_filter(input_jpeg, local_output, starts[taskid], ends[taskid], input_jpeg.width, 50.0f, 16.0f);

    if (taskid == MASTER) {
        // Master collects data from all processes
        unsigned char* complete_output = new unsigned char[total_pixels];
        MPI_Gather(local_output, ends[taskid] - starts[taskid], MPI_UNSIGNED_CHAR, complete_output, ends[taskid] - starts[taskid], MPI_UNSIGNED_CHAR, MASTER, MPI_COMM_WORLD);

        // Collect remaining parts from other processes
        for (int i = 1; i < numtasks; i++) {
            MPI_Recv(complete_output + starts[i], ends[i] - starts[i], MPI_UNSIGNED_CHAR, i, TAG_GATHER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        // Output results
        const char* output_filepath = argv[2];
        JPEGMeta output_jpeg{complete_output, input_jpeg.width, input_jpeg.height, input_jpeg.num_channels, input_jpeg.color_space};
        if (!export_jpeg(output_jpeg, output_filepath)) {
            std::cerr << "Failed to write output JPEG to file\n";
            delete[] complete_output;
            MPI_Finalize();
            return -1;
        }

        std::cout << "Transformation Complete!" << std::endl;
        std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";
        delete[] complete_output;
    } else {
        // Send local processing result to master
        MPI_Send(local_output, ends[taskid] - starts[taskid], MPI_UNSIGNED_CHAR, MASTER, TAG_GATHER, MPI_COMM_WORLD);
    }

    delete[] local_output;
    delete[] input_jpeg.buffer;

    MPI_Finalize();
    return 0;
}