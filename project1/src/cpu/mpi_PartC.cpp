// Created by Yang Yufan on 2023/10/6.
// Email: yufanyang1@link.cuhk.edu.cn
//
// MPI implementation of applying bilateral filter to a JPEG image
//

#include <mpi.h> // MPI Header
#include <chrono>
#include <cstring>
#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h> // OpenMP Header

#include "utils.hpp" // Assume this handles JPEG read/write and provides necessary structs

#define MASTER 0
#define TAG_GATHER 0

// Function to apply bilateral filtering on a segment of the image
void apply_bilateral_filter(JPEGMeta& input_jpeg, unsigned char* output, int start, int end, int width, double sigma_color, double sigma_space) {
    int filter_radius = 3;
    #pragma omp parallel for
    for (int index = start; index < end; index++) {
        int x = index % width;
        int y = index / width;
        double sum_weights = 0.0;
        double filtered_value = 0.0;
        for (int dy = -filter_radius; dy <= filter_radius; dy++) {
            for (int dx = -filter_radius; dx <= filter_radius; dx++) {
                int nx = x + dx;
                int ny = y + dy;
                if (nx >= 0 && nx < width && ny >= 0 && ny < input_jpeg.height) {
                    int neighbor_index = ny * width + nx;
                    double distance = sqrt(dx * dx + dy * dy);
                    double color_difference = input_jpeg.buffer[index] - input_jpeg.buffer[neighbor_index];
                    double space_weight = exp(-(distance * distance) / (2 * sigma_space * sigma_space));
                    double color_weight = exp(-(color_difference * color_difference) / (2 * sigma_color * sigma_color));
                    double weight = space_weight * color_weight;
                    filtered_value += input_jpeg.buffer[neighbor_index] * weight;
                    sum_weights += weight;
                }
            }
        }
        output[index - start] = static_cast<unsigned char>(filtered_value / sum_weights);
    }
}

void set_filtered_image(unsigned char* filtered_image, unsigned char* image, int width, int num_channels, int start_line, int end_line, int offset) {
    apply_bilateral_filter(reinterpret_cast<JPEGMeta&>(*image), filtered_image, start_line * width, end_line * width, width, 50.0, 16.0);
}

int main(int argc, char** argv)
{
    // Verify input argument format
    if (argc != 3) {
        std::cerr << "Invalid argument, should be: ./executable /path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }

    // Start the MPI
    MPI_Init(&argc, &argv);
    // How many processes are running
    int numtasks;
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    // What's my rank?
    int taskid;
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    // Which node am I running on?
    int len;
    char hostname[MPI_MAX_PROCESSOR_NAME];
    MPI_Get_processor_name(hostname, &len);
    MPI_Status status;

    // Read JPEG File
    const char* input_filepath = argv[1];
    auto input_jpeg = read_from_jpeg(input_filepath);
    if (input_jpeg.buffer == NULL) {
        std::cerr << "Failed to read input JPEG image\n";
        return -1;
    }

    // Divide the task
    int total_line_num = input_jpeg.height - 2;
    int line_per_task = total_line_num / numtasks;
    int left_line_num = total_line_num % numtasks;

    std::vector<int> cuts(numtasks + 1, 1);
    int divided_left_line_num = 0;
    for (int i = 0; i < numtasks; i++) {
        if (divided_left_line_num < left_line_num) {
            cuts[i + 1] = cuts[i] + line_per_task + 1;
            divided_left_line_num++;
        } else {
            cuts[i + 1] = cuts[i] + line_per_task;
        }
    }

    // The tasks for the master executor
    if (taskid == MASTER) {
        std::cout << "Input file from: " << input_filepath << "\n";
        auto filteredImage = new unsigned char[input_jpeg.width * input_jpeg.height * input_jpeg.num_channels];
        memset(filteredImage, 0, input_jpeg.width * input_jpeg.height * input_jpeg.num_channels);

        auto start_time = std::chrono::high_resolution_clock::now();

        // Filter the first division of the contents
        set_filtered_image(filteredImage, input_jpeg.buffer, input_jpeg.width, input_jpeg.num_channels, cuts[taskid], cuts[taskid + 1], 0);

        // Receive the filtered contents from slave executors
        for (int i = MASTER + 1; i < numtasks; i++) {
            int line_width = input_jpeg.width * input_jpeg.num_channels;
            unsigned char* start_pos = filteredImage + cuts[i] * line_width;
            int length = (cuts[i + 1] - cuts[i]) * line_width;
            MPI_Recv(start_pos, length, MPI_CHAR, i, TAG_GATHER, MPI_COMM_WORLD, &status);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        // Save output JPEG image
        const char* output_filepath = argv[2];
        std::cout << "Output file to: " << output_filepath << "\n";
        JPEGMeta output_jpeg{filteredImage, input_jpeg.width, input_jpeg.height, input_jpeg.num_channels, input_jpeg.color_space};
        if (!export_jpeg(output_jpeg, output_filepath)) {
            std::cerr << "Failed to write output JPEG\n";
            return -1;
        }
        // Post-processing
        delete[] input_jpeg.buffer;
        delete[] filteredImage;
        std::cout << "Transformation Complete!" << std::endl;
        std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";
    }
    // The tasks for the slave executor
    else {
        // Initialize the filtered image
        int length = input_jpeg.width * (cuts[taskid + 1] - cuts[taskid]) * input_jpeg.num_channels;
        int offset = input_jpeg.width * cuts[taskid] * input_jpeg.num_channels;

        auto filteredImage = new unsigned char[length];
        memset(filteredImage, 0, length);

        // Filter a corresponding division
        set_filtered_image(filteredImage, input_jpeg.buffer, input_jpeg.width, input_jpeg.num_channels, cuts[taskid], cuts[taskid + 1], offset);

        // Send the filtered image back to the master
        MPI_Send(filteredImage, length, MPI_CHAR, MASTER, TAG_GATHER, MPI_COMM_WORLD);

        // Release the memory
        delete[] filteredImage;
    }

    MPI_Finalize();
    return 0;
}