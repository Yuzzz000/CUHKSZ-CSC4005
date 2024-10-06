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

// Function to apply bilateral filtering on a segment of the image (JpegAOS format)
void apply_bilateral_filter(Pixel* input_pixels, Pixel* output_pixels, int start, int end, int width) {
    #pragma omp parallel for
    for (int i = start; i < end; i++) {
        int x = i % width;
        int y = i / width;

        // Apply bilateral filter for each channel (R, G, B)
        for (int channel = 0; channel < 3; channel++) {
            ColorValue filtered_value = bilateral_filter(input_pixels, y, x, width, channel);
            output_pixels[i].set_channel(channel, filtered_value);
        }
    }
}

// Wrapper to filter a segment of the image and store the result in the output array
void set_filtered_image(Pixel* filtered_image, const Pixel* image, int width, int start_line, int end_line) {
    apply_bilateral_filter(const_cast<Pixel*>(image), filtered_image, start_line * width, end_line * width, width);
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
    int numtasks, taskid;
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);

    MPI_Status status;

    // Read JPEG File (Using JpegAOS structure from utils.cpp)
    const char* input_filepath = argv[1];
    JpegAOS input_jpeg = read_jpeg_aos(input_filepath);
    if (input_jpeg.pixels == nullptr) {
        std::cerr << "Failed to read input JPEG image\n";
        return -1;
    }

    // Divide the task
    int total_line_num = input_jpeg.height;
    int line_per_task = total_line_num / numtasks;
    int left_line_num = total_line_num % numtasks;

    std::vector<int> cuts(numtasks + 1, 0);
    for (int i = 0; i < numtasks; i++) {
        cuts[i + 1] = cuts[i] + line_per_task + (i < left_line_num ? 1 : 0);
    }

    // Allocate filtered image memory
    Pixel* filtered_image = new Pixel[input_jpeg.width * input_jpeg.height];

    if (taskid == MASTER) {
        std::cout << "Input file from: " << input_filepath << "\n";
        auto start_time = std::chrono::high_resolution_clock::now();

        // Filter the first division of the contents
        set_filtered_image(filtered_image, input_jpeg.pixels, input_jpeg.width, cuts[taskid], cuts[taskid + 1]);

        // Receive the filtered contents from slave executors
        for (int i = 1; i < numtasks; i++) {
            int length = input_jpeg.width * (cuts[i + 1] - cuts[i]);
            MPI_Recv(&filtered_image[cuts[i] * input_jpeg.width], length * sizeof(Pixel), MPI_BYTE, i, TAG_GATHER, MPI_COMM_WORLD, &status);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        // Save output JPEG image
        const char* output_filepath = argv[2];
        std::cout << "Output file to: " << output_filepath << "\n";
        JpegAOS output_jpeg{filtered_image, input_jpeg.width, input_jpeg.height, input_jpeg.num_channels, input_jpeg.color_space};
        if (export_jpeg(output_jpeg, output_filepath) != 0) {
            std::cerr << "Failed to write output JPEG\n";
            return -1;
        }

        // Post-processing
        delete[] input_jpeg.pixels;
        delete[] filtered_image;

        std::cout << "Transformation Complete!" << std::endl;
        std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";
    }
    else {
        // Allocate space for local filtered image
        int length = input_jpeg.width * (cuts[taskid + 1] - cuts[taskid]);
        Pixel* local_filtered_image = new Pixel[length];

        // Filter a corresponding division
        set_filtered_image(local_filtered_image, input_jpeg.pixels, input_jpeg.width, cuts[taskid], cuts[taskid + 1]);

        // Send the filtered image back to the master
        MPI_Send(local_filtered_image, length * sizeof(Pixel), MPI_BYTE, MASTER, TAG_GATHER, MPI_COMM_WORLD);

        // Release memory
        delete[] local_filtered_image;
    }

    MPI_Finalize();
    return 0;
}
