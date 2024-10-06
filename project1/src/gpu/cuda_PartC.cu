#include <iostream>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>
#include "utils.hpp"  // 确保路径正确

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

    // 读取JPEG图像
    auto input_jpeg = read_from_jpeg(input_filename);
    if (input_jpeg.buffer == nullptr) {
        std::cerr << "Failed to read input JPEG image\n";
        return -1;
    }

    int width = input_jpeg.width;
    int height = input_jpeg.height;
    int num_channels = input_jpeg.num_channels;
    size_t buffer_size = width * height * num_channels;

    unsigned char* filteredImage = new unsigned char[buffer_size];

    // 分配GPU内存
    unsigned char* d_input_buffer;
    unsigned char* d_filtered_image;

    cudaMalloc((void**)&d_input_buffer, buffer_size);
    cudaMalloc((void**)&d_filtered_image, buffer_size);

    cudaMemset(d_filtered_image, 0, buffer_size);

    // 将数据从主机传输到设备
    cudaMemcpy(d_input_buffer, input_jpeg.buffer, buffer_size, cudaMemcpyHostToDevice);

    // 设置CUDA网格和块大小
    dim3 blockDim(32, 32);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                 (height + blockDim.y - 1) / blockDim.y);

    // 记录CUDA事件
    cudaEvent_t start, stop;
    float gpuDuration;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 执行双边滤波
    cudaEventRecord(start, 0); // GPU开始时间
    apply_bilateral_filter_kernel<<<gridDim, blockDim>>>(d_input_buffer, d_filtered_image, width, height, num_channels, SIGMA_S, SIGMA_R);
    cudaEventRecord(stop, 0); // GPU结束时间
    cudaEventSynchronize(stop);

    // 获取GPU计算时间
    cudaEventElapsedTime(&gpuDuration, start, stop);

    // 将数据从设备传输回主机
    cudaMemcpy(filteredImage, d_filtered_image, buffer_size, cudaMemcpyDeviceToHost);

    // 保存JPEG输出图像
    JPEGMeta output_jpeg{filteredImage, width, height, num_channels, input_jpeg.color_space};
    if (export_jpeg(output_jpeg, output_filename)) {
        std::cerr << "Failed to write output JPEG\n";
        return -1;
    }

    // 清理内存
    cudaFree(d_input_buffer);
    cudaFree(d_filtered_image);
    delete[] input_jpeg.buffer;
    delete[] filteredImage;

    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "GPU Execution Time: " << gpuDuration << " milliseconds" << std::endl;

    // 销毁CUDA事件
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
