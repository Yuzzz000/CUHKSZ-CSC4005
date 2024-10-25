//
// Created by Yang Yufan on 2023/10/07.
// Email: yufanyang1@link.cuhk.edu.cn
//
// SIMD + Reordering Matrix Multiplication
//

#include <immintrin.h>
#include <stdexcept>
#include <chrono>
#include "matrix.hpp"

Matrix matrix_multiply_simd(const Matrix& matrix1, const Matrix& matrix2) {
    if (matrix1.getCols() != matrix2.getRows()) {
        throw std::invalid_argument(
            "Matrix dimensions are not compatible for multiplication.");
    }

    size_t M = matrix1.getRows(), K = matrix1.getCols(), N = matrix2.getCols();

    Matrix result(M, N);

    // Your Code Here!
    // Optimizing Matrix Multiplication 
    // In addition to Memory Locality and Cache Missing,
    // Further Applying SIMD

    const size_t blockSize = 64; 

    for (size_t i = 0; i < M; i += blockSize) {
        for (size_t k = 0; k < K; k += blockSize) {
            for (size_t j = 0; j < N; j += blockSize) {
                for (size_t ii = i; ii < std::min(i + blockSize, M); ii++) {
                    for (size_t kk = k; kk < std::min(k + blockSize, K); ++kk) {
                        int a_elem = matrix1[ii][kk];
                        __m256i a_vec = _mm256_set1_epi32(a_elem);
                        for (size_t jj = j; jj < std::min(j + blockSize, N); jj += 8) {
                            __m256i b_vec = _mm256_loadu_si256((__m256i*)&matrix2[kk][jj]);
                            __m256i r_vec = _mm256_loadu_si256((__m256i*)&result[ii][jj]);
                            __m256i res_vec = _mm256_add_epi32(_mm256_mullo_epi32(a_vec, b_vec), r_vec);
                            _mm256_storeu_si256((__m256i*)&result[ii][jj], res_vec);
                        }
                    }
                }
            }
        }
    }


    return result;
}

int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 4) {
        throw std::invalid_argument(
            "Invalid argument, should be: ./executable "
            "/path/to/matrix1 /path/to/matrix2 /path/to/multiply_result\n");
    }

    const std::string matrix1_path = argv[1];

    const std::string matrix2_path = argv[2];

    const std::string result_path = argv[3];

    Matrix matrix1 = Matrix::loadFromFile(matrix1_path);

    Matrix matrix2 = Matrix::loadFromFile(matrix2_path);

    auto start_time = std::chrono::high_resolution_clock::now();

    Matrix result = matrix_multiply_simd(matrix1, matrix2);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

    result.saveToFile(result_path);

    std::cout << "Output file to: " << result_path << std::endl;

    std::cout << "Multiplication Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
              << std::endl;

    return 0;
}