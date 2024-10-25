#include <mpi.h>  // MPI Header
#include <omp.h>
#include <immintrin.h>
#include <stdexcept>
#include <chrono>
#include "matrix.hpp"

#define MASTER 0

Matrix matrix_multiply_mpi(const Matrix& matrix1, const Matrix& matrix2, int taskid, int numtasks, int thread_num) {
    if (matrix1.getCols() != matrix2.getRows()) {
        throw std::invalid_argument("Matrix dimensions are not compatible for multiplication.");
    }

    size_t M = matrix1.getRows(), K = matrix1.getCols(), N = matrix2.getCols();
    size_t rows_per_process = M / numtasks;
    size_t start_row = taskid * rows_per_process;
    size_t end_row = (taskid == numtasks - 1) ? M : start_row + rows_per_process;

    Matrix local_result(rows_per_process, N);

    omp_set_num_threads(thread_num);
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = start_row; i < end_row; i += 64) { // Loop blocks for better cache usage
        for (size_t j = 0; j < N; j += 64) {
            for (size_t k = 0; k < K; k += 64) {
                for (size_t ii = i; ii < std::min(i + 64, end_row); ++ii) {
                    for (size_t jj = j; jj < std::min(j + 64, N); jj += 8) { // SIMD vector width
                        __m256i sum_vec = _mm256_loadu_si256((__m256i*)&local_result[ii - start_row][jj]);
                        for (size_t kk = k; kk < std::min(k + 64, K); ++kk) {
                            __m256i a_vec = _mm256_set1_epi32(matrix1[ii][kk]);
                            __m256i b_vec = _mm256_loadu_si256((__m256i*)&matrix2[kk][jj]);
                            sum_vec = _mm256_add_epi32(sum_vec, _mm256_mullo_epi32(a_vec, b_vec));
                        }
                        _mm256_storeu_si256((__m256i*)&local_result[ii - start_row][jj], sum_vec);
                    }
                }
            }
        }
    }

    Matrix result(M, N);
    if (taskid == MASTER) {
        MPI_Gather(local_result[0], rows_per_process * N, MPI_INT,
                   result[0], rows_per_process * N, MPI_INT, MASTER, MPI_COMM_WORLD);
    } else {
        MPI_Gather(local_result[0], rows_per_process * N, MPI_INT,
                   nullptr, 0, MPI_INT, MASTER, MPI_COMM_WORLD);
    }

    return result;
}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "Invalid argument, should be: ./executable thread_num "
                  "/path/to/matrix1 /path/to/matrix2 /path/to/multiply_result\n";
        return 1;
    }

    MPI_Init(&argc, &argv);
    int numtasks, taskid;
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);

    int thread_num = atoi(argv[1]);
    Matrix matrix1 = Matrix::loadFromFile(argv[2]);
    Matrix matrix2 = Matrix::loadFromFile(argv[3]);
    std::string result_path = argv[4];

    auto start_time = std::chrono::high_resolution_clock::now();

    Matrix result = matrix_multiply_mpi(matrix1, matrix2, taskid, numtasks, thread_num);

    if (taskid == MASTER) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        result.saveToFile(result_path);
        std::cout << "Output file to: " << result_path << std::endl;
        std::cout << "Multiplication Complete! Execution Time: " << elapsed_time.count() << " milliseconds" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
