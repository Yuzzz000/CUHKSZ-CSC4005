//
// Created by Yang Yufan on 2023/10/31.
// Email: yufanyang1@link.cuhk.edu.cn
//
// Modified by Liu Yuxuan on 2024/10/26
// Email: yuxuanliu1@link.cuhk.edu.cn
//
// Task #2: Parallel Quick Sort with K-Way Merge using MPI
//

#include <iostream>
#include <vector>
#include <queue>
#include <tuple>

#include <mpi.h>

#include "../utils.hpp"
#include <climits>  // for INT_MAX

#define MASTER 0

int partition(std::vector<int> &vec, int low, int high) {
    int pivot = vec[high];
    int i = low - 1;
    for (int j = low; j < high; j++) {
        if (vec[j] <= pivot) {
            i++;
            std::swap(vec[i], vec[j]);
        }
    }
    std::swap(vec[i + 1], vec[high]);
    return i + 1;
}

/**
 * TODO: Implement parallel quick sort with MPI
 */

void localQuickSort(std::vector<int>& vec, int low, int high) {
    if (low < high) {
        int pivotIndex = partition(vec, low, high);
        localQuickSort(vec, low, pivotIndex - 1);
        localQuickSort(vec, pivotIndex + 1, high);
    }
}
void quickSort(std::vector<int>& vec, int numtasks, int taskid, MPI_Status* status) {
    int global_size = vec.size();
    int local_size = (global_size + numtasks - 1) / numtasks; // ceil to handle non-divisible cases
    std::vector<int> local_vec(local_size);

    if (taskid == numtasks - 1) { // Adjust for last process
        local_size = global_size - (numtasks - 1) * local_size;
    }
    local_vec.resize(local_size);

    // Scatter the global vector to all processes
    MPI_Scatter(vec.data(), local_size, MPI_INT, local_vec.data(), local_size, MPI_INT, MASTER, MPI_COMM_WORLD);

    // Sort locally using quickSort
    localQuickSort(local_vec, 0, local_size - 1);

    // Prepare to gather at the master
    std::vector<int> sorted_vec(global_size);
    std::vector<int> recvcounts(numtasks, (global_size + numtasks - 1) / numtasks);
    recvcounts[numtasks - 1] = local_size; // Adjust last segment size if not divisible

    std::vector<int> displs(numtasks);
    int displacement = 0;
    for (int i = 0; i < numtasks; i++) {
        displs[i] = displacement;
        displacement += recvcounts[i];
    }

    // Gather all local vectors back to the master
    MPI_Gatherv(local_vec.data(), local_size, MPI_INT, sorted_vec.data(), recvcounts.data(), displs.data(), MPI_INT, MASTER, MPI_COMM_WORLD);

    // K-way merge using a min-heap on the master process
    if (taskid == MASTER) {
        std::priority_queue<std::tuple<int, int, int>, std::vector<std::tuple<int, int, int>>, std::greater<std::tuple<int, int, int>>> min_heap;
        for (int i = 0, offset = 0; i < numtasks; i++) {
            if (recvcounts[i] > 0) {
                min_heap.push(std::make_tuple(sorted_vec[offset], i, 0));
                offset += recvcounts[i];
            }
        }

        int current_index = 0;
        while (!min_heap.empty()) {
            auto [value, sub_index, elem_index] = min_heap.top();
            min_heap.pop();
            vec[current_index++] = value;
            if (elem_index + 1 < recvcounts[sub_index]) {
                min_heap.push(std::make_tuple(sorted_vec[displs[sub_index] + elem_index + 1], sub_index, elem_index + 1));
            }
        }
    }
}


int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 3) {
        throw std::invalid_argument(
            "Invalid argument, should be: ./executable dist_type vector_size\n"
            );
    }
    const DistType dist_type = str_2_dist_type(std::string(argv[1]));
    const int size = atoi(argv[2]);
    std::vector<int> vec = genRandomVec(size, dist_type); // use default seed
    std::vector<int> vec_clone = vec;

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

    auto start_time = std::chrono::high_resolution_clock::now();
    
    quickSort(vec, numtasks, taskid, &status);

    if (taskid == MASTER) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time);
        
        std::cout << "Quick Sort Complete!" << std::endl;
        std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
                << std::endl;
        
        checkSortResult(vec_clone, vec);
    }

    MPI_Finalize();
    return 0;
}
