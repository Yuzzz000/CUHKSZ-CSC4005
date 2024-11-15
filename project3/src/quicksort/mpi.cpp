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


void quickSortLocal(std::vector<int> &vec, int low, int high) {
    if (low < high) {
        int pivotIndex = partition(vec, low, high);
        quickSortLocal(vec, low, pivotIndex - 1);
        quickSortLocal(vec, pivotIndex + 1, high);
    }
}

void quickSort(std::vector<int>& vec, int numtasks, int taskid, MPI_Status* status) {
    int size = vec.size();
    std::vector<int> cuts = createCuts(0, size, numtasks);
    
    // Calculate local size for each process
    int local_size = (taskid == numtasks - 1) ? 
        cuts[taskid + 1] - cuts[taskid] : 
        cuts[taskid + 1] - cuts[taskid];
    
    // Allocate local buffer
    std::vector<int> local_data(local_size);
    
    // Master process distributes data
    if (taskid == MASTER) {
        // Master keeps its own portion
        for (int i = 0; i < local_size; i++) {
            local_data[i] = vec[i];
        }
        
        // Send portions to other processes
        for (int i = 1; i < numtasks; i++) {
            int start = cuts[i];
            int count = cuts[i + 1] - cuts[i];
            MPI_Send(&vec[start], count, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
    } else {
        // Other processes receive their portion
        MPI_Recv(local_data.data(), local_size, MPI_INT, 
                 MASTER, 0, MPI_COMM_WORLD, status);
    }
    
    // Each process performs local quick sort
    if (local_size > 1) {
        quickSortLocal(local_data, 0, local_size - 1);
    }
    
    // Gather results back to master
    if (taskid == MASTER) {
        // Master keeps its sorted portion
        for (int i = 0; i < local_size; i++) {
            vec[i] = local_data[i];
        }
        
        // Receive sorted portions from other processes
        for (int i = 1; i < numtasks; i++) {
            int start = cuts[i];
            int count = cuts[i + 1] - cuts[i];
            MPI_Recv(&vec[start], count, MPI_INT, i, 1, MPI_COMM_WORLD, status);
        }
        
        // Perform k-way merge using min heap
        std::vector<int> merged(size);
        
        // Create vectors to track positions in each sorted portion
        std::vector<int> positions(numtasks);
        for (int i = 0; i < numtasks; i++) {
            positions[i] = cuts[i];
        }
        
        // Merge
        int merged_idx = 0;
        while (merged_idx < size) {
            int min_val = INT_MAX;
            int min_portion = -1;
            
            // Find minimum value among current positions
            for (int i = 0; i < numtasks; i++) {
                if (positions[i] < cuts[i + 1]) {
                    if (vec[positions[i]] < min_val) {
                        min_val = vec[positions[i]];
                        min_portion = i;
                    }
                }
            }
            
            merged[merged_idx++] = min_val;
            positions[min_portion]++;
        }
        
        // Copy merged result back to original vector
        vec = merged;
    } else {
        // Other processes send their sorted portion to master
        MPI_Send(local_data.data(), local_size, MPI_INT, 
                 MASTER, 1, MPI_COMM_WORLD);
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
