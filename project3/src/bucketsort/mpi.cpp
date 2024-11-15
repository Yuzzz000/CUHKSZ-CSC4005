//
// Created by Yang Yufan on 2023/10/31.
// Email: yufanyang1@link.cuhk.edu.cn
//
// Modified by Liu Yuxuan on 2024/10/26
// Email: yuxuanliu1@link.cuhk.edu.cn
//
// Task #1: Parallel Bucket Sort with MPI
//

#include <iostream>
#include <vector>
#include <mpi.h>
#include "../utils.hpp"

#define MASTER 0

void insertionSort(std::vector<int> &bucket)
{
    /* You may print out the data size in each bucket here to see how severe the load imbalance is */
    for (int i = 1; i < bucket.size(); ++i)
    {
        int key = bucket[i];
        int j = i - 1;

        while (j >= 0 && bucket[j] > key)
        {
            bucket[j + 1] = bucket[j];
            j--;
        }

        bucket[j + 1] = key;
    }
}

/**
 * TODO: Parallel Bucket Sort with MPI
 * @param vec: input vector for sorting
 * @param num_buckets: number of buckets
 * @param numtasks: number of processes for sorting
 * @param taskid: the rank of the current process
 * @param status: MPI_Status for message passing
 */
void bucketSort(std::vector<int>& vec, int num_buckets, int numtasks, int taskid, MPI_Status* status) {
    int vec_size = vec.size();
    // Broadcast vector size to all processes
    MPI_Bcast(&vec_size, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    
    // Distribute the vector evenly among processes
    int local_size = vec_size / numtasks;
    int remainder = vec_size % numtasks;
    int start_idx = taskid * local_size + std::min(taskid, remainder);
    if (taskid < remainder) local_size++;
    
    std::vector<int> local_vec(local_size);
    
    // Scatter data to all processes
    if (taskid == MASTER) {
        // Master sends portions to other processes
        for (int i = 0; i < numtasks; i++) {
            int proc_size = vec_size / numtasks + (i < remainder ? 1 : 0);
            int proc_start = i * (vec_size / numtasks) + std::min(i, remainder);
            
            if (i == MASTER) {
                // Copy master's portion
                std::copy(vec.begin() + proc_start, 
                         vec.begin() + proc_start + proc_size, 
                         local_vec.begin());
            } else {
                MPI_Send(&vec[proc_start], proc_size, MPI_INT, i, 0, MPI_COMM_WORLD);
            }
        }
    } else {
        // Other processes receive their portion
        MPI_Recv(local_vec.data(), local_size, MPI_INT, MASTER, 0, MPI_COMM_WORLD, status);
    }

    // Find global max and min
    int local_max = *std::max_element(local_vec.begin(), local_vec.end());
    int local_min = *std::min_element(local_vec.begin(), local_vec.end());
    int global_max, global_min;
    
    MPI_Allreduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&local_min, &global_min, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    int range = global_max - global_min + 1;
    int small_bucket_size = range / num_buckets;
    int large_bucket_size = small_bucket_size + 1;
    int large_bucket_num = range - small_bucket_size * num_buckets;
    int boundary = global_min + large_bucket_num * large_bucket_size;

    // Initialize local buckets
    std::vector<std::vector<int>> local_buckets(num_buckets);

    // Distribute elements to local buckets
    for (int num : local_vec) {
        int index;
        if (num < boundary) {
            index = (num - global_min) / large_bucket_size;
        } else {
            index = large_bucket_num + (num - boundary) / small_bucket_size;
        }
        if (index >= num_buckets) {
            index = num_buckets - 1;
        }
        local_buckets[index].push_back(num);
    }

    // Now we need to merge corresponding buckets across processes
    for (int bucket_idx = 0; bucket_idx < num_buckets; bucket_idx++) {
        int target_proc = bucket_idx % numtasks;
        
        if (taskid == target_proc) {
            // This process is responsible for this bucket
            std::vector<int> merged_bucket = local_buckets[bucket_idx];
            
            // Receive from other processes
            for (int src = 0; src < numtasks; src++) {
                if (src != taskid) {
                    int incoming_size;
                    MPI_Recv(&incoming_size, 1, MPI_INT, src, bucket_idx, MPI_COMM_WORLD, status);
                    if (incoming_size > 0) {
                        std::vector<int> temp(incoming_size);
                        MPI_Recv(temp.data(), incoming_size, MPI_INT, src, bucket_idx, MPI_COMM_WORLD, status);
                        merged_bucket.insert(merged_bucket.end(), temp.begin(), temp.end());
                    }
                }
            }
            
            // Sort the merged bucket
            insertionSort(merged_bucket);
            local_buckets[bucket_idx] = merged_bucket;
        } else {
            // Send this bucket to the target process
            int size = local_buckets[bucket_idx].size();
            MPI_Send(&size, 1, MPI_INT, target_proc, bucket_idx, MPI_COMM_WORLD);
            if (size > 0) {
                MPI_Send(local_buckets[bucket_idx].data(), size, MPI_INT, target_proc, bucket_idx, MPI_COMM_WORLD);
            }
            local_buckets[bucket_idx].clear();  // Free memory
        }
    }

    // Gather bucket sizes first
    std::vector<int> bucket_sizes(num_buckets, 0);
    for (int i = 0; i < num_buckets; i++) {
        int target_proc = i % numtasks;
        int local_size = (taskid == target_proc) ? local_buckets[i].size() : 0;
        MPI_Allreduce(&local_size, &bucket_sizes[i], 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    }

    // Calculate offsets for final gathering
    std::vector<int> offsets(num_buckets);
    offsets[0] = 0;
    for (int i = 1; i < num_buckets; i++) {
        offsets[i] = offsets[i-1] + bucket_sizes[i-1];
    }

    // Gather all sorted buckets to master
    if (taskid == MASTER) {
        for (int i = 0; i < num_buckets; i++) {
            int target_proc = i % numtasks;
            if (target_proc == MASTER) {
                // Copy local bucket directly
                std::copy(local_buckets[i].begin(), local_buckets[i].end(), 
                         vec.begin() + offsets[i]);
            } else {
                // Receive from other process
                if (bucket_sizes[i] > 0) {
                    MPI_Recv(&vec[offsets[i]], bucket_sizes[i], MPI_INT, 
                            target_proc, i + num_buckets, MPI_COMM_WORLD, status);
                }
            }
        }
    } else {
        // Other processes send their sorted buckets
        for (int i = 0; i < num_buckets; i++) {
            if (i % numtasks == taskid && bucket_sizes[i] > 0) {
                MPI_Send(local_buckets[i].data(), bucket_sizes[i], MPI_INT, 
                        MASTER, i + num_buckets, MPI_COMM_WORLD);
            }
        }
    }
}

int main(int argc, char **argv)
{
    // Verify input argument format
    if (argc != 4)
    {
        throw std::invalid_argument(
            "Invalid argument, should be: ./executable dist_type vector_size bucket_num\n");
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

    const DistType dist_type = str_2_dist_type(std::string(argv[1]));
    const int size = atoi(argv[2]);
    const int bucket_num = atoi(argv[3]);
    std::vector<int> vec = genRandomVec(size, dist_type); // use default seed
    std::vector<int> vec_clone = vec;

    auto start_time = std::chrono::high_resolution_clock::now();

    bucketSort(vec, bucket_num, numtasks, taskid, &status);

    if (taskid == MASTER)
    {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time);

        std::cout << "Bucket Sort Complete!" << std::endl;
        std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
                  << std::endl;

        checkSortResult(vec_clone, vec);
    }

    MPI_Finalize();
    return 0;
}
