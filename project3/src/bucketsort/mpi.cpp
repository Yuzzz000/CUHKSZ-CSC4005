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
void bucketSort(std::vector<int> &vec, int num_buckets, int numtasks, int taskid, MPI_Status *status)
{
    /* Your codes here! */
    int n = vec.size();
    int max_val = *std::max_element(vec.begin(), vec.end());
    int min_val = *std::min_element(vec.begin(), vec.end());
    int range = max_val - min_val + 1;
    int bucket_range = range / num_buckets;

    std::vector<std::vector<int>> buckets(num_buckets);

    for (int i = 0; i < num_buckets; i++) {
        buckets[i].reserve(n / num_buckets + 1); // Reserve space to avoid reallocations
    }

    // Distribute elements into buckets based on their value
    for (int num : vec) {
        int index = (num - min_val) / bucket_range;
        if (index >= num_buckets) index = num_buckets - 1;  // Handle edge case for max value
        buckets[index].push_back(num);
    }

    // Each process sorts its portion of buckets
    std::vector<int> local_sorted;
    for (int i = taskid; i < num_buckets; i += numtasks) {
        insertionSort(buckets[i]);
        local_sorted.insert(local_sorted.end(), buckets[i].begin(), buckets[i].end());
    }

    // Gather all sorted sub-vectors at the master
    std::vector<int> all_sorted;
    std::vector<int> recvcounts(numtasks);
    std::vector<int> displacements(numtasks);

    int local_size = local_sorted.size();
    MPI_Gather(&local_size, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, MASTER, MPI_COMM_WORLD);

    if (taskid == MASTER) {
        int total_size = 0;
        for (int i = 0; i < numtasks; ++i) {
            displacements[i] = total_size;
            total_size += recvcounts[i];
        }
        all_sorted.resize(total_size);
    }

    MPI_Gatherv(local_sorted.data(), local_size, MPI_INT,
                all_sorted.data(), recvcounts.data(), displacements.data(), MPI_INT,
                MASTER, MPI_COMM_WORLD);

    if (taskid == MASTER) {
        vec = std::move(all_sorted);  // Place the sorted data back into the original vector
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
