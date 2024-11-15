//
// Created by Liu Yuxuan on 2024/10/26
// Email: yuxuanliu1@link.cuhk.edu.cn
//
// Task #3: Parallel Sorting with Regular Sampling using MPI
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

#include <mpi.h>

#include "../utils.hpp"

#define MASTER 0

/**
 * The following are variables for each process
 * Feel free to delete them or create new ones 
 */
std::vector<int> DATA;				  // Input dataset for sorting
std::vector<int> SortedData;		  // Output sorted data
std::vector<int> localData;			  // Data that each process gets
std::vector<int> localRegularSamples; // Regular samples of each process
std::vector<int> regularSamples;	  // Global regular samples in master
std::vector<int> pivots;			  // Pivots for load balancing
std::vector<int> splitters;			  // splitting indices
std::vector<int> mergedArray;		  // locally merged array
std::vector<int> lengths;			  // lengths of splitted array pieces
std::vector<int> obtainedKeys;		  // obtained keys from other processors

DistType DIST_TYPE;		  // Distribution type of the input dataset
int obtainedKeysSize = 0; // data size obtained from other processes
int dataCurrentProc;	  // Data Size the Current Process Gets
int T;					  // Total number of processes for sorting
int SIZE;				  // Size of the dataset to sort
int rank;				  // Rank of the current process

// data distribution phase
void phase_0()
{
	// regular data size that a processor will get
	int dataPerProc = std::ceil(SIZE / T);
	// the actual data size that the processor will get
	dataCurrentProc = (rank == T - 1) ? SIZE - (T - 1) * dataPerProc : dataPerProc;
	// allocate sufficient memory for the local array
	localData = std::vector<int>(dataCurrentProc, 0);
	std::vector<int> lenEachProc(T, 0); // Array size for each process
	std::vector<int> displacements(T);	// Displacement index for each proc
	if (rank == MASTER)
	{
		DATA = genRandomVec(SIZE, DIST_TYPE); // use default seed
		for (int index = 0; index < T; index++)
		{
			lenEachProc[index] = (index == T - 1) ? SIZE - (T - 1) * dataPerProc : dataPerProc;
		}
		displacements = prefixSum(lenEachProc);
	}
	// Scatter dataset to each processor
	MPI_Scatterv(&DATA[0], &lenEachProc[0], &displacements[0], MPI_INT, &localData[0], dataCurrentProc, MPI_INT, MASTER, MPI_COMM_WORLD);
}

/**
 * TODO: local sorting and regular sampling phase
 * 1. You need to sort the local partition by any algorithm you want
 * 2. You need to pick T local regular samples
 */
void phase_1() {
    // Sort local data using quicksort
    std::sort(localData.begin(), localData.end());
    
    // Calculate regular sample size and interval
    int w = dataCurrentProc / T;
    localRegularSamples.resize(T);
    
    // Select regular samples
    for (int i = 0; i < T; i++) {
        int index = i * w;
        if (index < dataCurrentProc) {
            localRegularSamples[i] = localData[index];
        } else {
            // If we run out of elements, use the last element
            localRegularSamples[i] = localData[dataCurrentProc - 1];
        }
    }
}

void phase_2() {
    if (rank == MASTER) {
        regularSamples.resize(T * T);
    }
    
    // Gather all samples to master process
    MPI_Gather(localRegularSamples.data(), T, MPI_INT,
               regularSamples.data(), T, MPI_INT,
               MASTER, MPI_COMM_WORLD);
    
    if (rank == MASTER) {
        // Sort all samples
        std::sort(regularSamples.begin(), regularSamples.end());
        
        // Select T-1 pivots
        pivots.resize(T-1);
        for (int i = 0; i < T-1; i++) {
            pivots[i] = regularSamples[(i+1)*T];
        }
    }
    
    // Broadcast pivots to all processes
    if (rank != MASTER) {
        pivots.resize(T-1);
    }
    MPI_Bcast(pivots.data(), T-1, MPI_INT, MASTER, MPI_COMM_WORLD);
}

void phase_3() {
    // Find splitting points in local sorted data
    splitters.resize(T+1);
    splitters[0] = 0;
    splitters[T] = dataCurrentProc;
    
    for (int i = 1; i < T; i++) {
        splitters[i] = std::lower_bound(localData.begin(), localData.end(), pivots[i-1]) - localData.begin();
    }
    
    // Calculate lengths of partitions
    lengths.resize(T);
    for (int i = 0; i < T; i++) {
        lengths[i] = splitters[i+1] - splitters[i];
    }
    
    // Allocate receive buffer
    std::vector<int> recvLengths(T);
    MPI_Alltoall(lengths.data(), 1, MPI_INT,
                 recvLengths.data(), 1, MPI_INT,
                 MPI_COMM_WORLD);
    
    // Calculate total size of data to receive
    obtainedKeysSize = 0;
    for (int i = 0; i < T; i++) {
        obtainedKeysSize += recvLengths[i];
    }
    obtainedKeys.resize(obtainedKeysSize);
    
    // Calculate displacements for MPI_Alltoallv
    std::vector<int> sendDispl(T), recvDispl(T);
    sendDispl[0] = 0;
    recvDispl[0] = 0;
    for (int i = 1; i < T; i++) {
        sendDispl[i] = sendDispl[i-1] + lengths[i-1];
        recvDispl[i] = recvDispl[i-1] + recvLengths[i-1];
    }
    
    // Exchange data
    MPI_Alltoallv(localData.data(), lengths.data(), sendDispl.data(), MPI_INT,
                  obtainedKeys.data(), recvLengths.data(), recvDispl.data(), MPI_INT,
                  MPI_COMM_WORLD);
}

void phase_4() {
    // Merge the obtained keys
    mergedArray = obtainedKeys;
    std::sort(mergedArray.begin(), mergedArray.end());
}

void phase_merge() {
    std::vector<int> recvCounts(T);
    MPI_Gather(&obtainedKeysSize, 1, MPI_INT,
               recvCounts.data(), 1, MPI_INT,
               MASTER, MPI_COMM_WORLD);
               
    if (rank == MASTER) {
        SortedData.resize(SIZE);
        std::vector<int> displacements(T, 0);
        for (int i = 1; i < T; i++) {
            displacements[i] = displacements[i-1] + recvCounts[i-1];
        }
        
        MPI_Gatherv(mergedArray.data(), obtainedKeysSize, MPI_INT,
                    SortedData.data(), recvCounts.data(), displacements.data(), MPI_INT,
                    MASTER, MPI_COMM_WORLD);
    } else {
        MPI_Gatherv(mergedArray.data(), obtainedKeysSize, MPI_INT,
                    nullptr, nullptr, nullptr, MPI_INT,
                    MASTER, MPI_COMM_WORLD);
    }
}

/**
 * You can measure the time of each phase with this function
 */
void measureTime(void (*fun)(), char *processorName, char *title, int shouldLog)
{
	if (shouldLog)
	{
		auto start_time = std::chrono::high_resolution_clock::now();
		fun();
		auto end_time = std::chrono::high_resolution_clock::now();
		auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
			end_time - start_time);
		printf("[%s:%d] %s took %ld ms\n", processorName, rank, title, elapsed_time.count());
	}
	else
	{
		fun();
	}
}

int main(int argc, char *argv[])
{
	// Verify input argument format
	if (argc != 3)
	{
		throw std::invalid_argument(
			"Invalid argument, should be: ./executable dist_type vector_size\n");
	}
	DIST_TYPE = str_2_dist_type(std::string(argv[1]));
	SIZE = atoi(argv[2]); // data size to sort

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &T);	  // how many processors are available
	MPI_Comm_rank(MPI_COMM_WORLD, &rank); // what's my rank?
	char processor_name[MPI_MAX_PROCESSOR_NAME];
	int name_len;
	MPI_Get_processor_name(processor_name, &name_len); // What's processor name

	// Phase 0: Data distribution
	measureTime(phase_0, processor_name, "Phase 0", rank == 0);
	MPI_Barrier(MPI_COMM_WORLD);

	auto start_time = std::chrono::high_resolution_clock::now();

	// PHASE 1
    // For now, measureTime function does not print the time consumption for each phase, change the last param to 1 if you want to print it out
	measureTime(phase_1, processor_name, "Phase 1", 0);
	// PHASE 2
	measureTime(phase_2, processor_name, "Phase 2", 0);
	// PHASE 3
	measureTime(phase_3, processor_name, "Phase 3", 0);
	// PHASE 4
	measureTime(phase_4, processor_name, "Phase 4", 0);

	MPI_Barrier(MPI_COMM_WORLD);

	// PHASE Merge
	measureTime(phase_merge, processor_name, "Phase Merge", rank == 0);

	if (rank == MASTER)
	{
		auto end_time = std::chrono::high_resolution_clock::now();
		auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
			end_time - start_time);
		std::cout << "Sorting Complete!" << std::endl;
		std::cout << "Execution Time: " << elapsed_time.count() << " 		milliseconds" << std::endl;
		checkSortResult(DATA, SortedData); // check if sorted
	}

	MPI_Finalize();
	return 0;
}