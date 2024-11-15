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
void phase_1()
{
    // Sort local data
    std::sort(localData.begin(), localData.end());

    // Calculate sampling interval
    int sampleInterval = std::ceil(localData.size() / static_cast<float>(T));
    localRegularSamples.clear();
    for (int i = 0; i < T && (i * sampleInterval) < localData.size(); ++i)
    {
        localRegularSamples.push_back(localData[i * sampleInterval]);
    }
}

/**
 * TODO: Pivot Selection Phase
 * 1. Gather all the local samples from each process
 * 2. Select (T - 1) global pivots
 */
void phase_2()
{
    // Resize on all processes
    pivots.resize(T - 1);  

    if (rank == MASTER)
    {
        std::sort(regularSamples.begin(), regularSamples.end());
        for (int i = 1; i < T; i++)  // Select (T-1) pivots
        {
            pivots[i - 1] = regularSamples[i * T / (T - 1) - 1];
        }
    }

    // Debugging output to confirm pivots are ready
    //std::cout << "Rank " << rank << " ready to broadcast pivots, size: " << pivots.size() << std::endl;

    // Ensure MPI_Bcast has a valid buffer
    MPI_Bcast(pivots.data(), pivots.size(), MPI_INT, MASTER, MPI_COMM_WORLD);
}


/**
 * TODO: Split the data pieces and exchange them across processes
 */
void phase_3()
{
    // Create T partitions based on the pivots
    splitters = std::vector<int>(T + 1, 0);  // +1 to handle boundary
    int j = 0;
    for (int i = 0; i < localData.size(); ++i)
    {
        if (j < pivots.size() && localData[i] > pivots[j])
        {
            splitters[++j] = i;
        }
    }
    splitters[T] = localData.size();

    // Exchange data segments with other processes
    lengths.resize(T);
    obtainedKeys.clear();
    for (int i = 0; i < T; ++i)
    {
        int segmentSize = splitters[i + 1] - splitters[i];
        MPI_Sendrecv(&segmentSize, 1, MPI_INT, i, 0, &lengths[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    obtainedKeys.resize(T * dataCurrentProc);  // overestimate size
    for (int i = 0; i < T; ++i)
    {
        MPI_Sendrecv(&localData[splitters[i]], lengths[i], MPI_INT, i, 0,
                     &obtainedKeys[obtainedKeysSize], lengths[i], MPI_INT, i, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        obtainedKeysSize += lengths[i];
    }
}


/**
 * TODO: Merge local partitions
 * You can use k-way merge in Task #2 if you want
 */
void phase_4()
{
    mergedArray = std::vector<int>(obtainedKeys.begin(), obtainedKeys.begin() + obtainedKeysSize);
    std::sort(mergedArray.begin(), mergedArray.end());
}


/**
 * TODO: Merge all local arrays into SortedData in master process
 */
void phase_merge()
{
    // Compute displacements for each process
    std::vector<int> displacements = prefixSum(lengths);  // make sure prefixSum correctly handles boundary

    if (rank == MASTER)
    {
        SortedData.resize(SIZE);
        MPI_Gatherv(&mergedArray[0], mergedArray.size(), MPI_INT,
                    &SortedData[0], &lengths[0], &displacements[0], MPI_INT, MASTER, MPI_COMM_WORLD);
    }
}



/**
 * You can measure the time of each phase with this function
 */
void measureTime(void (*fun)(), const char *processorName, const char *title, int shouldLog)
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
