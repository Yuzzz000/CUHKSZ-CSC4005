#!/bin/bash
#SBATCH -o ./Project3-uniform-results.txt
#SBATCH -p Project
#SBATCH -J Project3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:1

DATA_SIZE=100000000
BUCKET_SIZE=1000000

# Create a directory to store perf data files
mkdir -p perf_data_uniform

# std::sort Sequential
## Uniformly distributed dataset
echo "std::sort Sequential for Uniform dataset (Optimized with -O2)"
perf stat -e cache-misses,cache-references,page-faults -o ./perf_data_uniform/std_sort_uniform_perf.txt -- srun -n 1 --cpus-per-task 1 ./build/src/std_sort uniform $DATA_SIZE
echo ""

# Task 1: Parallel Bucket Sort
## Uniformly distributed dataset
echo "Bucket Sort MPI for Uniformly dataset (Optimized with -O2)"
for num_processes in 1 4 16 32
do
  echo "Number of processes: $num_processes"
  perf stat -e cache-misses,cache-references,page-faults -o ./perf_data_uniform/bucket_sort_${num_processes}_uniform_perf.txt -- srun -n $num_processes --cpus-per-task 1 --mpi=pmi2 ./build/src/bucketsort/bucketsort_mpi uniform $DATA_SIZE $BUCKET_SIZE
done
echo ""

# Task 2: Parallel Quick Sort with K-Way Merge
## Uniformly distributed dataset
echo "Quick Sort MPI for Uniform dataset (Optimized with -O2)"
for num_processes in 1 4 16 32
do
  echo "Number of cores: $num_processes"
  perf stat -e cache-misses,cache-references,page-faults -o ./perf_data_uniform/quicksort_${num_processes}_uniform_perf.txt -- srun -n $num_processes --cpus-per-task 1 --mpi=pmi2 ./build/src/quicksort/quicksort_mpi uniform $DATA_SIZE
done
echo ""

# Task 3: PSRS
## Uniformly distributed dataset
echo "PSRS MPI for Uniform dataset (Optimized with -O2)"
for num_processes in 1 4 16 32
do
  echo "Number of cores: $num_processes"
  perf stat -e cache-misses,cache-references,page-faults -o ./perf_data_uniform/psrs_${num_processes}_uniform_perf.txt -- srun -n $num_processes --cpus-per-task 1 --mpi=pmi2 ./build/src/psrs/psrs_mpi uniform $DATA_SIZE
done
echo ""

# Task 4: Parallel Merge Sort
## Uniformly distributed dataset
echo "Merge Sort OpenMP for Uniform dataset (Optimized with -O2)"
for num_threads in 1 4 16 32
do
  echo "Number of threads: $num_threads"
  perf stat -e cache-misses,cache-references,page-faults -o ./perf_data_uniform/merge_sort_${num_threads}_uniform_perf.txt -- srun -n 1 --cpus-per-task $num_threads ./build/src/mergesort/mergesort_openmp uniform $num_threads $DATA_SIZE
done
echo ""
