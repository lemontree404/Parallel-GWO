

# Link

[Github](https://github.com/lemontree404/Parallel-GWO)

# Problem

​			Parallelization of the gray wolf algorithm as well as a modified grey wolf algorithm with multiple packs for increased exploration.

# Approach

​	Both single and multi-pack algorithms were implemented thrice, once serially, once using openmp and once using CUDA.

While implementing single pack gwo parallely, wolves were equally split between threads, to ensure maximum speedup. Similarly, for multiple packs, each pack was split equally among the threads.

The constructs used in this process were:

* Openmp
  * omp parallel num_threads
  * omp critical
  * omp barrier
  * omp_get_thread_num
  * omp_get_wtime
* Cuda
  * []

# Performance Metric

Two performance metrics were used for this project, number of iterations and number of wolves per pack

The average speed ups with regards to number of iterations can be seen below:





The average speed ups with regards to number of wolves per pack can be seen below: