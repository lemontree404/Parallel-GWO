#include <bits/stdc++.h>
#include <vector>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <cmath>
#include <omp.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include "multi_pack_gwo_parallel_cuda.h"
# define PI	    3.14159265358979323846
# define CLIP   10
using namespace std;

int main(int argc, char *argv[]){
    
    cudaEvent_t prog,start,end;

    cudaEventCreate(&prog);
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(prog,0);
    
    int num_packs = stoi(argv[1]);
    int wolves_per_pack = stoi(argv[2]);
    int num_iterations = stoi(argv[3]);
    int k = 3;
    int a = 2;

    float *wolves = (float *) malloc(sizeof(float) * num_packs * wolves_per_pack * 2);

    float *fitness_scores = (float *) malloc(sizeof(float) * num_packs * wolves_per_pack);

    float *guides = (float *) malloc(sizeof(float) * num_packs * k * 2);

    float *omega = (float *) malloc(sizeof(float) * num_packs * 2);

    float *sigma = (float *) malloc(sizeof(float) * num_packs * 2);

    for(int pack = 0; pack < num_packs; pack++){
        for(int wolf = 0; wolf < wolves_per_pack; wolf++){
            wolves[pack * wolves_per_pack * 2 + wolf * 2] = get_rand_num(-10,10);
            wolves[pack * wolves_per_pack * 2 + wolf * 2 + 1] = get_rand_num(-10,10);
        }
    }

    float *d_wolves, *d_fitness_scores, *d_guides, *d_omega, *d_sigma;

    cudaMalloc(&d_wolves, sizeof(float) * num_packs * wolves_per_pack * 2);
    cudaMalloc(&d_fitness_scores, sizeof(float) * num_packs * wolves_per_pack);
    cudaMalloc(&d_guides,sizeof(float) * num_packs * k * 2);
    cudaMalloc(&d_omega,sizeof(float) * num_packs * 2);
    cudaMalloc(&d_sigma,sizeof(float) * num_packs * 2);

    cudaMemcpy(d_wolves,wolves,sizeof(float) * num_packs * wolves_per_pack * 2,cudaMemcpyHostToDevice);

    cudaEventRecord(start,0);

    for(int iter = 0; iter < num_iterations; iter++){

        get_fitness<<<1,3 * 10>>>(d_wolves, d_fitness_scores, num_packs, wolves_per_pack, k);
        get_guides_omega_sigma<<<1,3 * 10>>>(d_wolves, d_fitness_scores, d_guides, d_omega, d_sigma, num_packs, wolves_per_pack, k);
        update<<<1,3 * 10>>>(d_wolves,d_guides,d_omega,d_sigma,num_packs,wolves_per_pack,k,a,get_rand_num(0,100));

    }

    cudaMemcpy(guides,d_guides,sizeof(float) * num_packs * k * 2,cudaMemcpyDeviceToHost);

    cudaEventRecord(end,0);

    float prog_time, par_time;
`
    cudaEventSynchronize(end);

    cudaEventElapsedTime(&prog_time,prog,end);
    cudaEventElapsedTime(&par_time,start,end);

    printf("%d,%d,%d,%f,%f\n", num_packs, wolves_per_pack, num_iterations,prog_time,par_time);

    // for(int i = 0; i<1; i++){
    //     for(int j = 0; j < k; j++){
    //         printf("%f %f\n",guides[i * k * 2 + j],guides[i * k * 2 + j + 1]);
    //     }
    // }
}   