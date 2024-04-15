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
# define PI	    3.14159265358979323846
# define CLIP   10
using namespace std;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

float get_rand_num(float low, float high)
{
    random_device rd;  // Will be used to obtain a seed for the random number engine
    mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    uniform_real_distribution<> dis(low, high);
    return (float)dis(gen);
}

__device__ void argsort(float *arr, int *indices, int n) {
    int i, j, temp;
    float tempFloat;
    for (i = 0; i < n-1; i++) {
        for (j = 0; j < n-i-1; j++) {
            if (arr[j] > arr[j+1]) {
                // Swap the elements in the float array
                tempFloat = arr[j];
                arr[j] = arr[j+1];
                arr[j+1] = tempFloat;
                
                // Swap the corresponding indices
                temp = indices[j];
                indices[j] = indices[j+1];
                indices[j+1] = temp;
            }
        }
    }
}


__device__ void get_guides(float *wolves, float *fitness_scores, float *guides, int pack, int wolves_per_pack, int k){
    
    float *pack_fitness = (float *) malloc(sizeof(float) * wolves_per_pack);
    memcpy(pack_fitness,fitness_scores + pack * wolves_per_pack, sizeof(float) * wolves_per_pack);

    int *indices  = (int *) malloc(sizeof(float) * wolves_per_pack);
    for(int i = 0; i < wolves_per_pack; i++){
        indices[i] = i;
    }

    argsort(fitness_scores, indices, wolves_per_pack);

    for(int i = 0; i < k; i++){
        guides[pack * k * 2 + i * 2] = wolves[pack * k * 2 + indices[i] * 2];
        guides[pack * k * 2 + i * 2 + 1] = wolves[pack * k * 2 + indices[i] * 2 + 1];
    }
}

__device__ void get_omega(float *omega, float *guides, int pack, int k){
    
    float x,y;
    x = 0; y = 0;

    for(int i = 0; i < k; i++){
        x += guides[pack * k * 2 + i * 2];
        y += guides[pack * k * 2 + i * 2 + 1];
    }

    omega[pack * 2] = x/k;
    omega[pack * 2 + 1] = y/k;
}

__device__ void get_sigma(float *sigma, float *omega, float *guides, int pack, int k){

    float x,y;
    x = 0; y = 0;

    for(int i = 0; i < k; i++){
        x += pow(guides[pack * k * 2 + i * 2] - omega[pack * 2],2);
        y += pow(guides[pack * k * 2 + i * 2 + 1] - omega[pack * 2 + 1],2);
    }

    x /= k;
    y /= k;

    sigma[pack * 2] = sqrt(x);
    sigma[pack * 2 + 1] = sqrt(y);
}

__device__ void get_repulsion(float *repulsion, float *sigma, float *omega, int num_packs, int pack){
    
    float x,y;
    x = 0; y = 0;
    float norm[] = {0,0};

    for(int i=0; i < num_packs; i++){
        norm[0] += sigma[i * 2];
        norm[1] += sigma[i * 2 + 1];
    }

    for(int i = 0; i < num_packs; i++){
        if(i != pack){
            x += omega[pack * 2] / (sigma[pack * 2] / norm[0]);
            y += omega[pack * 2 + 1] / (sigma[pack * 2 + 1] / norm[1]);
        }
    }

    repulsion[0] = x;
    repulsion[1] = y;
}

__device__ float clip(float num,float lower, float upper){
    return max(lower, min(num, upper));
}

__global__ void get_fitness(float *wolves, float *fitness_scores, int num_packs, int wolves_per_pack, int k){

    int blocknum = blockIdx.z * (gridDim.x * gridDim.y) + blockIdx.y * (gridDim.x) + blockIdx.x;
    int threadnum = threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * (blockDim.x) + threadIdx.x;
    
    int id = blocknum * (blockDim.x * blockDim.y * blockDim.z) + threadnum;

    if(id < num_packs * wolves_per_pack){
        
        int pack = id / wolves_per_pack;
        int wolf = id % wolves_per_pack;

        float x,y;

        x = wolves[pack * wolves_per_pack * 2 + wolf * 2];
        y = wolves[pack * wolves_per_pack * 2 + wolf * 2 + 1];

        fitness_scores[id] = -0.0001 * pow(abs(sin(x) * sin(y) * exp(abs(100 - (sqrt(x*x + y*y)/PI)))) + 1, 0.1);

    }
}

__global__ void get_guides_omega_sigma(float *wolves, float *fitness_scores, float *guides, float *omega, float *sigma, int num_packs, int wolves_per_pack,int k){

    int blocknum = blockIdx.z * (gridDim.x * gridDim.y) + blockIdx.y * (gridDim.x) + blockIdx.x;
    int threadnum = threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * (blockDim.x) + threadIdx.x;
    
    int id = blocknum * (blockDim.x * blockDim.y * blockDim.z) + threadnum;

    if(id < num_packs){
        get_guides(wolves, fitness_scores, guides, id, wolves_per_pack, k);
        get_omega(omega, guides, id, k);
        get_sigma(sigma, omega, guides, id, k);
    }
}

__global__ void update(float *wolves, float *guides, float *omega, float *sigma, int num_packs, int wolves_per_pack, int k, int a, int seed){
    int blocknum = blockIdx.z * (gridDim.x * gridDim.y) + blockIdx.y * (gridDim.x) + blockIdx.x;
    int threadnum = threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * (blockDim.x) + threadIdx.x;
    
    int id = blocknum * (blockDim.x * blockDim.y * blockDim.z) + threadnum;

    curandState state;

    if(id < num_packs * wolves_per_pack){

        int pack = id / wolves_per_pack;
        int wolf = id % wolves_per_pack;



        float *r1 = (float *) malloc(sizeof(float) * 2);
        float *r2 = (float *) malloc(sizeof(float) * 2);
        
        float *A = (float *) malloc(sizeof(float) * 2);
        float *C = (float *) malloc(sizeof(float) * 2);
        float *D = (float *) malloc(sizeof(float) * 2);
        float *X = (float *) malloc(sizeof(float) * k * 2);

        float *repulsion = (float *) malloc(sizeof(float) * 2);

        get_repulsion(repulsion, omega, sigma,num_packs, pack);

        curand_init(seed, id, 0, &state);
        for(int i = 0; i < k; i++){
            r1[0] = curand_uniform(&state); r1[1] = curand_uniform(&state);
            r2[0] = curand_uniform(&state); r2[1] = curand_uniform(&state);

            A[0] = 2 * a * r1[0] - a; A[1] = 2 * a * r1[1] - a;
            C[0] = 2 * r2[0]; C[1] = 2 * r2[1];
            D[0] = abs(C[0] * guides[pack * k * 2 + i * 2] - wolves[pack * wolves_per_pack * 2 + wolf * 2]);
            D[1] = abs(C[1] * guides[pack * k * 2 + i * 2 + 1] - wolves[pack * wolves_per_pack * 2 + wolf * 2 + 1]);
            X[i * 2] = guides[pack * k * 2 + i * 2] - A[0] * D[0];
            X[i * 2 + 1] = guides[pack * k * 2 + i * 2 + 1] - A[1] * D[1];
        }

        for(int i = 0; i < k; i++){
            wolves[pack * wolves_per_pack * 2 + wolf * 2] += X[i * 2];
            wolves[pack * wolves_per_pack * 2 + wolf * 2 + 1] += X[i * 2 + 1];
        }

        wolves[pack * wolves_per_pack * 2 + wolf * 2] -= sigma[pack * 2] / repulsion[0];
        wolves[pack * wolves_per_pack * 2 + wolf * 2 + 1] -= sigma[pack * 2 + 1] / repulsion[1];
        
        wolves[pack * wolves_per_pack * 2 + wolf * 2] /= 4;
        wolves[pack * wolves_per_pack * 2 + wolf * 2 + 1] /= 4;
        
        wolves[pack * wolves_per_pack * 2 + wolf * 2]  = clip(wolves[pack * wolves_per_pack * 2 + wolf * 2],-CLIP,CLIP);
        wolves[pack * wolves_per_pack * 2 + wolf * 2 + 1] =     clip(wolves[pack * wolves_per_pack * 2 + wolf * 2 + 1],-CLIP,CLIP);

    }
}