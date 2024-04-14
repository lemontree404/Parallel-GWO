#include <bits/stdc++.h>
#include <vector>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <random>
#include "multi_pack_gwo.hpp"
#define CLIP 10
using namespace std;

int main(int argc, char *argv[]){

    int num_packs = stoi(argv[1]);
    int wolves_per_pack = stoi(argv[2]);
    int num_iterations = stoi(argv[3]);
    int k = 3;
    int a = 2;

    vector <vector <vector <float>>> wolves(num_packs, vector <vector <float>> (wolves_per_pack, vector <float> (2)));
    vector <vector <float>> fitness_scores(num_packs, vector <float> (wolves_per_pack));
    vector <vector <vector <float>>> guides(num_packs, vector <vector <float>>(k,vector <float> (2)));
    vector <vector <float>> omega(num_packs, vector <float> (2));
    vector <vector <float>> sigma(num_packs, vector <float> (2));

    for(int pack = 0; pack < num_packs; pack++){
        for(int wolf = 0; wolf < wolves_per_pack; wolf++){
            wolves[pack][wolf] = {get_rand_num(-10,10),get_rand_num(-10,10)};
            fitness_scores[pack][wolf] = fitness(wolves[pack][wolf]);
        }
    }

    for(int pack = 0; pack < num_packs; pack++){
        guides[pack] = get_guides(fitness_scores[pack],k,wolves[pack]);
    }

    double start = omp_get_wtime();

    #pragma omp parallel num_threads(num_packs * wolves_per_pack)
    {

        int thread_id = omp_get_thread_num();
        
        int pack = thread_id / wolves_per_pack;
        int wolf = thread_id % wolves_per_pack;

        for(int iter = 0; iter < num_iterations; iter++){

            fitness_scores[pack][wolf] = fitness(wolves[pack][wolf]);
            
            #pragma omp critical
            {
                guides[pack] = get_guides(fitness_scores[pack],k,wolves[pack]);
            }

            omega[pack] = get_omega(guides[pack],k);

            sigma[pack] = get_sigma(guides[pack],omega[pack],k); 

            #pragma omp barrier

            vector <vector <float>> X (k, vector <float> (2));
            vector <float> repulsion = {0,0};

            for(int i = 0; i < k; i++){

                vector <float> r1 = {get_rand_num(0,1),get_rand_num(0,1)};
                vector <float> r2 = {get_rand_num(0,1),get_rand_num(0,1)};

                vector <float> A = {2 * a * r1[0] - a, 2 * a * r1[1] - a};
                vector <float> C = {2 * r2[0], 2 * r2[1]};
                vector <float> D = {abs(C[0] * guides[pack][i][0] - wolves[pack][wolf][0]), abs(C[1] * guides[pack][i][1] - wolves[pack][wolf][1])};
                X[i] = {guides[pack][i][0] - A[0] * D[0], guides[pack][i][1] - A[1] * D[1]};
            }

            repulsion = get_repulsion(sigma, omega, num_packs, pack);

            wolves[pack][wolf] = {0,0};

            for(int i = 0; i < k; i++){
                wolves[pack][wolf][0] += X[i][0];
                wolves[pack][wolf][1] += X[i][1];
            }

            wolves[pack][wolf][0] -= sigma[pack][0] / repulsion[0];
            wolves[pack][wolf][1] -= sigma[pack][1] / repulsion[1];

            wolves[pack][wolf][0] /= 4;
            wolves[pack][wolf][1] /= 4;

            wolves[pack][wolf] = {clip(wolves[pack][wolf][0],-CLIP,CLIP),clip(wolves[pack][wolf][1],-CLIP,CLIP)};

            #pragma omp barrier            
        }
    }
    
    double end = omp_get_wtime();

    printf("Packs: %d\n", num_packs);
    printf("Wolves per Pack: %d\n", wolves_per_pack);
    printf("Iterations %d\n",num_iterations);
    printf("Time Taken: %f\n",end-start);
    
    FILE *file = fopen("wolves.txt", "w+");
    if(file == NULL) {
        printf("Error opening file!\n");
        return 1;
    }

    for(int pack = 0; pack < num_packs; pack++){
        for(int wolf = 0; wolf < wolves_per_pack; wolf++){
            fprintf(file,"%f %f\n",wolves[pack][wolf][0], wolves[pack][wolf][1]);
        }
    }

    fclose(file);

    for(int pack = 0; pack < num_packs; pack++){
        printf("\nPack %d:\n",pack);
        for(int guide = 0; guide < k; guide++){
            printf("Guide %d: %f %f\tFitness: %f\n", guide, wolves[pack][guide][0], wolves[pack][guide][1],fitness_scores[pack][guide]);
            fprintf(file,"%f %f\n",wolves[pack][guide][0], wolves[pack][guide][1]);
        }
    }

}