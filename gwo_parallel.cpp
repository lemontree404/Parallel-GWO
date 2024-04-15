#include <bits/stdc++.h>
#include <vector>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <cmath>
#include <omp.h>
#define CLIP 20
using namespace std;

float get_rand_num(float low, float high)
{
    random_device rd;  // Will be used to obtain a seed for the random number engine
    mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    uniform_real_distribution<> dis(low, high);
    return (float)dis(gen);
}

float fitness(vector<float> & wolf){
    
    float x = wolf[0];
    float y = wolf[1];
    // return 0.2 * (x * x + y * y) + 0.8 * pow(sin(2 * x),2) - 0.7 * cos(3 * y) - 1;
    return -0.0001 * pow(abs(sin(x) * sin(y) * exp(abs(100 - (sqrt(x*x + y*y)/M_PI)))) + 1, 0.1);
}

vector<int> get_guide_indices(vector <float> &fitness_scores, int k)
{
    vector<pair<float, int>> n_arr;
    for(int i=0; i<fitness_scores.size(); i++) n_arr.push_back(make_pair(fitness_scores[i], i));

    sort(n_arr.begin(), n_arr.end());

    vector <int> ans;
    for(int i=0; i < k; i++) ans.push_back(n_arr[i].second);

    return ans;
}

float clip(float n, float lower, float upper) {
  return max(lower, min(n, upper));
}

int main(int argc, char *argv[]){

    double prog = omp_get_wtime();

    int population = stoi(argv[1]);
    int nIterations = stoi(argv[2]);

    vector <vector<float>> wolves(population, vector<float> (2));

    vector<float> fitness_scores(population);
    
    for(int i = 0; i < population; i++){
        
        wolves[i] = {get_rand_num(-10,10),get_rand_num(-10,10)};
        fitness_scores[i] = fitness(wolves[i]);
    }

    vector <int> index = get_guide_indices(fitness_scores, 3);

    vector< vector <float>> guides;

    for(auto i: index) guides.push_back(wolves[i]);

    double start = omp_get_wtime();

    #pragma omp parallel num_threads(population)
    {

        int id = omp_get_thread_num();

        for(int iter = 0; iter < nIterations; iter++){

            fitness_scores[id] = fitness(wolves[id]);

            if(id == 0){
                index = get_guide_indices(fitness_scores, 3);

                guides.clear();

                for(auto i: index) guides.push_back(wolves[i]);
            }

            #pragma omp barrier

            vector<vector<float>> r1(3, vector <float> (2));
            vector<vector<float>> r2(3, vector <float> (2));

            vector<vector <float>> A(3, vector <float> (2));
            vector<vector <float>> C(3, vector <float> (2));
            vector<vector <float>> D(3, vector <float> (2));
            vector<vector <float>> X(3, vector <float> (2));

            float a = 2;

            for(int i = 0; i < 3; i++){
                r1[i] = {get_rand_num(0,1),get_rand_num(0,1)};
                r2[i] = {get_rand_num(0,1),get_rand_num(0,1)};

                A[i] = {2  * a * r1[i][0] - a, 2  * a * r1[i][1] - a};
                C[i] = {2 * r2[i][0], 2 * r2[i][1]};

                D[i] = {abs(C[i][0] * guides[i][0] - wolves[id][0]), abs(C[i][1] * guides[i][1] - wolves[id][1])};
                X[i] = {guides[i][0] - A[i][0] * D[i][0], guides[i][1] - A[i][1] * D[i][1]};
            }

            wolves[id] = {clip((X[0][0] + X[1][0] + X[2][0]) / 3,-CLIP,CLIP), clip((X[0][1] + X[1][1] + X[2][1]) / 3,-CLIP,CLIP)};

            #pragma omp barrier

        }
    }

    double end = omp_get_wtime();

    printf("%d,%d,%d,%f,%f,%f\n", 1, population, nIterations,prog,start,end);

    // if(fitness(guides[0]) == -1.700000) printf("Optimality Reached\n");
    // else printf("Optimality Not Reached.\n");

    // printf("Population: %d\n", population);
    // printf("Iterations %d\n",nIterations);
    // printf("Time Taken: %f\n",end-start);
    

    // // for(int i = 0; i < population; i++){
    // //     printf("%f %f\tFitness: %f\n",wolves[i][0], wolves[i][1], fitness_scores[i]);
    // // }

    // printf("Guides:\n");

    // for(auto i : index){
    //     printf("%f %f\tFitness: %f\n", wolves[i][0], wolves[i][1], fitness_scores[i]);
    // }

    


}