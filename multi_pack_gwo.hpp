#include <bits/stdc++.h>
#include <vector>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <cmath>
#include <omp.h>
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

vector <vector <float>> get_guides(vector <float> &fitness_scores, int k,vector <vector <float>> wolves){

    vector <int> indices = get_guide_indices(fitness_scores,k);

    vector< vector <float>> guides;

    for(auto i: indices) guides.push_back(wolves[i]);

    return guides;

}

float clip(float n, float lower, float upper) {
  return max(lower, min(n, upper));
}

vector <float> get_omega(vector< vector<float>> &pack_guides, int k){
    float x,y;

    x = 0; y = 0;

    for(int i = 0; i < k; i++){
            x += pack_guides[i][0];
            y += pack_guides[i][1];
        }
        x /= k;
        y /= k;
    
    return {x,y};

}

vector <float> get_sigma(vector< vector<float>> &pack_guides, vector< float> &omega, int k){
    float x,y;

    x = 0; y = 0;

    for(int i = 0; i < k; i++){
        x += pow(pack_guides[i][0] - omega[0],2);
        y += pow(pack_guides[i][1] - omega[1],2);
    }

    x /= k;
    y /= k;

    x = sqrt(x);
    y = sqrt(y);

    return {x,y};

}

vector <float> get_repulsion(vector <vector <float>> sigma, vector <vector <float>> omega, int num_packs, int pack){
    
    float x,y;

    x = 0; y = 0;

    vector <float> norm = {0,0};

    for(int i = 0; i < num_packs; i++){
        norm[0] += sigma[i][0];
        norm[1] += sigma[i][1];
    }

    for(int i = 0; i < num_packs; i++){
        if(i != pack){
            x += omega[pack][0] / (sigma[pack][0] / norm[0]) ;
            y += omega[pack][1] / (sigma[pack][1] / norm[1]);
        }
    }

    return {x,y};
}
