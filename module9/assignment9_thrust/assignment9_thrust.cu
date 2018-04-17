/*
Author: Andrew DiPrinzio 
Course: EN605.417.FA
Assignment: Module 9
Resources:
https://github.com/thrust/thrust/blob/master/examples/saxpy.cu
*/

#include <thrust/transform.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>

#include <iostream>
#include <iterator>
#include <algorithm>
#include <helper_string.h>

struct saxpy_functor : public thrust::binary_function<float,float,float>
{
    const float a;

    saxpy_functor(float _a) : a(_a) {}

    __host__ __device__
        float operator()(const float& x, const float& y) const { 
            return a * x + y;
        }
};

// Helper function to generate a random number within a defined range
float random(int max){
    return  (float)rand()/(float)(RAND_MAX/max);
}

void saxpy_fast(float A, thrust::device_vector<float>& X, thrust::device_vector<float>& Y)
{
    // Y <- A * X + Y
    thrust::transform(X.begin(), X.end(), Y.begin(), Y.begin(), saxpy_functor(A));
}

int main(int argc, char *argv[])
{
    int n = 100;

    //set the mask size 
    if (checkCmdLineFlag(argc, (const char **)argv, "elements"))
    {
        n = getCmdLineArgumentInt(argc, (const char **)argv, "elements");
    }

    // initialize host arrays
    float x[n];
    float y[n];

    for(int index = 0; index < n; index++) 
    {
        x[index] = random(100);
        y[index] = random(100);
    }

    // transfer to device
    thrust::device_vector<float> X(x, x + n);
    thrust::device_vector<float> Y(y, y + n);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    saxpy_fast(2.0, X, Y);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);                                              
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "GPU Execution time: " << milliseconds <<"ms"<< std::endl;

    return 0;
}