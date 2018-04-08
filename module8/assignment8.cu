#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <assert.h>
#include <time.h>
#include <math.h>

/* we need these includes for CUDA's random number stuff */
#include <curand.h>
#include <curand_kernel.h>
#include <cuda.h>

#include "cublas_v2.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_functions.h"
#include "helper_cuda.h"

/*
Author: Andrew DiPrinzio 
Course: EN605.417.FA
Assignment: Module 8
Resources: 
http://docs.nvidia.com/cuda/curand/host-api-overview.html
https://developer.nvidia.com/sites/default/files/akamai/cuda/files/Misc/mygpu.pdf
*/


static void usage(){    
    printf("Usage: ./assignment8 [-h]\n");
}

// Parse the command line arguments using getopt and return an Argument structure
// GetOpt requies the POSIX C Library
void parse_arguments(const int argc, char ** argv){   
    // Argument format string for getopt
    static const char * _ARG_STR = "h";
    // Initialize arguments to their default values    
    // Arguments args;    
    // args.num_threads = DEFAULT_NUM_THREADS;    
    // args.block_size = DEFAULT_BLOCK_SIZE;
    // Parse any command line options
    int c;
    //int value;
    while ((c = getopt(argc, argv, _ARG_STR)) != -1) {
        switch (c) {
            case 'h':
                // 'help': print usage, then exit
                // note the fall through
                usage();
            default:
                exit(-1);
        }
    }
}

/* ******************* CURAND Section *******************/

#define MAX_RAND 100
 
#define N_RAND 100

/* this GPU kernel function is used to initialize the random states */
__global__
void init(unsigned int seed, curandState_t* states)
{
    /* we have to initialize the state */
    curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
                blockIdx.x, /* the sequence number should be different for each core (unless you want all
                                cores to get the same sequence of numbers for some reason - use thread id! */
                0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
                &states[blockIdx.x]);
}
 
/* this GPU kernel takes an array of states, and an array of ints, and puts a random int into each */
__global__
void randoms(curandState_t* states, unsigned int* numbers)
{
    /* curand works like rand - except that it takes a state as a parameter */
    numbers[blockIdx.x] = curand(&states[blockIdx.x]) % 100;
}

__host__
void run_cuda_rand_kernel()
{
    // create events for timing
    cudaEvent_t startEvent, stopEvent; 
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    /* CUDA's random number library uses curandState_t to keep track of the seed value
    we will store a random state for every thread  */
    curandState_t* states;

    /* allocate space on the GPU for the random states */
    cudaMalloc((void**) &states, N_RAND * sizeof(curandState_t));

    /* invoke the GPU to initialize all of the random states */
    init<<<N_RAND, 1>>>(time(0), states);

    /* allocate an array of unsigned ints on the CPU and GPU */
    unsigned int cpu_nums[N_RAND];
    unsigned int* gpu_nums;
    cudaMalloc((void**) &gpu_nums, N_RAND * sizeof(unsigned int));

    /* invoke the kernel to get some random numbers */
    cudaEventRecord(startEvent, 0);
    randoms<<<N_RAND, 1>>>(states, gpu_nums);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);

    /* copy the random numbers back */
    cudaMemcpy(cpu_nums, gpu_nums, N_RAND * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    /* print them out */
    //   for (int i = 0; i < N_RAND; i++) {
    //     printf("%u\n", cpu_nums[i]);
    //   }

    float time;
    cudaEventElapsedTime(&time, startEvent, stopEvent);
    printf("cuRAND deviceAPI Exec Time: %f ms\n", time);

    /* free the memory we allocated for the states and numbers */
    cudaFree(states);
    cudaFree(gpu_nums);
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
}

void run_cuda_rand_hostAPI( ) {
    size_t n = N_RAND;
    
    curandGenerator_t gen;

    // create events for timing
    cudaEvent_t startEvent, stopEvent; 
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    float * devData, * hostData; /* Allocate n floats on host */

    hostData = (float * ) calloc(n, sizeof(float)); /* Allocate n floats on device */

    cudaMalloc((void * * ) & devData, n * sizeof(float)); /* Create pseudo-random number generator */
    curandCreateGenerator( & gen, CURAND_RNG_PSEUDO_DEFAULT); /* Set seed */

    cudaEventRecord(startEvent, 0);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL); /* Generate n floats on device */
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);

    curandGenerateUniform(gen, devData, n); /* Copy device memory to host */
    cudaMemcpy(hostData, devData, n * sizeof(float), cudaMemcpyDeviceToHost); /* Show result */

    float time;
    cudaEventElapsedTime(&time, startEvent, stopEvent);
    printf("cuRAND HostAPI Exec Time: %f ms\n", time);
    
    // size_t i;
    // for (i = 0; i < n; i++) {
    //     printf("%1.4f ", hostData[i]);
    // }
    
    printf("\n"); /* Cleanup */
    curandDestroyGenerator(gen);
    cudaFree(devData);
    free(hostData);
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
}

/* ******************* CUBLAS Section *******************/

// Helper function to generate a random number within a defined range
float random(int max){
    return  (float)rand()/(float)(RAND_MAX/max);
}

void run_cuBLAS_saxpy() {
    // create events for timing
    cudaEvent_t startEvent, stopEvent; 
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    cublasHandle_t handle; // CUBLAS context

    int array_size = 320000;
    const unsigned int array_size_in_bytes = array_size * sizeof(int);

    /* Randomly generate input vectors and dynamically allocate their memory */
    float * x; 
    float * y;
    
    x = (float*)malloc(array_size * sizeof(float));
    y = (float*)malloc(array_size * sizeof(float));

    int i;
    for (i = 0; i < array_size; i++) {
        x[i] = random(100);
    }
    for (i = 0; i < array_size; i++) {
        y[i] = random(100);
    }

    /* Declare pointers for GPU based params */
    float *x_d;
    float *y_d;

    cudaMalloc((void**)&x_d, array_size_in_bytes);
    cudaMalloc((void**)&y_d, array_size_in_bytes);
    cudaMemcpy( x_d, x, array_size_in_bytes, cudaMemcpyHostToDevice );
    cudaMemcpy( y_d, y, array_size_in_bytes, cudaMemcpyHostToDevice );

    cublasCreate(&handle); // initialize CUBLAS context
    cublasSetVector(array_size, sizeof( *x), x, 1, x_d, 1); // cp x- >x_d
    cublasSetVector(array_size, sizeof( *y), y, 1, y_d, 1); // cp y- >y_d
    float al = 2.0; // al =2
    // multiply the vector x_d by the scalar al and add to y_d
    // y_d = al*x_d + y_d , x_d ,y_d - n- vectors ; al - scalar
    cudaEventRecord(startEvent, 0);
    cublasSaxpy(handle, array_size, &al, x_d, 1, y_d, 1);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);

    cublasGetVector(array_size, sizeof(float), y_d, 1, y, 1); // cp y_d - >y

    float time;
    cudaEventElapsedTime(&time, startEvent, stopEvent);
    printf("cuBLAS Saxpy Exec Time: %f ms\n", time);

    // printf("y after Saxpy :\n"); // print y after Saxpy
    // for (j = 0; j < array_size; j++)
    //     printf(" %2.0f,", y[j]);
    
    printf("\n");
    cudaFree(x_d); // free device memory
    cudaFree(y_d); // free device memory
    cublasDestroy(handle); // destroy CUBLAS context
    free(x); // free host memory
    free(y); // free host memory
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
}

//Kernel that performs saxpy
__global__
void custom_saxpy(const float * a , const float *x, float *y)
{
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	y[thread_idx] = (*a) * x[thread_idx] + y[thread_idx] ;
}

void run_custom_saxpy(){
    printf("Running custom_saxpy\n");
    int array_size = 320000;
    const unsigned int array_size_in_bytes = array_size * sizeof(int);

    // create events for timing
    cudaEvent_t startEvent, stopEvent; 
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    /* Randomly generate input vectors and dynamically allocate their memory */
    float * x; 
    float * y;
   	float * a;
    
    x = (float*)malloc(array_size * sizeof(float));
    y = (float*)malloc(array_size * sizeof(float));
    a = (float*)malloc(sizeof(float));

    int i;
    for (i = 0; i < array_size; i++) {
        x[i] = random(100);
    }
    for (i = 0; i < array_size; i++) {
        y[i] = random(100);
    }
	a[0] = (float) 2;

	/* Declare pointers for GPU based params */
	float *x_d;
    float *y_d;
    float *a_d;

    cudaMalloc((void**)&x_d, array_size_in_bytes);
    cudaMalloc((void**)&y_d, array_size_in_bytes);
    cudaMalloc((void**)&a_d, sizeof(float));
	cudaMemcpy( x_d, x, array_size_in_bytes, cudaMemcpyHostToDevice );
    cudaMemcpy( y_d, y, array_size_in_bytes, cudaMemcpyHostToDevice );
    cudaMemcpy( a_d, a, sizeof(float), cudaMemcpyHostToDevice );

	const unsigned int num_blocks = array_size / 32;
    const unsigned int num_threads_per_blk = array_size/num_blocks;
    
    cudaEventRecord(startEvent, 0);
	/* Execute our kernel */
	custom_saxpy<<<num_blocks, num_threads_per_blk>>>(a_d,x_d,y_d);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);

    float time;
    cudaEventElapsedTime(&time, startEvent, stopEvent);
    printf("Custom Saxpy Exec Time: %f ms\n", time);

	/* Free the arrays on the GPU as now we're done with them */
    cudaMemcpy(y, y_d, array_size_in_bytes, cudaMemcpyDeviceToHost );
    
	cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(a_d);
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
}

int main(int argc, char ** argv)
{
    parse_arguments(argc, argv);

    printf("Runing cuRAND Experements \n");

    printf("cuRAND device API \n");
    run_cuda_rand_kernel();

    printf("\ncuRAND host API \n");
    run_cuda_rand_hostAPI();

    printf("Runing cuBLAS Experement\n");
    run_cuBLAS_saxpy();

    run_custom_saxpy();

	return EXIT_SUCCESS;
}