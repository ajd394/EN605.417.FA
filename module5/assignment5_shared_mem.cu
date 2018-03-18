#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <assert.h>
#include <time.h>
#include <math.h>

/*
Author: Andrew DiPrinzio 
Course: EN605.417.FA
Assignment: Module 5
Resources: 
https://devblogs.nvidia.com/using-shared-memory-cuda-cc/
http://www.nvidia.com/docs/IO/116711/sc11-cuda-c-basics.pdf
*/
static const uint32_t DEFAULT_NUM_THREADS = 1024;
static const uint32_t DEFAULT_BLOCK_SIZE = 32;

#define RADIUS 3

static void usage(){    
    printf("Usage: ./assignment5 [-t <num_threads>] [-b <block_size>] [-h]\n");
   
    printf("\t-t: Specify the number of threads. <num_threads> must be greater than 0. Optional (default %u)\n", DEFAULT_NUM_THREADS);
   
    printf("\t-b: Specify the size of each block. <block_size> must be greater than 0. Optional (default %u)\n", DEFAULT_BLOCK_SIZE);    
}

// Structure that holds program arguments specifying number of threads/blocks
// to use.
typedef struct {    
    uint32_t num_threads;
    uint32_t block_size;
} Arguments;

// Parse the command line arguments using getopt and return an Argument structure
// GetOpt requies the POSIX C Library
static Arguments parse_arguments(const int argc, char ** argv){   
    // Argument format string for getopt
    static const char * _ARG_STR = "ht:b:";
    // Initialize arguments to their default values    
    Arguments args;    
    args.num_threads = DEFAULT_NUM_THREADS;    
    args.block_size = DEFAULT_BLOCK_SIZE;
    // Parse any command line options
    int c;
    int value;
    while ((c = getopt(argc, argv, _ARG_STR)) != -1) {
        switch (c) {
            case 't':
                value = atoi(optarg);
                args.num_threads = value;
                break;
            case 'b':
                // Normal argument
                value = atoi(optarg);
                args.block_size = value;
                break;
            case 'h':
                // 'help': print usage, then exit
                // note the fall through
                usage();
            default:
                exit(-1);
        }
    }
    return args;
}

__global__ 
void stencil_1d_shared(int *in, int *out)
{
    extern __shared__ int temp[];
    int gindex = threadIdx.x + blockIdx.x * blockDim.x;
    int lindex = threadIdx.x + RADIUS;
    // Read input elements into shared memory
    temp[lindex] = in[gindex];
    if (threadIdx.x < RADIUS) {
        temp[lindex - RADIUS] = in[gindex - RADIUS];
        temp[lindex + blockDim.x] = in[gindex + blockDim.x];
    }
    // Synchronize (ensure all the data is available)
    __syncthreads();
    // Apply the stencil
    int result = 0;
    for (int offset = -RADIUS ; offset <= RADIUS ; offset++)
        result += temp[lindex + offset];
    // Store the result
    out[gindex] = result;
}

__global__ 
void stencil_1d_global(const int *in, int *out)
{
    int gindex = threadIdx.x + blockIdx.x * blockDim.x;

    int result = 0;
    for (int offset = -RADIUS ; offset <= RADIUS ; offset++)
        result += in[gindex + offset];
    // Store the result
    out[gindex] = result;
}

// Helper function to generate a random number within a defined range
int random(int min, int max){
    return min + rand() / (RAND_MAX / (max - min + 1) + 1);
}

void measure_shared_kern_speed(Arguments args, int * a, int * b, int * a_d, int * b_d)
{
    // create events for timing
    cudaEvent_t startEvent, stopEvent; 
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    float time;

    int array_size = args.num_threads;
    const unsigned int array_size_in_bytes = array_size * sizeof(int);
	const unsigned int num_blocks = array_size / args.block_size;
    const unsigned int num_threads_per_blk = args.block_size;
    
	cudaMemcpy( a_d, a, array_size_in_bytes, cudaMemcpyHostToDevice );
   
    cudaEventRecord(startEvent, 0);

    stencil_1d_shared<<<num_blocks, num_threads_per_blk, (num_threads_per_blk + (2 * RADIUS)) * sizeof(int) >>>(a_d, b_d);

    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&time, startEvent, stopEvent);
    printf("  Function using Shared memory %f ms\n", time);

    //Print Shared sums
    
    cudaMemcpy(b, b_d, array_size_in_bytes, cudaMemcpyDeviceToHost );

    // for(unsigned int i = 0; i < array_size; i++)
	// {
	// 	printf("Sum #%d: %d\n",i,b[i]);
	// }

    //Global Test
    cudaEventRecord(startEvent, 0);

    stencil_1d_global<<<num_blocks, num_threads_per_blk>>>(a_d, b_d);

    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&time, startEvent, stopEvent);
    printf("  Function using Global memory %f ms\n", time);

    cudaMemcpy(b, b_d, array_size_in_bytes, cudaMemcpyDeviceToHost );

    // for(unsigned int i = 0; i < array_size; i++)
	// {
	// 	printf("Sum #%d: %d\n",i,b[i]);
	// }
    
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
}

void run_shared_experemnt(Arguments args){
    int *h_aPinned, *h_bPinned;

    int array_size = args.num_threads;
    const unsigned int array_size_in_bytes = array_size * sizeof(int);

    /* Randomly generate input vectors and dynamically allocate their memory */
    cudaMallocHost((void**)&h_aPinned, array_size_in_bytes);
    cudaMallocHost((void**)&h_bPinned, array_size_in_bytes);

    int i;
    for (i = 0; i < array_size; i++) {
        h_aPinned[i] = random(0,100);
    }

    /* Declare pointers for GPU based params */
    int *a_d;
    int *b_d;

    cudaMalloc((void**)&a_d, array_size_in_bytes);
    cudaMalloc((void**)&b_d, array_size_in_bytes);

    measure_shared_kern_speed(args, h_aPinned, h_bPinned, a_d, b_d);

    //free memory
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFreeHost(h_aPinned);
    cudaFreeHost(h_bPinned);
}

// void run_constant_experemnt(Arguments args){
//     int *h_aPinned, *h_bPinned;

//     __constant__ unsigned int const_data_gpu[32];
//     static unsigned int const_data_host[32];

//     int array_size = args.num_threads;
//     const unsigned int array_size_in_bytes = array_size * sizeof(int);

//     /* Randomly generate input vectors and dynamically allocate their memory */
//     cudaMallocHost((void**)&h_aPinned, array_size_in_bytes);
//     cudaMallocHost((void**)&h_bPinned, array_size_in_bytes);

//     int i;
//     for (i = 0; i < array_size; i++) {
//         h_aPinned[i] = random(0,100);
//         h_bPinned[i] = random(0,100);
//     }

//     for (i = 0; i < 32; i++) {
//         const_data_host[i] = random(0,100);
//     }

//     /* Declare pointers for GPU based params */
//     int *a_d;
//     int *b_d;

//     cudaMalloc((void**)&a_d, array_size_in_bytes);
//     cudaMalloc((void**)&b_d, array_size_in_bytes);

//     measure_kern_speed(args, h_aPinned, h_bPinned, a_d, b_d);

//     cudaMemcpyToSymbol(const_data_gpu, h_bPinned, array_size_in_bytes);

//     //free memory
//     cudaFree(a_d);
//     cudaFree(b_d);
//     cudaFreeHost(h_aPinned);
//     cudaFreeHost(h_bPinned);
// }

int main(int argc, char ** argv)
{
    Arguments args = parse_arguments(argc, argv);
    printf("Num Threads: %u, Block Size: %u\n", args.num_threads, args.block_size);

    run_shared_experemnt(args);

    cudaDeviceReset();

    // run_constant_experemnt(args);
    
	return EXIT_SUCCESS;
}
