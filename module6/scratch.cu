/*
Author: Andrew DiPrinzio 
Course: EN605.417.FA
Assignment: Module 6
*/

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <assert.h>
#include <time.h>
#include <math.h>

#define KERNEL_LOOP 128

static const uint32_t DEFAULT_NUM_THREADS = 1024;
static const uint32_t DEFAULT_BLOCK_SIZE = 16;

static void usage(){    
    printf("Usage: ./assignment6 [-t <num_threads>] [-b <block_size>] [-h]\n");
   
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


// __host__ void generate_rand_data(unsigned int * host_data_ptr)
// {
//     for(unsigned int i=0; i < KERNEL_LOOP; i++)
//     {
//             host_data_ptr[i] = (unsigned int) rand();
//     }
// }

// Helper function to generate a random number within a defined range
__host__ unsigned int random(int min, int max){
    return (unsigned int) min + rand() / (RAND_MAX / (max - min + 1) + 1);
}

__global__ void test_gpu_register(unsigned int * const data, const unsigned int num_elements)
{
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(tid < num_elements)
    {
            unsigned int d_tmp = data[tid];
            d_tmp = d_tmp * 2;
            data[tid] = d_tmp;
    }
}

__global__ void test_gpu_global(unsigned int * const data, const unsigned int num_elements)
{
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(tid < num_elements)
    {
            unsigned int d_tmp = data[tid];
            d_tmp = d_tmp * 2;
            data[tid] = d_tmp;
    }
}

__host__ void test_register_mem(Arguments args)
{
	const unsigned int num_elements = args.num_threads;
	const unsigned int num_threads = args.block_size;
	const unsigned int num_blocks = num_elements / num_threads;
	const unsigned int num_bytes = num_elements * sizeof(unsigned int);

    unsigned int * data_gpu;

    unsigned int host_packed_array[num_elements];
    unsigned int host_packed_array_output[num_elements];

    cudaMalloc(&data_gpu, num_bytes);

    generate_rand_data(host_packed_array);

    cudaMemcpy(data_gpu, host_packed_array, num_bytes,cudaMemcpyHostToDevice);

    test_gpu_register <<<num_blocks, num_threads>>>(data_gpu, num_elements);

    cudaThreadSynchronize();        // Wait for the GPU launched work to complete
    cudaGetLastError();

    cudaMemcpy(host_packed_array_output, data_gpu, num_bytes,cudaMemcpyDeviceToHost);

    for (int i = 0; i < num_elements; i++){
            printf("Input value: %x, device output: %x\n", host_packed_array[i], host_packed_array_output[i]);
    }

    cudaFree((void* ) data_gpu);
    cudaDeviceReset();
}

int main(int argc, char ** argv)
{
    Arguments args = parse_arguments(argc, argv);
    printf("Num Threads: %u, Block Size: %u\n", args.num_threads, args.block_size);

    test_register_mem(args);
    
    return EXIT_SUCCESS;
}
