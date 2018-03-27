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
*/

static const uint32_t DEFAULT_NUM_THREADS = 1024;
static const uint32_t DEFAULT_BLOCK_SIZE = 16;

static void usage(){    
    printf("Usage: ./assignment7 [-t <num_threads>] [-b <block_size>] [-h]\n");
   
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

// Helper function to generate a random number within a defined range
__host__
int random(int min, int max){
    return min + rand() / (RAND_MAX / (max - min + 1) + 1);
}

// simple kernel that adds vectors
__global__
void arrayAddition(int *device_a, int *device_b, int *device_result)
{
    int threadId = threadIdx.x + blockIdx.x * blockDim.x ;
    device_result[threadId]= device_a[threadId]+device_b[threadId]; 
} 

//Setup and run a timed experement using CUDA streams
__host__
void run_streaming(int run_index, Arguments args)
{
    int *host_a, *host_b, *host_result;
    int *device_a, *device_b, *device_result;
  
    int array_size = args.num_threads;
    const unsigned int array_size_in_bytes = array_size * sizeof(int);
    
    // create events for timing
    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate( &start ); 
    cudaEventCreate( &stop );

    //configure stream
    cudaStream_t stream; 
    cudaStreamCreate(&stream); 

    //Configure device and host memory
    cudaMalloc( ( void**)& device_a, array_size_in_bytes ); 
    cudaMalloc( ( void**)& device_b, array_size_in_bytes ); 
    cudaMalloc( ( void**)& device_result, array_size_in_bytes ); 

    cudaHostAlloc((void **)&host_a, array_size_in_bytes, cudaHostAllocDefault);
    cudaHostAlloc((void **)&host_b, array_size_in_bytes, cudaHostAllocDefault);
    cudaHostAlloc((void **)&host_result, array_size_in_bytes, cudaHostAllocDefault);

    for(int index = 0; index < array_size; index++) 
    {
        host_a[index] = random(0,100);
        host_b[index] = random(0,100);
    }

    const unsigned int num_blocks = array_size / args.block_size;
    const unsigned int num_threads_per_blk = array_size/num_blocks;
    
    cudaEventRecord(start, 0);
    
    //initialte streaming memory operation
    cudaMemcpyAsync(device_a, host_a, array_size_in_bytes, cudaMemcpyHostToDevice, stream); 
    cudaMemcpyAsync(device_b, host_b, array_size_in_bytes, cudaMemcpyHostToDevice, stream);

	/* Execute our kernel */
    arrayAddition<<<num_blocks, num_threads_per_blk, 0, stream>>>(device_a, device_b, device_result); 

    cudaMemcpyAsync(host_result, device_result, array_size_in_bytes, cudaMemcpyHostToDevice, stream); 

    cudaStreamSynchronize(stream);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop); 
    cudaEventElapsedTime(&elapsedTime, start, stop); 

    // Print input and output for debugging purposes
    /*
    for(int index = 0; index < array_size; index++) 
    {
        printf("a: %d\n", host_a[index]);
        printf("a: %d\n", host_b[index]);
        printf("Result: %d\n", host_result[index]);
    }
    */
    //Ensure all operations have completed and print sumary
    cudaDeviceSynchronize();
    printf("Execution: %d\n", run_index); 
    printf("\n Size of array : %d \n", array_size); 
    printf("\n Time taken: %3.1f ms \n", elapsedTime); 

    //memory cleanup
    cudaFreeHost(host_a); 
    cudaFreeHost(host_b); 
    cudaFreeHost(host_result); 
    cudaFree(device_a); 
    cudaFree(device_b); 
    cudaFree(device_result);
}

int main(int argc, char ** argv)
{
    Arguments args = parse_arguments(argc, argv);
    printf("Num Threads: %u, Block Size: %u\n", args.num_threads, args.block_size);

    //loop to run experement with differnt random data
    for(int index = 0; index < 5; index++) 
    {
        run_streaming(index,args);
    }

    cudaDeviceReset();
    
	return EXIT_SUCCESS;
}