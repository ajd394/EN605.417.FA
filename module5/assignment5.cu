/*
Author: Andrew DiPrinzio 
Course: EN605.417.FA
Assignment: Module 5
*/

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <assert.h>
#include <time.h>
#include <math.h>

static const uint32_t DEFAULT_NUM_THREADS = 1024;
static const uint32_t DEFAULT_BLOCK_SIZE = 16;

#define KERNEL_LOOP 4096

__constant__ unsigned int const_data_gpu[KERNEL_LOOP];
__device__ static unsigned int gmem_data_gpu[KERNEL_LOOP];
static unsigned int const_data_host[KERNEL_LOOP];

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

__global__ void const_test_gpu_gmem(unsigned int * const data, const unsigned int num_elements)
{
	const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(tid < num_elements)
	{
		unsigned int d = gmem_data_gpu[0];

		for(int i=0;i<KERNEL_LOOP;i++)
		{
			d ^= gmem_data_gpu[0];
			d |= gmem_data_gpu[1];
			d &= gmem_data_gpu[2];
			d |= gmem_data_gpu[3];
		}

		data[tid] = d;
	}
}


__global__ void const_test_gpu_const(unsigned int * const data, const unsigned int num_elements)
{
	const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(tid < num_elements)
	{
		unsigned int d = const_data_gpu[0];

		for(int i=0;i<KERNEL_LOOP;i++)
		{
			d ^= const_data_gpu[0];
			d |= const_data_gpu[1];
			d &= const_data_gpu[2];
			d |= const_data_gpu[3];
		}

		data[tid] = d;
	}
}

__host__ void wait_exit(void)
{
	char ch;

	printf("\nPress any key to exit");
	ch = getchar();
}

__host__ void cuda_error_check(const char * prefix, const char * postfix)
{
	if(cudaPeekAtLastError() != cudaSuccess)
	{
		printf("\n%s%s%s",prefix,cudaGetErrorString(cudaGetLastError()),postfix);
		cudaDeviceReset();
		wait_exit();
		exit(1);
	}
}

__host__ void generate_rand_data(unsigned int * host_data_ptr)
{
	for(unsigned int i=0; i < KERNEL_LOOP; i++)
	{
		host_data_ptr[i] = (unsigned int) rand();
	}
}
__host__ void test_const_mem(Arguments args)
{
	const unsigned int num_elements = (128*1024);
	const unsigned int num_threads = args.num_threads;
	const unsigned int num_blocks = (num_elements + (num_threads-1))/num_threads;
	const unsigned int num_bytes = num_elements * sizeof(unsigned int);
	int max_device_num;
	const int max_runs = 6;

	cudaGetDeviceCount(&max_device_num);

	for(int device_num=0; device_num < max_device_num; device_num++)
	{
		cudaSetDevice(device_num);

		unsigned int * data_gpu;
		cudaEvent_t kernel_start1, kernel_stop1;
		cudaEvent_t kernel_start2, kernel_stop2;
		float delta_time1 = 0.0F, delta_time2 = 0.0F;
		struct cudaDeviceProp device_prop;
		char device_prefix[261];

		cudaMalloc(&data_gpu, num_bytes);
		cudaEventCreate(&kernel_start1);
		cudaEventCreate(&kernel_start2);
		cudaEventCreateWithFlags(&kernel_stop1, cudaEventBlockingSync);
		cudaEventCreateWithFlags(&kernel_stop2, cudaEventBlockingSync);

		cudaGetDeviceProperties(&device_prop, device_num);
		sprintf(device_prefix, "ID: %d %s:", device_num, device_prop.name);

		for(int num_test=0; num_test < max_runs; num_test++)
		{
			generate_rand_data(const_data_host);

			cudaMemcpyToSymbol(const_data_gpu, const_data_host, KERNEL_LOOP * sizeof(unsigned int));

			const_test_gpu_gmem <<<num_blocks, num_threads>>>(data_gpu, num_elements);
			cuda_error_check("Error ", " returned from literal runtime  kernel!");

			cudaEventRecord(kernel_start1,0);

			const_test_gpu_gmem <<<num_blocks, num_threads>>>(data_gpu, num_elements);

			cuda_error_check("Error ", " returned from literal runtime  kernel!");

			cudaEventRecord(kernel_stop1,0);
			cudaEventSynchronize(kernel_stop1);
			cudaEventElapsedTime(&delta_time1, kernel_start1, kernel_stop1);

			cudaMemcpyToSymbol(gmem_data_gpu, const_data_host, KERNEL_LOOP * sizeof(unsigned int));
			const_test_gpu_const<<< num_blocks, num_threads >>>(data_gpu, num_elements);

			cuda_error_check("Error ", " returned from literal startup  kernel!");

			cudaEventRecord(kernel_start2,0);

			const_test_gpu_const<<< num_blocks, num_threads >>>(data_gpu, num_elements);

			cuda_error_check("Error ", " returned from literal startup  kernel!");

			cudaEventRecord(kernel_stop2,0);
			cudaEventSynchronize(kernel_stop2);
			cudaEventElapsedTime(&delta_time2, kernel_start2, kernel_stop2);

			if(delta_time1 > delta_time2)
			{
				printf("\n%sConstant version is faster by: %.2fms (G=%.2fms vs. C=%.2fms)",device_prefix, delta_time1-delta_time2, delta_time1, delta_time2);
			}
			else
			{
				printf("\n%sGMEM version is faster by: %.2fms (G=%.2fms vs. C=%.2fms)",device_prefix, delta_time2-delta_time1, delta_time1, delta_time2);
			}

		}

		cudaEventDestroy(kernel_start1);
		cudaEventDestroy(kernel_start2);
		cudaEventDestroy(kernel_stop1);
		cudaEventDestroy(kernel_stop2);
		cudaFree(data_gpu);

		cudaDeviceReset();
		printf("\n");
	}
	wait_exit();
}

int main(int argc, char ** argv)
{
    Arguments args = parse_arguments(argc, argv);
    printf("Num Threads: %u, Block Size: %u\n", args.num_threads, args.block_size);

    test_const_mem(args);
    
	return EXIT_SUCCESS;
}
