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

__host__ void wait_exit(void)
{
    char ch;

    printf("\nPress any key to exit");
    ch = getchar();
}

__host__ void generate_rand_data(unsigned int * host_data_ptr)
{
    for(unsigned int i=0; i < KERNEL_LOOP; i++)
    {
            host_data_ptr[i] = (unsigned int) rand();
    }
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

__host__ void gpu_kernel(void)
{
    const unsigned int num_elements = KERNEL_LOOP;
    const unsigned int num_threads = KERNEL_LOOP;
    const unsigned int num_blocks = num_elements/num_threads;
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
            printf("Input value: %x, device output: %x\n",host_packed_array[i], host_packed_array_output[i]);
    }

    cudaFree((void* ) data_gpu);
    cudaDeviceReset();
    wait_exit();
}

void execute_host_functions()
{

}

void execute_gpu_functions()
{
	gpu_kernel();
}

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
int main(void) {
	execute_host_functions();
	execute_gpu_functions();

	return EXIT_SUCCESS;
}
