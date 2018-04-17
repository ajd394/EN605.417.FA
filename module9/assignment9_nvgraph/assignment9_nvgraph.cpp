/*
Author: Andrew DiPrinzio 
Course: EN605.417.FA
Assignment: Module 9
Resources:
https://docs.nvidia.com/cuda/nvgraph/index.html#nvgraph-code-examples
*/

#include <iostream>
#include "stdlib.h"
#include "inttypes.h"
#include "stdio.h"

#include "nvgraph.h"
#include <helper_string.h>


void check(nvgraphStatus_t status)
{
    if (status != NVGRAPH_STATUS_SUCCESS) {
        printf("ERROR : %d\n",status);
        exit(1);
    }
}
int main(int argc, char **argv)
{
    const size_t  n = 6, nnz = 10, vertex_numsets = 1, edge_numsets = 1;
    float *sssp_1_h;
    void** vertex_dim;

    // nvgraph variables
    nvgraphStatus_t status; 
    nvgraphHandle_t handle;
    nvgraphGraphDescr_t graph;
    nvgraphCSCTopology32I_t CSC_input;
    cudaDataType_t edge_dimT = CUDA_R_32F;
    cudaDataType_t* vertex_dimT;

    // Init host data
    sssp_1_h = (float*)malloc(n*sizeof(float));
    vertex_dim  = (void**)malloc(vertex_numsets*sizeof(void*));
    vertex_dimT = (cudaDataType_t*)malloc(vertex_numsets*sizeof(cudaDataType_t));
    CSC_input = (nvgraphCSCTopology32I_t) malloc(sizeof(struct nvgraphCSCTopology32I_st));
    vertex_dim[0]= (void*)sssp_1_h; 
    vertex_dimT[0] = CUDA_R_32F;

    float weights_h[nnz];
    for(int i = 0; i < nnz; i++){
        weights_h[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    }
    //weights_h = {0.333333, 0.5, 0.333333, 0.5, 0.5, 1.0, 0.333333, 0.5, 0.5, 0.5};
    //int destination_offsets_h[n + 1];
    int destination_offsets_h[] = {0, 1, 3, 4, 6, 8, nnz};
    //int source_indices_h[n];
    int source_indices_h[] = {2, 0, 2, 0, 4, 5, 2, 3, 3, 4};

    check(nvgraphCreate(&handle));
    check(nvgraphCreateGraphDescr (handle, &graph));

    CSC_input->nvertices = n;
    CSC_input->nedges = nnz;
    CSC_input->destination_offsets = destination_offsets_h;
    CSC_input->source_indices = source_indices_h;

    // Set graph connectivity and properties (tranfers)
    check(nvgraphSetGraphStructure(handle, graph, (void*)CSC_input, NVGRAPH_CSC_32));
    check(nvgraphAllocateVertexData(handle, graph, vertex_numsets, vertex_dimT));
    check(nvgraphAllocateEdgeData  (handle, graph, edge_numsets, &edge_dimT));
    check(nvgraphSetEdgeData(handle, graph, (void*)weights_h, 0));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Solve
    int source_vert = 0;
    check(nvgraphSssp(handle, graph, 0,  &source_vert, 0));

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);                                              
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "GPU Execution time: " << milliseconds <<"ms"<< std::endl;

    // Get and print result
    check(nvgraphGetVertexData(handle, graph, (void*)sssp_1_h, 0));

    //Clean 
    free(sssp_1_h); free(vertex_dim);
    free(vertex_dimT); free(CSC_input);
    check(nvgraphDestroyGraphDescr(handle, graph));
    check(nvgraphDestroy(handle));
    return 0;
}

