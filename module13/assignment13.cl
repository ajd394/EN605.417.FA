/*
Author: Andrew DiPrinzio 
Course: EN605.417.FA
Assignment: Module 13
Resources:
*/

__kernel void assignment13_kernel(__global const float *a,
						__global const float *b,
						__global float *result)
{
    int gid = get_global_id(0);

    result[gid] = a[gid] + b[gid];
}