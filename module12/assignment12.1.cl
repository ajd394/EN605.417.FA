/*
Author: Andrew DiPrinzio 
Course: EN605.417.FA
Assignment: Module 12
Resources:
*/

// simple.cl
//
//    This is a simple example demonstrating buffers and sub-buffer usage

__kernel void avg4(__global const float * input, __global float * buffer_out)
{
	size_t id = get_global_id(0);
	//computes the average of when executed on 4 numbers
	buffer_out[0] += (input[id] * 0.25);
}