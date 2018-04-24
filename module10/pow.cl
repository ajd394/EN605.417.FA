
__kernel void pow_kernel(__global const float *a,
						__global const float *b,
						__global float *result)
{
    int gid = get_global_id(0);

    result[gid] = pow(a[gid],b[gid]);
}
