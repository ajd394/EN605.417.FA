/*
Author: Andrew DiPrinzio 
Course: EN605.417.FA
Assignment: Module 12
Resources:
*/


// raytracer.cpp
//
//    This is a (very) simple raytracer that is intended to demonstrate 
//    using OpenCL buffers.

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "info.hpp"

#define NUM_BUFFER_ELEMENTS 16

// Function to check and handle OpenCL errors
inline void 
checkErr(cl_int err, const char * name)
{
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " <<  name << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

///
//  Create an OpenCL context on the first available platform using
//  either a GPU or CPU depending on what is available.
//
cl_context CreateContext()
{
    cl_int errNum;
    cl_uint numPlatforms;
    cl_platform_id firstPlatformId;
    cl_context context = NULL;

    // First, select an OpenCL platform to run on.  For this example, we
    // simply choose the first available platform.  Normally, you would
    // query for all available platforms and select the most appropriate one.
    errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);
    if (errNum != CL_SUCCESS || numPlatforms <= 0)
    {
        std::cerr << "Failed to find any OpenCL platforms." << std::endl;
        return NULL;
    }

    // Next, create an OpenCL context on the platform.  Attempt to
    // create a GPU-based context, and if that fails, try to create
    // a CPU-based context.
    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)firstPlatformId,
        0
    };
    context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU,
                                      NULL, NULL, &errNum);
    if (errNum != CL_SUCCESS)
    {
        std::cout << "Could not create GPU context, trying CPU..." << std::endl;
        context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_CPU,
                                          NULL, NULL, &errNum);
        if (errNum != CL_SUCCESS)
        {
            std::cerr << "Failed to create an OpenCL GPU or CPU context." << std::endl;
            return NULL;
        }
    }

    return context;
}

///
//  Create a command queue on the first device available on the
//  context
//
cl_command_queue CreateCommandQueue(cl_context context, cl_device_id *device)
{
    cl_int errNum;
    cl_device_id *devices;
    cl_command_queue commandQueue = NULL;
    size_t deviceBufferSize = -1;

    // First get the size of the devices buffer
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Failed call to clGetContextInfo(...,GL_CONTEXT_DEVICES,...)";
        return NULL;
    }

    if (deviceBufferSize <= 0)
    {
        std::cerr << "No devices available.";
        return NULL;
    }

    // Allocate memory for the devices buffer
    devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL);
    if (errNum != CL_SUCCESS)
    {
        delete [] devices;
        std::cerr << "Failed to get device IDs";
        return NULL;
    }

    // In this example, we just choose the first available device.  In a
    // real program, you would likely use all available devices or choose
    // the highest performance device based on OpenCL device queries
    commandQueue = clCreateCommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, NULL);
    if (commandQueue == NULL)
    {
        delete [] devices;
        std::cerr << "Failed to create commandQueue for device 0";
        return NULL;
    }

    *device = devices[0];
    delete [] devices;
    return commandQueue;
}

///
//  Create an OpenCL program from the kernel source file
//
cl_program CreateProgram(cl_context context, cl_device_id device, const std::string fileName)
{
    cl_int errNum;
    cl_program program;

    std::ifstream kernelFile(fileName.c_str(), std::ios::in);
    if (!kernelFile.is_open())
    {
        std::cerr << "Failed to open file for reading: " << fileName << std::endl;
        return NULL;
    }

    std::ostringstream oss;
    oss << kernelFile.rdbuf();

    std::string srcStdStr = oss.str();
    const char *srcStr = srcStdStr.c_str();
    program = clCreateProgramWithSource(context, 1,
                                        (const char**)&srcStr,
                                        NULL, NULL);
    if (program == NULL)
    {
        std::cerr << "Failed to create CL program from source." << std::endl;
        return NULL;
    }

    errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (errNum != CL_SUCCESS)
    {
        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                              sizeof(buildLog), buildLog, NULL);

        std::cerr << "Error in kernel: " << std::endl;
        std::cerr << buildLog;
        clReleaseProgram(program);
        return NULL;
    }

    return program;
}

///
//	main() for simple buffer and sub-buffer example
//
int main(int argc, char** argv)
{
    cl_int errNum;
    cl_uint numPlatforms;
    cl_uint numDevices;
    cl_platform_id * platformIDs;
    cl_device_id * deviceIDs;
    cl_device_id device;
    cl_context context;
    cl_program program;
    cl_command_queue queue;
    std::vector<cl_kernel> kernels;
    std::vector<cl_mem> buffers;
    std::vector<cl_mem> buffers_out;
    float input[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    float output_ind[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    const std::string filename = "assignment12";

    int platform = 0;

    std::cout << filename << std::endl;

    context = CreateContext();
    if (context == NULL)
    {
        std::cerr << "Failed to create OpenCL context." << std::endl;
        exit;
    }

    // Create a command-queue on the first device available
    // on the created context
    queue = CreateCommandQueue(context, &device);
    if (queue == NULL)
    {
        std::cerr << "Failed to create OpenCL queue." << std::endl;
        exit;
    }
    
    // Create OpenCL program from corresponding .cl kernel source
    program = CreateProgram(context, device, filename + ".cl");
    if (program == NULL)
    {
        exit;
    }

    // create buffers and sub-buffers
    int sub_buf_size = 4;
    int numSubBufs = NUM_BUFFER_ELEMENTS - sub_buf_size + 1;
    int output [numSubBufs];

    std::cout << "Input Buffer Size: " << NUM_BUFFER_ELEMENTS << std::endl;
    std::cout << "Size of Sub-buffers: " << sub_buf_size << std::endl;
    std::cout << "Number of Sub-buffers: " << numSubBufs << std::endl;

    // create a single buffer to cover all the input data
    cl_mem buffer = clCreateBuffer(
        context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * NUM_BUFFER_ELEMENTS,
        input,
        &errNum);
    checkErr(errNum, "clCreateBuffer");

    cl_mem buffer_out = clCreateBuffer(
        context,
        CL_MEM_WRITE_ONLY,
        sizeof(float) * numSubBufs,
        NULL,
        &errNum);
    checkErr(errNum, "clCreateBuffer");

    for (unsigned int i = 0; i <= NUM_BUFFER_ELEMENTS - sub_buf_size; i++)
    {
        //input subbuffer
        cl_buffer_region region = 
            {
                i * sizeof(float), 
                sub_buf_size * sizeof(float)
            };

        cl_mem buff = clCreateSubBuffer(
            buffer,
            CL_MEM_READ_ONLY,
            CL_BUFFER_CREATE_TYPE_REGION,
            &region,
            &errNum);
        
        //output subbuffer
        cl_buffer_region region_out = 
            {
                i * sizeof(float), 
                sizeof(float)
            };

        cl_mem buff_out = clCreateSubBuffer(
            buffer_out,
            CL_MEM_WRITE_ONLY,
            CL_BUFFER_CREATE_TYPE_REGION,
            &region_out,
            &errNum);
        
        checkErr(errNum, "clCreateSubBuffer");

        buffers.push_back(buff);
        buffers_out.push_back(buff_out);
    }
    
    // Create kernels
    for (unsigned int i = 0; i < numSubBufs; i++)
    {
        cl_kernel kernel = clCreateKernel(
            program,
            "avg4",
            &errNum);
        checkErr(errNum, "clCreateKernel(avg4)");
        
        errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&buffers[i]);
        checkErr(errNum, "clSetKernelArg(avg4)");

        errNum = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&buffers_out[i]);
        //errNum = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&buffers_out);
        checkErr(errNum, "clSetKernelArg(avg4)");

        kernels.push_back(kernel);
    }

    
    // std::vector<cl_event> events;
    // for (unsigned int i = 0; i < numSubBufs; i++)
    // {
    //     cl_event event;

    //     size_t gWI = sub_buf_size;

    //     errNum = clEnqueueNDRangeKernel(
    //         queue, 
    //         kernels[i], 
    //         1, 
    //         NULL,
    //         (const size_t*)&gWI, 
    //         (const size_t*)NULL, 
    //         0, 
    //         0, 
    //         &event);
    //     checkErr(errNum, "clEnqueueNDRangeKernel");

    //     events.push_back(event);
    // }

        std::vector<cl_event> events;
    for (unsigned int i = 0; i < numSubBufs; i++)
    {
        cl_event event;

        size_t gWI = 1;

        errNum = clEnqueueNDRangeKernel(
            queue, 
            kernels[i], 
            1, 
            NULL,
            (const size_t*)&gWI, 
            (const size_t*)NULL, 
            0, 
            0, 
            &event);
        checkErr(errNum, "clEnqueueNDRangeKernel");

        events.push_back(event);
    }

    // Technically don't need this as we are doing a blocking read
    // with in-order queue.
    clWaitForEvents(events.size(), &events[0]);

    // Read back computed data
    errNum = clEnqueueReadBuffer(
        queue,
        buffer_out,
        CL_TRUE,
        0,
        sizeof(float) * numSubBufs,
        (void*)output,
        0,
        NULL,
        NULL);
    checkErr(errNum, "clEnqueueReadBuffer(buffer_out)");

    // Display output
    for (unsigned i = 0; i < numSubBufs; i++)
    {
        std::cout << " " << output[i];
    }

    std::cout << std::endl;

    std::cout << "Program completed successfully" << std::endl;

    return 0;
}
