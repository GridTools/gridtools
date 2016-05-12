#pragma once
/**
@file
@brief definition of macros for host/GPU
*/
#ifdef _USE_GPU_
#include <cuda_runtime.h>
#else
#define __host__
#define __device__
#endif

#ifdef __CUDACC__
#define GT_FUNCTION __host__ __device__ __forceinline__
#else
#define GT_FUNCTION inline
#endif

#define GT_FUNCTION_WARNING __host__ __device__
