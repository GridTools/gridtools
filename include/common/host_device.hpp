#pragma once
/**
@file
@brief definition of macros for host/GPU
*/
#ifdef _USE_GPU_
# include <cuda_runtime.h>
#else
# ifndef __host__
#  define __host__
# endif
# ifndef __device__
#  define __device__
# endif
#endif

#ifdef __CUDACC__
#define GT_FUNCTION __host__ __device__ __forceinline__
#else
#define GT_FUNCTION inline
#endif

#define GT_FUNCTION_WARNING __host__ __device__
