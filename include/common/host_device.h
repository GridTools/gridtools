#pragma once

#ifdef _GCL_GPU_
#include <cuda_runtime.h>
#else
#define __host__
#define __device__
#endif

#define GT_FUNCTION __host__ __device__
#define GT_FUNCTION_WARNING __host__ __device__
