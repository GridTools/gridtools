#pragma once

#ifdef __CUDACC__
#include <cuda_runtime.h>
#else
#define __host__
#define __device__
#endif

#define GT_FUNCTION __host__ __device__
