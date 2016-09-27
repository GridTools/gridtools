// cuda test cases
#ifdef __CUDACC__

#define __Size0 52
#define __Size1 52
#define __Size2 60
#define BACKEND_BLOCK
#define CUDA_EXAMPLE
#define TESTCLASS stencil_cuda
#include "test_stencils.hpp"
#undef TESTCLASS
#undef BACKEND_BLOCK
#undef CUDA_EXAMPLE
#else

#define __Size0 12
#define __Size1 33
#define __Size2 61
#define BACKEND_BLOCK
#define TESTCLASS stencil_block
#include "test_stencils.hpp"
#undef BACKEND_BLOCK
#undef TESTCLASS
#define TESTCLASS stencil
#include "test_stencils.hpp"
#undef TESTCLASS

#endif