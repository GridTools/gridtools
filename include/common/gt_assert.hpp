#pragma once

#ifdef __CUDACC__
  #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 200)
  // we take the cuda assert for arch greater than 2.x
    #include <assert.h>
  #else
    #undef assert
    #define assert(e)
  #endif
#else
  #include <cassert>
#endif
