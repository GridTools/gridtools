#pragma once
#include "host_device.hpp"

#ifndef NDEBUG
#include <stdio.h>
#endif

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

#ifdef __GNUC__

#define GTASSERT(cond) gt_assert(cond, __LINE__, __FILE__)

namespace gridtools {
    GT_FUNCTION
    void gt_assert(bool cond, int line, const char *filename) {
#ifndef NDEBUG
        if (!cond)
            printf("Assert triggered in %s:%d \n", filename, line);
#endif
        assert(cond);
    }
}

#else

#define GTASSERT(cond) gt_assert(cond)

namespace gridtools {
    GT_FUNCTION
    void gt_assert(bool cond) { assert(cond); }
}

#endif
