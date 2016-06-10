/*
   Copyright 2016 GridTools Consortium

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
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
