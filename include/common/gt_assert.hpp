/*
  GridTools Libraries

  Copyright (c) 2017, GridTools Consortium
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
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
