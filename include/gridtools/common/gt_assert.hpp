/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
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
#include <stdexcept>
#include <strstream>

#ifdef __CUDACC__

#include <cuda_runtime.h>

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

namespace gridtools {
    namespace _impl {
        template < class T >
        void do_throw(T const &msg, const char *fun, const char *file, size_t line, const char *cond) {
            std::ostringstream strm;
            strm << "in function \"" << fun << "\" [" << file << ":" << line << "] condition \"" << cond
                 << "\" failed with message: " << msg;
            throw std::runtime_error(strm.str());
        }
    }
}

#ifdef __CUDA_ARCH__
#define ASSERT_OR_THROW(cond, msg) assert(cond)
#else
#define ASSERT_OR_THROW(cond, msg) \
    if (!cond)                     \
        ::gridtools::_impl::do_throw(msg, __func__, __FILE__, __LINE__, #cond);
#endif

#ifdef __CUDACC__

namespace gridtools {
    namespace _impl {
        std::string cuda_error_msg(cudaError_t err) {
            std::ostringstream strm;
            strm << "CUDA error: " << cudaGetErrorName(err) << "(" << cudaGetErrorString(err) << ")";
            return strm.str();
        }
    }
}

#define CHECK_CUDA_ERROR(err) ASSERT_OR_THROW((err) == cudaSuccess, ::gridtools::_impl::cuda_error_msg(err))

#endif
