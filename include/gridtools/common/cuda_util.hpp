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

#include <type_traits>

namespace gridtools {
    namespace cuda_util {
        template <class>
        struct is_cloneable;

#ifndef __INTEL_COMPILER
        template <class T>
        struct is_cloneable : std::is_trivially_copyable<T> {};
#endif
    } // namespace cuda_util
} // namespace gridtools

#ifdef __CUDACC__

#include <cassert>
#include <memory>
#include <sstream>

#include <cuda_runtime.h>

#include "defs.hpp"

#define GT_CUDA_CHECK(expr)                                                                    \
    do {                                                                                       \
        cudaError_t err = expr;                                                                \
        if (err != cudaSuccess)                                                                \
            ::gridtools::cuda_util::_impl::on_error(err, #expr, __func__, __FILE__, __LINE__); \
    } while (false)

namespace gridtools {
    namespace cuda_util {
        namespace _impl {
            struct deleter_f {
                template <class T>
                void operator()(T *ptr) const {
                    cudaFree(ptr);
                }
            };

            void on_error(cudaError_t err, const char snippet[], const char fun[], const char file[], int line) {
                std::ostringstream strm;
                strm << "cuda failure: \"" << cudaGetErrorString(err) << "\" [" << cudaGetErrorName(err) << "(" << err
                     << ")] in \"" << snippet << "\" function: " << fun << ", location: " << file << "(" << line << ")";
                throw std::runtime_error(strm.str());
            }

        } // namespace _impl

        template <class T, class Res = std::unique_ptr<T, _impl::deleter_f>>
        Res make_clone(T const &src) {
            GRIDTOOLS_STATIC_ASSERT(std::is_trivially_copyable<T>::value, GT_INTERNAL_ERROR);
            T *ptr;
            GT_CUDA_CHECK(cudaMalloc(&ptr, sizeof(T)));
            Res res{ptr};
            GT_CUDA_CHECK(cudaMemcpy(ptr, &src, sizeof(T), cudaMemcpyHostToDevice));
            return res;
        }

        template <class T>
        T from_clone(std::unique_ptr<T, _impl::deleter_f> const &clone) {
            GRIDTOOLS_STATIC_ASSERT(std::is_trivially_copyable<T>::value, GT_INTERNAL_ERROR);
            T res;
            GT_CUDA_CHECK(cudaMemcpy(&res, clone.get(), sizeof(T), cudaMemcpyDeviceToHost));
            return res;
        }

    } // namespace cuda_util
} // namespace gridtools

#endif
