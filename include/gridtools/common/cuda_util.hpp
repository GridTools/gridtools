/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <cassert>
#include <memory>
#include <sstream>
#include <type_traits>

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
        template <class T>
        struct is_cloneable : std::is_trivially_copyable<T> {};

        namespace _impl {
            inline void on_error(cudaError_t err, const char snippet[], const char fun[], const char file[], int line) {
                std::ostringstream strm;
                strm << "cuda failure: \"" << cudaGetErrorString(err) << "\" [" << cudaGetErrorName(err) << "(" << err
                     << ")] in \"" << snippet << "\" function: " << fun << ", location: " << file << "(" << line << ")";
                throw std::runtime_error(strm.str());
            }
        } // namespace _impl

        template <class T>
        using unique_cuda_ptr = std::unique_ptr<T, std::integral_constant<decltype(&cudaFree), &cudaFree>>;

        template <class T>
        unique_cuda_ptr<T> cuda_malloc(size_t size = 1) {
            T *ptr;
            GT_CUDA_CHECK(cudaMalloc(&ptr, size * sizeof(T)));
            return unique_cuda_ptr<T>{ptr};
        }

        template <class T>
        unique_cuda_ptr<T> make_clone(T const &src) {
            static_assert(std::is_trivially_copyable<T>::value, GT_INTERNAL_ERROR);
            unique_cuda_ptr<T> res = cuda_malloc<T>();
            GT_CUDA_CHECK(cudaMemcpy(res.get(), &src, sizeof(T), cudaMemcpyHostToDevice));
            return res;
        }

        template <class T>
        T from_clone(unique_cuda_ptr<T> const &clone) {
            static_assert(std::is_trivially_copyable<T>::value, GT_INTERNAL_ERROR);
            T res;
            GT_CUDA_CHECK(cudaMemcpy(&res, clone.get(), sizeof(T), cudaMemcpyDeviceToHost));
            return res;
        }
    } // namespace cuda_util
} // namespace gridtools
