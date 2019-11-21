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

#include "defs.hpp"
#include "hip_wrappers.hpp"

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

            struct cuda_free {
                void operator()(void *ptr) const { cudaFree(ptr); }
            };

        } // namespace _impl

        template <class T>
        using unique_cuda_ptr = std::unique_ptr<T, _impl::cuda_free>;

        template <class Arr, class T = std::remove_extent_t<Arr>>
        unique_cuda_ptr<Arr> cuda_malloc(size_t size) {
            T *ptr;
            GT_CUDA_CHECK(cudaMalloc(&ptr, size * sizeof(T)));
            return unique_cuda_ptr<Arr>{ptr};
        }

        template <class T, std::enable_if_t<!std::is_array<T>::value, int> = 0>
        unique_cuda_ptr<T> cuda_malloc() {
            T *ptr;
            GT_CUDA_CHECK(cudaMalloc(&ptr, sizeof(T)));
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
