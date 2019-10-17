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

#include "integral_constant.hpp"

#ifdef __NVCC__
#define GT_CONSTEXPR
#else
#define GT_CONSTEXPR constexpr
#endif

namespace gridtools {
    using int_t = int;
    using uint_t = unsigned int;

    namespace naive {
        struct backend {};
    } // namespace naive

    namespace cuda {
        template <class IBlockSize = integral_constant<int_t, 64>, class JBlockSize = integral_constant<int_t, 8>>
        struct backend {
            using i_block_size_t = IBlockSize;
            using j_block_size_t = JBlockSize;

            static constexpr i_block_size_t i_block_size() { return {}; }
            static constexpr j_block_size_t j_block_size() { return {}; }
        };
    } // namespace cuda

    namespace mc {
        struct backend {};
    } // namespace mc

    namespace x86 {
        template <class IBlockSize = integral_constant<int_t, 8>, class JBlockSize = integral_constant<int_t, 8>>
        struct backend {
            using i_block_size_t = IBlockSize;
            using j_block_size_t = JBlockSize;

            static constexpr i_block_size_t i_block_size() { return {}; }
            static constexpr j_block_size_t j_block_size() { return {}; }
        };
    } // namespace x86

    /** tags specifying the backend to use */
    namespace backend {
        using cuda = cuda::backend<>;
        using mc = mc::backend;
        using x86 = x86::backend<>;
        using naive = naive::backend;
    } // namespace backend

#define GT_INTERNAL_ERROR                                                                                       \
    "GridTools encountered an internal error. Please submit the error message produced by the compiler to the " \
    "GridTools Development Team."

#define GT_INTERNAL_ERROR_MSG(x) GT_INTERNAL_ERROR "\nMessage\n\n" x

#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ < 9 || __CUDACC_VER_MAJOR__ == 9 && __CUDACC_VER_MINOR__ < 2)
#define GT_DECLARE_DEFAULT_EMPTY_CTOR(class_name)                          \
    __forceinline__ __host__ __device__ constexpr class_name() noexcept {} \
    static_assert(1, "")
#else
#define GT_DECLARE_DEFAULT_EMPTY_CTOR(class_name) class_name() = default
#endif
} // namespace gridtools
