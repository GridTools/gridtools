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

namespace gridtools {
    using int_t = int;
    using uint_t = unsigned int;
} // namespace gridtools

#ifdef __NVCC__
#define GT_CONSTEXPR
#else
#define GT_CONSTEXPR constexpr
#endif

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
