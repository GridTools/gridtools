/*
 * GridTools
 *
 * Copyright (c) 2014-2023, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once
#include <cassert>

namespace gridtools {
    using int_t = int;
    using uint_t = unsigned int;
} // namespace gridtools

#if defined(__CUDACC__)
#define GT_CUDACC
#ifdef __CUDA_ARCH__
#define GT_CUDA_ARCH __CUDA_ARCH__
#endif
#elif defined(__HIP__)
#define GT_CUDACC
#ifdef __HIP_DEVICE_COMPILE__
#define GT_CUDA_ARCH 1
#endif
#endif

#if defined(__has_builtin)
#if __has_builtin(__builtin_assume)
#define GT_ASSUME(x) __builtin_assume(x)
#else
#define GT_ASSUME(x)
#endif
#else
#define GT_ASSUME(x)
#endif

#ifdef NDEBUG
#define GT_PROMISE(x) GT_ASSUME(x)
#else
#define GT_PROMISE(x) assert(x)
#endif

#ifdef __cpp_consteval
#define GT_CONSTEVAL consteval
#else
#define GT_CONSTEVAL constexpr
#endif

#define GT_INTERNAL_ERROR                                                                                       \
    "GridTools encountered an internal error. Please submit the error message produced by the compiler to the " \
    "GridTools Development Team."

#define GT_INTERNAL_ERROR_MSG(x) GT_INTERNAL_ERROR "\nMessage\n\n" x

#ifdef __NVCC__
#define GT_NVCC_DIAG_STR(x) #x
#ifdef __NVCC_DIAG_PRAGMA_SUPPORT__
#define GT_NVCC_DIAG_PUSH_SUPPRESS(x) _Pragma("nv_diagnostic push") _Pragma(GT_NVCC_DIAG_STR(nv_diag_suppress x))
#define GT_NVCC_DIAG_POP_SUPPRESS(x) _Pragma("nv_diagnostic pop")
#else
#define GT_NVCC_DIAG_PUSH_SUPPRESS(x) _Pragma(GT_NVCC_DIAG_STR(diag_suppress = x))
#define GT_NVCC_DIAG_POP_SUPPRESS(x) _Pragma(GT_NVCC_DIAG_STR(diag_default = x))
#endif
#else
#define GT_NVCC_DIAG_PUSH_SUPPRESS(x)
#define GT_NVCC_DIAG_POP_SUPPRESS(x)
#endif

#if defined(__NVCC__) && (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 1 && __CUDACC_VER_MINOR__ <= 4)
// enables workaround for CTAD/constexpr issues in CUDA 12.1, 12.2, 12.3, 12.4
// (https://github.com/GridTools/gridtools/issues/1766)
#define GT_NVCC_WORKAROUND_1766 1
#else
#define GT_NVCC_WORKAROUND_1766 0
#endif
