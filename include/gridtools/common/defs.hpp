/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
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

#ifdef __NVCC__
#define GT_CONSTEXPR
#else
#define GT_CONSTEXPR constexpr
#endif

#define GT_INTERNAL_ERROR                                                                                       \
    "GridTools encountered an internal error. Please submit the error message produced by the compiler to the " \
    "GridTools Development Team."

#define GT_INTERNAL_ERROR_MSG(x) GT_INTERNAL_ERROR "\nMessage\n\n" x
