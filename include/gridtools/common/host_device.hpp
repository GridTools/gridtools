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
/**
@file
@brief definition of macros for host/GPU
*/
/** \ingroup common
    @{
    \defgroup hostdevice Host-Device Macros
    @{
*/

#include "defs.hpp"

#ifdef __HIPCC__
#include "cuda_runtime.hpp"
#endif

#if defined(__NVCC__)
#define GT_FORCE_INLINE __forceinline__
#define GT_FORCE_INLINE_LAMBDA
#elif defined(__GNUC__)
#define GT_FORCE_INLINE inline __attribute__((always_inline))
#define GT_FORCE_INLINE_LAMBDA __attribute__((always_inline))
#else
#define GT_FORCE_INLINE inline
#define GT_FORCE_INLINE_LAMBDA
#endif

/**
 * @def GT_FUNCTION
 * Function attribute macro to be used for host-device functions.
 */
/**
 * @def GT_FUNCTION_HOST
 * Function attribute macro to be used for host-only functions.
 */
/**
 * @def GT_FUNCTION_DEVICE
 * Function attribute macro to be used for device-only functions.
 */

#ifdef GT_CUDACC
#define GT_HOST_DEVICE __host__ __device__
#ifdef __NVCC__ // NVIDIA CUDA compilation
#define GT_DEVICE __device__
#define GT_HOST __host__
#else // Clang CUDA or HIP compilation
#define GT_DEVICE __device__ __host__
#define GT_HOST __host__
#endif
#else
#define GT_HOST_DEVICE
#define GT_HOST
#endif

#ifndef GT_FUNCTION
#define GT_FUNCTION GT_HOST_DEVICE GT_FORCE_INLINE
#endif

#ifndef GT_FUNCTION_HOST
#define GT_FUNCTION_HOST GT_HOST GT_FORCE_INLINE
#endif

#ifndef GT_FUNCTION_DEVICE
#define GT_FUNCTION_DEVICE GT_DEVICE GT_FORCE_INLINE
#endif

/** @} */
/** @} */
