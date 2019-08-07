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
#include <stdexcept>

#include "hip_wrappers.hpp"

/** \ingroup common
    @{
    \defgroup assert Assertion
    @{
*/

#if defined(__clang__) && defined(__CUDACC__) // Clang CUDA compilation
namespace gt_assert_impl_ {
    __host__ inline void throw_error(const std::string &msg) { throw std::runtime_error(msg); }

    __device__ void throw_error(const char *);
} // namespace gt_assert_impl_

#ifdef __CUDA_ARCH__
#define GT_ASSERT_OR_THROW(cond, msg) assert(cond)
#else
#define GT_ASSERT_OR_THROW(cond, msg) \
    if (!(cond))                      \
    gt_assert_impl_::throw_error(msg)
#endif
#else // NVIDIA CUDA compilation or host-only compilation
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

#ifdef __CUDA_ARCH__
#define GT_ASSERT_OR_THROW(cond, msg) assert(cond)
#else
#define GT_ASSERT_OR_THROW(cond, msg) \
    if (!(cond))                      \
    throw std::runtime_error(msg)
#endif
#endif
/** @} */
/** @} */
