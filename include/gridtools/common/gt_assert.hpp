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
#include <stdexcept>

#include "hip_wrappers.hpp"

/** \ingroup common
    @{
    \defgroup assert Assertion
    @{
*/

#ifdef __CUDA_ARCH__ // device version of GT_ASSERT_OR_THROW
#define GT_ASSERT_OR_THROW(cond, msg) assert(cond)
#elif defined(__CUDACC__) && defined(__clang__) && !defined(__APPLE_CC__) // Clang-CUDA host version
namespace gt_assert_impl_ {
    __host__ inline void throw_error(const std::string &msg) { throw std::runtime_error(msg); }

    __device__ void throw_error(const char *);
} // namespace gt_assert_impl_
#define GT_ASSERT_OR_THROW(cond, msg) \
    if (!(cond))                      \
    gt_assert_impl_::throw_error(msg)
#else // NVCC host/host-only version
#define GT_ASSERT_OR_THROW(cond, msg) \
    if (!(cond))                      \
    throw std::runtime_error(msg)
#endif
/** @} */
/** @} */
