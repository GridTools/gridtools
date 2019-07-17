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

#if defined(__clang__) && defined(__CUDACC__) // Clang CUDA compilation
namespace gt_assert_impl_ {
__host__ inline void throw_error(const std::string& msg) {
    throw std::runtime_error(msg);
}

__device__ void throw_error(const char*);
}

#ifdef __CUDA_ARCH__
#define GT_ASSERT_OR_THROW(cond, msg) assert(cond)
#else
#define GT_ASSERT_OR_THROW(cond, msg) \
    if (!(cond))                      \
    gt_assert_impl_::throw_error(msg)
#endif
#else // NVIDIA CUDA compilation or host-only compilation
#ifdef __CUDA_ARCH__
#if __CUDACC_VER_MAJOR__ == 9 && __CUDACC_VER_MINOR__ == 2
// we define this macro to an empty string for CUDA 9.2 because in certain cases, CUDA 9.2 tries to compile device
// instantiations of certain constexpr function templates, which can lead to compile-time errors like "cannot use an
// entity undefined in device code".
#define __PRETTY_FUNCTION__ ""
#endif
#define GT_ASSERT_OR_THROW(cond, msg) assert(cond)
#else
#define GT_ASSERT_OR_THROW(cond, msg) \
    if (!(cond))                      \
    throw std::runtime_error(msg)
#endif
#endif
/** @} */
/** @} */
