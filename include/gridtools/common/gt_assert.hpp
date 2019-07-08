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

/** \ingroup common
    @{
    \defgroup assert Assertion
    @{
*/

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
/** @} */
/** @} */
