/*
 * GridTools
 *
 * Copyright (c) 2019, ETH Zurich
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
#define GT_ASSERT_OR_THROW(cond, msg) assert(cond)
#else
#define GT_ASSERT_OR_THROW(cond, msg) \
    if (!(cond))                      \
    throw std::runtime_error(msg)
#endif
/** @} */
/** @} */
