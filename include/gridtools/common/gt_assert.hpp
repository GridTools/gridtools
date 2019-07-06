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

/** \ingroup common
    @{
    \defgroup assert Assertion
    @{
*/

#ifdef __CUDA_ARCH__
#define host_assert(e)
#define GT_ASSERT_OR_THROW(cond, msg) assert(cond)
#else
#define host_assert(e) assert(e)
#define GT_ASSERT_OR_THROW(cond, msg) \
    if (!(cond))                      \
    throw std::runtime_error(msg)
#endif
/** @} */
/** @} */
