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

// some compilers have the problem that template alias instantiations have exponential complexity
#if !defined(GT_BROKEN_TEMPLATE_ALIASES)
#if defined(__CUDACC_VER_MAJOR__)
// CUDA 9.0 and 9.1 have an different problem (not related to the exponential complexity of template alias
// instantiation) see https://github.com/eth-cscs/gridtools/issues/976
#define GT_BROKEN_TEMPLATE_ALIASES (__CUDACC_VER_MAJOR__ < 9)
#elif defined(__INTEL_COMPILER)
#define GT_BROKEN_TEMPLATE_ALIASES (__INTEL_COMPILER < 1800)
#elif defined(__clang__)
#define GT_BROKEN_TEMPLATE_ALIASES 0
#elif defined(__GNUC__) && defined(__GNUC_MINOR__)
#define GT_BROKEN_TEMPLATE_ALIASES (__GNUC__ * 10 + __GNUC_MINOR__ < 47)
#else
#define GT_BROKEN_TEMPLATE_ALIASES 0
#endif
#endif
