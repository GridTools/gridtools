/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

// DON'T USE #pragma once HERE!!!

#if !defined(GT_FILENAME)
#error GT_FILENAME is not defined
#endif

#if defined(GT_TARGET_ITERATING)
#error nesting target iterating is not supported
#endif

#if defined(GT_TARGET)
#error GT_TARGET should not be defined outside of this file
#endif

#if defined(GT_TARGET_NAMESPACE)
#error GT_TARGET_NAMESPACE should not be defined outside of this file
#endif

#if defined(GT_TARGET_CONSTEXPR)
#error GT_TARGET_NAMESPACE should not be defined outside of this file
#endif

#define GT_TARGET_ITERATING

#ifdef GT_CUDACC

#define GT_TARGET_NAMESPACE_NAME host
#define GT_TARGET_NAMESPACE inline namespace host
#define GT_TARGET GT_HOST

#if defined(__NVCC__) && defined(__CUDACC_VER_MAJOR__) && \
    (__CUDACC_VER_MAJOR__ < 10 || __CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ < 2)
// Sometimes NVCC 10.2 compilation fails with internal compiler error if constexpr functions are used in runtime
// context even they are host only.
#define GT_TARGET_CONSTEXPR
#else
#define GT_TARGET_CONSTEXPR constexpr
#endif

#include GT_FILENAME
#undef GT_TARGET
#undef GT_TARGET_NAMESPACE
#undef GT_TARGET_NAMESPACE_NAME
#undef GT_TARGET_CONSTEXPR

#define GT_TARGET_NAMESPACE_NAME host_device
#define GT_TARGET_NAMESPACE namespace host_device
#define GT_TARGET GT_HOST_DEVICE
#define GT_TARGET_CONSTEXPR
#define GT_TARGET_HAS_DEVICE
#include GT_FILENAME
#undef GT_TARGET_HAS_DEVICE
#undef GT_TARGET
#undef GT_TARGET_NAMESPACE
#undef GT_TARGET_NAMESPACE_NAME
#undef GT_TARGET_CONSTEXPR

#define GT_TARGET_NAMESPACE_NAME device
#define GT_TARGET_NAMESPACE namespace device
#define GT_TARGET GT_DEVICE
#define GT_TARGET_HAS_DEVICE
#define GT_TARGET_CONSTEXPR
#include GT_FILENAME
#undef GT_TARGET_HAS_DEVICE
#undef GT_TARGET
#undef GT_TARGET_NAMESPACE
#undef GT_TARGET_NAMESPACE_NAME
#undef GT_TARGET_CONSTEXPR

#else

#define GT_TARGET_NAMESPACE_NAME host
#define GT_TARGET_NAMESPACE   \
    inline namespace host {}  \
    namespace device {        \
        using namespace host; \
    }                         \
    namespace host_device {   \
        using namespace host; \
    }                         \
    inline namespace host
#define GT_TARGET GT_HOST
#define GT_TARGET_CONSTEXPR constexpr
#include GT_FILENAME
#undef GT_TARGET
#undef GT_TARGET_NAMESPACE
#undef GT_TARGET_NAMESPACE_NAME
#undef GT_TARGET_CONSTEXPR
#endif

#undef GT_TARGET_ITERATING
