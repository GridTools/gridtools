/*
 * GridTools
 *
 * Copyright (c) 2014-2023, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <string_view>

#include <boost/preprocessor/stringize.hpp>

#include <gridtools/common/defs.hpp>

#if defined(__NVCC__) && (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 2 || __CUDACC__VER_MAJOR__ > 11)

#if !defined(GT_ASSUME)
#error "GT_ASSUME is undefined"
#else
static_assert(std::string_view(BOOST_PP_STRINGIZE(GT_ASSUME(x))) ==
    std::string_view(BOOST_PP_STRINGIZE(__builtin_assume(x))),
        BOOST_PP_STRINGIZE(GT_ASSUME(x)) " != " BOOST_PP_STRINGIZE(__builtin_assume(x)));
#endif

#endif
