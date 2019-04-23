/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
/**
@file
@brief Unstructured collection of small generic purpose functors and related helpers.

   All functors here are supplied with the inner result or result_type to follow boost::result_of requirement
   for the cases if BOOST_RESULT_OF_USE_DECLTYPE is not defined [It is so for nvcc8]. This makes those functors
   usable in the context of boost::fuison high order functions.

*/

#ifndef GT_TARGET_ITERATING
// DON'T USE #pragma once HERE!!!
#ifndef GT_COMMON_FUNCTIONAL_HPP_
#define GT_COMMON_FUNCTIONAL_HPP_

#include <utility>

#include "./host_device.hpp"

#define GT_FILENAME <gridtools/common/functional.hpp>
#include GT_ITERATE_ON_TARGETS()
#undef GT_FILENAME

#endif
#else

namespace gridtools {
    /** \ingroup common
        @{
        \defgroup functional Functional
        @{
    */
    GT_TARGET_NAMESPACE {

        /// Forward the args to constructor.
        //
        template <typename T>
        struct ctor {
            template <typename... Args>
            GT_TARGET GT_FORCE_INLINE GT_HOST_CONSTEXPR T operator()(Args &&... args) const {
                return T{std::forward<Args>(args)...};
            }

#ifndef BOOST_RESULT_OF_USE_DECLTYPE
            using result_type = T;
#endif
        };

        /// Do nothing.
        //
        struct noop {
            template <typename... Args>
            GT_TARGET GT_FORCE_INLINE void operator()(Args &&...) const {}

#ifndef BOOST_RESULT_OF_USE_DECLTYPE
            using result_type = void;
#endif
        };

        /// Perfectly forward the argument.
        //
        struct identity {
            template <typename Arg>
            GT_TARGET GT_FORCE_INLINE GT_HOST_CONSTEXPR Arg operator()(Arg &&arg) const {
                return arg;
            }

#ifndef BOOST_RESULT_OF_USE_DECLTYPE
            template <typename>
            struct result;
            template <typename Arg>
            struct result<identity(Arg &&)> {
                using type = Arg;
            };
#endif
        };

        /// Copy the argument.
        //
        struct clone {
            template <typename Arg>
            GT_TARGET GT_FORCE_INLINE GT_HOST_CONSTEXPR Arg operator()(Arg const &arg) const {
                return arg;
            }
#ifndef BOOST_RESULT_OF_USE_DECLTYPE
            template <typename>
            struct result;
            template <typename Arg>
            struct result<clone(Arg const &)> {
                using type = Arg;
            };
#endif
        };
    }

    /** @} */
    /** @} */
} // namespace gridtools

#endif
