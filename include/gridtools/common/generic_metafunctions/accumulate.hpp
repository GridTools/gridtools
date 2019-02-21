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
#include "../defs.hpp"
#include "binary_ops.hpp"
#include <algorithm>
#include <boost/mpl/at.hpp>

/**@file
   @brief Implementation of a compile-time accumulator and constexpr max and min functions

   The accumulator allows to perform operations on static const value to
   be passed as template argument.
*/
namespace gridtools {

/** \ingroup common
    @{
    \ingroup allmeta
    @{
    \defgroup accumulate Accumulate
    @{
*/
/*
 * find/replace constexpr_max(...) -> max({...}) once we are c++14
 */
#if __cplusplus >= 201402L
    template <typename... Ts>
    GT_FUNCTION constexpr typename std::common_type<Ts...>::type constexpr_max(Ts... vals) {
        return std::max({vals...});
    }
    template <typename... Ts>
    GT_FUNCTION constexpr typename std::common_type<Ts...>::type constexpr_min(Ts... vals) {
        return std::min({vals...});
    }
#else
    template <typename T>
    GT_FUNCTION constexpr T constexpr_max(T v) {
        return v;
    }

    /**
     * @brief constexpr max of a sequence of values
     */
    template <typename T0, typename T1, typename... Ts>
    GT_FUNCTION constexpr typename std::common_type<T0, T1, Ts...>::type constexpr_max(T0 v0, T1 v1, Ts... vals) {
        return (v0 > v1) ? constexpr_max(v0, vals...) : constexpr_max(v1, vals...);
    }

    template <typename T>
    GT_FUNCTION constexpr T constexpr_min(T v) {
        return v;
    }

    /**
     * @brief constexpr min of a sequence of values
     */
    template <typename T0, typename T1, typename... Ts>
    GT_FUNCTION constexpr typename std::common_type<T0, T1, Ts...>::type constexpr_min(T0 v0, T1 v1, Ts... vals) {
        return (v0 < v1) ? constexpr_min(v0, vals...) : constexpr_min(v1, vals...);
    }
#endif

    /**@brief operation to be used inside the accumulator*/
    struct multiplies {
        GT_FUNCTION
        constexpr multiplies() {}
        template <typename T1, typename T2>
        GT_FUNCTION constexpr T1 operator()(const T1 &x, const T2 &y) const {
            return x * y;
        }
    };

    /**@brief operation to be used inside the accumulator*/
    struct plus_functor {
        GT_FUNCTION
        constexpr plus_functor() {}
        template <class T>
        GT_FUNCTION constexpr T operator()(const T &x, const T &y) const {
            return x + y;
        }
    };

    /**@brief accumulator recursive implementation*/
    template <typename Operator, typename First, typename... Args>
    GT_FUNCTION static constexpr First accumulate(Operator op, First first, Args... args) {
        return op(first, accumulate(op, args...));
    }

    /**@brief specialization to stop the recursion*/
    template <typename Operator, typename First>
    GT_FUNCTION static constexpr First accumulate(Operator op, First first) {
        return first;
    }

    /** @} */
    /** @} */
    /** @} */

} // namespace gridtools
