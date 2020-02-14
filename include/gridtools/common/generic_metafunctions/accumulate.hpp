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

#include <algorithm>

#include "../host_device.hpp"

/**@file
   @brief Implementation of a compile-time accumulator

   The accumulator allows to perform operations on static const value to be passed as template argument.
*/
namespace gridtools {

    /** \ingroup common
        @{
        \ingroup allmeta
        @{
        \defgroup accumulate Accumulate
        @{
    */

    template <typename... Ts>
    constexpr auto constexpr_max(Ts const &... vals) {
        return std::max({vals...});
    }
    template <typename... Ts>
    constexpr auto constexpr_min(Ts const &... vals) {
        return std::min({vals...});
    }

    /**@brief operation to be used inside the accumulator*/
    struct plus_functor {
        template <typename T, typename U>
        GT_FUNCTION constexpr auto operator()(T const &x, U const &y) const {
            return x + y;
        }
    };

    /**@brief operation to be used inside the accumulator*/
    struct logical_and {
        GT_FUNCTION constexpr bool operator()(bool x, bool y) const { return x && y; }
    };

    /**@brief operation to be used inside the accumulator*/
    struct logical_or {
        GT_FUNCTION constexpr bool operator()(bool x, bool y) const { return x || y; }
    };

    /**@brief accumulator recursive implementation*/
    template <typename Operator, typename First, typename... Args>
    GT_FUNCTION static constexpr auto accumulate(Operator op, First const &first, Args const &... args) {
        return op(first, accumulate(op, args...));
    }

    /**@brief specialization to stop the recursion*/
    template <typename Operator, typename First>
    GT_FUNCTION static constexpr auto accumulate(Operator, First const &first) {
        return first;
    }

    /** @} */
    /** @} */
    /** @} */

} // namespace gridtools
