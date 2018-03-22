/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/
#pragma once
#include <boost/mpl/at.hpp>
#include "../defs.hpp"
#include "binary_ops.hpp"
#include <algorithm>

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
    template < typename... Ts >
    GT_FUNCTION constexpr typename std::common_type< Ts... >::type constexpr_max(Ts... vals) {
        return std::max({vals...});
    }
    template < typename... Ts >
    GT_FUNCTION constexpr typename std::common_type< Ts... >::type constexpr_min(Ts... vals) {
        return std::min({vals...});
    }
#else
    template < typename T >
    GT_FUNCTION constexpr T constexpr_max(T v) {
        return v;
    }

    /**
     * @brief constexpr max of a sequence of values
     */
    template < typename T0, typename T1, typename... Ts >
    GT_FUNCTION constexpr typename std::common_type< T0, T1, Ts... >::type constexpr_max(T0 v0, T1 v1, Ts... vals) {
        return (v0 > v1) ? constexpr_max(v0, vals...) : constexpr_max(v1, vals...);
    }

    template < typename T >
    GT_FUNCTION constexpr T constexpr_min(T v) {
        return v;
    }

    /**
     * @brief constexpr min of a sequence of values
     */
    template < typename T0, typename T1, typename... Ts >
    GT_FUNCTION constexpr typename std::common_type< T0, T1, Ts... >::type constexpr_min(T0 v0, T1 v1, Ts... vals) {
        return (v0 < v1) ? constexpr_min(v0, vals...) : constexpr_min(v1, vals...);
    }
#endif

    /**@brief operation to be used inside the accumulator*/
    struct multiplies {
        GT_FUNCTION
        constexpr multiplies() {}
        template < typename T1, typename T2 >
        GT_FUNCTION constexpr T1 operator()(const T1 &x, const T2 &y) const {
            return x * y;
        }
    };

    /**@brief operation to be used inside the accumulator*/
    struct plus_functor {
        GT_FUNCTION
        constexpr plus_functor() {}
        template < class T >
        GT_FUNCTION constexpr T operator()(const T &x, const T &y) const {
            return x + y;
        }
    };

    /**@brief accumulator recursive implementation*/
    template < typename Operator, typename First, typename... Args >
    GT_FUNCTION static constexpr First accumulate(Operator op, First first, Args... args) {
        return op(first, accumulate(op, args...));
    }

    /**@brief specialization to stop the recursion*/
    template < typename Operator, typename First >
    GT_FUNCTION static constexpr First accumulate(Operator op, First first) {
        return first;
    }

    /** @} */
    /** @} */
    /** @} */

} // namespace gridtools
