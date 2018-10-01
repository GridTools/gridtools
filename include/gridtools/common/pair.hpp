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

#include <type_traits>
#include <utility>

#include "defs.hpp"
#include "host_device.hpp"

namespace gridtools {
    /** \ingroup common
        @{
        \defgroup pair Simple Pair
        @{
    */

    /**
       @brief simple pair with constexpr constructor

       NOTE: can be replaced by std::pair
     */
    template <typename T1, typename T2>
    struct pair {
        pair() = default;

        template <class U1, class U2>
        constexpr GT_FUNCTION pair(const std::pair<U1, U2> &p) : pair(p.first, p.second) {}

        template <class U1, class U2>
        constexpr GT_FUNCTION pair(std::pair<U1, U2> &&p) : pair(std::move(p.first), std::move(p.second)) {}

        template <class U1, class U2>
        constexpr GT_FUNCTION pair(U1 &&t1_, U2 &&t2_) : first(std::forward<U1>(t1_)), second(std::forward<U2>(t2_)) {}

        template <class U1, class U2, typename std::enable_if<!std::is_same<pair<U1, U2>, pair>::value, int>::type = 0>
        constexpr GT_FUNCTION pair(const pair<U1, U2> &p) : first(p.first), second(p.second) {}

        template <class U1, class U2, typename std::enable_if<!std::is_same<pair<U1, U2>, pair>::value, int>::type = 0>
        constexpr GT_FUNCTION pair(pair<U1, U2> &&p) : first(std::move(p.first)), second(std::move(p.second)) {}

        template <typename U1,
            typename U2,
            typename std::enable_if<!std::is_same<pair<U1, U2>, pair>::value, int>::type = 0>
        GT_FUNCTION pair &operator=(const pair<U1, U2> &other) {
            first = other.first;
            second = other.second;
            return *this;
        }

        template <typename U1,
            typename U2,
            typename std::enable_if<!std::is_same<pair<U1, U2>, pair>::value, int>::type = 0>
        GT_FUNCTION pair &operator=(pair<U1, U2> &&other) noexcept {
            first = std::move(other.first);
            second = std::move(other.second);
            return *this;
        }

        T1 first;
        T2 second;
    };

    template <typename T1, typename T2>
    constexpr GT_FUNCTION bool operator==(const pair<T1, T2> &lhs, const pair<T1, T2> &rhs) {
        return lhs.first == rhs.first && lhs.second == rhs.second;
    }

    template <typename T1, typename T2>
    constexpr GT_FUNCTION bool operator!=(const pair<T1, T2> &lhs, const pair<T1, T2> &rhs) {
        return !(lhs == rhs);
    }

    template <typename T1, typename T2>
    constexpr GT_FUNCTION bool operator<(const pair<T1, T2> &lhs, const pair<T1, T2> &rhs) {
        return lhs.first < rhs.first || (!(rhs.first < lhs.first) && lhs.second < rhs.second);
    }

    template <typename T1, typename T2>
    constexpr GT_FUNCTION bool operator>(const pair<T1, T2> &lhs, const pair<T1, T2> &rhs) {
        return rhs < lhs;
    }

    template <typename T1, typename T2>
    constexpr GT_FUNCTION bool operator<=(const pair<T1, T2> &lhs, const pair<T1, T2> &rhs) {
        return !(rhs < lhs);
    }

    template <typename T1, typename T2>
    constexpr GT_FUNCTION bool operator>=(const pair<T1, T2> &lhs, const pair<T1, T2> &rhs) {
        return !(lhs < rhs);
    }

    template <typename T1, typename T2>
    constexpr GT_FUNCTION pair<T1, T2> make_pair(T1 t1_, T2 t2_) {
        return pair<T1, T2>(t1_, t2_);
    }
    /** @} */
    /** @} */

    template <typename T>
    struct tuple_size;

    template <typename T1, typename T2>
    struct tuple_size<pair<T1, T2>> : std::integral_constant<size_t, 2> {};

    template <size_t I, typename T>
    struct tuple_element;

    template <typename T1, typename T2>
    struct tuple_element<0, pair<T1, T2>> {
        using type = T1;
    };

    template <typename T1, typename T2>
    struct tuple_element<1, pair<T1, T2>> {
        using type = T2;
    };

    namespace impl_ {
        template <size_t I>
        struct pair_get;

        template <>
        struct pair_get<0> {
            template <typename T1, typename T2>
            static constexpr GT_FUNCTION const T1 &const_get(const pair<T1, T2> &p) noexcept {
                return p.first;
            }
            template <typename T1, typename T2>
            static constexpr GT_FUNCTION T1 &get(pair<T1, T2> &p) noexcept {
                return p.first;
            }
            template <typename T1, typename T2>
            static constexpr GT_FUNCTION T1 &&move_get(pair<T1, T2> &&p) noexcept {
                return std::move(p.first);
            }
        };
        template <>
        struct pair_get<1> {
            template <typename T1, typename T2>
            static constexpr GT_FUNCTION const T2 &const_get(const pair<T1, T2> &p) noexcept {
                return p.second;
            }
            template <typename T1, typename T2>
            static constexpr GT_FUNCTION T2 &get(pair<T1, T2> &p) noexcept {
                return p.second;
            }
            template <typename T1, typename T2>
            static constexpr GT_FUNCTION T2 &&move_get(pair<T1, T2> &&p) noexcept {
                return std::move(p.second);
            }
        };
    } // namespace impl_

    template <size_t I, class T1, class T2>
    constexpr GT_FUNCTION auto get(pair<T1, T2> &p) noexcept GT_AUTO_RETURN(impl_::pair_get<I>::get(p));

    template <size_t I, class T1, class T2>
    constexpr GT_FUNCTION auto get(const pair<T1, T2> &p) noexcept GT_AUTO_RETURN(impl_::pair_get<I>::const_get(p));

    template <size_t I, class T1, class T2>
    constexpr GT_FUNCTION auto get(pair<T1, T2> &&p) noexcept GT_AUTO_RETURN(
        impl_::pair_get<I>::move_get(std::move(p)));

} // namespace gridtools
