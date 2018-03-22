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

#include "defs.hpp"
#include "host_device.hpp"

namespace gridtools {
    /**
       @brief simple wrapper for a pair of types
     */
    template < typename T, typename U >
    struct pair_type {
        typedef T first;
        typedef U second;
    };

    /**
       @brief simple wrapper for a pair of integral types
     */
    template < typename Value, Value T, Value U >
    struct ipair_type {
        static constexpr Value first = T;
        static constexpr Value second = U;
    };

    /**
       @brief simple pair with constexpr constructor

       NOTE: can be replaced by std::pair
     */
    template < typename T1, typename T2 >
    struct pair {
        constexpr GT_FUNCTION pair(T1 t1_, T2 t2_) : first(t1_), second(t2_) {}
        constexpr GT_FUNCTION pair() : first(), second() {}

        T1 first;
        T2 second;
    };

    template < typename T1, typename T2 >
    constexpr GT_FUNCTION bool operator==(const pair< T1, T2 > &lhs, const pair< T1, T2 > &rhs) {
        return lhs.first == rhs.first && lhs.second == rhs.second;
    }

    template < typename T1, typename T2 >
    constexpr GT_FUNCTION bool operator!=(const pair< T1, T2 > &lhs, const pair< T1, T2 > &rhs) {
        return !(lhs == rhs);
    }

    template < typename T1, typename T2 >
    constexpr pair< T1, T2 > make_pair(T1 t1_, T2 t2_) {
        return pair< T1, T2 >(t1_, t2_);
    }

    template < typename T >
    class tuple_size;

    template < typename T1, typename T2 >
    class tuple_size< pair< T1, T2 > > : public gridtools::static_size_t< 2 > {};

    namespace impl_ {
        template < size_t I >
        struct pair_get;

        template <>
        struct pair_get< 0 > {
            template < typename T1, typename T2 >
            static constexpr const T1 &const_get(const pair< T1, T2 > &p) noexcept {
                return p.first;
            }
        };
        template <>
        struct pair_get< 1 > {
            template < typename T1, typename T2 >
            static constexpr const T2 &const_get(const pair< T1, T2 > &p) noexcept {
                return p.second;
            }
        };
    }

    template < size_t I, class T1, class T2 >
    constexpr auto get(const pair< T1, T2 > &p) noexcept GT_AUTO_RETURN(impl_::pair_get< I >::const_get(p));

} // namespace gridtools
