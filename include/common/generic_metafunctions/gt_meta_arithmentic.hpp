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

namespace gridtools {

    template < typename T, typename U >
    struct gt_greater {
        using type = std::integral_constant< bool, (T::value > U::value) >;
    };

    template < typename T, typename U >
    using gt_greater_t = typename gt_greater< T, U >::type;

    template < typename T, typename U >
    struct gt_greater_equal {
        using type = std::integral_constant< bool, (T::value >= U::value) >;
    };

    template < typename T, typename U >
    using gt_greater_equal_t = typename gt_greater_equal< T, U >::type;

    template < typename T, typename U >
    struct gt_less {
        using type = std::integral_constant< bool, (T::value < U::value) >;
    };

    template < typename T, typename U >
    using gt_less_t = typename gt_less< T, U >::type;

    template < typename T, typename U >
    struct gt_less_equal {
        using type = std::integral_constant< bool, (T::value <= U::value) >;
    };

    template < typename T, typename U >
    using gt_less_equal_t = typename gt_less_equal< T, U >::type;

    template < typename T, typename U >
    struct gt_plus {
        using type = std::integral_constant< typename T::value_type, T::value + U::value >;
    };

    template < typename T, typename U >
    using gt_plus_t = typename gt_plus< T, U >::type;

    template < typename T, typename U >
    struct gt_minus {
        using type = std::integral_constant< typename T::value_type, T::value - U::value >;
    };

    template < typename T, typename U >
    using gt_minus_t = typename gt_minus< T, U >::type;

    template < typename T, typename U >
    struct gt_min {
        using type = std::integral_constant< typename T::value_type, (T::value < U::value) ? T::value : U::value >;
    };

    template < typename T, typename U >
    using gt_min_t = typename gt_min< T, U >::type;

    template < typename T, typename U >
    struct gt_max {
        using type = std::integral_constant< typename T::value_type, (T::value > U::value) ? T::value : U::value >;
    };

    template < typename T, typename U >
    using gt_max_t = typename gt_max< T, U >::type;
}
