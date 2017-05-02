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

#include "layout_map_cxx11.hpp"

namespace gridtools {
    template < typename LayoutMap >
    struct reverse_map;

    template < short_t... Is >
    struct reverse_map< layout_map< Is... > > {
        template < short_t I, short_t Max >
        struct new_value {
            static const short_t value = (Max - I) > Max ? I : Max - I;
        };

        template < typename Current, typename Next >
        struct get_max {
            using type = boost::mpl::int_< (Current::value > Next::value) ? Current::value : Next::value >;
        };

        using max = typename boost::mpl::fold< typename layout_map< Is... >::layout_vector_t,
            boost::mpl::int_< -1 >,
            get_max< boost::mpl::_1, boost::mpl::_2 > >::type;

        typedef layout_map< new_value< Is, max::value >::value... > type;
    };

    template < typename DATALO, typename PROCLO >
    struct layout_transform;

    template < short_t I1, short_t I2, short_t P1, short_t P2 >
    struct layout_transform< layout_map< I1, I2 >, layout_map< P1, P2 > > {
        typedef layout_map< I1, I2 > L1;
        typedef layout_map< P1, P2 > L2;

        static const short_t N1 = boost::mpl::at_c< typename L1::layout_vector_t, P1 >::type::value;
        static const short_t N2 = boost::mpl::at_c< typename L1::layout_vector_t, P2 >::type::value;

        typedef layout_map< N1, N2 > type;
    };

    template < short_t I1, short_t I2, short_t I3, short_t P1, short_t P2, short_t P3 >
    struct layout_transform< layout_map< I1, I2, I3 >, layout_map< P1, P2, P3 > > {
        typedef layout_map< I1, I2, I3 > L1;
        typedef layout_map< P1, P2, P3 > L2;

        static const short_t N1 = boost::mpl::at_c< typename L1::layout_vector_t, P1 >::type::value;
        static const short_t N2 = boost::mpl::at_c< typename L1::layout_vector_t, P2 >::type::value;
        static const short_t N3 = boost::mpl::at_c< typename L1::layout_vector_t, P3 >::type::value;

        typedef layout_map< N1, N2, N3 > type;
    };

    template < short_t D >
    struct default_layout_map;

    template <>
    struct default_layout_map< 1 > {
        typedef layout_map< 0 > type;
    };

    template <>
    struct default_layout_map< 2 > {
        typedef layout_map< 0, 1 > type;
    };

    template <>
    struct default_layout_map< 3 > {
        typedef layout_map< 0, 1, 2 > type;
    };

    template <>
    struct default_layout_map< 4 > {
        typedef layout_map< 0, 1, 2, 3 > type;
    };
} // namespace gridtools
