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

#include <stencil-composition/iterate_on_esfs.hpp>

#include <type_traits>
#include <boost/mpl/vector.hpp>

#include <stencil-composition/accessor.hpp>
#include <stencil-composition/arg.hpp>
#include <stencil-composition/backend.hpp>
#include <stencil-composition/independent_esf.hpp>
#include <stencil-composition/make_stage.hpp>
#include <stencil-composition/make_stencils_cxx11.hpp>
#include <storage/storage-facility.hpp>

#include "gtest/gtest.h"

#ifdef __CUDACC__
#define BACKEND \
    ::gridtools::backend<::gridtools::enumtype::Cuda, ::gridtools::enumtype::GRIDBACKEND, ::gridtools::enumtype::Block >
#else
#ifdef BACKEND_BLOCK
#define BACKEND \
    ::gridtools::backend<::gridtools::enumtype::Host, ::gridtools::enumtype::GRIDBACKEND, ::gridtools::enumtype::Block >
#else
#define BACKEND \
    ::gridtools::backend<::gridtools::enumtype::Host, ::gridtools::enumtype::GRIDBACKEND, ::gridtools::enumtype::Naive >
#endif
#endif

namespace gridtools {
    namespace {
        using boost::mpl::vector;

        template < int I >
        struct functor {
            static const int thevalue = I;
            using arg_list = vector< in_accessor< 0 >, inout_accessor< 1 > >;
        };

        typedef storage_traits< BACKEND::s_backend_id >::storage_info_t< 0, 3 > storage_info_t;
        typedef storage_traits< BACKEND::s_backend_id >::data_store_t< float_type, storage_info_t > data_store_t;

        typedef arg< 0, data_store_t > p_in;
        typedef arg< 1, data_store_t > p_out;

        template < int I >
        using an_esf = decltype(make_stage< functor< I > >(p_in{}, p_out{}));

        template < class... Esfs >
        using an_independent = independent_esf< boost::mpl::vector< Esfs... > >;

        template < class... Esfs >
        using an_mss = decltype(make_multistage(enumtype::execute< enumtype::forward >(), std::declval< Esfs >()...));

        template < typename StencilOp >
        struct is_even : std::integral_constant< int, !(StencilOp::esf_function::thevalue % 2) > {};

        template < typename StencilOp >
        struct is_odd : std::integral_constant< int, !!(StencilOp::esf_function::thevalue % 2) > {};

        template < typename A, typename B >
        struct sum : std::integral_constant< int, A::value + B::value > {};

        template < typename Msses >
        using get_even = typename with_operators< is_even, sum >::iterate_on_esfs< boost::mpl::int_< 0 >, Msses >::type;

        template < typename Msses >
        using get_odd = typename with_operators< is_odd, sum >::iterate_on_esfs< boost::mpl::int_< 0 >, Msses >::type;

        using basic_t = vector< an_mss< an_esf< 0 >, an_esf< 1 >, an_esf< 2 >, an_esf< 3 >, an_esf< 4 > > >;
        static_assert(get_even< basic_t >::value == 3, "");
        static_assert(get_odd< basic_t >::value == 2, "");

        using two_multistages_t =
            vector< an_mss< an_esf< 0 >, an_esf< 1 >, an_esf< 2 > >, an_mss< an_esf< 3 >, an_esf< 4 > > >;
        static_assert(get_even< two_multistages_t >::value == 3, "");
        static_assert(get_odd< two_multistages_t >::value == 2, "");

        using two_multistages_independent_t = vector< an_mss< an_esf< 0 >, an_independent< an_esf< 1 >, an_esf< 2 > > >,
            an_mss< an_esf< 3 >, an_esf< 4 > > >;
        static_assert(get_even< two_multistages_independent_t >::value == 3, "");
        static_assert(get_odd< two_multistages_independent_t >::value == 2, "");

        using just_independent_t =
            vector< an_mss< an_independent< an_esf< 1 >, an_esf< 2 > > >, an_mss< an_esf< 4 > > >;
        static_assert(get_even< just_independent_t >::value == 2, "");
        static_assert(get_odd< just_independent_t >::value == 1, "");

        TEST(dummy, dummy) {}
    }
}
