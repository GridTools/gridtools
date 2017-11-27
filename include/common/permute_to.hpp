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

#include <boost/fusion/include/is_sequence.hpp>
#include <boost/fusion/include/mpl.hpp>
#include <boost/fusion/include/nview.hpp>

#include <boost/mpl/back_inserter.hpp>
#include <boost/mpl/begin.hpp>
#include <boost/mpl/count_if.hpp>
#include <boost/mpl/distance.hpp>
#include <boost/mpl/equal_to.hpp>
#include <boost/mpl/find.hpp>
#include <boost/mpl/size.hpp>
#include <boost/mpl/transform.hpp>
#include <boost/mpl/vector_c.hpp>

#include "defs.hpp"

namespace gridtools {
    namespace impl_ {
        template < typename Sec, typename T >
        struct get_position {
            using type = typename boost::mpl::distance< typename boost::mpl::begin< Sec >::type,
                typename boost::mpl::find< Sec, T >::type >::type;
        };
    }

    /** For each type in Res find the element in src of the same type, place those elements in correct order and
     *  construct the Res instance from them.
     *
     *  This utility is handy when we have all elements of the Res, but not in the right order.
     *
     *  Requirments:
     *      - Res and Src should model fusion sequence;
     *      - Res type should have a ctor from a fusion sequence;
     *      - all types from the Res should present in the Src;
     *
     *  Example:
     *      auto what_we_have = boost::fusion::make_vector(42, 80, 'a', .1, "other_stuff", 79, .4);
     *      using what_we_need_t = boost::fusion<char, double, int>;
     *      what_we_need_t expected {'a', .1, 42};
     *      auto actual = permute_to<what_we_need_t>(what_we_have);
     *      EXPECT_EQ(actual, expected);
     */
    template < typename Res, typename Src >
    Res permute_to(Src &&src) {
        namespace f = boost::fusion;
        namespace m = boost::mpl;
        using src_t = typename std::decay< Src >::type;
        GRIDTOOLS_STATIC_ASSERT(f::traits::is_sequence< Res >::value, "Output type should model fusion sequence.");
        GRIDTOOLS_STATIC_ASSERT(f::traits::is_sequence< src_t >::value, "Input type should model fusion sequence.");
        using positions_t = typename m::transform< Res,
            impl_::get_position< src_t, m::_ >,
            m::back_inserter< m::vector_c< int > > >::type;
        GRIDTOOLS_STATIC_ASSERT((m::count_if< positions_t, m::equal_to< m::_, m::size< src_t > > >::value == 0),
            "All types from the result should present in the source.");
        return Res{f::nview< typename std::remove_reference< Src >::type, positions_t >(src)};
    };
}
