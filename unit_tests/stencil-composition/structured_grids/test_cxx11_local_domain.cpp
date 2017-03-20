/*
  GridTools Libraries

  Copyright (c) 2017, GridTools Consortium
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
/*
 * test_local_domain.cpp
 *
 *  Created on: Apr 9, 2015
 *      Author: carlosos
 */

//#define BOOST_NO_CXX11_RVALUE_REFERENCES

#include <gridtools.hpp>
#include <boost/mpl/equal.hpp>
#include <boost/fusion/include/make_vector.hpp>

#include "gtest/gtest.h"

#include <stencil-composition/stencil-composition.hpp>

using namespace gridtools;
using gridtools::level;
using gridtools::accessor;
using gridtools::extent;
using gridtools::arg;

using namespace gridtools;
using namespace enumtype;

namespace local_domain_stencil {
    // This is the definition of the special regions in the "vertical" direction
    typedef gridtools::interval< level< 0, -1 >, level< 1, -1 > > x_interval;
    typedef gridtools::interval< level< 0, -2 >, level< 1, 1 > > axis;

    // These are the stencil operators that compose the multistage stencil in this test
    struct dummy_functor {
        typedef accessor< 0, gridtools::enumtype::inout > in;
        typedef accessor< 1 > out;
        typedef boost::mpl::vector< in, out > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {}
    };

    std::ostream &operator<<(std::ostream &s, dummy_functor const) { return s << "dummy_function"; }
}

TEST(test_local_domain, merge_mss_local_domains) {
    using namespace local_domain_stencil;

    typedef layout_map< 2, 1, 0 > layout_ijk_t;
    typedef layout_map< 0, 1, 2 > layout_kji_t;
    typedef backend< Host, GRIDBACKEND, Naive >::storage_info< 0, layout_ijk_t > meta_ijk_t;
    typedef backend< Host, GRIDBACKEND, Naive >::storage_info< 0, layout_kji_t > meta_kji_t;
    typedef backend< Host, GRIDBACKEND, Naive >::storage_type< float_type, meta_ijk_t >::type storage_type;
    typedef backend< Host, GRIDBACKEND, Naive >::storage_type< float_type, meta_kji_t >::type storage_buff_type;

    typedef arg< 0, storage_type > p_in;
    typedef arg< 1, storage_buff_type > p_buff;
    typedef arg< 2, storage_type > p_out;
    typedef boost::mpl::vector< p_in, p_buff, p_out > accessor_list;

    uint_t d1 = 1;
    uint_t d2 = 1;
    uint_t d3 = 1;

    meta_ijk_t meta_ijk(gridtools::array< uint_t, 3 >{d1, d2, d3});
    storage_type in(meta_ijk, -3.5, "in");
    meta_kji_t meta_kji(gridtools::array< uint_t, 3 >{d1, d2, d3});
    storage_buff_type buff(meta_kji, 1.5, "buff");
    storage_type out(meta_ijk, 1.5, "out");

    gridtools::aggregator_type< accessor_list > domain((p_in() = in), (p_buff() = buff), (p_out() = out));

    uint_t di[5] = {0, 0, 0, d1 - 1, d1};
    uint_t dj[5] = {0, 0, 0, d2 - 1, d2};

    gridtools::grid< local_domain_stencil::axis > grid(di, dj);
    grid.value_list[0] = 0;
    grid.value_list[1] = d3 - 1;

    typedef intermediate< gridtools::backend< Host, GRIDBACKEND, Naive >,
        meta_array< boost::mpl::vector< decltype(gridtools::make_multistage // mss_descriptor
                        (execute< forward >(),
                            gridtools::make_stage< local_domain_stencil::dummy_functor >(p_in(), p_buff()),
                            gridtools::make_stage< local_domain_stencil::dummy_functor >(p_buff(), p_out()))) >,
                              boost::mpl::quote1< gridtools::is_computation_token > >,
        gridtools::aggregator_type< accessor_list >,
        gridtools::grid< local_domain_stencil::axis >,
        boost::fusion::set<>,
        gridtools::notype,
        false > intermediate_t;

    typedef intermediate_backend< intermediate_t >::type backend_t;
    typedef intermediate_aggregator_type< intermediate_t >::type domain_t;
    typedef intermediate_mss_components_array< intermediate_t >::type mss_components_array_t;

    typedef mss_components_array_t::elements mss_elements_t;

    typedef intermediate_mss_local_domains< intermediate_t >::type mss_local_domains_t;

    BOOST_STATIC_ASSERT((boost::mpl::size< mss_local_domains_t >::value == 2));

    typedef boost::mpl::front< mss_local_domains_t >::type mss_local_domain1_t;

    BOOST_STATIC_ASSERT((boost::mpl::size< mss_local_domain1_t::unfused_local_domain_sequence_t >::value == 1));
    BOOST_STATIC_ASSERT((boost::mpl::size< mss_local_domain1_t::fused_local_domain_sequence_t >::value == 1));

    // the merged local domain should contain the args used by all the esfs
    BOOST_STATIC_ASSERT((boost::mpl::equal<
        local_domain_esf_args< boost::mpl::front< mss_local_domain1_t::unfused_local_domain_sequence_t >::type >::type,
        boost::mpl::vector2< p_in, p_buff > >::value));

    typedef boost::mpl::at< mss_local_domains_t, boost::mpl::int_< 1 > >::type mss_local_domain2_t;

    BOOST_STATIC_ASSERT((boost::mpl::size< mss_local_domain2_t::unfused_local_domain_sequence_t >::value == 1));
    BOOST_STATIC_ASSERT((boost::mpl::size< mss_local_domain2_t::fused_local_domain_sequence_t >::value == 1));

    // the merged local domain should contain the args used by all the esfs
    BOOST_STATIC_ASSERT((boost::mpl::equal<
        local_domain_esf_args< boost::mpl::front< mss_local_domain2_t::unfused_local_domain_sequence_t >::type >::type,
        boost::mpl::vector2< p_buff, p_out > >::value));

    EXPECT_TRUE(true);
}
