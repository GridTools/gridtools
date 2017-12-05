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
/*
 * test_local_domain.cpp
 *
 *  Created on: Apr 9, 2015
 *      Author: carlosos
 */

#include <gridtools.hpp>
#include <boost/mpl/equal.hpp>
#include <boost/fusion/include/make_vector.hpp>

#include "gtest/gtest.h"

#include <stencil-composition/stencil-composition.hpp>

#include "backend_select.hpp"

using namespace gridtools;
using gridtools::level;
using gridtools::accessor;
using gridtools::extent;
using gridtools::arg;

using namespace gridtools;
using namespace enumtype;

namespace local_domain_stencil {
    // These are the stencil operators that compose the multistage stencil in this test
    struct dummy_functor {
        typedef accessor< 0, gridtools::enumtype::inout > in;
        typedef accessor< 1 > out;
        typedef boost::mpl::vector< in, out > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval) {}
    };

    std::ostream &operator<<(std::ostream &s, dummy_functor const) { return s << "dummy_function"; }
}

// helper function to check intermediate types for backends which fuse esfs
template < typename Intermediate, typename PIn, typename PBuff, typename POut >
typename boost::enable_if< typename backend_traits_from_id<
    intermediate_backend< Intermediate >::type::s_backend_id >::mss_fuse_esfs_strategy >::type
check_intermediate() {
    typedef typename intermediate_backend< Intermediate >::type backend_t;
    typedef typename intermediate_aggregator_type< Intermediate >::type domain_t;
    typedef typename intermediate_mss_components_array< Intermediate >::type mss_components_array_t;

    typedef typename mss_components_array_t::elements mss_elements_t;

    typedef typename intermediate_mss_local_domains< Intermediate >::type mss_local_domains_t;

    BOOST_STATIC_ASSERT((boost::mpl::size< mss_local_domains_t >::value == 1));

    typedef typename boost::mpl::front< mss_local_domains_t >::type mss_local_domain1_t;

    BOOST_STATIC_ASSERT(
        (boost::mpl::size< typename mss_local_domain1_t::unfused_local_domain_sequence_t >::value == 2));
    BOOST_STATIC_ASSERT((boost::mpl::size< typename mss_local_domain1_t::fused_local_domain_sequence_t >::value == 1));

    // the merged local domain should contain the args used by all the esfs
    BOOST_STATIC_ASSERT(
        (boost::mpl::equal< typename local_domain_esf_args< typename boost::mpl::front<
                                typename mss_local_domain1_t::unfused_local_domain_sequence_t >::type >::type,
            boost::mpl::vector2< PIn, PBuff > >::value));

    BOOST_STATIC_ASSERT(
        (boost::mpl::equal< typename local_domain_esf_args< typename boost::mpl::back<
                                typename mss_local_domain1_t::unfused_local_domain_sequence_t >::type >::type,
            boost::mpl::vector2< PBuff, POut > >::value));

    BOOST_STATIC_ASSERT(
        (boost::mpl::equal< typename local_domain_esf_args< typename boost::mpl::front<
                                typename mss_local_domain1_t::fused_local_domain_sequence_t >::type >::type,
            boost::mpl::vector3< PIn, PBuff, POut > >::value));
}

// helper function to check intermediate types for backends which do not fuse esfs
template < typename Intermediate, typename PIn, typename PBuff, typename POut >
typename boost::disable_if< typename backend_traits_from_id<
    intermediate_backend< Intermediate >::type::s_backend_id >::mss_fuse_esfs_strategy >::type
check_intermediate() {
    typedef typename intermediate_backend< Intermediate >::type backend_t;
    typedef typename intermediate_aggregator_type< Intermediate >::type domain_t;
    typedef typename intermediate_mss_components_array< Intermediate >::type mss_components_array_t;

    typedef typename mss_components_array_t::elements mss_elements_t;

    typedef typename intermediate_mss_local_domains< Intermediate >::type mss_local_domains_t;

    BOOST_STATIC_ASSERT((boost::mpl::size< mss_local_domains_t >::value == 2));

    typedef typename boost::mpl::front< mss_local_domains_t >::type mss_local_domain1_t;

    BOOST_STATIC_ASSERT(
        (boost::mpl::size< typename mss_local_domain1_t::unfused_local_domain_sequence_t >::value == 1));
    BOOST_STATIC_ASSERT((boost::mpl::size< typename mss_local_domain1_t::fused_local_domain_sequence_t >::value == 1));

    // the merged local domain should contain the args used by all the esfs
    BOOST_STATIC_ASSERT(
        (boost::mpl::equal< typename local_domain_esf_args< typename boost::mpl::front<
                                typename mss_local_domain1_t::unfused_local_domain_sequence_t >::type >::type,
            boost::mpl::vector2< PIn, PBuff > >::value));

    typedef typename boost::mpl::at< mss_local_domains_t, boost::mpl::int_< 1 > >::type mss_local_domain2_t;

    BOOST_STATIC_ASSERT(
        (boost::mpl::size< typename mss_local_domain2_t::unfused_local_domain_sequence_t >::value == 1));
    BOOST_STATIC_ASSERT((boost::mpl::size< typename mss_local_domain2_t::fused_local_domain_sequence_t >::value == 1));

    // the merged local domain should contain the args used by all the esfs
    BOOST_STATIC_ASSERT(
        (boost::mpl::equal< typename local_domain_esf_args< typename boost::mpl::front<
                                typename mss_local_domain2_t::unfused_local_domain_sequence_t >::type >::type,
            boost::mpl::vector2< PBuff, POut > >::value));
}

TEST(test_local_domain, merge_mss_local_domains) {
    using namespace local_domain_stencil;

    typedef gridtools::layout_map< 2, 1, 0 > layout_ijk_t;
    typedef gridtools::layout_map< 0, 1, 2 > layout_kji_t;
    typedef gridtools::storage_traits< backend_t::s_backend_id >::custom_layout_storage_info_t< 0, layout_ijk_t >
        meta_ijk_t;
    typedef gridtools::storage_traits< backend_t::s_backend_id >::custom_layout_storage_info_t< 0, layout_kji_t >
        meta_kji_t;
    typedef gridtools::storage_traits< backend_t::s_backend_id >::data_store_t< float_type, meta_ijk_t > storage_t;
    typedef gridtools::storage_traits< backend_t::s_backend_id >::data_store_t< float_type, meta_kji_t > storage_buff_t;

    typedef arg< 0, storage_t > p_in;
    typedef arg< 1, storage_buff_t > p_buff;
    typedef arg< 2, storage_t > p_out;
    typedef boost::mpl::vector< p_in, p_buff, p_out > accessor_list;

    uint_t d1 = 1;
    uint_t d2 = 1;
    uint_t d3 = 1;

    meta_ijk_t meta_ijk(d1, d2, d3);
    storage_t in(meta_ijk, -3.5);
    meta_kji_t meta_kji(d1, d2, d3);
    storage_buff_t buff(meta_kji, 1.5);
    storage_t out(meta_ijk, 1.5);

    gridtools::aggregator_type< accessor_list > domain(in, buff, out);

    auto grid = gridtools::make_grid(d1, d2, d3);

    typedef intermediate< backend_t,
        meta_array< boost::mpl::vector< decltype(gridtools::make_multistage // mss_descriptor
                        (execute< forward >(),
                            gridtools::make_stage< local_domain_stencil::dummy_functor >(p_in(), p_buff()),
                            gridtools::make_stage< local_domain_stencil::dummy_functor >(p_buff(), p_out()))) >,
                              boost::mpl::quote1< gridtools::is_computation_token > >,
        gridtools::aggregator_type< accessor_list >,
        gridtools::grid< gridtools::axis< 1 >::axis_interval_t >,
        boost::fusion::set<>,
        gridtools::notype,
        false > intermediate_t;

    check_intermediate<intermediate_t, p_in, p_buff, p_out>();
}
