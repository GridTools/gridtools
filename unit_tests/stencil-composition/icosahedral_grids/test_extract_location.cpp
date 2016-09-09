/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
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
#include "gtest/gtest.h"

#include <stencil-composition/stencil-composition.hpp>

using namespace gridtools;
using namespace enumtype;

namespace el_test{

    using backend_t = ::gridtools::backend< Host, icosahedral, Naive >;
    using icosahedral_topology_t = ::gridtools::icosahedral_topology<backend_t>;

    typedef gridtools::interval<level<0,-1>, level<1,-1> > x_interval;
    typedef gridtools::interval<level<0,-2>, level<1,1> > axis;

    template < uint_t Color >
    struct test_functor {
        typedef in_accessor< 0, icosahedral_topology_t::cells, extent< 1 > > in;
        typedef boost::mpl::vector<in> arg_list;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {}
    };
}

using namespace el_test;
TEST(extract_location, test) {

    using cell_storage_type = typename backend_t::storage_t<icosahedral_topology_t::cells, double>;

    typedef arg<0, cell_storage_type> p_in_cells;
    typedef arg<1, cell_storage_type> p_out_cells;

    auto esf1 =
        gridtools::make_stage< test_functor, icosahedral_topology_t, icosahedral_topology_t::cells >(p_in_cells());

    auto esf2 =
        gridtools::make_stage< test_functor, icosahedral_topology_t, icosahedral_topology_t::vertexes >(p_in_cells());

    auto esf3 =
        gridtools::make_stage< test_functor, icosahedral_topology_t, icosahedral_topology_t::cells >(p_out_cells());

    auto esf4 =
        gridtools::make_stage< test_functor, icosahedral_topology_t, icosahedral_topology_t::vertexes >(p_in_cells());

    using esf1_t = decltype(esf1);
    using esf2_t = decltype(esf2);
    using esf3_t = decltype(esf3);
    using esf4_t = decltype(esf4);

    using cell_location_t = extract_location_type<boost::mpl::vector2<esf1_t, esf3_t> >::type;
    using vertex_location_t = extract_location_type<boost::mpl::vector2<esf2_t, esf4_t> >::type;

    GRIDTOOLS_STATIC_ASSERT((boost::is_same<cell_location_t, icosahedral_topology_t::cells>::value), "Error: wrong location type");
    GRIDTOOLS_STATIC_ASSERT((boost::is_same<vertex_location_t, icosahedral_topology_t::vertexes>::value), "Error: wrong location type");

    ASSERT_TRUE(true);
}
