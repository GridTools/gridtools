#include "gtest/gtest.h"

#include <stencil-composition/stencil-composition.hpp>

using namespace gridtools;
using namespace enumtype;

namespace el_test{

    using backend_t = ::gridtools::backend< Host, icosahedral, Naive >;
    using icosahedral_topology_t = ::gridtools::icosahedral_topology<backend_t>;

    typedef gridtools::interval<level<0,-1>, level<1,-1> > x_interval;
    typedef gridtools::interval<level<0,-2>, level<1,1> > axis;

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

    auto esf1 = gridtools::make_esf<test_functor, icosahedral_topology_t, icosahedral_topology_t::cells>(
        p_in_cells() );

    auto esf2 = gridtools::make_esf<test_functor, icosahedral_topology_t, icosahedral_topology_t::vertexes>(
        p_in_cells() );

    auto esf3 = gridtools::make_esf<test_functor, icosahedral_topology_t, icosahedral_topology_t::cells>(
        p_out_cells() );

    auto esf4 = gridtools::make_esf<test_functor, icosahedral_topology_t, icosahedral_topology_t::vertexes>(
        p_in_cells() );

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
