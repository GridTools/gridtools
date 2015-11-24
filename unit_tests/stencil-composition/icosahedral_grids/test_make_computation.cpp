/*
 * test_computation.cpp
 *
 *  Created on: Mar 9, 2015
 *      Author: carlosos
 */

#define BOOST_NO_CXX11_RVALUE_REFERENCES

#include <gridtools.hpp>
#include <boost/mpl/equal.hpp>
#include <boost/fusion/include/make_vector.hpp>

#include "gtest/gtest.h"

#include <stencil-composition/stencil-composition.hpp>
#include <stencil-composition/domain_type.hpp>

#ifdef CXX11_ENABLED

using namespace gridtools;

namespace make_computation_test{

    typedef gridtools::interval<level<0,-1>, level<1,-1> > x_interval;

    struct test_functor {
        typedef ro_accessor<0, radius<1> > in;
        typedef boost::mpl::vector1<in> arg_list;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {}
    };
}

TEST(test_make_computation, get_mss_array) {

    using namespace gridtools;

    using backend_t = backend<enumtype::Host, enumtype::Block >;

    typedef gridtools::layout_map<2,1,0> layout_t;
    using icosahedral_topology_t = gridtools::icosahedral_topology<backend_t>;

    using cell_storage_type = typename backend_t::storage_t<icosahedral_topology_t::cells, double>;

    typedef arg<0, cell_storage_type> in_cells;
    typedef arg<1, cell_storage_type> out_cells;

    typedef boost::mpl::vector<in_cells, out_cells> accessor_list_t;

    typedef decltype(
        gridtools::make_mss // mss_descriptor
        (
                enumtype::execute<enumtype::forward>(),
                gridtools::make_esf<make_computation_test::test_functor, icosahedral_topology_t, icosahedral_topology_t::cells>(in_cells())
        )) mss1_t;

    typedef decltype(
        gridtools::make_mss // mss_descriptor
        (
                enumtype::execute<enumtype::forward>(),
                gridtools::make_esf<make_computation_test::test_functor, icosahedral_topology_t, icosahedral_topology_t::cells>(out_cells())
        )) mss2_t;

    typedef gridtools::interval<level<0,-2>, level<1,1> > axis_t;
    typedef gridtools::coordinates<axis_t, icosahedral_topology_t> coords_t;

    typedef gridtools::domain_type<accessor_list_t> domain_t;
    typedef boost::mpl::vector5<int, domain_t, mss2_t, coords_t, mss1_t> ListTypes;

    typedef _impl::get_mss_array<ListTypes>::type MssArray;

    BOOST_STATIC_ASSERT(( boost::mpl::equal<MssArray::elements, boost::mpl::vector2<mss2_t, mss1_t> >::value));
    EXPECT_TRUE(true);
}

#endif
