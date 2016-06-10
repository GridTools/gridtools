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

#include <stencil_composition/stencil_composition.hpp>
#include <stencil_composition/aggregator_type.hpp>

#ifdef CXX11_ENABLED

using namespace gridtools;

namespace make_computation_test{

    typedef gridtools::interval<level<0,-1>, level<1,-1> > x_interval;
    using backend_t = backend< enumtype::Host, enumtype::icosahedral, enumtype::Block >;
    using icosahedral_topology_t = gridtools::icosahedral_topology<backend_t>;

    struct test_functor {
        using in = in_accessor< 0, icosahedral_topology_t::cells, extent< 1 > >;
        using arg_list = boost::mpl::vector1<in>;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {}
    };
}

#endif
