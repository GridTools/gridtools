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
#include "stencil-composition/backend.hpp"
#include "stencil-composition/make_computation.hpp"
#include "stencil-composition/make_stencils.hpp"
#include "stencil-composition/reductions/reductions.hpp"

using namespace gridtools;
using namespace enumtype;

namespace make_reduction_test{

    typedef gridtools::interval<level<0,-1>, level<1,-1> > x_interval;

    struct test_functor {
        typedef accessor<0> in;
        typedef boost::mpl::vector1<in> arg_list;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {}
    };
}

TEST(test_make_reduction, make_reduction) {

    using namespace gridtools;

    #define BACKEND backend<enumtype::Host, enumtype::structured, enumtype::Block >

    typedef gridtools::layout_map<2,1,0> layout_t;
    typedef gridtools::BACKEND::storage_type<float_type, gridtools::BACKEND::storage_info<0,layout_t> >::type storage_type;

    typedef arg<0, storage_type> p_in;
    typedef arg<1, storage_type> p_out;
    typedef boost::mpl::vector<p_in, p_out> accessor_list_t;

    typedef decltype(
        gridtools::make_reduction<make_reduction_test::test_functor, binop::sum>(0.0, p_in())
    ) red_t;

    typedef reduction_descriptor<
        double,
        binop::sum,
        boost::mpl::vector1<
            esf_descriptor<make_reduction_test::test_functor, boost::mpl::vector1<p_in> >
        >
    > red_ref_t;

    GRIDTOOLS_STATIC_ASSERT((red_t::is_reduction_t::value), "ERROR");
    GRIDTOOLS_STATIC_ASSERT((boost::mpl::equal<red_t::esf_sequence_t, red_ref_t::esf_sequence_t,
                             esf_equal<boost::mpl::_1, boost::mpl::_2> >::type::value), "ERROR");

    EXPECT_TRUE(true);
}
