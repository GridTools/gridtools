/*
 * test_computation.cpp
 *
 *  Created on: Mar 9, 2015
 *      Author: carlosos
 */

#define BOOST_NO_CXX11_RVALUE_REFERENCES

#include <gridtools.h>
#include <boost/mpl/equal.hpp>

#include "gtest/gtest.h"

#ifdef CUDA_EXAMPLE
#include <stencil-composition/backend_cuda.h>
#else
#include <stencil-composition/backend_host.h>
#endif

#ifdef CXX11_ENABLED

using namespace gridtools;

namespace make_computation_test{

typedef gridtools::interval<level<0,-1>, level<1,-1> > x_interval;

struct test_functor {
    typedef const arg_type<0> in;
    typedef boost::mpl::vector1<in> arg_list;

    template <typename Evaluation>
    GT_FUNCTION
    static void Do(Evaluation const & eval, x_interval) {}
};
}

TEST(test_make_computation, get_mss_array) {

    using namespace gridtools;

    #define BACKEND backend<enumtype::Host, enumtype::Block >

    typedef gridtools::layout_map<2,1,0> layout_t;
    typedef gridtools::BACKEND::storage_type<float_type, layout_t >::type storage_type;

    typedef arg<0, storage_type> p_in;
    typedef arg<1, storage_type> p_out;
    typedef boost::mpl::vector<p_in, p_out> arg_type_list_t;

    typedef decltype(
        gridtools::make_mss // mss_descriptor
        (
                enumtype::execute<enumtype::forward>(),
                gridtools::make_esf<make_computation_test::test_functor>(p_in())
        )) mss1_t;

    typedef decltype(
        gridtools::make_mss // mss_descriptor
        (
                enumtype::execute<enumtype::forward>(),
                gridtools::make_esf<make_computation_test::test_functor>(p_in())
        )) mss2_t;

    typedef gridtools::interval<level<0,-2>, level<1,1> > axis_t;
    typedef gridtools::coordinates<axis_t> coords_t;

    typedef gridtools::domain_type<arg_type_list_t> domain_t;
    typedef boost::mpl::vector5<int, domain_t, mss2_t, coords_t, mss1_t> ListTypes;

    typedef _impl::get_mss_array<ListTypes>::type MssArray;

    BOOST_STATIC_ASSERT(( boost::mpl::equal<MssArray::elements_t, boost::mpl::vector2<mss2_t, mss1_t> >::value));
    EXPECT_TRUE(true);
}

#endif
