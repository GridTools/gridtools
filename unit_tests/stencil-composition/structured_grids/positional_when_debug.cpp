/*
 * test_computation.cpp
 *
 *  Created on: Mar 9, 2015
 *      Author: carlosos
 */

#define BOOST_NO_CXX11_RVALUE_REFERENCES

#ifdef NDEBUG
#undef NDEBUG
#define __WAS_DEBUG
#endif

#include <gridtools.hpp>
#include <boost/mpl/equal.hpp>
#include <boost/fusion/include/make_vector.hpp>

#include "gtest/gtest.h"

#include <stencil-composition/stencil-composition.hpp>
#include "stencil-composition/backend.hpp"
#include "stencil-composition/make_computation.hpp"
#include "stencil-composition/make_stencils.hpp"


namespace positional_when_debug_test{

    typedef gridtools::interval<gridtools::level<0,-1>, gridtools::level<1,-1> > x_interval;
    typedef gridtools::interval<gridtools::level<0,-2>, gridtools::level<1,1> > axis_t;
    typedef gridtools::grid<axis_t> grid_t;


    struct test_functor {
        typedef gridtools::accessor<0> in;
        typedef boost::mpl::vector1<in> arg_list;

        template <typename Evaluation>
        GT_FUNCTION
        static void Do(Evaluation const & eval, x_interval) {
            eval.i();
            eval.j();
            eval.k();
        }
    };
}

TEST(test_make_computation, positional_when_debug) {

    using namespace gridtools;
    using namespace gridtools::enumtype;
#ifdef __CUDACC__
#define BACKEND backend< Cuda, GRIDBACKEND, Block >
#else
#define BACKEND backend< Host, GRIDBACKEND, Block >
#endif

    typedef layout_map<2,1,0> layout_t;
    typedef BACKEND::storage_type<int, BACKEND::storage_info<0,layout_t> >::type storage_type;
    BACKEND::storage_info<0,layout_t> sinfo(3,3,3);
    storage_type a_storage(sinfo, 0, "test");

    typedef arg<0, storage_type> p_in;
    typedef boost::mpl::vector<p_in> accessor_list_t;

    /* canot use the assignment since with a single placeholder the wrong constructor is picked.
       This is a TODO in domain_type.hpp */
    domain_type<accessor_list_t> dm( boost::fusion::make_vector(&a_storage));
#ifdef CXX11_ENABLED
    auto
#else
#ifdef __CUDACC__
    computation*
#else
    boost::shared_ptr<gridtools::computation>
#endif
#endif
        test_computation = make_computation<BACKEND>
        (
         dm,
         positional_when_debug_test::grid_t({0,0,0,0,0}, {0,0,0,0,0}),
         make_mss // mss_descriptor
         (
          execute<forward>(),
          make_esf<positional_when_debug_test::test_functor>(p_in())
          )
         );

    EXPECT_TRUE(true);
}


#ifdef __WAS_DEBUG
#undef __WAS_DEBUG
#define NDEBUG
#endif
