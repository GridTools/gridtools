/*
 * test_cache_metafunctions.cpp
 *
 *  Created on: Jul 17, 2015
 *      Author: cosuna
 */

#include "gtest/gtest.h"
#include "defs.hpp"
#include <stencil-composition/backend.hpp>
#include <stencil-composition/caches/cache_metafunctions.hpp>
#include <stencil-composition/interval.hpp>
#include <stencil-composition/make_computation.hpp>


using namespace gridtools;
using namespace enumtype;

// This is the definition of the special regions in the "vertical" direction
typedef gridtools::interval<gridtools::level<0,-1>, gridtools::level<1,-1> > x_interval;
struct functor1 {
    typedef const accessor<0, range<0,0,0,0>, 6> in;
    typedef accessor<1, range<0,0,0,0>, 5> buff;
    typedef boost::mpl::vector<in,buff> arg_list;

    template <typename Evaluation>
    GT_FUNCTION
    static void Do(Evaluation const & eval, x_interval) {}
};

TEST(cache_metafunction, cache_used_by_esfs)
{
    typedef layout_map<0,1> layout_ij_t;
    typedef gridtools::backend<enumtype::Host, enumtype::Naive >::storage_type<float_type, layout_ij_t >::type storage_type;

    typedef arg<0, storage_type> p_in;
    typedef arg<2, storage_type> p_out;
    typedef arg<1, storage_type> p_buff;
    typedef arg<3, storage_type> p_notin;

    typedef decltype(gridtools::make_esf<functor1>(p_in() ,p_buff()) ) esf1_t;
    typedef decltype(gridtools::make_esf<functor1>(p_buff(), p_out()) ) esf2_t;

    typedef boost::mpl::vector2<esf1_t, esf2_t> esf_sequence_t;
    typedef cache<IJ, p_in, cFill> cache1_t;
    typedef cache<IJ, p_buff, cFill> cache2_t;
    typedef cache<K, p_notin, cLocal> cache3_t;

    GRIDTOOLS_STATIC_ASSERT((cache_used_by_esfs<cache1_t, esf_sequence_t>::value),"ERROR");
    GRIDTOOLS_STATIC_ASSERT((cache_used_by_esfs<cache2_t, esf_sequence_t>::value),"ERROR");
    GRIDTOOLS_STATIC_ASSERT((! cache_used_by_esfs<cache3_t, esf_sequence_t>::value),"ERROR");

    ASSERT_TRUE(true);
}
