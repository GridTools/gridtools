#include "gtest/gtest.h"
#include <boost/mpl/equal.hpp>
// #include "stencil-composition/caches/cache_metafunctions.hpp"
// #include "stencil-composition/caches/define_caches.hpp"
// #include "stencil-composition/interval.hpp"
#include <stencil-composition/stencil-composition.hpp>


using namespace gridtools;
using namespace enumtype;


// This is the definition of the special regions in the "vertical" direction
typedef gridtools::interval<gridtools::level<0,-1>, gridtools::level<1,-1> > x_interval;

struct functor1 {
    typedef accessor<0> in;
    typedef accessor<1> out;
    typedef boost::mpl::vector<in,out> arg_list;

    template <typename Evaluation>
    GT_FUNCTION
    static void Do(Evaluation const & eval, x_interval) {}
};

#ifdef __CUDACC__
#define BACKEND backend< Cuda, GRIDBACKEND, Block >
#else
#define BACKEND backend< Host, GRIDBACKEND, Block >
#endif

typedef layout_map<2,1,0> layout_ijk_t;
typedef gridtools::BACKEND::storage_type<float_type, gridtools::BACKEND::storage_info<0,layout_ijk_t> >::type storage_type;
typedef gridtools::BACKEND::temporary_storage_type<float_type, gridtools::BACKEND::storage_info<0,layout_ijk_t> >::type tmp_storage_type;

typedef arg<0, storage_type> p_in;
typedef arg<1, storage_type> p_out;
typedef arg<2, tmp_storage_type> p_buff;

TEST(mss_metafunctions, extract_mss_caches_and_esfs)
{
    typename storage_type::storage_info_type meta_(10, 10, 10);
    storage_type in(meta_, 1.0, "in"), out(meta_, 1.0, "out");

    typedef decltype(make_esf<functor1>(p_in(), p_buff())) esf1_t;
    typedef decltype(make_esf<functor1>(p_buff(), p_out())) esf2_t;

    typedef decltype(make_multistage // mss_descriptor
        (execute< forward >(),
            define_caches(cache< IJ, local >(p_buff(), p_out())),
            esf1_t(), // esf_descriptor
            esf2_t()  // esf_descriptor
            )) mss_t;
    GRIDTOOLS_STATIC_ASSERT((boost::mpl::equal<
            mss_t::esf_sequence_t, boost::mpl::vector2<esf1_t, esf2_t>
        >::value), "ERROR");

#ifndef __DISABLE_CACHING__
    GRIDTOOLS_STATIC_ASSERT((boost::mpl::equal< mss_t::cache_sequence_t,
                                boost::mpl::vector2< detail::cache_impl< IJ, p_buff, local >,
                                                    detail::cache_impl< IJ, p_out, local > > >::value),
        "ERROR\nLists do not match");
#else
    GRIDTOOLS_STATIC_ASSERT((boost::mpl::empty<mss_t::cache_sequence_t>::value), "ERROR\nList not empty");
#endif


    ASSERT_TRUE(true);
}
