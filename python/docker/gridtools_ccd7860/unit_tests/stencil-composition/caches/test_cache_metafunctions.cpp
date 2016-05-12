/*
 * test_cache_metafunctions.cpp
 *
 *  Created on: Jul 17, 2015
 *      Author: cosuna
 */

#include "gtest/gtest.h"
#include <boost/mpl/equal.hpp>
#include "common/defs.hpp"
#include "stencil-composition/backend.hpp"
#include "stencil-composition/caches/cache_metafunctions.hpp"
#include "stencil-composition/interval.hpp"
#include "stencil-composition/make_computation.hpp"
#include "common/generic_metafunctions/fusion_map_to_mpl_map.hpp"

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

typedef layout_map<0,1> layout_ij_t;
typedef gridtools::backend<enumtype::Host, enumtype::Naive >::storage_type<float_type, layout_ij_t >::type storage_type;

typedef arg<0, storage_type> p_in;
typedef arg<2, storage_type> p_out;
typedef arg<1, storage_type> p_buff;
typedef arg<3, storage_type> p_notin;

typedef decltype(gridtools::make_esf<functor1>(p_in() ,p_buff()) ) esf1_t;
typedef decltype(gridtools::make_esf<functor1>(p_buff(), p_out()) ) esf2_t;

typedef boost::mpl::vector2<esf1_t, esf2_t> esf_sequence_t;

typedef cache<IJ, p_in, fill> cache1_t;
typedef cache<IJ, p_buff, fill> cache2_t;
typedef cache<K, p_out, local> cache3_t;
typedef cache<K, p_notin, local> cache4_t;
typedef boost::mpl::vector4<cache1_t, cache2_t, cache3_t, cache4_t> caches_t;

TEST(cache_metafunctions, cache_used_by_esfs)
{
    typedef caches_used_by_esfs<esf_sequence_t, caches_t>::type caches_used_t;

    GRIDTOOLS_STATIC_ASSERT((boost::mpl::equal<caches_used_t, boost::mpl::vector3<cache1_t, cache2_t, cache3_t> >::value), "WRONG");
    ASSERT_TRUE(true);
}

TEST(cache_metafunctions, extract_ranges_for_caches)
{
    typedef boost::mpl::vector3<p_in, p_buff, p_out> esf_args_t;
    typedef local_domain< boost::mpl::void_, esf_args_t, false> local_domain_t;

    typedef boost::mpl::vector2< range<-1,2,-2,1>, range<-2,1,-3,2> > ranges_t;

    typedef iterate_domain_arguments< local_domain_t, esf_sequence_t, ranges_t, caches_t, block_size<32,4> > iterate_domain_arguments_t;

    typedef extract_ranges_for_caches<iterate_domain_arguments_t>::type ranges_map_t;

    GRIDTOOLS_STATIC_ASSERT((boost::mpl::equal<ranges_map_t,
            boost::mpl::map3<
                boost::mpl::pair<cache1_t, range<-1,2,-2,1> >,
                boost::mpl::pair<cache2_t, range<-2,2,-3,2> >,
                boost::mpl::pair<cache3_t, range<-2,1,-3,2> >
            > >::value), "ERROR"
    );
}

TEST(cache_metafunctions, get_cache_storage_tuple)
{
    typedef boost::mpl::vector3<storage_type, storage_type, storage_type> storages_t;
    typedef boost::fusion::result_of::as_vector<storages_t>::type storages_tuple_t;
    typedef boost::mpl::vector3<p_in, p_buff, p_out> esf_args_t;
    typedef local_domain< storages_tuple_t, esf_args_t, false> local_domain_t;

    typedef boost::mpl::vector2< range<-1,2,-2,1>, range<-2,1,-3,2> > ranges_t;

    typedef iterate_domain_arguments< local_domain_t, esf_sequence_t, ranges_t, caches_t, block_size<32,4> > iterate_domain_arguments_t;

    typedef extract_ranges_for_caches<iterate_domain_arguments_t>::type ranges_map_t;

    typedef get_cache_storage_tuple<IJ, caches_t, ranges_map_t, block_size<32,4>, local_domain_t>::type cache_storage_tuple_t;

    // fusion::result_of::at_key<cache_storage_tuple_t, p_in::index_type> does not compile,
    // therefore we convert into an mpl map and do all the metaprogramming operations on that map
    typedef fusion_map_to_mpl_map<cache_storage_tuple_t>::type cache_storage_mpl_map_t;

    GRIDTOOLS_STATIC_ASSERT((
        boost::mpl::equal<
            cache_storage_tuple_t,
            boost::fusion::map<
                boost::fusion::pair<p_in::index_type, cache_storage<float_type, block_size<32,4>, range<-1,2,-2,1> > >,
                boost::fusion::pair<p_buff::index_type, cache_storage<float_type, block_size<32,4>, range<-2,2,-3,2> > >
            >
        >::value),"ERROR");
}
