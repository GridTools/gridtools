/**
   @file
   @brief File containing metafunctions and traits common to all cache classes
*/

#pragma once

#include <boost/mpl/copy_if.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/map.hpp>
#include <boost/mpl/range_c.hpp>
#include <boost/fusion/container/map/convert.hpp>
#include <boost/fusion/sequence/intrinsic/at_key.hpp>
#include <boost/fusion/include/at_key.hpp>
#include <boost/fusion/include/mpl.hpp>
#include <boost/fusion/support/pair.hpp>
#include <boost/fusion/include/pair.hpp>
#include <boost/fusion/algorithm/transformation/push_back.hpp>
#include <boost/fusion/include/push_back.hpp>
#include <boost/fusion/container/vector.hpp>
#include <boost/fusion/include/vector.hpp>
#include <boost/fusion/container/vector/vector_fwd.hpp>
#include <boost/fusion/include/vector_fwd.hpp>

#include <stencil-composition/caches/cache.hpp>
#include <stencil-composition/caches/cache_storage.hpp>
#include <stencil-composition/esf_metafunctions.hpp>
#include <stencil-composition/local_domain.hpp>

#include <common/generic_metafunctions/is_there_in_sequence_if.hpp>

namespace gridtools {

/**
 * @brief metafunction determining if a type is a cache type
 */
template<typename T> struct is_cache : boost::mpl::false_{};

template<CacheType cacheType, typename Arg, CacheIOPolicy cacheIOPolicy>
struct is_cache<cache<cacheType, Arg, cacheIOPolicy> > : boost::mpl::true_{};

/**
 * @brief trait returning the parameter Arg type of a user provided cache
 */
template<typename T> struct cache_parameter;

template<CacheType cacheType, typename Arg, CacheIOPolicy cacheIOPolicy>
struct cache_parameter<cache<cacheType, Arg, cacheIOPolicy> >
{
    typedef Arg type;
};

/**
 * @brief metafunction that return an accessor with the right position of the parameter being cached within the local domain.
 * This is used as key to retrieve later the cache elements from the map.
 * @param Cache cache being converted into an accessor
 */
template<typename Cache, typename LocalDomain>
struct cache_to_accessor
{
    GRIDTOOLS_STATIC_ASSERT((is_cache<Cache>::value), "Internal Error: wrong type");
    GRIDTOOLS_STATIC_ASSERT((is_local_domain<LocalDomain>::value), "Internal Error: wrong type");

    typedef typename boost::mpl::find<typename LocalDomain::esf_args, typename cache_parameter<Cache>::type >::type arg_pos_t;

    typedef accessor< arg_pos_t::pos::value> type;
};

template<typename EsfSequence, typename CacheSequence>
struct caches_used_by_esfs
{
    // remove caches which are not used by the stencil stages
    typedef typename boost::mpl::copy_if<
        CacheSequence,
        is_there_in_sequence_if<EsfSequence, esf_has_parameter_h<cache_parameter<boost::mpl::_> > >
    >::type type;

};

template<CacheType cacheType>
struct cache_is_type
{
    template<typename Cache>
    struct apply
    {
        GRIDTOOLS_STATIC_ASSERT((is_cache<Cache>::value), "Internal Error: wrong type");
        typedef typename boost::is_same< enumtype::enum_type<CacheType, cacheType>, typename Cache::cache_type_t >::type type;
        BOOST_STATIC_CONSTANT(bool, value=(type::value));
    };
};

template<typename IterateDomainArguments>
struct extract_ranges_for_caches
{
    typedef typename IterateDomainArguments::cache_sequence_t cache_sequence_t;
    typedef typename IterateDomainArguments::range_sizes_t ranges_t;
    typedef typename IterateDomainArguments::esf_sequence_t esf_sequence_t;

    template<typename RangesMap, typename Cache>
    struct insert_range_for_cache
    {
        GRIDTOOLS_STATIC_ASSERT((is_cache<Cache>::value), "ERROR");
        template<typename RangesMap_, typename Range>
        struct update_range_map
        {
            GRIDTOOLS_STATIC_ASSERT((is_range<Range>::value), "ERROR");

            typedef typename boost::mpl::if_<
                boost::mpl::has_key<RangesMap_, Cache>,
                typename boost::mpl::at<RangesMap_, Cache>::type,
                range<0,0,0,0>
            >::type default_range_t;

            typedef typename boost::mpl::insert<
                typename boost::mpl::erase_key<RangesMap_, Cache>::type,
                boost::mpl::pair<Cache, typename enclosing_range<default_range_t, Range>::type >
            >::type type;

        };

        template<typename RangesMap_, typename EsfIdx>
        struct insert_range_for_cache_esf
        {
            GRIDTOOLS_STATIC_ASSERT((boost::mpl::size<ranges_t>::value > EsfIdx::value), "ERROR");
            GRIDTOOLS_STATIC_ASSERT((boost::mpl::size<esf_sequence_t>::value > EsfIdx::value), "ERROR");

            typedef typename boost::mpl::at<ranges_t, EsfIdx>::type range_t;
            typedef typename boost::mpl::at<esf_sequence_t, EsfIdx>::type esf_t;

            typedef typename boost::mpl::if_<
                is_there_in_sequence<typename esf_t::args_t, typename cache_parameter<Cache>::type >,
                typename update_range_map<RangesMap_, range_t>::type,
                RangesMap_
            >::type type;
        };

        typedef typename boost::mpl::fold<
            boost::mpl::range_c<int, 0, boost::mpl::size<esf_sequence_t>::value >,
            RangesMap,
            insert_range_for_cache_esf<boost::mpl::_1, boost::mpl::_2>
        >::type type;
    };

    typedef typename boost::mpl::fold<
        cache_sequence_t,
        boost::mpl::map0<>,
        insert_range_for_cache<boost::mpl::_1, boost::mpl::_2>
    >::type type;

};

template<CacheType cacheType, typename CacheSequence, typename CacheRangesMap, typename BlockSize, typename LocalDomain>
struct get_cache_storage_tuple
{
    GRIDTOOLS_STATIC_ASSERT((is_sequence_of<CacheSequence, is_cache>::value), "Internal Error: Wrong Type");
    GRIDTOOLS_STATIC_ASSERT((is_block_size<BlockSize>::value), "Internal Error: Wrong Type");
    GRIDTOOLS_STATIC_ASSERT((is_local_domain<LocalDomain>::value), "Internal Error: Wrong Type");

    template<typename Cache>
    struct get_cache_storage
    {
        typedef cache_storage<float_type, BlockSize, typename boost::mpl::at<CacheRangesMap, Cache>::type > type;
    };

    //first we build an mpl vector
    typedef typename boost::mpl::fold<
        CacheSequence,
        boost::mpl::vector0<>,
        boost::mpl::eval_if<
            typename cache_is_type<cacheType>::template apply< boost::mpl::_2 >,
            boost::mpl::push_back<
                boost::mpl::_1, boost::mpl::pair<cache_to_accessor<boost::mpl::_2, LocalDomain>, get_cache_storage<boost::mpl::_2> >
            >,
            boost::mpl::identity<boost::mpl::_1>
        >
    >::type mpl_type;

    template<typename FusionSeq, typename Pair>
    struct insert_pair_into_fusion_vector
    {
        typedef typename boost::fusion::result_of::push_back<
            FusionSeq,
            boost::fusion::pair<
                typename boost::mpl::first<Pair>::type,
                typename boost::mpl::second<Pair>::type
            >
        >::type type;
    };

    // then we transform the mpl vector into a fusion vector
    typedef typename boost::mpl::fold<
        mpl_type,
        boost::fusion::vector0<>,
        insert_pair_into_fusion_vector<boost::mpl::_1, boost::mpl::_2>
    >::type type;
};

} // namespace gridtools
