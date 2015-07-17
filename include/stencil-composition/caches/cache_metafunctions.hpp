/**
   @file
   @brief File containing metafunctions and traits common to all cache classes
*/

#pragma once

#include "cache.hpp"
#include "../../common/generic_metafunctions/is_there_in_sequence.hpp"

namespace gridtools {
/**
 * @brief metafunction determining if a type is a cache type
 */
template<typename T> struct is_cache : boost::mpl::false_{};

template<CacheType cacheType, typename Arg, CacheIOPolicy cacheIOPolicy>
struct is_cache<cache<cacheType, Arg, cacheIOPolicy> > : boost::mpl::true_{};

template<typename Cache, typename EsfSequence>
struct cache_used_by_esfs
{
    GRIDTOOLS_STATIC_ASSERT((is_cache<Cache>::value), "internal error: wrong type");

    typedef typename boost::mpl::fold<
        EsfSequence,
        boost::mpl::false_,
        boost::mpl::or_<
            boost::mpl::_1,
            is_there_in_sequence<EsfSequence, boost::mpl::_2>
        >
    >::type type;
    BOOST_STATIC_CONSTANT(bool, value=(type::value));
};


}
