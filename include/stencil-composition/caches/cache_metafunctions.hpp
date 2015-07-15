/**
   @file
   @brief File containing metafunctions and traits common to all cache classes
*/

#pragma once

#include "cache.hpp"

namespace gridtools {
/**
 * @brief metafunction determining if a type is a cache type
 */
template<typename T> struct is_cache : boost::mpl::false_{};

template<CacheType cacheType, typename Arg, CacheIOPolicy cacheIOPolicy>
struct is_cache<cache<cacheType, Arg, cacheIOPolicy> > : boost::mpl::true_{};

}
