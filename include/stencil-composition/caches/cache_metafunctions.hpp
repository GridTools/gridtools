/**
   @file
   @brief File containing metafunctions and traits common to all cache classes
*/

#pragma once

#include <boost/mpl/copy_if.hpp>
#include <stencil-composition/caches/cache.hpp>
#include <stencil-composition/esf_metafunctions.hpp>

#include <common/generic_metafunctions/is_there_in_sequence_if.hpp>

namespace gridtools {


template<typename T> struct cache_parameter;

template<CacheType cacheType, typename Arg, CacheIOPolicy cacheIOPolicy>
struct cache_parameter<cache<cacheType, Arg, cacheIOPolicy> >
{
    typedef Arg type;
};

/**
 * @brief metafunction determining if a type is a cache type
 */
template<typename T> struct is_cache : boost::mpl::false_{};

template<CacheType cacheType, typename Arg, CacheIOPolicy cacheIOPolicy>
struct is_cache<cache<cacheType, Arg, cacheIOPolicy> > : boost::mpl::true_{};


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

}
