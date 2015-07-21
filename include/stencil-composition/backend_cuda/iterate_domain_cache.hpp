/*
 * @file
 * @brief file containing infrastructure to provide cache functionality to the iterate domain.
 * All caching operations performance by the iterate domain are delegated to the code written here
 *
 */

#pragma once

#include <common/defs.hpp>
//#include <boost/fusion/adapted/mpl.hpp>
//#include <boost/fusion/include/vector.hpp>
#include <boost/fusion/container/map/convert.hpp>
#include <boost/fusion/include/as_map.hpp>
//#include <boost/fusion/container/map.hpp>
//#include <boost/fusion/include/map.hpp>
//#include <boost/fusion/container/map/map_fwd.hpp>
//#include <boost/fusion/include/map_fwd.hpp>
//#include <boost/fusion/include/mpl.hpp>
#include <boost/mpl/copy_if.hpp>
#include <stencil-composition/run_functor_arguments.hpp>
#include <common/generic_metafunctions/vector_to_map.hpp>

namespace gridtools {

/**
 * @class iterate_domain_cache
 * class that provides all the caching functionality needed by the iterate domain.
 * It keeps in type information all the caches setup by the user and provides methods to access cache storage and perform
 * all the caching operations, like filling, sliding or flushing.
 */
template<typename IterateDomainArguments>
class iterate_domain_cache
{
    DISALLOW_COPY_AND_ASSIGN(iterate_domain_cache);

    GRIDTOOLS_STATIC_ASSERT((is_iterate_domain_arguments<IterateDomainArguments>::value), "Internal error: wrong type");
    typedef typename IterateDomainArguments::esf_sequence_t esf_sequence_t;
    typedef typename IterateDomainArguments::cache_sequence_t cache_sequence_t;
private:

public:
    iterate_domain_cache() {}
    ~iterate_domain_cache() {}

    template<typename T> struct printp{BOOST_MPL_ASSERT_MSG((false), YYYYYYY, (T));};

    // remove caches which are not used by the stencil stages
    typedef typename boost::mpl::copy_if<
        cache_sequence_t,
        is_there_in_sequence<esf_sequence_t, boost::mpl::_>
    >::type caches_t;

    typedef typename extract_ranges_for_caches<IterateDomainArguments>::type cache_ranges_t;

    typedef typename get_cache_storage_tuple<
        IJ,
        caches_t,
        cache_ranges_t,
        typename IterateDomainArguments::physical_domain_block_size_t
    >::type ij_caches_vector_t;

    typedef typename vector_to_map<ij_caches_vector_t>::type ij_caches_map_t;

    typedef typename boost::fusion::result_of::as_map<ij_caches_vector_t>::type ij_caches_tuple_t;

};

} // namespace gridtools
