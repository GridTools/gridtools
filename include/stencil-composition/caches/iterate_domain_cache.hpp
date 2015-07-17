/*
 * @file
 * @brief file containing infrastructure to provide cache functionality to the iterate domain.
 * All caching operations performance by the iterate domain are delegated to the code written here
 *
 */

#pragma once
#include <boost/mpl/copy_if.hpp>
#include "../run_functor_arguments.hpp"

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

    // remove caches which are not used by the stencil stages
    typedef typename boost::mpl::copy_if<
        cache_sequence_t,
        cache_used_by_esfs<boost::mpl::_, esf_sequence_t>
    >::type caches_t;



//    typedef typename boost::mpl::copy_if<
//        Caches,
//        is_ijk_cache<boost::mpl::_>
//    >::type IJKCaches;

};

} // namespace gridtools
