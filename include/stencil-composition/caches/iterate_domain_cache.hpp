/*
 * @file
 * @brief file containing infrastructure to provide cache functionality to the iterate domain.
 * All caching operations performance by the iterate domain are delegated to the code written here
 *
 */

#pragma once

/**
 * @class iterate_domain_cache
 * class that provides all the caching functionality needed by the iterate domain.
 * It keeps in type information all the caches setup by the user and provides methods to access cache storage and perform
 * all the caching operations, like filling, sliding or flushing.
 */
template<typename CacheSequence>
class iterate_domain_cache
{
    DISALLOW_COPY_AND_ASSIGN(iterate_domain_cache);

private:

public:
    iterate_domain_cache() {}
    ~iterate_domain_cache() {}

//    // remove caches which are not used by the stencil stages
//    typedef typename boost::mpl::copy_if<
//        TCaches,
//        stencil_stages_have_parameter<TStencilStages, FullDomain, cache_index<boost::mpl::_> >
//    >::type Caches;

//    typedef typename boost::mpl::copy_if<
//        Caches,
//        is_ijk_cache<boost::mpl::_>
//    >::type IJKCaches;

};
