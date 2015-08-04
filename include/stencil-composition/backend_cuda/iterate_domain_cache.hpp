/*
 * @file
 * @brief file containing infrastructure to provide cache functionality to the iterate domain.
 * All caching operations performance by the iterate domain are delegated to the code written here
 *
 */

#pragma once

#include "common/defs.hpp"
#include <boost/fusion/container/map/convert.hpp>
#include <boost/fusion/include/as_map.hpp>
#include <boost/fusion/support/pair.hpp>
#include <boost/fusion/include/pair.hpp>
#include <boost/mpl/copy_if.hpp>
#include "stencil-composition/run_functor_arguments.hpp"
#include "common/generic_metafunctions/vector_to_map.hpp"
#include "common/generic_metafunctions/fusion_map_to_mpl_map.hpp"

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
    //checks if an arg is used by any of the esfs within a sequence
    template<typename EsfSequence, typename Arg>
    struct is_arg_used_in_esf_sequence
    {
        typedef typename boost::mpl::fold<
            EsfSequence,
            boost::mpl::false_,
            boost::mpl::or_<
                boost::mpl::_1,
                boost::mpl::contains< esf_args<boost::mpl::_2>, Arg>
            >
        >::type type;
    };

public:
    iterate_domain_cache() {}
    ~iterate_domain_cache() {}

    // remove caches which are not used by the stencil stages
    typedef typename boost::mpl::copy_if<
        cache_sequence_t,
        is_arg_used_in_esf_sequence<esf_sequence_t, cache_parameter<boost::mpl::_> >
    >::type caches_t;

    //extract a sequence of ranges for each cache
    typedef typename extract_ranges_for_caches<IterateDomainArguments>::type cache_ranges_map_t;

    //compute the fusion vector of pair<index_type, cache_storage>
    typedef typename get_cache_storage_tuple<
        IJ,
        caches_t,
        cache_ranges_map_t,
        typename IterateDomainArguments::physical_domain_block_size_t,
        typename IterateDomainArguments::local_domain_t
    >::type ij_caches_vector_t;

    //extract a fusion map from the fusion vector of pairs
    typedef typename boost::fusion::result_of::as_map<ij_caches_vector_t>::type ij_caches_tuple_t;

    // compute an mpl from the previous fusion vector, to be used for compile time meta operations
    typedef typename fusion_map_to_mpl_map<ij_caches_tuple_t>::type ij_caches_map_t;
};

} // namespace gridtools
