/*
   Copyright 2016 GridTools Consortium

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
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
#include "common/generic_metafunctions/vector_to_map.hpp"
#include "common/generic_metafunctions/fusion_map_to_mpl_map.hpp"
#include "stencil-composition/iterate_domain_fwd.hpp"
#include "../caches/cache_metafunctions.hpp"
#include "../caches/extract_extent_caches.hpp"

namespace gridtools {

    /**
     * @class iterate_domain_cache
     * class that provides all the caching functionality needed by the iterate domain.
     * It keeps in type information all the caches setup by the user and provides methods to access cache storage and
     * perform
     * all the caching operations, like filling, sliding or flushing.
     */
    template < typename IterateDomainArguments >
    class iterate_domain_cache {
        DISALLOW_COPY_AND_ASSIGN(iterate_domain_cache);

        GRIDTOOLS_STATIC_ASSERT(
            (is_iterate_domain_arguments< IterateDomainArguments >::value), "Internal error: wrong type");
        typedef typename IterateDomainArguments::esf_sequence_t esf_sequence_t;
        typedef typename IterateDomainArguments::cache_sequence_t cache_sequence_t;

      private:
        // checks if an arg is used by any of the esfs within a sequence
        template < typename EsfSequence, typename Arg >
        struct is_arg_used_in_esf_sequence {
            typedef typename boost::mpl::fold< EsfSequence,
                boost::mpl::false_,
                boost::mpl::or_< boost::mpl::_1, boost::mpl::contains< esf_args< boost::mpl::_2 >, Arg > > >::type type;
        };

      public:
        GT_FUNCTION
        iterate_domain_cache() {}

        GT_FUNCTION
        ~iterate_domain_cache() {}

        // remove caches which are not used by the stencil stages
        typedef typename boost::mpl::copy_if< cache_sequence_t,
            is_arg_used_in_esf_sequence< esf_sequence_t, cache_parameter< boost::mpl::_ > > >::type caches_t;

        // extract a sequence of extents for each cache
        typedef typename extract_extents_for_caches< IterateDomainArguments >::type cache_extents_map_t;

        // compute the fusion vector of pair<index_type, cache_storage>
        typedef typename get_cache_storage_tuple< IJ,
            caches_t,
            cache_extents_map_t,
            typename IterateDomainArguments::physical_domain_block_size_t,
            typename IterateDomainArguments::local_domain_t >::type ij_caches_vector_t;

        // extract a fusion map from the fusion vector of pairs
        typedef typename boost::fusion::result_of::as_map< ij_caches_vector_t >::type ij_caches_tuple_t;

        // compute an mpl from the previous fusion vector, to be used for compile time meta operations
        typedef typename fusion_map_to_mpl_map< ij_caches_tuple_t >::type ij_caches_map_t;

        typedef
            typename get_cache_set_for_type< bypass, caches_t, typename IterateDomainArguments::local_domain_t >::type
                bypass_caches_set_t;

        // associative container with all caches
        typedef typename get_cache_set< caches_t, typename IterateDomainArguments::local_domain_t >::type all_caches_t;
    };

    template < typename IterateDomainArguments >
    struct is_iterate_domain_cache< iterate_domain_cache< IterateDomainArguments > > : boost::mpl::true_ {};

} // namespace gridtools
