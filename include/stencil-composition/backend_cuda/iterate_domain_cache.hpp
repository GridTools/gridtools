/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/
/*
 * @file
 * @brief file containing infrastructure to provide cache functionality to the iterate domain.
 * All caching operations performance by the iterate domain are delegated to the code written here
 *
 */

#pragma once

#include "../caches/cache_metafunctions.hpp"
#include "../caches/extract_extent_caches.hpp"
#include "common/defs.hpp"
#include "common/generic_metafunctions/fusion_map_to_mpl_map.hpp"
#include "common/generic_metafunctions/vector_to_map.hpp"
#include "stencil-composition/iterate_domain_fwd.hpp"
#include <boost/fusion/container/map/convert.hpp>
#include <boost/fusion/include/as_map.hpp>
#include <boost/fusion/include/pair.hpp>
#include <boost/fusion/support/pair.hpp>
#include <boost/mpl/copy_if.hpp>

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

        // compute the fusion vector of pair<index_t, cache_storage>
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
