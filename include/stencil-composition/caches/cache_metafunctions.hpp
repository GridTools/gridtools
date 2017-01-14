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
/**
   @file
   @brief File containing metafunctions and traits common to all cache classes
*/

#pragma once

#include <boost/mpl/copy_if.hpp>
#include <boost/mpl/pair.hpp>
#include <boost/mpl/void.hpp>
#include <boost/mpl/insert.hpp>
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

#include "stencil-composition/caches/cache.hpp"
#include "stencil-composition/caches/cache_storage.hpp"
#include "stencil-composition/esf_metafunctions.hpp"

#include "common/generic_metafunctions/is_there_in_sequence_if.hpp"

#include "../accessor_fwd.hpp"

namespace gridtools {

    /**
     * @struct is_cache
     * metafunction determining if a type is a cache type
     */
    template < typename T >
    struct is_cache : boost::mpl::false_ {};

    template < cache_type cacheType, typename Arg, cache_io_policy cacheIOPolicy, typename Interval >
    struct is_cache< detail::cache_impl< cacheType, Arg, cacheIOPolicy, Interval > > : boost::mpl::true_ {};

    /**
     * @struct is_ij_cache
     * metafunction determining if a type is a cache of IJ type
     */
    template < typename T >
    struct is_ij_cache : boost::mpl::false_ {};

    template < typename Arg, cache_io_policy cacheIOPolicy, typename Interval >
    struct is_ij_cache< detail::cache_impl< IJ, Arg, cacheIOPolicy, Interval > > : boost::mpl::true_ {};

    /**
     * @struct is_k_cache
     * metafunction determining if a type is a cache of K type
     */
    template < typename T >
    struct is_k_cache : boost::mpl::false_ {};

    template < typename Arg, cache_io_policy cacheIOPolicy, typename Interval >
    struct is_k_cache< detail::cache_impl< K, Arg, cacheIOPolicy, Interval > > : boost::mpl::true_ {};

    /**
     * @struct cache_parameter
     *  trait returning the parameter Arg type of a user provided cache
     */
    template < typename T >
    struct cache_parameter;

    template < cache_type cacheType, typename Arg, cache_io_policy cacheIOPolicy, typename Interval >
    struct cache_parameter< detail::cache_impl< cacheType, Arg, cacheIOPolicy, Interval > > {
        typedef Arg type;
    };

    /**
     * @struct cache_to_index
     * metafunction that return the index type with the right position of the parameter being cached within the local
     * domain.
     * This is used as key to retrieve later the cache elements from the map.
     * @tparam Cache cache being converted into an accessor
     * @tparam LocalDomain local domain that contains a list of esf args which are used to determine the position of the
     *        cache within the sequence
     */
    template < typename Cache, typename LocalDomain >
    struct cache_to_index {
        GRIDTOOLS_STATIC_ASSERT((is_cache< Cache >::value), "Internal Error: wrong type");
        GRIDTOOLS_STATIC_ASSERT((is_local_domain< LocalDomain >::value), "Internal Error: wrong type");

        typedef typename boost::mpl::find< typename LocalDomain::esf_args,
            typename cache_parameter< Cache >::type >::type arg_pos_t;
        typedef static_uint< arg_pos_t::pos::value > type;
    };

    /**
     * @struct caches_used_by_esfs
     * metafunction that filter from a sequence of caches those that are used in a least one esf of the esf sequence
     * @tparam EsfSequence sequence of esf
     * @tparam CacheSequence original sequence of caches
     */
    template < typename EsfSequence, typename CacheSequence >
    struct caches_used_by_esfs {
        GRIDTOOLS_STATIC_ASSERT(
            (is_sequence_of< EsfSequence, is_esf_descriptor >::value), "Internal Error: wrong type");
        GRIDTOOLS_STATIC_ASSERT((is_sequence_of< CacheSequence, is_cache >::value), "Internal Error: wrong type");

        // remove caches which are not used by the stencil stages
        typedef typename boost::mpl::copy_if< CacheSequence,
            is_there_in_sequence_if< EsfSequence, esf_has_parameter_h< cache_parameter< boost::mpl::_ > > > >::type
            type;
    };

    /**
     * @struct cache_is_type
     * high order metafunction that determines if a cache is of the same type as provided as argument
     * @tparam cacheType type of cache that cache should equal
     */
    template < cache_type cacheType >
    struct cache_is_type {
        template < typename Cache >
        struct apply {
            GRIDTOOLS_STATIC_ASSERT((is_cache< Cache >::value), "Internal Error: wrong type");
            typedef typename boost::is_same< enumtype::enum_type< cache_type, cacheType >,
                typename Cache::cache_type_t >::type type;
            BOOST_STATIC_CONSTANT(bool, value = (type::value));
        };
    };

    /**
     * @struct get_cache_storage_tuple
     * metafunction that computes a fusion vector of pairs of <static_uint<index>, cache_storage> for all the caches,
     * where the index is the position of the parameter associated to the cache within the local domain
     * This fusion tuple will be used by the iterate domain to directly retrieve the corresponding cache storage
     * of a given accessor
     * @tparam cacheType type of cache
     * @tparam CacheSequence sequence of caches specified by the user
     * @tparam CacheExtendsMap map of <cache, extent> determining the extent size of each cache
     * @tparam BlockSize the physical domain block size
     * @tparam LocalDomain the fused local domain
     */

    template < cache_type cacheType,
        typename CacheSequence,
        typename CacheExtendsMap,
        typename BlockSize,
        typename LocalDomain >
    struct get_cache_storage_tuple {
        GRIDTOOLS_STATIC_ASSERT((is_sequence_of< CacheSequence, is_cache >::value), "Internal Error: Wrong Type");
        GRIDTOOLS_STATIC_ASSERT((is_block_size< BlockSize >::value), "Internal Error: Wrong Type");
        GRIDTOOLS_STATIC_ASSERT((is_local_domain< LocalDomain >::value), "Internal Error: Wrong Type");

        /** metafunction extracting the storage type corresponding to an index from the local_domain*/
        template < typename LocDom, typename Index >
        struct get_storage {
            GRIDTOOLS_STATIC_ASSERT(is_local_domain< LocDom >::value, "wrong type");
            GRIDTOOLS_STATIC_ASSERT((boost::mpl::size< typename LocDom::mpl_storages >::value > Index::value),
                "accessing a storage which is not in the list");
            typedef typename boost::mpl::at< typename LocDom::mpl_storages, Index >::type type;
        };

        // In order to build a fusion vector here, we first create an mpl vector of pairs, which is then transformed
        // into a fusion vector.
        // Note: building a fusion map using result_of::as_map<mpl_vector_of_pair> did not work due to a clash between
        // fusion pair and mpl pairs (the algorithm as_map expect fusion pairs). That is the reason of the workaround
        // here
        // mpl vector -> fusion vector -> fusion map (with result_of::as_map)

        template < typename Cache, typename StoragePtr >
        struct get_cache_storage {
            GRIDTOOLS_STATIC_ASSERT(is_cache< Cache >::value, "wrong type");
            GRIDTOOLS_STATIC_ASSERT(is_pointer< StoragePtr >::value, "wrong type");
            GRIDTOOLS_STATIC_ASSERT(is_storage< typename StoragePtr::value_type >::value, "wrong type");

            typedef cache_storage< BlockSize, typename boost::mpl::at< CacheExtendsMap, Cache >::type, StoragePtr >
                type;
        };

        // first we build an mpl vector of pairs
        typedef typename boost::mpl::fold<
            CacheSequence,
            boost::mpl::vector0<>,
            boost::mpl::eval_if< typename cache_is_type< cacheType >::template apply< boost::mpl::_2 >,
                boost::mpl::push_back< boost::mpl::_1,
                                     boost::mpl::pair< cache_to_index< boost::mpl::_2, LocalDomain >,
                                           get_cache_storage< boost::mpl::_2,
                                                           get_storage< LocalDomain,
                                                                  cache_to_index< boost::mpl::_2, LocalDomain > > > > >,
                boost::mpl::identity< boost::mpl::_1 > > >::type mpl_type;

        // here we insert an mpl pair into a fusion vector. The mpl pair is converted into a fusion pair
        template < typename FusionSeq, typename Pair >
        struct insert_pair_into_fusion_vector {
            typedef
                typename boost::fusion::result_of::push_back< FusionSeq,
                    boost::fusion::pair< typename boost::mpl::first< Pair >::type,
                                                                  typename boost::mpl::second< Pair >::type > >::type
                    type;
        };

        // then we transform the mpl vector into a fusion vector
        typedef typename boost::mpl::fold< mpl_type,
            boost::fusion::vector0<>,
            insert_pair_into_fusion_vector< boost::mpl::_1, boost::mpl::_2 > >::type type;
    };

    /**
     * @struct get_cache_set_for_type
     * metafunction that computes a set of integers with position of each cache in the local domain,
     * for all caches of a given type
     * @tparam cacheType type of cache that is used to filter the sequence of caches
     * @tparam CacheSequence sequence of caches used to extract the set of their positions
     * @tparam LocalDomain local domain that contains all parameters
     */
    template < cache_type cacheType, typename CacheSequence, typename LocalDomain >
    struct get_cache_set_for_type {
        GRIDTOOLS_STATIC_ASSERT((is_sequence_of< CacheSequence, is_cache >::value), "Internal Error: Wrong Type");
        GRIDTOOLS_STATIC_ASSERT((is_local_domain< LocalDomain >::value), "Internal Error: Wrong Type");

        typedef typename boost::mpl::fold<
            CacheSequence,
            boost::mpl::set0<>,
            boost::mpl::if_< typename cache_is_type< cacheType >::template apply< boost::mpl::_2 >,
                boost::mpl::insert< boost::mpl::_1, cache_to_index< boost::mpl::_2, LocalDomain > >,
                boost::mpl::_1 > >::type type;
    };

    /**
     * @struct get_cache_set
     * metafunction that computes a set of integers with position of each cache in the local domain
     * @tparam CacheSequence sequence of caches used to extract the set of their positions
     * @tparam LocalDomain local domain that contains all parameters
     */
    template < typename CacheSequence, typename LocalDomain >
    struct get_cache_set {

        GRIDTOOLS_STATIC_ASSERT((is_sequence_of< CacheSequence, is_cache >::value), "Internal Error: Wrong Type");
        GRIDTOOLS_STATIC_ASSERT(is_local_domain< LocalDomain >::value, "wrong type");

        typedef typename boost::mpl::fold< CacheSequence,
            boost::mpl::set0<>,
            boost::mpl::insert< boost::mpl::_1, cache_to_index< boost::mpl::_2, LocalDomain > > >::type type;
    };

} // namespace gridtools
