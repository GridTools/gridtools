/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
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
#pragma once

#include "../grid_traits_fwd.hpp"
#include "../extent_metafunctions.hpp"

namespace gridtools {

    namespace impl {
        // update the entry associated to a cache within the map with a new extent.
        // if the key exist we compute and insert the enclosing extent, otherwise we just
        // insert the extent into a new entry of the map of <cache, extent>
        template < typename ExtendsMap_, typename Extend, typename Cache, typename BackendIds >
        struct update_extent_map {
            GRIDTOOLS_STATIC_ASSERT((is_extent< Extend >::value), "ERROR");
            GRIDTOOLS_STATIC_ASSERT((is_cache< Cache >::value), "ERROR");
            GRIDTOOLS_STATIC_ASSERT((is_backend_ids< BackendIds >::value), "ERROR");
            typedef typename boost::mpl::if_< boost::mpl::has_key< ExtendsMap_, Cache >,
                typename boost::mpl::at< ExtendsMap_, Cache >::type,
                typename grid_traits_from_id< BackendIds::s_grid_type_id >::null_extent_t >::type default_extent_t;

            typedef typename boost::mpl::insert<
                typename boost::mpl::erase_key< ExtendsMap_, Cache >::type,
                boost::mpl::pair< Cache,
#if defined(CXX11_ENABLED) && !defined(__CUDACC__)
                    typename enclosing_extent_full< default_extent_t, Extend >::type > >::type
#else
                    typename enclosing_extent< default_extent_t, Extend >::type > >::type
#endif
                type;
        };
    }
    /**
     * @struct extract_ij_extents_for_caches
     * metafunction that extracts the extents associated to each cache of the sequence of caches provided by the user.
     * The extent is determined as the enclosing extent of all the extents of esfs that use the cache.
     * It is used in order to allocate enough memory for each cache storage.
     * @tparam IterateDomainArguments iterate domain arguments type containing sequences of caches, esfs and extents
     * @return map<cache,extent>
     */
    template < typename IterateDomainArguments >
    struct extract_ij_extents_for_caches {
        typedef typename IterateDomainArguments::cache_sequence_t cache_sequence_t;
        typedef typename IterateDomainArguments::extent_sizes_t extents_t;
        typedef typename IterateDomainArguments::esf_sequence_t esf_sequence_t;
        typedef typename IterateDomainArguments::backend_ids_t backend_ids_t;

        // insert the extent associated to a Cache into the map of <cache, extent>
        template < typename ExtendsMap, typename Cache >
        struct insert_extent_for_cache {
            GRIDTOOLS_STATIC_ASSERT((is_cache< Cache >::value), GT_INTERNAL_ERROR);

            // given an Id within the sequence of esf and extents, extract the extent associated an inserted into
            // the map if the cache is used by the esf with that Id.
            template < typename ExtendsMap_, typename EsfIdx >
            struct insert_extent_for_cache_esf {
                GRIDTOOLS_STATIC_ASSERT((boost::mpl::size< extents_t >::value > EsfIdx::value), "ERROR");
                GRIDTOOLS_STATIC_ASSERT((boost::mpl::size< esf_sequence_t >::value > EsfIdx::value), "ERROR");

                typedef typename boost::mpl::at< extents_t, EsfIdx >::type extent_t;
                GRIDTOOLS_STATIC_ASSERT((extent_t::kminus::value == 0 && extent_t::kplus::value == 0),
                    "Error: IJ Caches can not have k extent values");

                typedef typename boost::mpl::at< esf_sequence_t, EsfIdx >::type esf_t;

                typedef typename boost::mpl::if_<
                    boost::mpl::contains< typename esf_t::args_t, typename cache_parameter< Cache >::type >,
                    typename impl::update_extent_map< ExtendsMap_, extent_t, Cache, backend_ids_t >::type,
                    ExtendsMap_ >::type type;
            };

            // loop over all esfs and insert the extent associated to the cache into the map
            typedef typename boost::mpl::fold< boost::mpl::range_c< int, 0, boost::mpl::size< esf_sequence_t >::value >,
                ExtendsMap,
                insert_extent_for_cache_esf< boost::mpl::_1, boost::mpl::_2 > >::type type;
        };

        typedef typename boost::mpl::fold<
            typename boost::mpl::filter_view< cache_sequence_t, is_ij_cache< boost::mpl::_ > >::type,
            boost::mpl::map0<>,
            insert_extent_for_cache< boost::mpl::_1, boost::mpl::_2 > >::type type;
    };

    /**
     * @struct extract_k_extents_for_caches
     * metafunction that extracts the extents associated to each cache of the sequence of caches provided by the user.
     * The extent is determined as the enclosing extent of all the extents of esfs that use the cache.
     * It is used in order to allocate enough memory for each cache storage.
     * @tparam IterateDomainArguments iterate domain arguments type containing sequences of caches, esfs and extents
     * @return map<cache,extent>
     */
    template < typename IterateDomainArguments >
    struct extract_k_extents_for_caches {
        typedef typename IterateDomainArguments::cache_sequence_t cache_sequence_t;
        typedef typename IterateDomainArguments::extent_sizes_t extents_t;
        typedef typename IterateDomainArguments::esf_sequence_t esf_sequence_t;
        typedef typename IterateDomainArguments::backend_ids_t backend_ids_t;

        // insert the extent associated to a Cache into the map of <cache, extent>
        template < typename ExtendsMap, typename Cache >
        struct insert_extent_for_cache {
            GRIDTOOLS_STATIC_ASSERT((is_cache< Cache >::value), "ERROR");

            typedef typename cache_parameter< Cache >::type cache_arg_t;

            // given an Id within the sequence of esf and extents, extract the extent associated an inserted into
            // the map if the cache is used by the esf with that Id.
            template < typename ExtendsMap_, typename EsfIdx >
            struct insert_extent_for_cache_esf {
                GRIDTOOLS_STATIC_ASSERT((boost::mpl::size< extents_t >::value > EsfIdx::value), GT_INTERNAL_ERROR);
                GRIDTOOLS_STATIC_ASSERT((boost::mpl::size< esf_sequence_t >::value > EsfIdx::value), GT_INTERNAL_ERROR);

                typedef typename boost::mpl::at< esf_sequence_t, EsfIdx >::type esf_t;

                typedef typename boost::mpl::if_< boost::mpl::has_key< typename esf_t::args_with_extents, cache_arg_t >,
                    typename boost::mpl::at< typename esf_t::args_with_extents, cache_arg_t >::type,
                    typename grid_traits_from_id< backend_ids_t::s_grid_type_id >::null_extent_t >::type extent_t;

                GRIDTOOLS_STATIC_ASSERT((extent_t::iminus::value == 0 && extent_t::iplus::value == 0 &&
                                            extent_t::jminus::value == 0 && extent_t::jplus::value == 0),
                    "Error: K Caches can not have ij extent values");

                typedef typename boost::mpl::if_< boost::mpl::contains< typename esf_t::args_t, cache_arg_t >,
                    typename impl::update_extent_map< ExtendsMap_, extent_t, Cache, backend_ids_t >::type,
                    ExtendsMap_ >::type type;
            };

            // loop over all esfs and insert the extent associated to the cache into the map
            typedef typename boost::mpl::fold< boost::mpl::range_c< int, 0, boost::mpl::size< esf_sequence_t >::value >,
                ExtendsMap,
                insert_extent_for_cache_esf< boost::mpl::_1, boost::mpl::_2 > >::type type;
        };

        typedef typename boost::mpl::fold<
            typename boost::mpl::filter_view< cache_sequence_t, is_k_cache< boost::mpl::_ > >::type,
            boost::mpl::map0<>,
            insert_extent_for_cache< boost::mpl::_1, boost::mpl::_2 > >::type type;
    };

} // namespace gridtools
