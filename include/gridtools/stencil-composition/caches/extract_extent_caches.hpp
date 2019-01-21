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

#include <boost/mpl/at.hpp>
#include <boost/mpl/contains.hpp>
#include <boost/mpl/erase_key.hpp>
#include <boost/mpl/filter_view.hpp>
#include <boost/mpl/fold.hpp>
#include <boost/mpl/has_key.hpp>
#include <boost/mpl/if.hpp>
#include <boost/mpl/insert.hpp>
#include <boost/mpl/map.hpp>
#include <boost/mpl/pair.hpp>
#include <boost/mpl/range_c.hpp>
#include <boost/mpl/size.hpp>

#include "../../common/defs.hpp"
#include "../../common/generic_metafunctions/copy_into_variadic.hpp"
#include "../../meta.hpp"
#include "../extent.hpp"
#include "./cache_traits.hpp"

namespace gridtools {

    namespace impl {
        // update the entry associated to a cache within the map with a new extent.
        // if the key exist we compute and insert the enclosing extent, otherwise we just
        // insert the extent into a new entry of the map of <cache, extent>
        template <typename ExtentsMap_, typename Extent, typename Cache>
        struct update_extent_map {
            GRIDTOOLS_STATIC_ASSERT((is_extent<Extent>::value), GT_INTERNAL_ERROR);
            GRIDTOOLS_STATIC_ASSERT((is_cache<Cache>::value), GT_INTERNAL_ERROR);
            typedef typename boost::mpl::if_<boost::mpl::has_key<ExtentsMap_, Cache>,
                typename boost::mpl::at<ExtentsMap_, Cache>::type,
                extent<>>::type default_extent_t;

            typedef typename boost::mpl::insert<typename boost::mpl::erase_key<ExtentsMap_, Cache>::type,
                boost::mpl::pair<Cache, GT_META_CALL(enclosing_extent, (default_extent_t, Extent))>>::type type;
        };
    } // namespace impl

    /**
     * @struct extract_ij_extents_for_caches
     * metafunction that extracts the extents associated to each cache of the sequence of caches provided by the user.
     * The extent is determined as the enclosing extent of all the extents of esfs that use the cache.
     * It is used in order to allocate enough memory for each cache storage.
     * @tparam IterateDomainArguments iterate domain arguments type containing sequences of caches, esfs and extents
     * @return map<cache,extent>
     */
    template <typename IterateDomainArguments>
    struct extract_ij_extents_for_caches {

        typedef typename IterateDomainArguments::cache_sequence_t cache_sequence_t;
        typedef typename IterateDomainArguments::extent_sizes_t extents_t;
        typedef typename IterateDomainArguments::esf_sequence_t esf_sequence_t;

        // metafunction to extract the extent of an ESF where an Arg is used.
        // If Arg is not used by the ESF, a null extent is returned, otherwise
        // the extent of the ESF (where the non ij extents are nullified) is returned
        template <typename ESFIdx, typename Arg>
        struct esf_extent_of_arg {
            using esf_t = typename boost::mpl::at<esf_sequence_t, ESFIdx>::type;

            using type = typename boost::mpl::if_<boost::mpl::has_key<typename esf_t::args_with_extents, Arg>,
                typename boost::mpl::at<extents_t, ESFIdx>::type,
                extent<>>::type;
        };

        // insert the extent associated to a Cache into the map of <cache, extent>
        template <typename ExtentsMap, typename Cache>
        struct insert_extent_for_cache {
            GRIDTOOLS_STATIC_ASSERT((is_cache<Cache>::value), GT_INTERNAL_ERROR);

            // given an Id within the sequence of esf and extents, extract the extent associated an inserted into
            // the map if the cache is used by the esf with that Id.
            template <typename ExtentsMap_, typename EsfIdx>
            struct insert_extent_for_cache_esf {
                GRIDTOOLS_STATIC_ASSERT(boost::mpl::size<extents_t>::value > EsfIdx::value, GT_INTERNAL_ERROR);
                GRIDTOOLS_STATIC_ASSERT(boost::mpl::size<esf_sequence_t>::value > EsfIdx::value, GT_INTERNAL_ERROR);

                using cache_arg_t = GT_META_CALL(cache_parameter, Cache);
                typedef typename boost::mpl::at<esf_sequence_t, EsfIdx>::type esf_t;

                // only extract the extent of the esf and push it into the cache if the arg of the cache is used in the
                // extent (note non ij extents are nullified for the ij caches)
                using extent_t = typename esf_extent_of_arg<EsfIdx, cache_arg_t>::type;

                typedef typename boost::mpl::if_<boost::mpl::contains<typename esf_t::args_t, cache_arg_t>,
                    typename impl::update_extent_map<ExtentsMap_, extent_t, Cache>::type,
                    ExtentsMap_>::type type;
            };

            // loop over all esfs and insert the extent associated to the cache into the map
            typedef typename boost::mpl::fold<boost::mpl::range_c<int, 0, boost::mpl::size<esf_sequence_t>::value>,
                ExtentsMap,
                insert_extent_for_cache_esf<boost::mpl::_1, boost::mpl::_2>>::type type;
        };

        typedef typename boost::mpl::fold<cache_sequence_t,
            boost::mpl::map0<>,
            insert_extent_for_cache<boost::mpl::_1, boost::mpl::_2>>::type type;
    };

    namespace extract_extent_caches_impl_ {
        template <class Arg>
        struct arg_extent_from_esf {

            template <class EsfArg, class Accessor>
            GT_META_DEFINE_ALIAS(
                get_extent, meta::if_, (std::is_same<Arg, EsfArg>, typename Accessor::extent_t, extent<>));

            template <class Esf,
                class Args = typename Esf::args_t,
                class Accessors = copy_into_variadic<typename esf_arg_list<Esf>::type, meta::list<>>,
                class Extents = GT_META_CALL(meta::transform, (get_extent, Args, Accessors))>
            GT_META_DEFINE_ALIAS(apply, meta::rename, (enclosing_extent, Extents));
        };
    } // namespace extract_extent_caches_impl_

    template <class Arg,
        class Esfs,
        class Extents = GT_META_CALL(
            meta::transform, (extract_extent_caches_impl_::arg_extent_from_esf<Arg>::template apply, Esfs))>
    GT_META_DEFINE_ALIAS(extract_k_extent_for_cache, meta::rename, (enclosing_extent, Extents));

} // namespace gridtools
