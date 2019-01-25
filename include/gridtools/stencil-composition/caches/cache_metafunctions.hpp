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
/**
   @file
   @brief File containing metafunctions and traits common to all cache classes
*/

#pragma once

#include <boost/fusion/include/pair.hpp>
#include <boost/mpl/at.hpp>

#include "../../common/defs.hpp"
#include "../../meta.hpp"
#include "../esf_fwd.hpp"
#include "./cache.hpp"
#include "./cache_storage.hpp"
#include "./cache_traits.hpp"
#include "./extract_extent_caches.hpp"

namespace gridtools {

    template <class Caches>
    GT_META_DEFINE_ALIAS(ij_caches, meta::filter, (is_ij_cache, Caches));

    template <class Caches>
    GT_META_DEFINE_ALIAS(ij_cache_args, meta::transform, (cache_parameter, GT_META_CALL(ij_caches, Caches)));

    template <class Caches>
    GT_META_DEFINE_ALIAS(k_caches, meta::filter, (is_k_cache, Caches));

    template <class Caches>
    GT_META_DEFINE_ALIAS(k_cache_args, meta::transform, (cache_parameter, GT_META_CALL(k_caches, Caches)));

    template <class Caches, class Esfs>
    struct get_k_cache_storage_tuple {
        GRIDTOOLS_STATIC_ASSERT((meta::all_of<is_cache, Caches>::value), GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT((meta::all_of<is_esf_descriptor, Esfs>::value), GT_INTERNAL_ERROR);

        template <class Cache,
            class Arg = typename Cache::arg_t,
            class Extent = GT_META_CALL(extract_k_extent_for_cache, (Arg, Esfs))>
        GT_META_DEFINE_ALIAS(
            make_item, meta::id, (boost::fusion::pair<Arg, typename make_k_cache_storage<Arg, Extent>::type>));

        using type = GT_META_CALL(meta::transform, (make_item, GT_META_CALL(k_caches, Caches)));
    };

    template <class Caches, class MaxExtent, int_t ITile, int_t JTile>
    struct get_ij_cache_storage_tuple {
        GRIDTOOLS_STATIC_ASSERT((meta::all_of<is_cache, Caches>::value), GT_INTERNAL_ERROR);

        template <class Cache, class Arg = typename Cache::arg_t>
        GT_META_DEFINE_ALIAS(make_item,
            meta::id,
            (boost::fusion::pair<Arg, typename make_ij_cache_storage<Arg, ITile, JTile, MaxExtent>::type>));

        using type = GT_META_CALL(meta::transform, (make_item, GT_META_CALL(ij_caches, Caches)));
    };
} // namespace gridtools
