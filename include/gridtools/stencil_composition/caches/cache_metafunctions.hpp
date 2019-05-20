/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
/**
   @file
   @brief File containing metafunctions and traits common to all cache classes
*/

#pragma once

#include <boost/fusion/include/pair.hpp>

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
    GT_META_DEFINE_ALIAS(ij_cache_args, meta::transform, (cache_parameter, ij_caches<Caches>));

    template <class Caches>
    GT_META_DEFINE_ALIAS(k_caches, meta::filter, (is_k_cache, Caches));

    template <class Caches>
    GT_META_DEFINE_ALIAS(k_cache_args, meta::transform, (cache_parameter, k_caches<Caches>));

    template <class Caches, class Esfs>
    struct get_k_cache_storage_tuple {
        GT_STATIC_ASSERT((meta::all_of<is_cache, Caches>::value), GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT((meta::all_of<is_esf_descriptor, Esfs>::value), GT_INTERNAL_ERROR);

        template <class Cache, class Arg = typename Cache::arg_t>
        GT_META_DEFINE_ALIAS(make_item,
            meta::id,
            (boost::fusion::pair<Arg,
                typename make_k_cache_storage<Arg, extract_k_extent_for_cache<Arg, Esfs>>::type>));

        using type = meta::transform<make_item, k_caches<Caches>>;
    };

    template <class Caches, class MaxExtent, int_t ITile, int_t JTile>
    struct get_ij_cache_storage_tuple {
        GT_STATIC_ASSERT((meta::all_of<is_cache, Caches>::value), GT_INTERNAL_ERROR);

        template <class Cache, class Arg = typename Cache::arg_t>
        GT_META_DEFINE_ALIAS(make_item,
            meta::id,
            (boost::fusion::pair<Arg, typename make_ij_cache_storage<Arg, ITile, JTile, MaxExtent>::type>));

        using type = meta::transform<make_item, ij_caches<Caches>>;
    };
} // namespace gridtools
