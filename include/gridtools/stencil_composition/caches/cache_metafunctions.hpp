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

#include "../../common/defs.hpp"
#include "../../common/hymap.hpp"
#include "../../meta.hpp"
#include "../esf_fwd.hpp"
#include "./cache.hpp"
#include "./cache_storage.hpp"
#include "./cache_traits.hpp"
#include "./extract_extent_caches.hpp"

namespace gridtools {

    template <class Caches>
    using ij_caches = meta::filter<is_ij_cache, Caches>;

    template <class Caches>
    using ij_cache_args = meta::transform<cache_parameter, ij_caches<Caches>>;

    template <class Caches>
    using k_caches = meta::filter<is_k_cache, Caches>;

    template <class Caches>
    using k_cache_args = meta::transform<cache_parameter, k_caches<Caches>>;

    namespace cache_metafunctions_impl_ {
        template <class MaxExtent, int_t ITile, int_t JTile>
        struct make_ij_cache_storage_f {
            template <class Arg>
            using apply = typename make_ij_cache_storage<Arg, ITile, JTile, MaxExtent>::type;
        };

        template <class Esfs>
        struct make_k_cache_storage_f {
            template <class Arg>
            using apply = typename make_k_cache_storage<Arg, extract_k_extent_for_cache<Arg, Esfs>>::type;
        };
    } // namespace cache_metafunctions_impl_

    template <class Caches, class MaxExtent, int_t ITile, int_t JTile, class Args = ij_cache_args<Caches>>
    using get_ij_cache_storage_map = hymap::from_keys_values<Args,
        meta::transform<cache_metafunctions_impl_::make_ij_cache_storage_f<MaxExtent, ITile, JTile>::template apply,
            Args>>;

    template <class Caches, class Esfs, class Args = k_cache_args<Caches>>
    using get_k_cache_storage_map = hymap::from_keys_values<Args,
        meta::transform<cache_metafunctions_impl_::make_k_cache_storage_f<Esfs>::template apply, Args>>;

} // namespace gridtools
