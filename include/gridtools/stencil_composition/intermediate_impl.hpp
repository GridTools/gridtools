/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <type_traits>

#include "../common/functional.hpp"
#include "../common/generic_metafunctions/for_each.hpp"
#include "../common/hymap.hpp"
#include "../common/tuple_util.hpp"
#include "../storage/sid.hpp"
#include "dim.hpp"
#include "esf_metafunctions.hpp"
#include "extract_placeholders.hpp"
#include "local_domain.hpp"
#include "mss_components.hpp"
#include "sid/concept.hpp"
#include "tmp_storage.hpp"

namespace gridtools {
    namespace _impl {
        template <class Mss>
        struct non_cached_tmp_f {
            using local_caches_t = meta::filter<is_local_cache, typename Mss::cache_sequence_t>;
            using cached_args_t = meta::transform<cache_parameter, local_caches_t>;

            template <class Arg>
            using apply = bool_constant<is_tmp_arg<Arg>::value && !meta::st_contains<cached_args_t, Arg>::value>;
        };

        template <class Mss>
        using extract_non_cached_tmp_args_from_mss =
            meta::filter<non_cached_tmp_f<Mss>::template apply, extract_placeholders_from_mss<Mss>>;

        template <class Msses, class ArgLists = meta::transform<extract_non_cached_tmp_args_from_mss, Msses>>
        using extract_non_cached_tmp_args_from_msses = meta::dedup<meta::flatten<ArgLists>>;

        template <class MaxExtent, class Backend>
        struct get_tmp_arg_storage_pair_generator {
            template <class ArgStoragePair>
            struct generator {
                template <class Grid>
                ArgStoragePair operator()(Grid const &grid) const {
                    return tmp_storage::make_tmp_data_store<MaxExtent>(
                        Backend{}, typename ArgStoragePair::arg_t{}, grid);
                }
            };

            template <class T>
            using apply = generator<T>;
        };

        template <class MaxExtent, class Backend, class Res, class Grid>
        Res make_tmp_arg_storage_pairs(Grid const &grid) {
            using generators =
                meta::transform<get_tmp_arg_storage_pair_generator<MaxExtent, Backend>::template apply, Res>;
            return tuple_util::generate<generators, Res>(grid);
        }

        template <class MssComponentsList,
            class Extents = meta::transform<get_max_extent_for_tmp_from_mss_components, MssComponentsList>>
        using get_max_extent_for_tmp = meta::rename<enclosing_extent, Extents>;

        template <class Mss>
        using rw_args_from_mss = compute_readwrite_args<unwrap_independent<typename Mss::esf_sequence_t>>;

        template <class Msses,
            class RwArgsLists = meta::transform<rw_args_from_mss, Msses>,
            class RawRwArgs = meta::flatten<RwArgsLists>>
        using all_rw_args = meta::dedup<RawRwArgs>;

    } // namespace _impl
} // namespace gridtools
