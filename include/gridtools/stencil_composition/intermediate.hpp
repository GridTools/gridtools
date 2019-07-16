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

#include "../common/defs.hpp"
#include "../common/tuple_util.hpp"
#include "../meta.hpp"
#include "arg.hpp"
#include "caches/cache_traits.hpp"
#include "compute_extents_metafunctions.hpp"
#include "esf_metafunctions.hpp"
#include "extent.hpp"
#include "extract_placeholders.hpp"
#include "fused_mss_loop.hpp"
#include "grid.hpp"
#include "local_domain.hpp"
#include "mss_components.hpp"
#include "mss_components_metafunctions.hpp"
#include "tmp_storage.hpp"

namespace gridtools {
    namespace intermediate_impl_ {
        template <class Mss>
        using get_esfs = typename Mss::esf_sequence_t;

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

        template <class Arg>
        using to_arg_storage_pair = arg_storage_pair<Arg, typename Arg::data_store_t>;

        template <class MaxExtent, class Backend>
        struct get_tmp_arg_storage_pair_generator {
            template <class ArgStoragePair>
            struct apply {
                template <class Grid>
                ArgStoragePair operator()(Grid const &grid) const {
                    return {
                        tmp_storage::make_tmp_data_store<MaxExtent>(Backend{}, typename ArgStoragePair::arg_t{}, grid)};
                }
            };
        };

        template <class MaxExtent, class Backend, class Res, class Grid>
        Res make_tmp_arg_storage_pairs(Grid const &grid) {
            using generators =
                meta::transform<get_tmp_arg_storage_pair_generator<MaxExtent, Backend>::template apply, Res>;
            return tuple_util::generate<generators, Res>(grid);
        }

        template <class LocalDomains>
        struct data_store_set_f {
            LocalDomains &m_local_domains;
            template <class Plh, class DataStore>
            void operator()(DataStore &data_store) const {
                tuple_util::for_each([&](auto &dst) { dst.set_data_store(Plh{}, data_store); }, m_local_domains);
            }
        };

        template <bool IsStateful, class Backend, class Grid, class Msses>
        class intermediate {
            GT_STATIC_ASSERT(is_grid<Grid>::value, GT_INTERNAL_ERROR);

            using placeholders_t = extract_placeholders_from_msses<Msses>;
            using tmp_placeholders_t = meta::filter<is_tmp_arg, placeholders_t>;
            using non_tmp_placeholders_t = meta::filter<meta::not_<is_tmp_arg>::apply, placeholders_t>;

            using non_cached_tmp_placeholders_t =
                meta::dedup<meta::flatten<meta::transform<extract_non_cached_tmp_args_from_mss, Msses>>>;

            using tmp_arg_storage_pair_tuple_t = meta::transform<to_arg_storage_pair,
                meta::if_<needs_allocate_cached_tmp<Backend>, tmp_placeholders_t, non_cached_tmp_placeholders_t>>;

            using esfs_t = meta::flatten<meta::transform<get_esfs, Msses>>;

            // First we need to compute the association between placeholders and extents.
            // This information is needed to allocate temporaries, and to provide the extent information to the user.
            using extent_map_t = get_extent_map<esfs_t>;

            using mss_components_array_t = build_mss_components_array<Msses, extent_map_t, typename Grid::axis_type>;

            using max_extent_for_tmp_t = meta::rename<enclosing_extent,
                meta::transform<get_max_extent_for_tmp_from_mss_components, mss_components_array_t>>;

            template <class MssComponents>
            using get_local_domain_f =
                get_local_domain<Backend, typename MssComponents::mss_descriptor_t, max_extent_for_tmp_t, IsStateful>;

            // creates a tuple of local domains
            using local_domains_t = meta::transform<get_local_domain_f, mss_components_array_t>;

            // member fields

            Grid m_grid;

            /// tuple with temporary storages
            //
            tmp_arg_storage_pair_tuple_t m_tmp_arg_storage_pair_tuple;

            /// Here are local domains (structures with raw pointers for passing to backend.
            //
            mutable local_domains_t m_local_domains;

          public:
            intermediate(Grid const &grid)
                : m_grid(grid),
                  m_tmp_arg_storage_pair_tuple(
                      make_tmp_arg_storage_pairs<max_extent_for_tmp_t, Backend, tmp_arg_storage_pair_tuple_t>(grid)) {
                tuple_util::for_each_in_cartesian_product(
                    [&](auto const &src, auto &dst) { dst.set_data_store(src.arg(), src.m_value); },
                    m_tmp_arg_storage_pair_tuple,
                    m_local_domains);
            }

            template <class DataStoreMap>
            void operator()(DataStoreMap const &data_store_map) const {
                hymap::for_each(data_store_set_f<local_domains_t>{m_local_domains}, data_store_map);
                fused_mss_loop<mss_components_array_t>(Backend{}, m_local_domains, m_grid);
            }
        };
    } // namespace intermediate_impl_

    template <class Backend, class IsStateful, class Grid, class Msses>
    intermediate_impl_::intermediate<IsStateful::value, Backend, Grid, Msses> make_intermediate(
        Backend, IsStateful, Grid const &grid, Msses) {
        return {grid};
    }
} // namespace gridtools
