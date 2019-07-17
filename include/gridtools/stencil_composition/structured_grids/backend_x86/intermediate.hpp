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

#include "../../../common/defs.hpp"
#include "../../../common/tuple_util.hpp"
#include "../../../meta.hpp"
#include "../../arg.hpp"
#include "../../compute_extents_metafunctions.hpp"
#include "../../esf_metafunctions.hpp"
#include "../../extent.hpp"
#include "../../extract_placeholders.hpp"
#include "../../grid.hpp"
#include "../../mss_components.hpp"
#include "../../mss_components_metafunctions.hpp"
#include "local_domain.hpp"
#include "mss_loop_x86.hpp"

namespace gridtools {
    namespace x86 {
        struct execution_info_x86 {
            int_t bi, bj;
        };

        template <typename MssComponentsArray, typename LocalDomains, typename Grid, typename ExecutionInfo>
        struct mss_functor {
            GT_STATIC_ASSERT((meta::all_of<is_mss_components, MssComponentsArray>::value), GT_INTERNAL_ERROR);
            GT_STATIC_ASSERT(is_grid<Grid>::value, GT_INTERNAL_ERROR);

            LocalDomains const &m_local_domains;
            Grid const &m_grid;
            ExecutionInfo m_execution_info;

            /**
             * \brief given the index of a functor in the functors list ,it calls a kernel on the GPU executing the
             * operations defined on that functor.
             */
            template <typename Index>
            void operator()(Index) const {
                GT_STATIC_ASSERT(Index::value < meta::length<MssComponentsArray>::value, GT_INTERNAL_ERROR);
                using mss_components_t = meta::at<MssComponentsArray, Index>;

                mss_loop<typename mss_components_t::loop_intervals_t, typename mss_components_t::execution_engine_t>(
                    std::get<Index::value>(m_local_domains), m_grid, m_execution_info);
            }
        };

        template <class MssComponentsArray, class LocalDomains, class Grid, class ExecutionInfo>
        void run_mss_functors(
            LocalDomains const &local_domains, Grid const &grid, ExecutionInfo const &execution_info) {
            for_each<meta::make_indices_for<MssComponentsArray>>(
                mss_functor<MssComponentsArray, LocalDomains, Grid, ExecutionInfo>{
                    local_domains, grid, execution_info});
        }

        template <class MssComponents, class LocalDomainListArray, class Grid>
        void fused_mss_loop(LocalDomainListArray const &local_domain_lists, const Grid &grid) {
            GT_STATIC_ASSERT((meta::all_of<is_mss_components, MssComponents>::value), GT_INTERNAL_ERROR);
            GT_STATIC_ASSERT(is_grid<Grid>::value, GT_INTERNAL_ERROR);
            int_t n = grid.i_size() - 1;
            int_t m = grid.j_size() - 1;

            int_t NBI = n / GT_DEFAULT_TILE_I;
            int_t NBJ = m / GT_DEFAULT_TILE_J;

#pragma omp parallel
            {
#pragma omp for nowait
                for (int_t bi = 0; bi <= NBI; ++bi) {
                    for (int_t bj = 0; bj <= NBJ; ++bj) {
                        run_mss_functors<MssComponents>(local_domain_lists, grid, execution_info_x86{bi, bj});
                    }
                }
            }
        }

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

        template <class StorageInfo>
        int_t get_i_size(int_t block_size) {
            static constexpr auto halo = StorageInfo::halo_t::template at<dim::i::value>();
            return (block_size + 2 * halo) * omp_get_max_threads();
        }

        template <class StorageInfo>
        int_t get_j_size(int_t block_size) {
            static constexpr auto halo = StorageInfo::halo_t::template at<dim::j::value>();
            return block_size + 2 * halo;
        }

        template <class ArgTag, class DataStore, int_t I, uint_t NColors, class Grid>
        DataStore make_tmp_data_store(plh<ArgTag, DataStore, location_type<I, NColors>, true>, Grid const &grid) {
            GT_STATIC_ASSERT(is_grid<Grid>::value, GT_INTERNAL_ERROR);
            using storage_info_t = typename DataStore::storage_info_t;
            return {storage_info_t(get_i_size<storage_info_t>(GT_DEFAULT_TILE_I),
                get_j_size<storage_info_t>(GT_DEFAULT_TILE_J),
                grid.k_total_length())};
        }

        template <class ArgStoragePair>
        struct arg_storage_pair_generator {
            template <class Grid>
            ArgStoragePair operator()(Grid const &grid) const {
                return {make_tmp_data_store(typename ArgStoragePair::arg_t(), grid)};
            }
        };

        template <class Res, class Grid>
        Res make_tmp_arg_storage_pairs(Grid const &grid) {
            using generators = meta::transform<arg_storage_pair_generator, Res>;
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

        template <class IsStateful>
        struct get_local_domain_f {
            template <class MssComponents>
            using apply = get_local_domain<typename MssComponents::mss_descriptor_t, IsStateful::value>;
        };

        template <class IsStateful, class Grid, class Msses>
        auto make_intermediate(backend, IsStateful, Grid const &grid, Msses) {
            return [grid](auto data_store_map) {
                using placeholders_t = extract_placeholders_from_msses<Msses>;
                using tmp_placeholders_t = meta::filter<is_tmp_arg, placeholders_t>;
                using non_tmp_placeholders_t = meta::filter<meta::not_<is_tmp_arg>::apply, placeholders_t>;

                using tmp_arg_storage_pair_tuple_t = meta::transform<to_arg_storage_pair, tmp_placeholders_t>;

                using esfs_t = meta::flatten<meta::transform<get_esfs, Msses>>;

                using extent_map_t = get_extent_map<esfs_t>;

                using mss_components_array_t =
                    build_mss_components_array<Msses, extent_map_t, typename Grid::axis_type>;

                using local_domains_t =
                    meta::transform<get_local_domain_f<IsStateful>::template apply, mss_components_array_t>;

                local_domains_t local_domains;
                auto tmp_arg_storage_pair_tuple = make_tmp_arg_storage_pairs<tmp_arg_storage_pair_tuple_t>(grid);

                tuple_util::for_each_in_cartesian_product(
                    [&](auto const &src, auto &dst) { dst.set_data_store(src.arg(), src.m_value); },
                    tmp_arg_storage_pair_tuple,
                    local_domains);

                hymap::for_each(data_store_set_f<local_domains_t>{local_domains}, data_store_map);
                fused_mss_loop<mss_components_array_t>(local_domains, grid);
            };
        }
    } // namespace x86
} // namespace gridtools
