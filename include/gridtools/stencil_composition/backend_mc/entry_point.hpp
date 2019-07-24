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

#include "../../common/defs.hpp"
#include "../../common/hymap.hpp"
#include "../../common/tuple_util.hpp"
#include "../../meta.hpp"
#include "../../storage/sid.hpp"
#include "../arg.hpp"
#include "../caches/cache_traits.hpp"
#include "../compute_extents_metafunctions.hpp"
#include "../dim.hpp"
#include "../esf_metafunctions.hpp"
#include "../extent.hpp"
#include "../extract_placeholders.hpp"
#include "../grid.hpp"
#include "../mss_components.hpp"
#include "../mss_components_metafunctions.hpp"
#include "../positional.hpp"
#include "../sid/block.hpp"
#include "../sid/composite.hpp"
#include "../sid/concept.hpp"
#include "../sid/sid_shift_origin.hpp"
#include "execinfo_mc.hpp"
#include "fused_mss_loop_mc.hpp"
#include "tmp_storage_sid.hpp"

namespace gridtools {
    namespace mc {
        namespace _impl {
            /**
             * @brief Meta function to check if an MSS can be executed in parallel along k-axis.
             */
            template <typename Mss>
            using is_mss_kparallel = execute::is_parallel<typename Mss::execution_engine_t>;

            /**
             * @brief Meta function to check if all MSS in an MssComponents array can be executed in parallel along
             * k-axis.
             */
            template <typename Msses>
            using all_mss_kparallel = meta::all_of<is_mss_kparallel, Msses>;
        } // namespace _impl

        struct block_f {
            hymap::keys<dim::i, dim::j>::values<int_t, int_t> m_block_sizes;

            block_f(execinfo_mc const &info) : m_block_sizes{info.i_block_size(), info.j_block_size()} {}

            template <class Grid>
            block_f(Grid const &grid) : block_f(execinfo_mc(grid)) {}

            template <class T>
            auto operator()(T &&data_store) const {
                return sid::block(std::forward<T>(data_store), m_block_sizes);
            }
        };

        template <class Grid, class DataStoreMap>
        auto block(Grid const &grid, DataStoreMap data_stores) {
            return tuple_util::transform(block_f(grid), std::move(data_stores));
        }

        template <class Msses, class Grid, class Allocator>
        auto make_temporaries(Grid const &grid, Allocator &allocator) {
            using plhs_t = meta::filter<is_tmp_arg, extract_placeholders_from_msses<Msses>>;
            using extent_map_t = get_extent_map_from_msses<Msses>;
            execinfo_mc info(grid);
            return tuple_util::transform(
                [&allocator,
                    block_size = make_pos3(
                        (size_t)info.i_block_size(), (size_t)info.j_block_size(), (size_t)grid.k_total_length())](
                    auto plh) {
                    using plh_t = decltype(plh);
                    using data_t = typename plh_t::data_t;
                    using extent_t = lookup_extent_map<extent_map_t, plh_t>;
                    return make_tmp_storage_mc<data_t, extent_t, _impl::all_mss_kparallel<Msses>::value>(
                        allocator, block_size);
                },
                hymap::from_keys_values<plhs_t, plhs_t>());
        }

        template <class DataStores>
        struct at_key_f {
            DataStores &m_src;
            template <class Plh>
            GT_FORCE_INLINE decltype(auto) operator()(Plh) const {
                return at_key<Plh>(m_src);
            }
        };

        template <class Plhs, class Src>
        auto filter_map(Src &src) {
            return tuple_util::transform(at_key_f<Src>{src}, hymap::from_keys_values<Plhs, Plhs>());
        }

        template <class Mss, class DataStores, class Positionals>
        auto make_composite(DataStores &data_stores, Positionals positionals) {
            using plhs_t = extract_placeholders_from_mss<Mss>;

            return hymap::concat(sid::composite::keys<>::values<>(), filter_map<plhs_t>(data_stores), positionals);
        }

        template <class Count, class Stages>
        struct loop_interval {
            Count m_count;
            GT_FORCE_INLINE Count count() const { return m_count; }
            template <class Ptr, class Strides>
            GT_FORCE_INLINE void operator()(Ptr const &GT_RESTRICT ptr, Strides const &GT_RESTRICT strides) const {
                for_each<Stages>([&](auto stage) { stage(ptr, strides); });
            }
        };

        template <class Grid>
        struct adapt_interval_f {
            Grid const &m_grid;
            template <class Interval>
            auto operator()(Interval) const {
                auto count = m_grid.count(meta::first<Interval>(), meta::second<Interval>());
                return loop_interval<decltype(count), meta::third<Interval>>{count};
            }
        };

        template <class MssComponents, class Grid>
        auto make_loop_intervals(Grid const &grid) {
            return tuple_util::transform(
                adapt_interval_f<Grid>{grid}, meta::rename<tuple, typename MssComponents::loop_intervals_t>());
        }

        template <class MssComponentsList, class Grid, class DataStortes, class Positionals>
        auto make_mss_parallel_loops(Grid const &grid, DataStortes &data_stores, Positionals positionals) {
            return tuple_util::transform(
                [&](auto mss_components) {
                    using mss_components_t = decltype(mss_components);
                    using extent_t = get_extent_from_loop_intervals<typename mss_components_t::loop_intervals_t>;

                    auto composite =
                        make_composite<typename mss_components_t::mss_descriptor_t>(data_stores, positionals);
                    auto loop_intervals = make_loop_intervals<mss_components_t>(grid);
                    return make_mss_parallel_loop<extent_t>(std::move(composite), std::move(loop_intervals));
                },
                MssComponentsList());
        }

        template <class MssComponentsList, class NeedPositionals, class Grid, class DataStortes, class Positionals>
        auto make_mss_serial_loops(Grid const &grid, DataStortes &data_stores, Positionals positionals) {
            return tuple_util::transform(
                [&](auto mss_components) {
                    using mss_components_t = decltype(mss_components);
                    using extent_t = get_extent_from_loop_intervals<typename mss_components_t::loop_intervals_t>;
                    using execution_t = typename mss_components_t::execution_engine_t;

                    auto composite =
                        make_composite<typename mss_components_t::mss_descriptor_t>(data_stores, positionals);
                    auto loop_intervals = make_loop_intervals<mss_components_t>(grid);

                    return make_mss_serial_loop<extent_t>(grid.k_total_length(),
                        execute::step<execution_t>,
                        std::move(composite),
                        std::move(loop_intervals));
                },
                MssComponentsList());
        }

        template <class MssComponentsList,
            class Grid,
            class DataStortes,
            class Positionals,
            std::enable_if_t<_impl::all_mss_kparallel<MssComponentsList>::value, int> = 0>
        void run_loops(Grid const &grid, DataStortes data_stores, Positionals positionals) {
            exec_parallel(execinfo_mc(grid),
                grid.k_total_length(),
                make_mss_parallel_loops<MssComponentsList>(grid, data_stores, positionals));
        }

        template <class MssComponentsList,
            class NeedPositionals,
            class Grid,
            class DataStortes,
            class Positionals,
            std::enable_if_t<!_impl::all_mss_kparallel<MssComponentsList>::value, int> = 0>
        void run_loops(Grid const &grid, DataStortes data_stores, Positionals positionals) {
            exec_serial(execinfo_mc(grid), make_mss_serial_loops<MssComponentsList>(grid, data_stores, positionals));
        }

        template <class Grid, class Msses, class DataStores, class Positionals>
        void gridtools_backend_entry_point(
            backend, Msses, Grid const &grid, DataStores external_data_stores, Positionals positionals) {
            tmp_allocator_mc alloc;
            auto data_stores =
                hymap::concat(block(grid, std::move(external_data_stores)), make_temporaries<Msses>(grid, alloc));

            using mss_components_array_t = build_mss_components_array<Msses, typename Grid::interval_t>;

            run_loops<mss_components_array_t>(grid, std::move(data_stores), block(grid, positionals));
        }
    } // namespace mc
} // namespace gridtools
