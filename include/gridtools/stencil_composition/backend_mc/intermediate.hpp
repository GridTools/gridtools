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
        template <class Mss>
        using get_esfs = typename Mss::esf_sequence_t;

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

        template <class Grid>
        auto make_block_map(Grid const &grid) {
            execinfo_mc info(grid);
            return tuple_util::make<hymap::keys<dim::i, dim::j>::values>(info.i_block_size(), info.j_block_size());
        }

        template <class Grid, class DataStoreMap>
        auto shift_origin_and_block(Grid const &grid, DataStoreMap data_stores) {
            return tuple_util::transform(
                [offsets = tuple_util::make<hymap::keys<dim::i, dim::j, dim::k>::values>(
                     grid.i_low_bound(), grid.j_low_bound(), grid.k_min()),
                    block_map = make_block_map(grid)](
                    auto &src) { return sid::block(sid::shift_sid_origin(std::ref(src), offsets), block_map); },
                std::move(data_stores));
        }

        template <class Msses, class Grid, class Allocator>
        auto make_temporaries(Grid const &grid, Allocator &allocator) {
            using plhs_t = meta::filter<is_tmp_arg, extract_placeholders_from_msses<Msses>>;
            using extent_map_t = get_extent_map<meta::flatten<meta::transform<get_esfs, Msses>>>;
            execinfo_mc info(grid);
            return tuple_util::transform(
                [&allocator,
                    block_size = make_pos3(
                        (size_t)info.i_block_size(), (size_t)info.j_block_size(), (size_t)grid.k_total_length())](
                    auto plh) {
                    using plh_t = decltype(plh);
                    using data_t = typename plh_t::data_store_t::data_t;
                    using extent_t = lookup_extent_map<extent_map_t, plh_t>;
                    return make_tmp_storage_mc<data_t, extent_t, _impl::all_mss_kparallel<Msses>::value>(
                        allocator, block_size);
                },
                hymap::from_keys_values<plhs_t, plhs_t>());
        }

        template <class Grid>
        auto make_positionals(meta::list<dim::i, dim::j, dim::k>, Grid const &grid) {
            using positionals_t = std::tuple<positional<dim::i>, positional<dim::j>, positional<dim::k>>;
            return tuple_util::transform(
                [block_map = make_block_map(grid)](auto pos) { return sid::block(pos, block_map); },
                hymap::convert_to<hymap::keys, positionals_t>(
                    positionals_t{grid.i_low_bound(), grid.j_low_bound(), grid.k_min()}));
        }

        template <class Grid>
        tuple<> make_positionals(meta::list<>, Grid const &) {
            return {};
        }

        template <class Plhs, class Src>
        auto filter_map(Src &src) {
            return tuple_util::transform([&](auto plh) -> decltype(auto) { return at_key<decltype(plh)>(src); },
                hymap::from_keys_values<Plhs, Plhs>());
        }

        template <class Mss, class NeedPositionals, class Grid, class DataStores>
        auto make_composite(Grid grid, DataStores &data_stores) {
            using positionals_t = meta::if_<NeedPositionals, meta::list<dim::i, dim::j, dim::k>, meta::list<>>;

            using plhs_t = extract_placeholders_from_mss<Mss>;

            return hymap::concat(sid::composite::keys<>::values<>(),
                filter_map<plhs_t>(data_stores),
                make_positionals(positionals_t(), grid));
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

        template <class MssComponents, class Grid>
        auto make_loop_intervals(Grid const &grid) {
            return tuple_util::transform(
                [&](auto interval) {
                    using interval_t = decltype(interval);
                    using stages_t = meta::third<interval_t>;
                    auto count = grid.count(meta::first<interval_t>{}, meta::second<interval_t>{});
                    return loop_interval<decltype(count), meta::third<interval_t>>{count};
                },
                meta::rename<tuple, typename MssComponents::loop_intervals_t>());
        }

        template <class MssComponentsList, class NeedPositionals, class Grid, class DataStortes>
        auto make_mss_parallel_loops(Grid const &grid, DataStortes &data_stores) {
            return tuple_util::transform(
                [&](auto mss_components) {
                    using mss_components_t = decltype(mss_components);
                    using extent_t = get_extent_from_loop_intervals<typename mss_components_t::loop_intervals_t>;

                    auto composite =
                        make_composite<typename mss_components_t::mss_descriptor_t, NeedPositionals>(grid, data_stores);
                    auto loop_intervals = make_loop_intervals<mss_components_t>(grid);
                    return make_mss_parallel_loop<extent_t>(std::move(composite), std::move(loop_intervals));
                },
                MssComponentsList());
        }

        template <class MssComponentsList, class NeedPositionals, class Grid, class DataStortes>
        auto make_mss_serial_loops(Grid const &grid, DataStortes &data_stores) {
            return tuple_util::transform(
                [&](auto mss_components) {
                    using mss_components_t = decltype(mss_components);
                    using extent_t = get_extent_from_loop_intervals<typename mss_components_t::loop_intervals_t>;
                    using execution_t = typename mss_components_t::execution_engine_t;

                    auto composite =
                        make_composite<typename mss_components_t::mss_descriptor_t, NeedPositionals>(grid, data_stores);
                    auto loop_intervals = make_loop_intervals<mss_components_t>(grid);

                    return make_mss_serial_loop<extent_t>(grid.k_total_length(),
                        execute::step<execution_t>,
                        std::move(composite),
                        std::move(loop_intervals));
                },
                MssComponentsList());
        }

        template <class MssComponentsList,
            class NeedPositionals,
            class Grid,
            class DataStortes,
            std::enable_if_t<_impl::all_mss_kparallel<MssComponentsList>::value, int> = 0>
        void run_loops(Grid const &grid, DataStortes data_stores) {
            exec_parallel(execinfo_mc(grid),
                grid.k_total_length(),
                make_mss_parallel_loops<MssComponentsList, NeedPositionals>(grid, data_stores));
        }

        template <class MssComponentsList,
            class NeedPositionals,
            class Grid,
            class DataStortes,
            std::enable_if_t<!_impl::all_mss_kparallel<MssComponentsList>::value, int> = 0>
        void run_loops(Grid const &grid, DataStortes data_stores) {
            exec_serial(
                execinfo_mc(grid), make_mss_serial_loops<MssComponentsList, NeedPositionals>(grid, data_stores));
        }

        template <class NeedPositionals, class Grid, class Msses>
        auto make_intermediate(backend, NeedPositionals, Grid const &grid, Msses) {
            return [grid](auto external_data_stores) {
                tmp_allocator_mc alloc;
                auto data_stores = hymap::concat(shift_origin_and_block(grid, std::move(external_data_stores)),
                    make_temporaries<Msses>(grid, alloc));

                using esfs_t = meta::flatten<meta::transform<get_esfs, Msses>>;
                using extent_map_t = get_extent_map<esfs_t>;
                using mss_components_array_t =
                    build_mss_components_array<Msses, extent_map_t, typename Grid::axis_type>;

                run_loops<mss_components_array_t, NeedPositionals>(grid, std::move(data_stores));
            };
        }
    } // namespace mc
} // namespace gridtools
